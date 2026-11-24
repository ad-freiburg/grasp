import json
import logging
from pathlib import Path

import re
import unicodedata

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.utils import derive_label_from_iri, camel_case_split
from grasp.tasks.entities import Entity
from grasp.tasks.om import AlignmentTaskInput, PotentialCorrespondences
from grasp.sparql.types import ObjType


def normalize_name(name: str) -> str:
    """
    Normalizes a label/alias for string-matching comparison: splits
    camelCase, treats "_"/"-" as word separators, collapses whitespace, and
    casefolds - so e.g. "hasAuthor", "has_author" and "Has Author" all
    normalize to the same string and therefore match each other.

    >>> normalize_name("hasAuthor")
    'has author'
    >>> normalize_name("Program_Committee-Chair")
    'program committee chair'
    >>> normalize_name("  extra   spaces  ")
    'extra spaces'
    """
    _WHITESPACE_RE = re.compile(r"\s+")

    name = unicodedata.normalize("NFKC", name)
    name = name.replace("_", " ").replace("-", " ")
    name = camel_case_split(name)
    name = _WHITESPACE_RE.sub(" ", name).strip()
    return name.casefold()


def entity_names(entity: Entity) -> list[str]:
    """
    All of `entity`'s own names for string-matching purposes: its label (if
    any) followed by its aliases. Returns an empty list if the entity has
    neither.

    >>> entity_names(Entity(identifier="http://cmt#Paper", entity="cmt:Paper", label="Paper", aliases=["Publication"]))
    ['Paper', 'Publication']
    >>> entity_names(Entity(identifier="http://cmt#Paper", entity="cmt:Paper"))
    []
    """
    names = [entity.label] if entity.label else []
    if entity.aliases:
        names.extend(entity.aliases)
    return names


def _entities_from_ids(
    kg_manager: KgManager, ids: list[str], info: dict[str, dict]
) -> dict[str, Entity]:
    """
    Builds an `Entity` per id, preferring the label from `info` and falling
    back to one derived from the IRI itself (e.g. via camelCase-splitting
    the local name) when `info` has none for that id.

    >>> class FakeManager:
    ...     prefixes = {}
    ...     def format_iri(self, iri):
    ...         return iri
    >>> result = _entities_from_ids(
    ...     FakeManager(),
    ...     ["http://cmt#Paper", "http://cmt#hasAuthor"],
    ...     {"http://cmt#Paper": {"label": "Paper", "alias": ["Publication"]}},
    ... )
    >>> result["http://cmt#Paper"].label, result["http://cmt#Paper"].aliases
    ('Paper', ['Publication'])
    >>> result["http://cmt#hasAuthor"].label  # no info entry -> derived from the IRI
    'has Author'
    """
    result: dict[str, Entity] = {}
    for iri in ids:
        entry = info.get(iri, {})
        label = entry.get("label") or derive_label_from_iri(iri, kg_manager.prefixes)

        result[iri] = Entity(
            identifier=iri,
            entity=kg_manager.format_iri(iri),
            label=label,
            aliases=entry.get("alias", []),
            infos=entry.get("other", []),
        )

    return result


def _filter_own_namespace(identifiers: list[str], kg_manager: KgManager) -> list[str]:
    """
    Keeps only identifiers within the KG's own namespace(s).

    >>> class FakeManager:
    ...     kg_prefixes = {"cmt": "http://cmt#"}
    >>> ids = ["http://cmt#Paper", "http://conference#Paper", "http://cmt#Review"]
    >>> _filter_own_namespace(ids, FakeManager())
    ['http://cmt#Paper', 'http://cmt#Review']

    No declared namespace at all means this fails open (returns everything
    unfiltered) rather than risk silently discarding every identifier:

    >>> class FakeManagerNoNamespace:
    ...     kg_prefixes = {}
    >>> _filter_own_namespace(ids, FakeManagerNoNamespace()) == ids
    True
    """
    own_namespaces = tuple(kg_manager.kg_prefixes.values())
    if not own_namespaces:
        # no declared KG-specific namespace to filter against - fail open
        # rather than risk filtering out everything
        return identifiers
    return [i for i in identifiers if i.startswith(own_namespaces)]


def retrieve_entities_and_properties(kg_manager: KgManager) -> dict[str, Entity]:
    """
    Retrieves all classes and properties of a KG using GRASP's own, already
    built entities/properties indices.
    Does not cover individuals/instances.

    An identifier that appears in *both* indices (the entities index may
    already include properties, depending on how the KG was set up) is only
    queried/enriched once, via the properties index's own info lookup:

    >>> from grasp.sparql.types import ObjType
    >>> class FakeManager:
    ...     prefixes = {}
    ...     kg_prefixes = {"cmt": "http://cmt#"}
    ...     def format_iri(self, iri):
    ...         return iri
    ...     def get_data(self, name):
    ...         if name == ObjType.ENTITY.index_name:
    ...             return [("http://cmt#Paper", None), ("http://cmt#hasAuthor", None)]
    ...         return [("http://cmt#hasAuthor", None)]
    ...     def get_info_for_identifiers_from_index(self, ids, name):
    ...         return {"http://cmt#Paper": {"label": "Paper"}} if "http://cmt#Paper" in ids else {}
    >>> result = retrieve_entities_and_properties(FakeManager())
    >>> sorted(result)
    ['http://cmt#Paper', 'http://cmt#hasAuthor']
    >>> result["http://cmt#hasAuthor"].label
    'has Author'
    """
    entities_data = kg_manager.get_data(ObjType.ENTITY.index_name)
    properties_data = kg_manager.get_data(ObjType.PROPERTY.index_name)

    property_ids = _filter_own_namespace(
        [identifier for identifier, _ in properties_data], kg_manager
    )
    property_id_set = set(property_ids)

    # the entities index may already include properties (depends on how the KG
    # was set up) - exclude those here so they are only queried/enriched once,
    # via the properties index's own, property-specific info SPARQL
    entity_ids = _filter_own_namespace(
        [identifier for identifier, _ in entities_data if identifier not in property_id_set],
        kg_manager,
    )

    entity_info = kg_manager.get_info_for_identifiers_from_index(
        entity_ids, ObjType.ENTITY.index_name
    )
    property_info = kg_manager.get_info_for_identifiers_from_index(
        property_ids, ObjType.PROPERTY.index_name
    )

    result = _entities_from_ids(kg_manager, entity_ids, entity_info)
    result.update(_entities_from_ids(kg_manager, property_ids, property_info))
    return result


def build_string_matching_dict(entities: dict[str, Entity]) -> dict[str, list[Entity]]:
    """
    Indexes `entities` by every one of their normalized names (label and
    aliases alike), so entities with multiple names are found under each of
    them.

    >>> paper = Entity(identifier="http://cmt#Paper", entity="cmt:Paper", label="Paper")
    >>> review = Entity(identifier="http://cmt#Review", entity="cmt:Review", label="Review")
    >>> d = build_string_matching_dict({paper.identifier: paper, review.identifier: review})
    >>> sorted(d)
    ['paper', 'review']
    >>> d["paper"] == [paper]
    True
    """
    result: dict[str, list[Entity]] = {}
    for entity in entities.values():
        names = entity_names(entity)
        for name in names:
            name = normalize_name(name)
            if result.get(name) is None:
                result[name] = []
            result[name].append(entity)

    return result


def perform_string_matching(
    entities_source: dict[str, Entity],
    entities_target: dict[str, Entity]
        ) -> tuple[list[PotentialCorrespondences], list[Entity]]:
    """
    Splits `entities_source` into those with at least one normalized-name
    match in `entities_target` (returned as `PotentialCorrespondences`,
    candidates in first-seen order, deduplicated) and those with none
    (returned as unmatched).

    >>> paper = Entity(identifier="http://cmt#Paper", entity="cmt:Paper", label="Paper")
    >>> review = Entity(identifier="http://cmt#Review", entity="cmt:Review", label="Review")
    >>> conf_paper = Entity(identifier="http://conference#Paper", entity="conference:Paper", label="Paper")
    >>> matches, unmatched = perform_string_matching(
    ...     {paper.identifier: paper, review.identifier: review},
    ...     {conf_paper.identifier: conf_paper},
    ... )
    >>> matches[0].source_entity.entity, [c.entity for c in matches[0].candidates]
    ('cmt:Paper', ['conference:Paper'])
    >>> [e.entity for e in unmatched]
    ['cmt:Review']
    """
    target_matching_dict = build_string_matching_dict(entities_target)
    matches: list[PotentialCorrespondences] = []
    unmatched: list[Entity] = []

    for src_entity in entities_source.values():
        src_names = entity_names(src_entity)
        matched_tgt_entities: list[Entity] = []
        for src_name in src_names:
            src_name = normalize_name(src_name)
            if src_name in target_matching_dict:
                for tgt_entity in target_matching_dict[src_name]:
                    if tgt_entity in matched_tgt_entities:
                        continue
                    matched_tgt_entities.append(tgt_entity)

        if len(matched_tgt_entities) == 0:
            unmatched.append(src_entity)

        else:
            matches.append(PotentialCorrespondences(source_entity=src_entity, candidates=matched_tgt_entities))

    return matches, unmatched


def write_jsonl_input(
    string_matches: list[PotentialCorrespondences],
    unmatched_entities: list[Entity],
    source_kg: str,
    target_kg: str,
    output_file: Path,
    batch_size: int = 1,
    limit: int | None = None,
) -> Path:
    """
    Writes one `AlignmentTaskInput` JSON record per line, `batch_size`
    entities at a time (string-matched ones first, then unmatched ones,
    same relative order as in `string_matches + unmatched_entities`). A
    batch straddling the boundary between the two correctly splits into its
    `potential_correspondences` and `unmatched_entities` parts rather than
    misclassifying either side.

    >>> import tempfile, json
    >>> paper_src = Entity(identifier="http://cmt#Paper", entity="cmt:Paper", label="Paper")
    >>> paper_tgt = Entity(identifier="http://conference#Paper", entity="conference:Paper", label="Paper")
    >>> review = Entity(identifier="http://cmt#Review", entity="cmt:Review", label="Review")
    >>> matches = [PotentialCorrespondences(source_entity=paper_src, candidates=[paper_tgt])]
    >>> out = Path(tempfile.mktemp(suffix=".jsonl"))
    >>> _ = write_jsonl_input(matches, [review], "cmt", "conference", out, batch_size=1)
    >>> len(out.read_text().splitlines())
    2
    >>> _ = write_jsonl_input(matches, [review], "cmt", "conference", out, batch_size=2)
    >>> record = json.loads(out.read_text().splitlines()[0])
    >>> [c["source_entity"]["entity"] for c in record["potential_correspondences"]]
    ['cmt:Paper']
    >>> [e["entity"] for e in record["unmatched_entities"]]
    ['cmt:Review']
    >>> out.unlink()
    """
    entities = string_matches + unmatched_entities
    if limit is not None:
        logging.info(f"Limiting source entities to the first {limit} entities.")
        entities = entities[:limit]

    with output_file.open("w", encoding="utf-8") as jsonl_file:
        for i in range(0, len(entities), batch_size):

            if len(string_matches) >= i + batch_size:
                potential_corrs = entities[i:i + batch_size]
                unmatched = []
            elif len(string_matches) > i:
                potential_corrs = entities[i:len(string_matches)]
                unmatched = entities[len(string_matches):i + batch_size]
            else:
                potential_corrs = []
                unmatched = entities[i:i + batch_size]

            record = AlignmentTaskInput(
                unmatched_entities=unmatched,
                potential_correspondences=potential_corrs,
                source_kg=source_kg,
                target_kg=target_kg
                ).model_dump()

            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(
        f"Wrote OM task with {len(entities)} entities "
        f"in batches of {batch_size} to {output_file}"
        )

    return output_file


def om_pretask(source_manager: KgManager, target_manager: KgManager, output_file: Path, config: GraspConfig):
    """
    End-to-end OM pretask: retrieves both KGs' entities/properties, string-
    matches the source ones against the target ones (unless
    `task_kwargs.om_pretask.skip_prematching` is set, in which case every
    source entity is treated as unmatched and `target_manager` is never
    queried for its own entities at all), and writes the result as the
    JSONL input `om_task` consumes.

    >>> import tempfile, json
    >>> from grasp.sparql.types import ObjType
    >>> class FakeManager:
    ...     def __init__(self, kg, label):
    ...         self.kg = kg
    ...         self.prefixes = {}
    ...         self.kg_prefixes = {kg: f"http://{kg}#"}
    ...         self._label = label
    ...     def format_iri(self, iri):
    ...         return iri
    ...     def get_data(self, name):
    ...         if name == ObjType.ENTITY.index_name:
    ...             return [(f"http://{self.kg}#Paper", None)]
    ...         return []
    ...     def get_info_for_identifiers_from_index(self, ids, name):
    ...         return {f"http://{self.kg}#Paper": {"label": self._label}}
    >>> config = GraspConfig(model="test", task_kwargs={"om_pretask": {"batch_size": 1}})
    >>> out = Path(tempfile.mktemp(suffix=".jsonl"))
    >>> om_pretask(FakeManager("cmt", "Paper"), FakeManager("conference", "Paper"), out, config)
    >>> record = json.loads(out.read_text().splitlines()[0])
    >>> record["source_kg"], record["target_kg"]
    ('cmt', 'conference')
    >>> [c["source_entity"]["entity"] for c in record["potential_correspondences"]]
    ['http://cmt#Paper']
    >>> out.unlink()
    """
    configs = config.task_kwargs.get("om_pretask", {})
    skip_prematching = configs.get("skip_prematching")
    batch_size = configs.get("batch_size")
    limit = configs.get("limit")
    source_entities = retrieve_entities_and_properties(source_manager)

    if skip_prematching:
        matches = []
        unmatched = list(source_entities.values())
    else:
        target_entities = retrieve_entities_and_properties(target_manager)
        matches, unmatched = perform_string_matching(source_entities, target_entities)

    write_jsonl_input(
        matches,
        unmatched,
        source_manager.kg,
        target_manager.kg,
        output_file,
        (batch_size if batch_size else 1),
        limit
    )
