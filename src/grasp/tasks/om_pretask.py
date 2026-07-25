import argparse
import json
import logging
from pathlib import Path
import random
from typing import Any
from enum import StrEnum

from pydantic import BaseModel
from universal_ml_utils.table import generate_table

import re
import unicodedata

from grasp.configs import GraspConfig
from grasp.functions import find_manager
from grasp.manager import KgManager, format_kgs
from grasp.model import Message
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.utils import FunctionCallException, derive_label_from_iri, camel_case_split, format_list, format_notes
from grasp.tasks.om import Entity, Correspondence, AlignmentTaskInput, PotentialCorrespondences
from grasp.sparql.types import SelectResult


def normalize_name(name: str) -> str:
    _WHITESPACE_RE = re.compile(r"\s+")

    name = unicodedata.normalize("NFKC", name)
    name = name.replace("_", " ").replace("-", " ")
    name = camel_case_split(name)
    name = _WHITESPACE_RE.sub(" ", name).strip()
    return name.casefold()


def entity_names(entity: Entity) -> list[str]:
    names = [entity.label] if entity.label else []
    if entity.aliases:
        names.extend(entity.aliases)
    return names


def retrieve_all_entities(kg_manager: KgManager) -> dict[str, Entity]:
    classes_sparql = """
SELECT DISTINCT ?class WHERE {
    { ?x a ?class } UNION { ?class a owl:Class } UNION { ?class a rdfs:Class }
    FILTER(isIRI(?class))
    FILTER(!STRSTARTS(STR(?class), STR(owl:)))
    FILTER(!STRSTARTS(STR(?class), STR(rdf:)))
    FILTER(!STRSTARTS(STR(?class), STR(rdfs:)))
}
"""

    properties_sparql = """
SELECT DISTINCT ?prop WHERE {
    { ?prop a owl:ObjectProperty } UNION { ?prop a owl:DatatypeProperty }
    UNION { ?prop a owl:AnnotationProperty } UNION { ?s ?prop ?o }
    FILTER(isIRI(?prop))
    FILTER(!STRSTARTS(STR(?prop), STR(owl:)))
    FILTER(!STRSTARTS(STR(?prop), STR(rdf:)))
    FILTER(!STRSTARTS(STR(?prop), STR(rdfs:)))
}
"""

    instances_sparql = """
SELECT DISTINCT ?instance WHERE {
  ?instance a ?class .
  FILTER(isIRI(?instance))
  FILTER(!STRSTARTS(STR(?instance), STR(owl:)))
  FILTER(!STRSTARTS(STR(?instance), STR(rdf:)))
  FILTER(!STRSTARTS(STR(?instance), STR(rdfs:)))
  FILTER(!STRSTARTS(STR(?class), STR(owl:)))
  FILTER(!STRSTARTS(STR(?class), STR(rdf:)))
  FILTER(!STRSTARTS(STR(?class), STR(rdfs:)))
}
"""

    # shared info query, following GRASP's own `*.entity.info.sparql` convention
    # (id/value/type rows), consumable directly by retrieve_info_for_identifiers
    info_sparql = """
SELECT DISTINCT ?id ?value ?type WHERE {
  {
    VALUES ?id { {IDS} }
    ?id rdfs:label ?value
    FILTER(LANG(?value) = "en")
    BIND("label" AS ?type)
  } UNION {
    VALUES ?id { {IDS} }
    ?id skos:altLabel ?value
    FILTER(LANG(?value) = "en")
    BIND("alias" AS ?type)
  } UNION {
    VALUES ?id { {IDS} }
    ?id rdfs:comment ?value
    FILTER(LANG(?value) = "en")
    BIND("info" AS ?type)
  } UNION {
    VALUES ?id { {IDS} }
    ?id a ?t .
    ?t rdfs:label ?tl
    FILTER(LANG(?tl) = "en")
    BIND(CONCAT("is a ", ?tl) AS ?value)
    BIND("info" AS ?type)
  }
}
ORDER BY ?id ?type ?value
"""

    classes = kg_manager.execute_sparql(classes_sparql)
    properties = kg_manager.execute_sparql(properties_sparql)
    instances = kg_manager.execute_sparql(instances_sparql)
    assert isinstance(classes, SelectResult), "classes_sparql must be a SELECT query"
    assert isinstance(properties, SelectResult), "properties_sparql must be a SELECT query"
    assert isinstance(instances, SelectResult), "instances_sparql must be a SELECT query"

    class_iris = [row["class"].value for row in classes.rows()]
    property_iris = [row["prop"].value for row in properties.rows()]
    instance_iris = [row["instance"].value for row in instances.rows()]
    all_iris = class_iris + property_iris + instance_iris

    info = kg_manager.retrieve_info_for_identifiers(all_iris, info_sparql)

    result: dict[str, Entity] = {}
    for iri in all_iris:
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


def build_string_matching_dict(entities: dict[str, Entity]) -> dict[str, list[Entity]]:
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
    source_entities = retrieve_all_entities(source_manager)
    target_entities = retrieve_all_entities(target_manager)

    matches, unmatched = perform_string_matching(source_entities, target_entities)

    batch_size = config.task_kwargs.get("om_pretask", {}).get("batch_size")
    limit = config.task_kwargs.get("om_pretask", {}).get("limit")

    write_jsonl_input(
        matches,
        unmatched,
        source_manager.kg,
        target_manager.kg,
        output_file,
        (batch_size if batch_size else 1),
        limit
    )
