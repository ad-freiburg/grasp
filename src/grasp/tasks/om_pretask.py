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
from grasp.tasks.entities import Entity
from grasp.tasks.om import Correspondence, AlignmentTaskInput, PotentialCorrespondences
from grasp.sparql.types import ObjType


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


def _entities_from_ids(
    kg_manager: KgManager, ids: list[str], info: dict[str, dict]
) -> dict[str, Entity]:
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


def retrieve_entities_and_properties(kg_manager: KgManager) -> dict[str, Entity]:
    """
    Retrieves all classes and properties of a KG using GRASP's own, already
    built entities/properties indices instead of custom SPARQL queries.
    Does not cover individuals/instances.
    """
    entities_data = kg_manager.get_data(ObjType.ENTITY.index_name)
    properties_data = kg_manager.get_data(ObjType.PROPERTY.index_name)

    property_ids = [identifier for identifier, _ in properties_data]
    property_id_set = set(property_ids)

    # the entities index may already include properties (depends on how the KG
    # was set up) - exclude those here so they are only queried/enriched once,
    # via the properties index's own, property-specific info SPARQL
    entity_ids = [
        identifier for identifier, _ in entities_data if identifier not in property_id_set
    ]

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
    source_entities = retrieve_entities_and_properties(source_manager)
    target_entities = retrieve_entities_and_properties(target_manager)

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
