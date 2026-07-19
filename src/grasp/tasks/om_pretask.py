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
from grasp.tasks.om import Entity
from grasp.sparql.types import SelectResult


def normalize_name(name: str) -> str:
    _WHITESPACE_RE = re.compile(r"\s+")

    name = unicodedata.normalize("NFKC", name)
    name = name.replace("_", " ").replace("-", " ")
    name = camel_case_split(name)
    name = _WHITESPACE_RE.sub(" ", name).strip()
    return name.casefold()


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
}
"""

    instances_sparql = """
SELECT DISTINCT ?instance WHERE {
  ?instance a ?class .
  FILTER(isIRI(?instance))
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

        result[iri] = Entity(
            identifier=iri,
            entity=kg_manager.format_iri(iri),
            label=entry.get("label"),
            aliases=entry.get("alias", []),
            infos=entry.get("other", []),
        )

    return result


def build_string_matching_dict(entities: dict[str, Entity], kg_manager: KgManager) -> dict[str, list[Entity]]:
    result: dict[str, list[Entity]] = {}
    for entity in entities.values():
        names = [entity.label if entity.label else derive_label_from_iri(entity.identifier, kg_manager.prefixes)]
        if entity.aliases:
            names.extend(entity.aliases)
        for name in names:
            name = normalize_name(name)
            if result.get(name) is None:
                result[name] = []
            result[name].append(entity)

    return result


def perform_string_matching(entities_source: dict[str, Entity], entities_target: dict[str, Entity]):
    ...


