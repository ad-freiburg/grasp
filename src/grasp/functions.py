import math
import time
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Iterable

from grammar_utils.parse import LR1Parser  # type: ignore
from search_rdf import EmbeddingIndex
from universal_ml_utils.ops import partition_by

from grasp.configs import GraspConfig, ShapeConfig
from grasp.manager import KgManager
from grasp.manager.normalizer import Normalizer
from grasp.manager.utils import get_common_sparql_prefixes
from grasp.shapes import PropertyProfile, Shapes
from grasp.sparql.item import parse_into_binding
from grasp.sparql.types import (
    Alternative,
    AskResult,
    Binding,
    Position,
    Selection,
    SelectResult,
    SelectRow,
    iri_indices,
)
from grasp.sparql.utils import (
    READ_TIMEOUT,
    REQUEST_TIMEOUT,
    find_all,
    has_scheme,
    parse_string,
    wrap_iri,
)
from grasp.utils import FunctionCallException, format_list

if TYPE_CHECKING:
    from grasp.tasks.base import GraspTask

# maximum number of results for constraining with sub indices
MAX_RESULTS = 131072

MODALITY_QUERY_TYPES = {
    "text": [("text", "textual search query")],
    "image": [("image", "URL pointing to an image")],
}


def kg_functions(
    managers: list[KgManager],
    fn_set: str,
    list_k: int,
    search_k: int,
    search_max_pages: int,
) -> list[dict]:
    assert fn_set in [
        "base",
        "search",
        "search_extended",
        "search_filter",
        "search_constraints",
        "all",
    ], f"Unknown function set {fn_set}"
    kgs = [manager.kg for manager in managers]

    known_indices = set()
    known_modalities = set()
    for manager in managers:
        known_indices.update(manager.index_names)

        for idx in manager.indices.values():
            if not isinstance(idx.index, EmbeddingIndex):
                continue
            known_modalities.update(idx.index.modality)

    assert all(mod in MODALITY_QUERY_TYPES for mod in known_modalities), (
        f"Unknown modality in {known_modalities}"
    )
    index_names = sorted(known_indices)

    fns = [
        {
            "name": "execute",
            "description": """\
Execute a SPARQL query and retrieve its results as a table if successful, \
and an error message otherwise.

For example, to execute a SPARQL query over Wikidata to find the jobs of \
Albert Einstein, do the following:
execute(kg="wikidata", sparql="SELECT ?job WHERE { wd:Q937 wdt:P106 ?job }")""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph to query",
                    },
                    "sparql": {
                        "type": "string",
                        "description": "The SPARQL query to execute",
                    },
                },
                "required": ["kg", "sparql"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "list",
            "description": f"""\
List triples from the knowledge graph satisfying the given subject, property, \
and object constraints. At most {list_k} results are returned per page (use \
pagintion to see more results).

For example, to find triples with Albert Einstein as the subject in Wikidata, \
do the following:
list(kg="wikidata", subject="wd:Q937")

Or to find examples of how the property "place of birth" is used in Wikidata, \
do the following:
list(kg="wikidata", property="wdt:P19")""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph to use",
                    },
                    "subject": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the subject (null for unconstrained)",
                    },
                    "property": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the property (null for unconstrained)",
                    },
                    "object": {
                        "type": ["string", "null"],
                        "description": "IRI or literal for constraining the object (null for unconstrained)",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-indexed) for paginating results (default should be 1)",
                    },
                    "unclipped": {
                        "type": "boolean",
                        "description": "Whether to show full unclipped literal values "
                        "(default should be false, typically only needed to inspect very long string literals)",
                    },
                },
                "required": [
                    "kg",
                    "page",
                    "subject",
                    "property",
                    "object",
                    "unclipped",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]

    if fn_set == "base":
        return fns

    shape_kgs = [m.kg for m in managers if m.shapes is not None]
    if shape_kgs:
        fns.append(class_shape_function_definition(shape_kgs))

    has_entity_index = "entities" in known_indices
    has_property_index = "properties" in known_indices
    has_literal_index = "literals" in known_indices
    has_class_index = "classes" in known_indices

    page_prop = {
        "type": "integer",
        "description": "Page number (1-indexed) for paginating results (default should be 1)",
    }

    if fn_set in ["search", "search_extended", "all"]:
        search_entity_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "page": page_prop,
        }
        search_entity_required = ["kg", "query", "page"]

        search_property_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "page": page_prop,
        }
        search_property_required = ["kg", "query", "page"]

        if has_entity_index:
            fns.append(
                {
                    "name": "search_entity",
                    "description": f"""\
Search for entities in the knowledge graph with a search query. \
At most {search_k} results are returned per page (use pagination \
up to page {search_max_pages} to see more results).

For example, to search for the entity Albert Einstein in Wikidata, \
do the following:
search_entity(kg="wikidata", query="albert einstein", page=1)""",
                    "parameters": {
                        "type": "object",
                        "properties": search_entity_props,
                        "required": search_entity_required,
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            )

        if has_property_index:
            fns.append(
                {
                    "name": "search_property",
                    "description": f"""\
Search for properties in the knowledge graph with a search query. \
At most {search_k} results are returned per page (use pagination \
up to page {search_max_pages} to see more results).

For example, to search for properties related to birth in Wikidata, do the following:
search_property(kg="wikidata", query="birth", page=1)""",
                    "parameters": {
                        "type": "object",
                        "properties": search_property_props,
                        "required": search_property_required,
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            )

        if has_literal_index:
            search_literal_props = {
                "kg": {
                    "type": "string",
                    "enum": kgs,
                    "description": "The knowledge graph to search",
                },
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "page": page_prop,
            }
            fns.append(
                {
                    "name": "search_literal",
                    "description": f"""\
Search for literal values (e.g. enumerable string values used as objects, \
not entity labels) in the knowledge graph with a search query. \
At most {search_k} results are returned per page (use pagination \
up to page {search_max_pages} to see more results). Results are \
shown as quoted strings, e.g. "restaurant", and can be pasted directly \
into a SPARQL filter.

For example, to search for the literal value "restaurant" used as an \
amenity in OSM, do the following:
search_literal(kg="osm-planet", query="restaurant", page=1)""",
                    "parameters": {
                        "type": "object",
                        "properties": search_literal_props,
                        "required": ["kg", "query", "page"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            )

        if has_class_index:
            fns.append(
                {
                    "name": "search_class",
                    "description": f"""\
Search for classes in the knowledge graph with a search query. \
At most {search_k} results are returned per page (use pagination \
up to page {search_max_pages} to see more results). Use get_class_shape \
to see which properties the instances of a class have.

For example, to search for the class of humans in Wikidata, \
do the following:
search_class(kg="wikidata", query="human", page=1)""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kg": {
                                "type": "string",
                                "enum": kgs,
                                "description": "The knowledge graph to search",
                            },
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                            "page": page_prop,
                        },
                        "required": ["kg", "query", "page"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            )

    if fn_set in ["search_extended", "all"]:
        if has_property_index:
            fns.append(
                {
                    "name": "search_property_of_entity",
                    "description": f"""\
Search for properties of a given entity in the knowledge graph. \
At most {search_k} results are returned per page (use pagination \
up to page {search_max_pages} to see more results).

For example, to search for properties related to birth for Albert Einstein \
in Wikidata, do the following:
search_property_of_entity(kg="wikidata", entity="wd:Q937", query="birth", page=1)""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kg": {
                                "type": "string",
                                "enum": kgs,
                                "description": "The knowledge graph to search",
                            },
                            "entity": {
                                "type": "string",
                                "description": "The entity to search properties for",
                            },
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                            "page": page_prop,
                        },
                        "required": ["kg", "entity", "query", "page"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            )

        if index_names:
            fns.append(
                {
                    "name": "search_object_of_property",
                    "description": f"""\
Search one of the indices for items at the object position of a given \
property in the knowledge graph. At most {search_k} results are returned \
per page (use pagination up to page {search_max_pages} to see more \
results).

For example, to search for football-related jobs in Wikidata, do the following:
search_object_of_property(kg="wikidata", property="wdt:P106", \
query="football", index="entities", page=1)

Or to search for restaurant-related amenities in OSM, do the following:
search_object_of_property(kg="osm", property="osmkey:amenity", \
query="restaurant", index="literals", page=1)""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kg": {
                                "type": "string",
                                "enum": kgs,
                                "description": "The knowledge graph to search",
                            },
                            "property": {
                                "type": "string",
                                "description": "The property to search objects for",
                            },
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                            "index": {
                                "type": "string",
                                "enum": index_names,
                                "description": "Which index to search for the property's objects",
                            },
                            "page": page_prop,
                        },
                        "required": ["kg", "property", "query", "index", "page"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            )

    # prepare query type arg
    query_types = [typ for mod in known_modalities for typ in MODALITY_QUERY_TYPES[mod]]
    query_type_prop = {
        "type": "string",
        "enum": sorted(qt for qt, _ in query_types),
        "description": "How to interpret the query string: "
        + ", ".join(f'"{qt}" for {desc}' for qt, desc in query_types),
    }

    if fn_set in ["search_filter", "all"] and index_names:
        search_filter_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "index": {
                "type": "string",
                "enum": index_names,
                "description": "The index to search in",
            },
            "sparql": {
                "type": ["string", "null"],
                "description": "The SPARQL query for filtering or null for an unconstrained search",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "page": page_prop,
        }
        search_filter_required = ["kg", "index", "sparql", "query", "page"]

        if len(query_types) > 1:
            search_filter_props["query_type"] = query_type_prop
            search_filter_required.append("query_type")

        search_filter_name = (
            "search" if fn_set == "search_filter" else "search_with_filter"
        )
        fns.append(
            {
                "name": search_filter_name,
                "description": f"""\
Search for knowledge graph items in a context-sensitive way by specifying a filter \
SPARQL query together with a search query. The SPARQL query must be a single-column SELECT \
query returning knowledge graph identifiers (IRIs or literals). The search is then restricted \
to the items matching those identifiers in the specified index. The SPARQL query can be null, in which case \
a search over the full index is performed. At most {search_k} results are returned per \
page (use pagination up to page {search_max_pages} to see more results).

For example, to search for Albert Einstein in Wikidata, do the following:
{search_filter_name}(kg="wikidata", index="entities", query="albert einstein", page=1)

Or to search for properties of Albert Einstein related to his birth in \
Wikidata, do the following:
{search_filter_name}(kg="wikidata", index="properties", sparql="SELECT DISTINCT ?p WHERE {{ wd:Q937 ?p ?o }}", \
query="birth", page=1)""",
                "parameters": {
                    "type": "object",
                    "properties": search_filter_props,
                    "required": search_filter_required,
                    "additionalProperties": False,
                },
                "strict": True,
            }
        )

    if fn_set == "search_constraints" and index_names:
        search_constraints_props = {
            "kg": {
                "type": "string",
                "enum": kgs,
                "description": "The knowledge graph to search",
            },
            "index": {
                "type": "string",
                "enum": index_names,
                "description": "The index to search in",
            },
            "position": {
                "type": "string",
                "enum": ["subject", "property", "object"],
                "description": "The position/type of item to look for",
            },
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "constraints": {
                "type": ["object", "null"],
                "description": "Constraints for the search (null for unconstrained)",
                "properties": {
                    "subject": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the subject (null for unconstrained)",
                    },
                    "property": {
                        "type": ["string", "null"],
                        "description": "IRI for constraining the property (null for unconstrained)",
                    },
                    "object": {
                        "type": ["string", "null"],
                        "description": "IRI or literal for constraining the object (null for unconstrained)",
                    },
                },
                "required": ["subject", "property", "object"],
                "additionalProperties": False,
            },
            "page": page_prop,
        }
        search_constraints_required = [
            "kg",
            "index",
            "position",
            "query",
            "constraints",
            "page",
        ]

        if len(query_types) > 1:
            search_constraints_props["query_type"] = query_type_prop
            search_constraints_required.append("query_type")

        search_constraints_name = (
            "search" if fn_set == "search_constraints" else "search_with_constraints"
        )
        fns.append(
            {
                "name": search_constraints_name,
                "description": f"""\
Search for knowledge graph items at a particular position (subject, property, or object) \
with optional constraints. If constraints are provided, they are used to limit the search \
space accordingly. At most {search_k} results are returned per page (use pagination \
up to page {search_max_pages} to see more results).

For example, to search for the subject Albert Einstein in Wikidata, do the following:
{search_constraints_name}(kg="wikidata", index="entities", position="subject", query="albert einstein", page=1)

Or to search for properties of Albert Einstein related to his birth in Wikidata, \
do the following:
{search_constraints_name}(kg="wikidata", index="properties", position="property", query="birth", \
constraints={{"subject": "wd:Q937"}}, page=1)""",
                "parameters": {
                    "type": "object",
                    "properties": search_constraints_props,
                    "required": search_constraints_required,
                    "additionalProperties": False,
                },
                "strict": True,
            },
        )

    return fns


def find_manager(
    managers: list[KgManager],
    kg: str,
) -> tuple[KgManager, list[KgManager]]:
    managers, others = partition_by(managers, lambda m: m.kg == kg)
    if not managers:
        raise FunctionCallException(f"Unknown knowledge graph {kg}")
    elif len(managers) > 1:
        raise FunctionCallException(f"Multiple managers found for knowledge graph {kg}")
    return managers[0], others


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    task: "GraspTask | None" = None,
    example_indices: dict | None = None,
) -> str:
    if fn_name == "execute":
        return execute_sparql(
            managers,
            fn_args["kg"],
            fn_args["sparql"],
            config.result_max_rows,
            config.result_max_columns,
            known,
            config.know_before_use,
            config.sparql_request_timeout,
            config.sparql_read_timeout,
            config.sparql_result_max_rows,
        ).formatted  # type: ignore

    elif fn_name == "list":
        return list_triples(
            managers,
            fn_args["kg"],
            fn_args.get("subject"),
            fn_args.get("property"),
            fn_args.get("object"),
            fn_args.get("page") or 1,
            fn_args.get("unclipped") or False,
            config.list_k,
            known,
            config.sparql_request_timeout,
            config.sparql_read_timeout,
        )

    elif fn_name == "search_entity":
        return search_entity(
            managers,
            fn_args["kg"],
            fn_args["query"],
            config.search_k,
            known,
            fn_args.get("query_type", "text"),
            page=fn_args.get("page") or 1,
            max_pages=config.search_max_pages,
        )

    elif fn_name == "search_property":
        return search_property(
            managers,
            fn_args["kg"],
            fn_args["query"],
            config.search_k,
            known,
            fn_args.get("query_type", "text"),
            page=fn_args.get("page") or 1,
            max_pages=config.search_max_pages,
        )

    elif fn_name == "search_class":
        return search_class(
            managers,
            fn_args["kg"],
            fn_args["query"],
            config.search_k,
            known,
            fn_args.get("query_type", "text"),
            page=fn_args.get("page") or 1,
            max_pages=config.search_max_pages,
        )

    elif fn_name == "search_literal":
        return search_literal(
            managers,
            fn_args["kg"],
            fn_args["query"],
            config.search_k,
            known,
            fn_args.get("query_type", "text"),
            page=fn_args.get("page") or 1,
            max_pages=config.search_max_pages,
        )

    elif fn_name == "search_property_of_entity":
        return search_with_constraints(
            managers,
            fn_args["kg"],
            "properties",
            "property",
            fn_args["query"],
            {"subject": fn_args["entity"]},
            config.search_k,
            known,
            fn_args.get("query_type", "text"),
            config.sparql_request_timeout,
            config.sparql_read_timeout,
            page=fn_args.get("page") or 1,
            max_pages=config.search_max_pages,
        )

    elif fn_name == "search_object_of_property":
        manager, _ = find_manager(managers, fn_args["kg"])
        if fn_args["index"] not in manager.indices:
            raise FunctionCallException(
                f"Knowledge graph {fn_args['kg']} has no {fn_args['index']} index"
            )
        return search_with_constraints(
            managers,
            fn_args["kg"],
            fn_args["index"],
            "object",
            fn_args["query"],
            {"property": fn_args["property"]},
            config.search_k,
            known,
            fn_args.get("query_type", "text"),
            config.sparql_request_timeout,
            config.sparql_read_timeout,
            page=fn_args.get("page") or 1,
            max_pages=config.search_max_pages,
        )

    elif fn_name == "search_with_constraints" or (
        fn_name == "search" and config.fn_set == "search_constraints"
    ):
        return search_with_constraints(
            managers,
            fn_args["kg"],
            fn_args["index"],
            fn_args["position"],
            fn_args["query"],
            fn_args.get("constraints"),
            config.search_k,
            known,
            fn_args.get("query_type", "text"),
            config.sparql_request_timeout,
            config.sparql_read_timeout,
            page=fn_args.get("page") or 1,
            max_pages=config.search_max_pages,
        )

    elif fn_name == "search_with_filter" or (
        fn_name == "search" and config.fn_set == "search_filter"
    ):
        return search_with_filter(
            managers,
            fn_args["kg"],
            fn_args["index"],
            fn_args["sparql"],
            fn_args["query"],
            config.search_k,
            known,
            fn_args.get("query_type", "text"),
            config.know_before_use,
            config.sparql_request_timeout,
            config.sparql_read_timeout,
            page=fn_args.get("page") or 1,
            max_pages=config.search_max_pages,
        )

    elif fn_name == "get_class_shape":
        manager, _ = find_manager(managers, fn_args["kg"])
        if manager.shapes is None:
            return f"No class shapes available for knowledge graph '{fn_args['kg']}'"
        return get_class_shape(
            manager,
            manager.shapes,
            fn_args["iri"],
            fn_args.get("direction") or "out",
            fn_args.get("query"),
            fn_args.get("page") or 1,
            config.shape_k,
            known,
        )

    elif task is not None:
        return task.call_function(fn_name, fn_args, known, example_indices)

    else:
        raise ValueError(f"Unknown function {fn_name}")


SHAPE_CAVEAT = (
    "Shapes are approximate (derived from instance data and/or schema "
    "declarations) and may be incomplete, so treat them as guidance. "
    "Cardinalities follow ShEx (none for exactly one, ? for at most one, "
    "+ for one or more, * for any number), and a leading ^ marks an inverse "
    "edge, where the class is the property's object."
)


def class_shape_function_definition(kgs: list[str]) -> dict:
    return {
        "name": "get_class_shape",
        "description": f"""\
Get the shape of a class in pseudo-ShEx, i.e., either the properties its \
instances have together with their values (direction "out"), or the properties \
pointing to its instances together with where they come from (direction "in"). \
{SHAPE_CAVEAT}

Properties are shown page by page. Without a query, the properties used by \
only few instances are left out; with a query, all properties are ranked by \
how well they match it.

For example, to get the shape of the class of humans in Wikidata, do the following:
get_class_shape(kg="wikidata", iri="wd:Q5", direction="out", query=null, page=1)

Or to find properties related to birth for humans in Wikidata, do the following:
get_class_shape(kg="wikidata", iri="wd:Q5", direction="out", query="birth", page=1)

Or to see which properties point to humans in Wikidata, do the following:
get_class_shape(kg="wikidata", iri="wd:Q5", direction="in", query=null, page=1)""",
        "parameters": {
            "type": "object",
            "properties": {
                "kg": {
                    "type": "string",
                    "enum": kgs,
                    "description": "The knowledge graph to look up the class in",
                },
                "iri": {
                    "type": "string",
                    "description": "The class IRI",
                },
                "direction": {
                    "type": "string",
                    "enum": ["out", "in"],
                    "description": "Whether to show the properties of the class "
                    "instances (out) or the properties pointing to them (in), "
                    'default should be "out"',
                },
                "query": {
                    "type": ["string", "null"],
                    "description": "Search query for ranking the properties of the "
                    "class (null to show the most common ones)",
                },
                "page": {
                    "type": "integer",
                    "description": "Page number (1-indexed) for paginating the "
                    "properties (default should be 1)",
                },
            },
            "required": ["kg", "iri", "direction", "query", "page"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def rank_class_properties(
    manager: KgManager,
    props: list[PropertyProfile],
    query: str,
) -> list[str]:
    iris = [p.iri for p in props]
    ranking = []
    if "properties" in manager.indices:
        ranking = manager.rank_identifiers("properties", query, iris)

    # properties outside the index can only match by their IRI
    ranked = set(ranking)
    ranking.extend(
        iri for iri in iris if iri not in ranked and query.lower() in iri.lower()
    )
    return ranking


def get_class_shape(
    manager: KgManager,
    shapes: Shapes,
    iri: str,
    direction: str,
    query: str | None,
    page: int,
    k: int,
    known: set[str] | None = None,
) -> str:
    from grasp.build.shapes import (
        DIRECTION_NAMES,
        collect_iris,
        compute_shape,
        render_shape,
        select_shape_view,
    )

    validate_page(page)
    if direction not in DIRECTION_NAMES:
        raise FunctionCallException('Direction must be "out" or "in"')

    # computing a shape builds SPARQL around the class term, so a non-IRI
    # argument would only fail later as an opaque parse error
    binding = parse_iri_or_literal(iri, manager.iri_literal_parser, manager.prefixes)
    if binding is None or binding.typ != "uri":
        return format_iri_or_literal_error(iri)

    shape_config = manager.shape_config or ShapeConfig()
    class_iri = binding.identifier()
    profile = shapes.get(class_iri)
    if profile is None:
        try:
            profile = compute_shape(
                class_iri,
                manager,
                instance_pattern=shapes.instance_pattern,
                schema_pattern=shapes.schema_pattern,
                shape_config=shape_config,
            )
        except Exception as e:
            return f"Failed to compute the shape of '{iri}':\n{e}"

        shapes.put(profile)

    ranking = None
    if query is not None and query.strip():
        props = profile.properties if direction == "out" else profile.incoming
        ranking = rank_class_properties(manager, props, query)

    view = select_shape_view(profile, shape_config, k, direction, page, ranking)
    if page > view.total_pages:
        raise FunctionCallException(
            f"Page number must be at most {view.total_pages} for this class"
        )

    if known is not None:
        update_known_from_index_iris(known, collect_iris(view, manager), manager)

    return render_shape(profile, view, manager, shape_config)


def update_known_from_index_iris(
    known: set[str],
    iris: list[str],
    manager: KgManager,
) -> None:
    for obj_type in iri_indices():
        update_known_from_iris(
            known,
            iris,
            manager.get_normalizer(obj_type.index_name),
        )


def validate_page(page: int, max_pages: int | None = None) -> None:
    if page < 1:
        raise FunctionCallException("Page number must be at least 1")
    elif max_pages is not None and page > max_pages:
        raise FunctionCallException(f"Page number must be at most {max_pages}")


def validate_query(query: str) -> None:
    if not query.strip():
        raise FunctionCallException("Query must not be empty")


def search_single_index(
    managers: list[KgManager],
    kg: str,
    index_name: str,
    query: str,
    k: int,
    known: set[str],
    query_type: str = "text",
    page: int = 1,
    max_pages: int = 10,
) -> str:
    validate_query(query)
    validate_page(page, max_pages)
    manager, _ = find_manager(managers, kg)

    alts = manager.search_index(
        index_name,
        query=query,
        k=k * max_pages + 1,
        query_type=query_type,
    )
    alts, total_pages, more = paginate_results(alts, k, page, max_pages)

    # update known items, for literals so the LLM can paste these quoted
    # literals straight into a SPARQL filter
    normalizer = manager.get_normalizer(index_name)
    update_known_from_alts(known, alts, normalizer)

    return format_index_alternatives(alts, k, page, total_pages, more)


def search_entity(
    managers: list[KgManager],
    kg: str,
    query: str,
    k: int,
    known: set[str],
    query_type: str = "text",
    page: int = 1,
    max_pages: int = 10,
) -> str:
    return search_single_index(
        managers,
        kg,
        "entities",
        query,
        k,
        known,
        query_type,
        page,
        max_pages,
    )


def search_property(
    managers: list[KgManager],
    kg: str,
    query: str,
    k: int,
    known: set[str],
    query_type: str = "text",
    page: int = 1,
    max_pages: int = 10,
) -> str:
    return search_single_index(
        managers,
        kg,
        "properties",
        query,
        k,
        known,
        query_type,
        page,
        max_pages,
    )


def search_literal(
    managers: list[KgManager],
    kg: str,
    query: str,
    k: int,
    known: set[str],
    query_type: str = "text",
    page: int = 1,
    max_pages: int = 10,
) -> str:
    return search_single_index(
        managers,
        kg,
        "literals",
        query,
        k,
        known,
        query_type,
        page,
        max_pages,
    )


def search_class(
    managers: list[KgManager],
    kg: str,
    query: str,
    k: int,
    known: set[str],
    query_type: str = "text",
    page: int = 1,
    max_pages: int = 10,
) -> str:
    return search_single_index(
        managers,
        kg,
        "classes",
        query,
        k,
        known,
        query_type,
        page,
        max_pages,
    )


COMMON_PREFIXES = get_common_sparql_prefixes()


def check_known(manager: KgManager, sparql: str, known: set[str]):
    parse, _ = parse_string(sparql, manager.sparql_parser)
    in_query = set()

    for iri in find_all(parse, {"IRIREF", "PNAME_NS", "PNAME_LN"}, skip={"Prologue"}):
        binding = parse_into_binding(
            iri["value"],
            manager.iri_literal_parser,
            manager.prefixes,
        )
        assert binding is not None, f"Failed to parse binding from {iri['value']}"
        assert binding.typ == "uri", f"Expected IRI, got {binding.typ}"

        identifier = binding.identifier()

        longest = manager.find_longest_prefix(identifier)
        if longest is None or longest[0] not in COMMON_PREFIXES:
            # unknown or uncommon prefix, should be known before use
            in_query.add(identifier)

    unknown_in_query = in_query - known
    if not unknown_in_query:
        return

    not_seen = []
    for iri in unknown_in_query:
        short_iri = manager.format_iri(iri)
        if short_iri == iri:
            not_seen.append(iri)
        else:
            not_seen.append(f"{iri} ({short_iri})")

    raise FunctionCallException(f"""\
The following knowledge graph items are used in the SPARQL query \
without being known from previous function call results. \
This does not mean they are invalid, but you should verify \
that they indeed exist (e.g., via listing example triples) \
in the knowledge graphs before trying again:
{format_list(not_seen)}""")


def update_known_from_iris(
    known: set[str],
    iris: Iterable[str],
    normalizer: Normalizer | None = None,
):
    for iri in iris:
        known.add(iri)
        if normalizer is None:
            continue

        norm = normalizer.normalize(iri)
        if norm is None:
            continue

        # also add normalized identifier
        known.add(norm[0])


def update_known_from_alts(
    known: set[str],
    alts: Iterable[Alternative],
    normalizer: Normalizer | None = None,
):
    for alt in alts:
        known.add(alt.identifier)
        if normalizer is None or not alt.variants:
            continue

        for var in alt.variants:
            denorm = normalizer.denormalize(alt.identifier, var)
            if denorm is None:
                continue
            known.add(denorm)


def row_iris(rows: Iterable[SelectRow]) -> list[str]:
    return [
        binding.identifier()
        for row in rows
        for binding in row.values()
        if binding.typ == "uri"
    ]


def update_known_from_selections(
    known: set[str],
    selections: list[Selection],
    manager: KgManager,
):
    for obj_type in iri_indices():
        update_known_from_alts(
            known,
            (sel.alternative for sel in selections if sel.obj_type == obj_type),
            manager.get_normalizer(obj_type.index_name),
        )


@dataclass
class ExecutionResult:
    sparql: str
    formatted: str
    result: SelectResult | AskResult | None = None


def execute_sparql(
    managers: list[KgManager],
    kg: str,
    sparql: str,
    max_rows: int,
    max_columns: int,
    known: set[str] | None = None,
    know_before_use: bool = False,
    request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
    read_timeout: float | None = READ_TIMEOUT,
    sparql_result_max_rows: int | None = None,
) -> ExecutionResult:
    manager, others = find_manager(managers, kg)

    # fix prefixes with managers
    sparql = manager.fix_prefixes(sparql)
    for other in others:
        sparql = other.fix_prefixes(sparql)

    if know_before_use and known is not None:
        check_known(manager, sparql, known)

    start = time.monotonic()
    try:
        result = manager.execute_sparql(
            sparql,
            request_timeout,
            read_timeout,
            sparql_result_max_rows=sparql_result_max_rows,
        )
    except Exception as e:
        diff = time.monotonic() - start
        error = f"SPARQL execution failed in {diff:.2f}s:\n{e}"
        return ExecutionResult(sparql, error)

    diff = time.monotonic() - start
    half_rows = math.ceil(max_rows / 2)
    half_columns = math.ceil(max_columns / 2)

    if isinstance(result, SelectResult) and known is not None:
        # only update with the bindings shown to the model
        shown_vars = result.variables[:half_columns] + result.variables[-half_columns:]
        rows = (
            {var: row[var] for var in shown_vars if var in row}
            for row in chain(
                result.rows(end=half_rows),
                result.rows(start=max(0, len(result) - half_rows)),
            )
        )

        update_known_from_index_iris(known, row_iris(rows), manager)

    formatted = manager.format_sparql_result(
        result,
        half_rows,
        half_rows,
        half_columns,
        half_columns,
        time=diff,
    )
    return ExecutionResult(sparql, formatted, result)


def parse_iri_or_literal(
    input: str,
    parser: LR1Parser,
    prefixes: dict[str, str] | None = None,
) -> Binding | None:
    binding = parse_into_binding(input, parser, prefixes)

    if binding is None and has_scheme(input):
        # fallback for full IRIs given without angle brackets
        binding = parse_into_binding(wrap_iri(input), parser, prefixes)

    if binding is None:
        # fallback for string literals because they are typically given without quotes
        # but the parser expects them to be quoted
        binding = parse_into_binding(f'"{input}"', parser, prefixes)

    return binding


def format_bnode_error(value: str) -> str:
    return (
        f'Value "{value}" is a blank node. Blank nodes cannot be used as constraints '
        "because they behave like variables in SPARQL and do not refer to a specific node."
    )


def format_iri_or_literal_error(value: str, position: Position | None = None) -> str:
    expected = "IRI or literal" if position == Position.OBJECT else "IRI"
    pos_str = f" for {position}" if position is not None else ""
    return f'Value "{value}"{pos_str} is not a valid {expected}. \
IRIs can be given in prefixed form, like wd:Q937, or in full form, \
like <http://www.wikidata.org/entity/Q937> or \
http://www.wikidata.org/entity/Q937. Be aware that not all IRIs can be \
used in prefixed form due to unsupported characters. If the value \
comes from a previous function call result, make sure to specify it exactly \
as given, with proper escaping and quoting.'


def list_triples(
    managers: list[KgManager],
    kg: str,
    subject: str | None,
    property: str | None,
    obj: str | None,
    page: int,
    unclipped: bool,
    k: int,
    known: set[str],
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
) -> str:
    validate_page(page)

    manager, _ = find_manager(managers, kg)

    triple = []
    bindings = []
    for pos, const in zip(Position, [subject, property, obj]):
        if const is None:
            triple.append(f"?{pos.value[0]}")
            continue

        ver_const = parse_iri_or_literal(
            const,
            manager.iri_literal_parser,
            manager.prefixes,
        )
        if ver_const is not None and ver_const.typ == "bnode":
            raise FunctionCallException(format_bnode_error(const))
        if ver_const is None or (pos != Position.OBJECT and ver_const.typ != "uri"):
            raise FunctionCallException(format_iri_or_literal_error(const, pos))

        bindings.append(f"BIND({ver_const.sparql()} AS ?{pos.value[0]})")
        triple.append(ver_const.sparql())

    triple = " ".join(triple)
    bindings = "\n".join(bindings)
    sparql = f"""\
SELECT ?s ?p ?o WHERE {{
    {triple}
    {bindings}
}} LIMIT {MAX_RESULTS + 1}"""

    try:
        result = manager.execute_sparql(sparql, request_timeout, read_timeout)
    except Exception as e:
        raise FunctionCallException(f"Failed to list triples with error:\n{e}") from e

    assert isinstance(result, SelectResult)
    result.truncate(MAX_RESULTS)
    more = not result.complete

    result.data = [result.data[i] for i in order_triples(result.rows(), manager)]

    # apply pagination
    total_pages = max(1, math.ceil(len(result.data) / k))
    start = (page - 1) * k
    end = page * k
    result.data = result.data[start:end]

    # update known
    update_known_from_index_iris(known, row_iris(result.rows()), manager)

    table = manager.format_sparql_result(
        result,
        show_top_rows=k,
        show_bottom_rows=0,
        show_left_columns=3,
        show_right_columns=0,
        # override column names
        column_names=["subject", "property", "object"],
        clip_literals=not unclipped,
        table_only=True,
    )
    suffix = format_page_suffix(page, total_pages, more)
    if not table:
        return f"No triples ({suffix})"

    return f"Triples ({suffix}):\n{table}"


# popular and diverse triples first, by the ids of the indices
# (lower id means more popular)
def order_triples(rows: Iterable[SelectRow], manager: KgManager) -> list[int]:
    def index_rank(bnd: Binding, index_names: list[str]) -> float:
        for index_name in index_names:
            norm = manager.normalize(bnd.identifier(), index_name)
            data = manager.try_get_data(index_name)
            if norm is None or data is None:
                continue
            id = data.id_from_identifier(norm[0])
            if id is not None:
                return id
        return math.inf

    def is_class(bnd: Binding) -> bool:
        norm = manager.normalize(bnd.identifier(), "classes")
        return norm is not None and manager.check_identifier(norm[0], "classes")

    # sort by property, then by subject or object, which are ranked as
    # entities or else as classes
    def sort_key(row: SelectRow) -> tuple[float, float]:
        ps = index_rank(row["p"], ["properties"])
        es = min(
            index_rank(row["s"], ["entities", "classes"]),
            index_rank(row["o"], ["entities", "classes"]),
        )
        return ps, es

    sorted_rows = sorted(enumerate(rows), key=lambda item: sort_key(item[1]))

    prop_norm = manager.get_normalizer("properties")
    ent_norm = manager.get_normalizer("entities")

    def normalize(bnd: Binding, normalizer: Normalizer) -> str:
        identifier = bnd.identifier()
        norm = normalizer.normalize(identifier)
        return norm[0] if norm is not None else identifier

    # rows with an unseen property or class come first (e.g., all classes
    # of an entity), then rows with unseen subjects or objects
    props_seen = set()
    ents_seen = set()
    classes_seen = set()
    permutation = []

    for i, row in sorted_rows:
        s = normalize(row["s"], ent_norm)
        p = normalize(row["p"], prop_norm)
        o = normalize(row["o"], ent_norm)
        classes = {ent for ent, var in ((s, "s"), (o, "o")) if is_class(row[var])}

        new = p not in props_seen or not classes <= classes_seen
        key = (not new, s in ents_seen or o in ents_seen)
        permutation.append((key, i))

        props_seen.add(p)
        ents_seen.add(s)
        ents_seen.add(o)
        classes_seen.update(classes)

    # stable sort keeps the popularity order within each group
    permutation.sort(key=lambda item: item[0])
    return [i for _, i in permutation]


def search_with_constraints(
    managers: list[KgManager],
    kg: str,
    index: str,
    position: str,
    query: str,
    constraints: dict[str, str | None] | None,
    k: int,
    known: set[str],
    query_type: str = "text",
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
    page: int = 1,
    max_pages: int = 10,
) -> str:
    validate_query(query)
    validate_page(page, max_pages)
    manager, _ = find_manager(managers, kg)

    if constraints is None:
        constraints = {}

    target_constr = constraints.get(position)
    if target_constr is not None:
        raise FunctionCallException(
            f"Cannot look for {position} and constrain it to "
            f'"{target_constr}" at the same time.'
        )

    num_constraints = sum(c is not None for c in constraints.values())
    if num_constraints > 2:
        raise FunctionCallException(
            "At most two of subject, property, and "
            "object should be constrained at once."
        )

    identifier_map = None
    info = ""
    if num_constraints > 0:
        pos_values = {}
        for pos in Position:
            const = constraints.get(pos.value)
            if const is None:
                pos_values[pos] = f"?{pos.value[0]}"
                continue

            elif pos.value == position:
                pos_values[pos] = "?search"
                continue

            ver_const = parse_iri_or_literal(
                const,
                manager.iri_literal_parser,
                manager.prefixes,
            )
            if ver_const is not None and ver_const.typ == "bnode":
                raise FunctionCallException(format_bnode_error(const))
            if ver_const is None or (pos != Position.OBJECT and ver_const.typ != "uri"):
                raise FunctionCallException(format_iri_or_literal_error(const, pos))

            pos_values[pos] = ver_const.sparql()

        select_var = f"?{position[0]}"

        sparql = f"""\
SELECT DISTINCT {select_var} WHERE {{
    {pos_values["subject"]}
    {pos_values["property"]}
    {pos_values["object"]}
}} LIMIT {MAX_RESULTS + 1}"""

        try:
            identifier_map = manager.get_candidate_ids(
                index,
                sparql,
                MAX_RESULTS,
                request_timeout,
                read_timeout,
            )
        except Exception as e:
            info = f"""\
Falling back to an unconstrained search on the full \
search index due to:
{e}

"""

    alternatives = manager.search_index(
        index,
        query,
        k * max_pages + 1,
        identifier_map,
        query_type=query_type,
    )
    alternatives, total_pages, more = paginate_results(alternatives, k, page, max_pages)

    # update known items
    normalizer = manager.get_normalizer(index)
    update_known_from_alts(known, alternatives, normalizer)

    return info + format_index_alternatives(alternatives, k, page, total_pages, more)


def format_page_suffix(page: int, total_pages: int, more: bool) -> str:
    return f"page {page} of {total_pages}{'+' if more else ''}"


def paginate_results(
    items: list,
    k: int,
    page: int,
    max_pages: int,
) -> tuple[list, int, bool]:
    more = len(items) > k * max_pages
    if more:
        items = items[: k * max_pages]
    total_pages = max(1, math.ceil(len(items) / k))
    page_items = items[(page - 1) * k : page * k]
    return page_items, total_pages, more


def format_index_alternatives(
    alternatives: list[Alternative],
    k: int,
    page: int,
    total_pages: int,
    more: bool = False,
) -> str:
    suffix = format_page_suffix(page, total_pages, more)
    if not alternatives:
        return f"No results ({suffix})"

    start_index = (page - 1) * k
    lines = "\n".join(
        f"{start_index + i + 1}. {alt.get_selection_string()}"
        for i, alt in enumerate(alternatives)
    )
    return f"Results ({suffix}):\n{lines}"


def search_with_filter(
    managers: list[KgManager],
    kg: str,
    index: str,
    sparql: str | None,
    query: str,
    k: int,
    known: set[str],
    query_type: str = "text",
    know_before_use: bool = False,
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
    page: int = 1,
    max_pages: int = 10,
) -> str:
    validate_query(query)
    validate_page(page, max_pages)
    manager, others = find_manager(managers, kg)

    identifier_map = None
    info = ""
    if sparql is not None:
        # fix prefixes with managers
        sparql = manager.fix_prefixes(sparql)
        for other in others:
            sparql = other.fix_prefixes(sparql)

        if know_before_use:
            check_known(manager, sparql, known)

        try:
            identifier_map = manager.get_candidate_ids(
                index,
                sparql,
                MAX_RESULTS,
                request_timeout,
                read_timeout,
            )
        except Exception as e:
            info = f"""\
Falling back to an unconstrained search on the full \
search index due to:
{e}

"""

    alternatives = manager.search_index(
        index,
        query,
        k=k * max_pages + 1,
        identifier_map=identifier_map,
        query_type=query_type,
    )
    alternatives, total_pages, more = paginate_results(alternatives, k, page, max_pages)

    # update known items
    normalizer = manager.get_normalizer(index)
    update_known_from_alts(known, alternatives, normalizer)

    return info + format_index_alternatives(alternatives, k, page, total_pages, more)
