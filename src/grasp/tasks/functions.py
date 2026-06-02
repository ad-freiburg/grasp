import math

from grasp.functions import (
    MAX_RESULTS,
    find_manager,
    format_iri_or_literal_error,
    format_page_suffix,
    parse_iri_or_literal,
    update_known_from_rows,
)
from grasp.manager import KgManager
from grasp.sparql.types import Position, SelectResult
from grasp.utils import FunctionCallException, format_enumerate


def find_frequent_function_definition(kgs: list[str], k: int) -> dict:
    return {
        "name": "find_frequent",
        "description": f"""\
List common IRIs or literals at a given position (subject, property, \
or object) in the knowledge graph for the given constraints. \
Results are ordered by descending by frequency, and at most {k} \
results are returned per page (use pagination to see more results).

For example, to find the most common types used in Wikidata:
find_frequent(kg="wikidata", position="object", property="rdf:type")

Or to find the most frequently used properties in Wikidata:
find_frequent(kg="wikidata", position="property")""",
        "parameters": {
            "type": "object",
            "properties": {
                "kg": {
                    "type": "string",
                    "enum": kgs,
                    "description": "The knowledge graph to query",
                },
                "position": {
                    "type": "string",
                    "enum": ["subject", "property", "object"],
                    "description": "The position to find common values for",
                },
                "subject": {
                    "type": ["string", "null"],
                    "description": "IRI constraint for subject position (null for unconstrained)",
                },
                "property": {
                    "type": ["string", "null"],
                    "description": "IRI constraint for property position (null for unconstrained)",
                },
                "object": {
                    "type": ["string", "null"],
                    "description": "IRI or literal constraint for object position (null for unconstrained)",
                },
                "page": {
                    "type": "integer",
                    "description": "Page number (1-indexed) for paginating results (default should be 1)",
                },
            },
            "required": ["kg", "position", "subject", "property", "object", "page"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def find_frequent(
    managers: list[KgManager],
    kg: str,
    position: str,
    subj: str | None,
    prop: str | None,
    obj: str | None,
    page: int,
    k: int,
    known: set[str],
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
) -> str:
    if page < 1:
        raise FunctionCallException("Page number must be at least 1")

    manager, _ = find_manager(managers, kg)

    constraints = {"subject": subj, "property": prop, "object": obj}

    if constraints.get(position) is not None:
        raise FunctionCallException(
            f"Cannot look for {position} values and constrain them "
            f'to "{constraints[position]}" at the same time.'
        )

    pos_values = []
    for pos, const in zip(Position, [subj, prop, obj]):
        if const is None:
            pos_values.append(f"?{pos.value[0]}")
            continue

        ver_const = parse_iri_or_literal(
            const,
            manager.iri_literal_parser,
            manager.prefixes,
        )
        if ver_const is None or (pos != Position.OBJECT and ver_const.typ != "uri"):
            raise FunctionCallException(format_iri_or_literal_error(const, pos))

        pos_values.append(ver_const.sparql())

    target_var = f"?{position[0]}"

    triple_pattern = " ".join(pos_values)
    sparql = f"""\
SELECT {target_var} ?freq WHERE {{
    {{ SELECT {target_var} (COUNT({target_var}) AS ?freq) {{ {triple_pattern} . }} GROUP BY {target_var} }}
}}
ORDER BY DESC(?freq) {target_var}
LIMIT {MAX_RESULTS + 1}"""

    try:
        result = manager.execute_sparql(sparql, request_timeout, read_timeout)
    except Exception as e:
        raise FunctionCallException(f"Failed to list common {position} values:\n{e}")

    assert isinstance(result, SelectResult)
    result.truncate(MAX_RESULTS)
    more = not result.complete

    # apply pagination
    total_pages = max(1, math.ceil(len(result.data) / k))
    start = (page - 1) * k
    end = page * k
    result.data = result.data[start:end]

    plural = "properties" if position == "property" else f"{position}s"
    suffix = format_page_suffix(page, total_pages, more)
    if len(result) == 0:
        return f"No {plural} ({suffix})"

    # update known
    update_known_from_rows(
        known,
        result.rows(),
        manager.get_normalizer("entities"),
    )
    update_known_from_rows(
        known,
        result.rows(),
        manager.get_normalizer("properties"),
    )

    target_var = position[0]
    items = []
    for row in result.rows():
        val = row[target_var]
        freq = row["freq"].value
        identifier = val.identifier()
        formatted = manager.format_identifier(identifier)

        if val.typ == "uri":
            label = manager.get_label(
                identifier,
                "entities",
            ) or manager.get_label(identifier, "properties")

            if label is not None:
                formatted = f"{label} ({formatted})"

        items.append(f"{formatted} - {int(freq):,} occurrences")

    return f"Most frequent {plural} ({suffix}):\n" + format_enumerate(
        items, start=(page - 1) * k
    )
