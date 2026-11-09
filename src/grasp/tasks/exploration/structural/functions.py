from grasp.functions import (
    find_manager,
    format_bnode_error,
    format_iri_or_literal_error,
    parse_iri_or_literal,
)
from grasp.manager import KgManager
from grasp.sparql.types import iri_indices
from grasp.tasks.exploration.functions import (
    note_function_definitions as base_note_function_definitions,
)
from grasp.utils import FunctionCallException, format_enumerate


def note_function_definitions(managers: list[KgManager]) -> list[dict]:
    kgs = [manager.kg for manager in managers]
    functions = base_note_function_definitions(managers, general=False)
    functions.extend(
        [
            {
                "name": "mark_explored",
                "description": "Mark an IRI as the seed for this exploration round. "
                "Exactly one seed must be marked per round.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph of the seed to mark as explored",
                        },
                        "iri": {
                            "type": "string",
                            "description": "The IRI of the seed to mark as explored",
                        },
                    },
                    "required": ["kg", "iri"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "show_explored",
                "description": "Show previously explored seeds, most recent first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph for which to show explored seeds",
                        },
                        "page": {
                            "type": "integer",
                            "description": "Page number (1-indexed) for paginating results (default should be 1)",
                        },
                    },
                    "required": ["kg", "page"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]
    )
    return functions


def mark_explored(
    managers: list[KgManager],
    kg: str,
    explored: dict[str, list[str]],
    iri: str,
    explored_this_round: bool,
) -> str:
    if explored_this_round:
        raise FunctionCallException("A seed was already marked as explored this round")

    manager, _ = find_manager(managers, kg)
    ver_iri = parse_iri_or_literal(iri, manager.iri_literal_parser, manager.prefixes)
    if ver_iri is not None and ver_iri.typ == "bnode":
        raise FunctionCallException(format_bnode_error(iri))
    if ver_iri is None or ver_iri.typ != "uri":
        raise FunctionCallException(format_iri_or_literal_error(iri))

    kg_explored = explored.setdefault(kg, [])
    if ver_iri in kg_explored:
        raise FunctionCallException(
            "This seed was already explored in a previous round"
        )

    kg_explored.append(ver_iri.identifier())
    return f'Marked "{iri}" as explored'


def show_explored(
    managers: list[KgManager],
    kg: str,
    explored: dict[str, list[str]],
    page: int,
    k: int,
) -> str:
    if page < 1:
        raise FunctionCallException("Page number must be at least 1")

    kg_explored = explored.get(kg, [])
    if not kg_explored:
        return "None"

    total_pages = (len(kg_explored) + k - 1) // k
    if page > total_pages:
        raise FunctionCallException(f"Page number exceeds maximum page {total_pages}")

    # most recent first
    kg_explored = list(reversed(kg_explored))

    start = (page - 1) * k
    end = page * k
    page_items = kg_explored[start:end]

    header = f"Most recently explored seeds (page {page} of {total_pages}):\n"

    manager, _ = find_manager(managers, kg)

    # group seeds by the first index containing them (entities otherwise)
    # to batch-fetch their infos
    groups: dict[str, list[str]] = {}
    for iri in page_items:
        groups.setdefault(seed_index_name(iri, manager), []).append(iri)

    infos: dict[str, dict] = {}
    for idx, ids in groups.items():
        if ids:
            infos.update(manager.get_info_for_identifiers_from_index(ids, idx))

    items = []
    for iri in page_items:
        alt = manager.build_alternative_with_info(iri, infos.get(iri))
        items.append(alt.get_selection_string())

    return header + format_enumerate(items, start=start)


def seed_index_name(iri: str, manager: KgManager) -> str:
    for obj_type in iri_indices():
        norm = manager.normalize(iri, obj_type.index_name)
        if norm is not None and manager.check_identifier(norm[0], obj_type.index_name):
            return obj_type.index_name
    return "entities"
