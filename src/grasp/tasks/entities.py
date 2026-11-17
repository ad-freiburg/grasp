from typing import Iterable

from pydantic import BaseModel

from grasp.functions import SearchItem, alternative_item, parse_iri_or_literal
from grasp.manager import KgManager, format_kgs
from grasp.model import ToolCallResult
from grasp.sparql.types import Alternative, ObjType
from grasp.utils import (
    FunctionCallException,
    format_enumerate,
    format_list,
    format_notes,
    format_section,
)

# placeholder for mentions that refer to an entity not in the knowledge graphs
NIL = "<NIL>"


class Entity(BaseModel):
    identifier: str
    entity: str
    label: str | None = None
    aliases: list[str] | None = None
    infos: list[str] | None = None

    def to_alternative(self) -> Alternative:
        return Alternative(
            self.identifier,
            short_identifier=self.entity,
            label=self.label,
            aliases=self.aliases,
            info=self.infos,
        )

    @property
    def is_nil(self) -> bool:
        return self.identifier == NIL


class AnnotationsResult(ToolCallResult):
    # message of annotate or delete_annotation, empty for show_annotations
    message: str = ""
    # the listed annotated entities, if any
    entities: list[SearchItem] = []


def nil_entity() -> Entity:
    return Entity(identifier=NIL, entity=NIL)


def prepare_entity(manager: KgManager, identifier: str) -> Entity:
    binding = parse_iri_or_literal(
        identifier, manager.iri_literal_parser, manager.prefixes
    )
    if binding is None or binding.typ != "uri":
        raise ValueError(f"{identifier} is not a valid IRI")

    identifier = binding.identifier()

    norm = manager.normalize(identifier, ObjType.ENTITY.index_name)
    if norm is not None:
        identifier, _ = norm

    all_infos = manager.get_info_for_identifiers_from_index(
        [identifier], ObjType.ENTITY.index_name
    )
    info = all_infos.get(identifier, {})

    # format normalized identifier again, so always
    # prefixed form is shown if available
    formatted_entity = manager.format_iri(identifier)
    label = info.get("label")
    aliases = info.get("alias", [])
    # note: the manager rewrites the "info" type to "other" when storing
    # (see KgManager.retrieve_info_for_identifiers), so "other" is the
    # correct key here; reading "info" always yielded an empty list
    infos = info.get("other", [])

    return Entity(
        identifier=identifier,
        entity=formatted_entity,
        label=label,
        aliases=aliases,
        infos=infos,
    )


def prepare_annotation(
    manager: KgManager,
    entity: str | None,
    known: set[str],
    know_before_annotate: bool,
) -> Entity:
    # None means the mention refers to an entity not in the knowledge graphs
    if entity is None:
        return nil_entity()

    annotation = prepare_entity(manager, entity)
    if know_before_annotate and annotation.identifier not in known:
        raise FunctionCallException(
            f"The entity {entity} cannot be used for annotation "
            "without being known from previous function call results. "
            "This does not mean it is invalid, but you should verify "
            "that it indeed exists (e.g., by listing example triples) "
            "in the knowledge graphs first."
        )
    return annotation


def annotated_entities(annotations: Iterable[Entity]) -> tuple[str, list[SearchItem]]:
    entities: dict[str, Alternative] = {}
    for annot in annotations:
        if annot.is_nil or annot.identifier in entities:
            continue

        entities[annot.identifier] = annot.to_alternative()

    if not entities:
        return "", []

    lines = []
    items = []
    for _, alt in sorted(entities.items()):
        line, item = alternative_item(alt)
        lines.append(line)
        items.append(item)

    return f"\n\nAnnotated entities:\n{format_list(lines)}", items


def format_annotated_entities(annotations: Iterable[Entity]) -> str:
    formatted, _ = annotated_entities(annotations)
    return formatted


def annotation_feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
    rules: list[str],
    kind: str,
) -> str:
    return "\n\n".join(
        [
            f"You are a {kind} annotation assistant providing feedback on the "
            f"output of a {kind} annotation system for a given input {kind}.",
            format_section(
                "Available knowledge graphs",
                format_kgs(managers, kg_notes),
            ),
            format_section(
                "General notes across knowledge graphs",
                format_notes(notes, enumerated=True),
            ),
            format_section(
                "Rules to follow",
                format_enumerate(rules) if rules else "None",
            ),
            "Provide your feedback with the give_feedback function.",
        ]
    )


def annotation_feedback_instructions(inputs: list[str], output: dict) -> str:
    assert inputs, "At least one input is required for feedback"

    sections = []
    if len(inputs) > 1:
        sections.append(
            format_section(
                "Previous inputs",
                "\n\n".join(i.strip() for i in inputs[:-1]),
            )
        )

    sections.append(format_section("Input", inputs[-1].strip()))
    sections.append(format_section("Annotations", output["formatted"]))
    return "\n\n".join(sections)
