from pydantic import BaseModel

from grasp.functions import parse_iri_or_literal
from grasp.manager import KgManager
from grasp.sparql.types import ObjType


class Entity(BaseModel):
    identifier: str
    entity: str
    label: str | None = None
    aliases: list[str] | None = None
    infos: list[str] | None = None

    def format(self) -> str:
        output = f"Full IRI: {self.identifier}"
        if self.entity != self.identifier:
            output += f", shortened IRI: {self.entity}"
        if self.label is not None:
            output += f", label: {self.label}"
        if self.aliases is not None and self.aliases != []:
            output += f", aliases: {self.aliases}"
        if self.infos is not None and self.infos != []:
            output += f", infos: {self.infos}"
        return output


def prepare_entity(manager: KgManager, identifier: str) -> Entity:
    binding = parse_iri_or_literal(identifier, manager.iri_literal_parser, manager.prefixes)
    if binding is None or binding.typ != "uri":
        raise ValueError(f"{identifier} is not a valid IRI")

    identifier = binding.identifier()

    norm = manager.normalize(identifier, ObjType.ENTITY.index_name)
    if norm is not None:
        identifier, _ = norm

    infos = manager.get_info_for_identifiers_from_index(
        [identifier], ObjType.ENTITY.index_name
    )
    info = infos.get(identifier, {})

    # format normalized identifier again, so always
    # prefixed form is shown if available
    formatted_entity = manager.format_iri(identifier)
    label = info.get("label")
    aliases = info.get("alias", [])
    infos = info.get("other", [])

    return Entity(
        identifier=identifier,
        entity=formatted_entity,
        label=label,
        aliases=aliases,
        infos=infos,
    )
