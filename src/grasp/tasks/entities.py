from pydantic import BaseModel

from grasp.functions import parse_iri_or_literal
from grasp.manager import KgManager
from grasp.sparql.types import Alternative, ObjType


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
