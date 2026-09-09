import sys
from dataclasses import dataclass
from itertools import chain
from typing import Optional

from grasp.manager import KgManager
from grasp.manager.utils import find_obj_type_from_prefixes, get_common_sparql_prefixes
from grasp.sparql.types import Alternative, ObjType, Position, Selection
from grasp.sparql.utils import (
    find_all,
    infer_position_from_prefix,
    parse_into_binding,
    parse_string,
)

__all__ = [
    "Item",
    "extract_sparql_items",
    "natural_sparql_from_items",
    "selections_from_items",
]


@dataclass
class Item:
    parse: dict
    item_span: tuple[int, int]
    prefix: str
    item: str
    suffix: str
    alternative: Alternative
    obj_type: ObjType
    variant: str | None

    def same_as(self, other: "Item") -> bool:
        return self.alternative.identifier == other.alternative.identifier

    @property
    def full_prefix(self) -> str:
        return self.prefix + self.item

    @property
    def is_literal(self) -> bool:
        return self.obj_type == ObjType.LITERAL

    @property
    def is_unindexed(self) -> bool:
        return self.obj_type == ObjType.UNINDEXED

    @property
    def is_unknown(self) -> bool:
        return self.obj_type == ObjType.UNKNOWN

    @property
    def is_common(self) -> bool:
        return self.obj_type == ObjType.COMMON

    @property
    def has_label(self) -> bool:
        return self.alternative.has_label()

    @property
    def is_entity_or_property(self) -> bool:
        return self.obj_type == ObjType.ENTITY or self.obj_type == ObjType.PROPERTY

    @property
    def selection(self) -> Selection:
        return Selection(self.alternative, self.obj_type, self.variant)

    def continuation(self, other: Optional["Item"]) -> str:
        end, _ = self.item_span
        if other is None:
            return self.full_prefix[:end]

        assert other.item_span < self.item_span, "other item must come before this one"
        assert self.prefix.startswith(other.prefix), "prefix mismatch"

        _, start = other.item_span
        return self.full_prefix[start:end]


def byte_span(parse: dict, start: int = sys.maxsize, end: int = 0) -> tuple[int, int]:
    if "children" in parse:
        for child in parse["children"]:
            start, end = byte_span(child, start, end)
        return start, end

    f, t = parse["byte_span"]
    return min(start, f), max(end, t)


COMMON_PREFIXES = get_common_sparql_prefixes()


def get_item(
    parse: dict,
    manager: KgManager,
    sparql_encoded: bytes,
) -> Item | None:
    # return tuple with identifier, variant, label, synonyms
    # and additional info
    (byte_start, byte_end) = byte_span(parse)
    prefix = sparql_encoded[:byte_start].decode()
    item = sparql_encoded[byte_start:byte_end].decode()
    suffix = sparql_encoded[byte_end:].decode()
    start = len(prefix)
    end = start + len(item)

    kwargs = {
        "parse": parse,
        "item": item,
        "item_span": (start, end),
        "prefix": prefix,
        "suffix": suffix,
    }

    binding = parse_into_binding(item, manager.iri_literal_parser, manager.prefixes)
    if binding is None:
        return None

    if binding.typ == "literal":
        if binding.datatype is not None:
            info = manager.format_iri(binding.datatype)
        elif binding.lang is not None:
            info = binding.lang
        else:
            info = None

        return Item(
            alternative=Alternative(
                identifier=binding.identifier(),
                short_identifier=binding.identifier(),
                label=binding.value,
                info=[info] if info else None,
            ),
            obj_type=ObjType.LITERAL,
            variant=None,
            **kwargs,
        )

    # we have an iri
    iri = binding.identifier()

    try:
        position = infer_position_from_prefix(prefix, manager.sparql_parser)
        if position in [Position.SUBJECT, Position.OBJECT]:
            obj_types = [ObjType.ENTITY]
        else:
            obj_types = [ObjType.PROPERTY]
    except Exception:
        obj_types = [ObjType.PROPERTY, ObjType.ENTITY]

    # check whether the iri is a valid entity or property
    for obj_type in obj_types:
        norm = manager.normalize(iri, obj_type.index_name)
        if norm is None:
            continue

        identifier, variant = norm
        if not manager.check_identifier(identifier, obj_type.index_name):
            continue

        infos = manager.get_info_for_identifiers_from_index(
            [identifier], obj_type.index_name
        )
        alternative = manager.build_alternative_with_info(
            identifier,
            infos.get(identifier, {}),
            variants=[variant] if variant else None,
        )

        return Item(
            alternative=alternative,
            obj_type=obj_type,
            variant=variant,
            **kwargs,
        )

    item_obj_type = find_obj_type_from_prefixes(iri, manager.prefixes, COMMON_PREFIXES)

    variant = None
    if item_obj_type == ObjType.UNINDEXED:
        # try to get infos from live endpoint
        infos = {}
        for obj_type in obj_types:
            norm = manager.normalize(iri, obj_type.index_name)
            if norm is None:
                continue

            identifier, obj_type_variant = norm
            if obj_type_variant:
                variant = obj_type_variant

            obj_type_infos = manager.get_info_for_identifiers_from_index(
                [identifier], obj_type.index_name
            )
            # merge infos across types
            for key, value in obj_type_infos.get(identifier, {}).items():
                if key not in infos or not isinstance(infos[key], list):
                    infos[key] = value
                else:
                    assert isinstance(value, list)
                    infos[key].extend(value)

    else:
        infos = None

    return Item(
        alternative=manager.build_alternative_with_info(
            identifier=iri,
            info=infos,
            variants=[variant] if variant else None,
        ),
        obj_type=item_obj_type,
        variant=None,
        **kwargs,
    )


def selections_from_items(item: list[Item]) -> list[Selection]:
    return [item.selection for item in item]


def selections_from_sparql(
    sparql: str,
    manager: KgManager,
) -> list[Selection]:
    _, items = extract_sparql_items(sparql, manager)
    return selections_from_items(items)


def natural_sparql_from_items(
    items: list[Item],
    is_prefix: bool = False,
    full_identifier: bool = False,
) -> str:
    prefix = ""
    for i, item in enumerate(items):
        prev = items[i - 1] if i > 0 else None
        prefix += item.continuation(prev)
        prefix += item.selection.get_natural_sparql_label(full_identifier)
        if i == len(items) - 1 and not is_prefix:
            prefix += item.suffix
    return prefix


def extract_sparql_items(
    sparql: str,
    manager: KgManager,
    is_prefix: bool = False,
) -> tuple[str, list[Item]]:
    sparql_encoded = sparql.encode()

    parse, _ = parse_string(
        sparql,
        manager.sparql_parser,
        collapse_single=False,
        skip_empty=True,
        is_prefix=is_prefix,
    )

    # get all items in triples
    items = filter(
        lambda item: item is not None,
        chain(
            (
                # get IRIs (excluding prefixes)
                get_item(iri, manager, sparql_encoded)
                for iri in find_all(parse, name="iri", skip={"Prologue"})
            ),
            # commented out for now, since not used anywhere
            # (
            #     # get literals in triples
            #     get_item(lit, manager, sparql_encoded)
            #     for triple in find_all(parse, name="TriplesSameSubject")
            #     for lit in find_all(
            #         triple,
            #         name={"RDFLiteral", "NumericLiteral", "BooleanLiteral"},
            #     )
            # ),
        ),
    )

    # by occurence position in the query
    return sparql, sorted(items, key=lambda item: item.item_span)  # type: ignore
