import sys
from dataclasses import dataclass
from itertools import chain

from grasp.manager import KgManager
from grasp.manager.utils import find_term_kind, get_common_sparql_prefixes
from grasp.sparql.types import KgItem, TermKind, Position, Selection, iri_indices
from grasp.sparql.utils import (
    find_all,
    infer_position_from_prefix,
    parse_into_binding,
    parse_string,
)

__all__ = [
    "Term",
    "extract_sparql_terms",
    "selections_from_terms",
]


@dataclass
class Term:
    parse: dict
    span: tuple[int, int]
    prefix: str
    text: str
    suffix: str
    kg_item: KgItem
    kind: TermKind
    variant: str | None
    # inferred for IRIs, None if unknown
    position: Position | None = None

    def same_as(self, other: "Term") -> bool:
        return self.kg_item.identifier == other.kg_item.identifier

    @property
    def is_literal(self) -> bool:
        return self.kind == TermKind.LITERAL

    @property
    def is_unindexed(self) -> bool:
        return self.kind == TermKind.UNINDEXED

    @property
    def is_unknown(self) -> bool:
        return self.kind == TermKind.UNKNOWN

    @property
    def is_common(self) -> bool:
        return self.kind == TermKind.COMMON

    @property
    def has_label(self) -> bool:
        return self.kg_item.has_label()

    @property
    def selection(self) -> Selection:
        return Selection(self.kg_item, self.kind, self.variant)


def byte_span(parse: dict, start: int = sys.maxsize, end: int = 0) -> tuple[int, int]:
    if "children" in parse:
        for child in parse["children"]:
            start, end = byte_span(child, start, end)
        return start, end

    f, t = parse["byte_span"]
    return min(start, f), max(end, t)


COMMON_PREFIXES = get_common_sparql_prefixes()


def get_term(
    parse: dict,
    manager: KgManager,
    sparql_encoded: bytes,
) -> Term | None:
    # return tuple with identifier, variant, label, synonyms
    # and additional info
    (byte_start, byte_end) = byte_span(parse)
    prefix = sparql_encoded[:byte_start].decode()
    text = sparql_encoded[byte_start:byte_end].decode()
    suffix = sparql_encoded[byte_end:].decode()
    start = len(prefix)
    end = start + len(text)

    kwargs = {
        "parse": parse,
        "text": text,
        "span": (start, end),
        "prefix": prefix,
        "suffix": suffix,
    }

    binding = parse_into_binding(text, manager.iri_literal_parser, manager.prefixes)
    if binding is None:
        return None

    if binding.typ == "literal":
        if binding.datatype is not None:
            info = manager.format_iri(binding.datatype)
        elif binding.lang is not None:
            info = binding.lang
        else:
            info = None

        return Term(
            kg_item=KgItem(
                identifier=binding.identifier(),
                short_identifier=binding.identifier(),
                label=binding.value,
                infos=[info] if info else [],
            ),
            kind=TermKind.LITERAL,
            variant=None,
            **kwargs,
        )

    # we have an iri
    iri = binding.identifier()

    try:
        position = infer_position_from_prefix(prefix, manager.sparql_parser)
    except Exception:
        position = None

    kinds = iri_indices(position)

    # check whether the iri is a valid entity, class, or property
    for kind in kinds:
        norm = manager.normalize(iri, kind.index_name)
        if norm is None:
            continue

        identifier, variant = norm
        if not manager.check_identifier(identifier, kind.index_name):
            continue

        infos = manager.get_info_for_identifiers_from_index(
            [identifier], kind.index_name
        )
        kg_item = manager.build_kg_item_with_info(
            identifier,
            infos.get(identifier, {}),
            variants=[variant] if variant else None,
        )

        return Term(
            kg_item=kg_item,
            kind=kind,
            variant=variant,
            position=position,
            **kwargs,
        )

    term_kind = find_term_kind(iri, manager.prefixes, COMMON_PREFIXES)

    variant = None
    if term_kind == TermKind.UNINDEXED:
        # try to get infos from live endpoint
        infos = {}
        for kind in kinds:
            norm = manager.normalize(iri, kind.index_name)
            if norm is None:
                continue

            identifier, norm_variant = norm
            if norm_variant:
                variant = norm_variant

            index_infos = manager.get_info_for_identifiers_from_index(
                [identifier], kind.index_name
            )
            # merge infos across types
            for key, value in index_infos.get(identifier, {}).items():
                if key not in infos or not isinstance(infos[key], list):
                    infos[key] = value
                else:
                    assert isinstance(value, list)
                    infos[key].extend(value)

    else:
        infos = None

    return Term(
        kg_item=manager.build_kg_item_with_info(
            identifier=iri,
            info=infos,
            variants=[variant] if variant else None,
        ),
        kind=term_kind,
        variant=None,
        position=position,
        **kwargs,
    )


def selections_from_terms(terms: list[Term]) -> list[Selection]:
    return [term.selection for term in terms]


def selections_from_sparql(
    sparql: str,
    manager: KgManager,
) -> list[Selection]:
    _, terms = extract_sparql_terms(sparql, manager)
    return selections_from_terms(terms)


def extract_sparql_terms(
    sparql: str,
    manager: KgManager,
    is_prefix: bool = False,
) -> tuple[str, list[Term]]:
    sparql_encoded = sparql.encode()

    parse, _ = parse_string(
        sparql,
        manager.sparql_parser,
        collapse_single=False,
        skip_empty=True,
        is_prefix=is_prefix,
    )

    # get all terms in triples
    terms = filter(
        lambda term: term is not None,
        chain(
            (
                # get IRIs (excluding prefixes)
                get_term(iri, manager, sparql_encoded)
                for iri in find_all(parse, name="iri", skip={"Prologue"})
            ),
            # commented out for now, since not used anywhere
            # (
            #     # get literals in triples
            #     get_term(lit, manager, sparql_encoded)
            #     for triple in find_all(parse, name="TriplesSameSubject")
            #     for lit in find_all(
            #         triple,
            #         name={"RDFLiteral", "NumericLiteral", "BooleanLiteral"},
            #     )
            # ),
        ),
    )

    # by occurence position in the query
    return sparql, sorted(terms, key=lambda term: term.span)  # type: ignore
