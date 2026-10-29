import random
from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby
from types import ModuleType
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field

from grasp.utils import clip, format_list


TrimMode = Literal["sample", "rank"]


class TermKind(StrEnum):
    ENTITY = "entity"
    PROPERTY = "property"
    CLASS = "class"
    COMMON = "common"
    UNINDEXED = "unindexed"
    UNKNOWN = "unknown"
    LITERAL = "literal"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    @property
    def index_name(self) -> str:
        if self == TermKind.ENTITY:
            return "entities"
        elif self == TermKind.PROPERTY:
            return "properties"
        elif self == TermKind.CLASS:
            return "classes"
        elif self == TermKind.LITERAL:
            return "literals"
        raise ValueError(f"TermKind {self.value} has no associated index")


class Position(StrEnum):
    SUBJECT = "subject"
    PROPERTY = "property"
    OBJECT = "object"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


# standard indices in setup order, add new ones here and to IRI_INDICES
STANDARD_INDICES = [
    TermKind.PROPERTY,
    TermKind.CLASS,
    TermKind.ENTITY,
    TermKind.LITERAL,
]
STANDARD_INDEX_NAMES = [kind.index_name for kind in STANDARD_INDICES]

# standard indices of IRIs in lookup order, classes before entities since
# entity indices often contain classes too
IRI_INDICES = [TermKind.CLASS, TermKind.ENTITY, TermKind.PROPERTY]
IRI_INDEX_NAMES = [kind.index_name for kind in IRI_INDICES]


# properties first at the predicate position
def iri_indices(position: Position | None = None) -> list[TermKind]:
    if position == Position.PROPERTY:
        return sorted(IRI_INDICES, key=lambda kind: kind != TermKind.PROPERTY)
    return list(IRI_INDICES)


@dataclass
class AskResult:
    boolean: bool

    def __len__(self) -> int:
        return 1

    @property
    def is_empty(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AskResult):
            return False

        return self.boolean == other.boolean


@dataclass
class Binding:
    typ: str
    value: str
    datatype: str | None = None
    lang: str | None = None

    def __hash__(self) -> int:
        return hash((self.typ, self.value, self.datatype, self.lang))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Binding):
            return False

        return (
            self.typ == other.typ
            and self.value == other.value
            and self.datatype == other.datatype
            and self.lang == other.lang
        )

    @staticmethod
    def from_dict(data: dict) -> "Binding":
        # virtuoso emits the pre-2013 "typed-literal" where the spec uses "literal"
        typ = data["type"]
        if typ == "typed-literal":
            typ = "literal"

        return Binding(
            typ=typ,
            value=data["value"],
            datatype=data.get("datatype"),
            lang=data.get("xml:lang"),
        )

    def identifier(self) -> str:
        assert self.typ in ["uri", "literal", "bnode"]
        match self.typ:
            case "uri":
                return self.value
            case "literal":
                if self.datatype is not None:
                    return f'"{self.value}"^^<{self.datatype}>'
                elif self.lang is not None:
                    return f'"{self.value}"@{self.lang}'
                else:
                    return f'"{self.value}"'
            case "bnode":
                return f"_:{self.value}"
            case _:
                raise ValueError(f"Unknown binding type: {self.typ}")

    def sparql(self) -> str:
        identifier = self.identifier()

        if self.typ == "uri":
            identifier = f"<{identifier}>"

        return identifier

    def __repr__(self) -> str:
        return self.identifier()


SelectRow = dict[str, Binding]


@dataclass
class TableValue:
    # a value of a result row; short_identifier and label are only set for
    # IRIs, label if the IRI has one in an index
    type: str
    value: str
    datatype: str | None = None
    lang: str | None = None
    short_identifier: str | None = None
    label: str | None = None

    def binding(self) -> Binding:
        return Binding(self.type, self.value, self.datatype, self.lang)


@dataclass
class TableColumn:
    # a column of a result table, name is the column name shown to the model
    variable: str
    name: str


# rows of a result table keep the values of all columns, also the omitted ones
TableRow = dict[str, TableValue]


@dataclass
class SelectResult:
    variables: list[str]
    data: list[dict | None]
    complete: bool = True

    @staticmethod
    def from_json(data: dict) -> "SelectResult":
        return SelectResult(
            variables=data["head"]["vars"],
            data=data["results"]["bindings"],
        )

    def truncate(self, max_rows: int) -> None:
        if len(self) <= max_rows:
            return

        self.data = self.data[:max_rows]
        self.complete = False

    def __len__(self) -> int:
        return len(self.data)

    def bindings(
        self,
        start: int = 0,
        end: int | None = None,
    ) -> Iterator[tuple[Binding, ...]]:
        for row in self.rows(start, end):
            bindings = tuple(row[var] for var in self.variables if var in row)
            yield bindings

    def rows(self, start: int = 0, end: int | None = None) -> Iterator[SelectRow]:
        start = max(start, 0)

        if end is None:
            end = len(self.data)
        else:
            end = min(end, len(self.data))

        for i in range(start, end):
            data = self.data[i]
            if data is None:
                yield {}
            else:
                yield {
                    var: Binding.from_dict(data[var])
                    for var in self.variables
                    if var in data
                }

    @property
    def num_rows(self) -> int:
        return len(self.data)

    @property
    def num_columns(self) -> int:
        return len(self.variables)

    @property
    def is_empty(self) -> bool:
        return not self.data

    def to_ask_result(self) -> AskResult:
        return AskResult(not self.is_empty)


@dataclass
class Example:
    question: str
    sparql: str


class KgItem(BaseModel):
    # an entity, property, or other item of a knowledge graph; num_aliases is
    # the number of aliases before trimming
    identifier: str
    short_identifier: str | None = None
    label: str | None = None
    variants: list[str] | None = None
    aliases: list[str] = []
    num_aliases: int | None = None
    infos: list[str] = []
    matched_label: str | None = None
    # search score, only comparable within one index type, not shown
    score: float | None = Field(default=None, exclude=True)

    def __hash__(self) -> int:
        # hash identifier
        return hash(self.identifier)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, KgItem):
            return False

        return self.identifier == other.identifier

    def __repr__(self) -> str:
        return f"KgItem({self.label}, {self.get_identifier()}, {self.variants})"

    def get_identifier(self) -> str:
        return self.short_identifier or self.identifier

    def has_label(self) -> bool:
        return bool(self.label)

    def get_label(self) -> str | None:
        return clip(self.label) if self.label else None

    def has_variants(self) -> bool:
        return bool(self.variants)

    def trim(
        self,
        max_aliases: int = 5,
        max_infos: int = 10,
        mode: TrimMode = "sample",
        seed: int | None = None,
    ) -> "KgItem":
        # copy with the aliases and infos shown to the model, rank keeps the
        # shortest aliases and the longest infos
        rng = random.Random(seed) if seed is not None else random
        aliases = trim_values(self.aliases, max_aliases, mode, rng)
        infos = trim_values(self.infos, max_infos, mode, rng, longest=True)

        return self.model_copy(
            update={
                "aliases": aliases,
                "num_aliases": max(self.num_aliases or 0, len(self.aliases)),
                "infos": infos,
            }
        )


def trim_values(
    values: list[str],
    n: int,
    mode: TrimMode,
    rng: random.Random | ModuleType,
    longest: bool = False,
) -> list[str]:
    # rank keeps the top n in rank order, sample draws n from the sorted
    # values and puts them in rank order; neither depends on the order of
    # the engine
    if n <= 0:
        return []

    if mode == "sample" and len(values) > n:
        values = rng.sample(sorted(values), n)

    values = sorted(values, key=lambda v: (-len(v) if longest else len(v), v))
    return values[:n]


def format_kg_item(
    item: KgItem,
    show_matched_label: bool = True,
    add_info: bool = True,
    variants: list[str] | None = None,
) -> str:
    # formats all aliases and infos of the item, trim it first to show
    # only some; variants replace the ones of the item if given
    s = item.get_label() or item.get_identifier()

    variants = item.variants if variants is None else variants
    parts = []
    if item.has_label() and not variants:
        parts.append(f"{item.get_identifier()}")
    elif not item.has_label() and variants:
        parts.append(f"as {'/'.join(variants)}")
    elif item.has_label() and variants:
        parts.append(f"{item.get_identifier()} as {'/'.join(variants)}")

    if (
        show_matched_label
        and item.matched_label is not None
        and item.matched_label != item.label
    ):
        parts.append(f'matched via "{clip(item.matched_label)}"')

    if parts:
        s += " ("
        s += ", ".join(parts)
        s += ")"

    if add_info and item.aliases:
        s += ", "
        if item.has_label():
            s += "also "

        s += "known as " + ", ".join(f'"{clip(alias)}"' for alias in item.aliases)

        if (item.num_aliases or 0) > len(item.aliases):
            s += ", etc."

    if add_info and item.infos:
        s += ":\n" + format_list((clip(info) for info in item.infos), indent=2)

    return s


class Selection:
    kg_item: KgItem
    kind: TermKind
    variant: str | None

    def __init__(
        self,
        kg_item: KgItem,
        kind: TermKind,
        variant: str | None = None,
    ) -> None:
        self.kg_item = kg_item
        self.kind = kind
        self.variant = variant

    def __repr__(self) -> str:
        return f"Selection({self.kg_item}, {self.kind}, {self.variant})"

    def __hash__(self) -> int:
        return hash((self.kg_item, self.kind, self.variant))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Selection):
            return False

        return (
            self.kg_item == other.kg_item
            and self.kind == other.kind
            and self.variant == other.variant
        )


def group_selections(
    selections: list[Selection],
) -> dict[TermKind, list[tuple[KgItem, list[str]]]]:
    def key(sel: Selection) -> tuple[str, str]:
        return sel.kg_item.identifier, sel.kind.name

    grouped = {}
    for _, group in groupby(sorted(selections, key=key), key=key):
        selections = list(group)
        kind = selections[0].kind

        if kind not in grouped:
            grouped[kind] = []

        variants = sorted(
            set(selection.variant for selection in selections if selection.variant)
        )
        item = selections[0].kg_item
        grouped[kind].append((item, variants))

    return grouped
