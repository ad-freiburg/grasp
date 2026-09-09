import random
from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby
from typing import Any, Iterator

from grasp.utils import clip, format_list


class ObjType(StrEnum):
    ENTITY = "entity"
    PROPERTY = "property"
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
        if self == ObjType.ENTITY:
            return "entities"
        elif self == ObjType.PROPERTY:
            return "properties"
        elif self == ObjType.LITERAL:
            return "literals"
        raise ValueError(f"ObjType {self.value} has no associated index")


class Position(StrEnum):
    SUBJECT = "subject"
    PROPERTY = "property"
    OBJECT = "object"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


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


class Alternative:
    def __init__(
        self,
        identifier: str,
        short_identifier: str | None = None,
        label: str | None = None,
        variants: list[str] | None = None,
        aliases: list[str] | None = None,
        info: list[str] | None = None,
        matched_label: str | None = None,
    ) -> None:
        self.identifier = identifier
        self.short_identifier = short_identifier
        self.label = label
        self.aliases = aliases
        self.variants = variants
        self.info = info
        self.matched_label = matched_label

    def __hash__(self) -> int:
        # hash identifier
        return hash(self.identifier)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Alternative):
            return False

        return self.identifier == other.identifier

    def __repr__(self) -> str:
        return f"Alternative({self.label}, {self.get_identifier()}, {self.variants})"

    def get_identifier(self) -> str:
        return self.short_identifier or self.identifier

    def has_label(self) -> bool:
        return bool(self.label)

    def get_label(self) -> str | None:
        return clip(self.label) if self.label else None

    def has_variants(self) -> bool:
        return bool(self.variants)

    def get_selection_string(
        self,
        max_aliases: int = 5,
        max_info: int = 10,
        show_matched_label: bool = True,
        add_info: bool = True,
        include_variants: list[str] | None = None,
    ) -> str:
        s = self.get_label() or self.get_identifier()

        variants = self.variants if include_variants is None else include_variants
        parts = []
        if self.has_label() and not variants:
            parts.append(f"{self.get_identifier()}")
        elif not self.has_label() and variants:
            parts.append(f"as {'/'.join(variants)}")
        elif self.has_label() and variants:
            parts.append(f"{self.get_identifier()} as {'/'.join(variants)}")

        if (
            show_matched_label
            and self.matched_label is not None
            and self.matched_label != self.label
        ):
            parts.append(f'matched via "{clip(self.matched_label)}"')

        if parts:
            s += " ("
            s += ", ".join(parts)
            s += ")"

        if add_info and self.aliases and max_aliases > 0:
            s += ", "
            if self.has_label():
                s += "also "

            if len(self.aliases) > max_aliases:
                aliases = random.sample(self.aliases, max_aliases)
            else:
                aliases = self.aliases

            s += "known as " + ", ".join(f'"{clip(alias)}"' for alias in aliases)

            if len(self.aliases) > max_aliases:
                s += ", etc."

        if add_info and self.info and max_info > 0:
            if len(self.info) > max_info:
                infos = random.sample(self.info, max_info)
            else:
                infos = self.info

            s += ":\n" + format_list((clip(info) for info in infos), indent=2)

        return s


class Selection:
    alternative: Alternative
    obj_type: ObjType
    variant: str | None

    def __init__(
        self,
        alternative: Alternative,
        obj_type: ObjType,
        variant: str | None = None,
    ) -> None:
        self.alternative = alternative
        self.obj_type = obj_type
        self.variant = variant

    def __repr__(self) -> str:
        return f"Selection({self.alternative}, {self.obj_type}, {self.variant})"

    def __hash__(self) -> int:
        return hash((self.alternative, self.obj_type, self.variant))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Selection):
            return False

        return (
            self.alternative == other.alternative
            and self.obj_type == other.obj_type
            and self.variant == other.variant
        )

    @property
    def is_entity_or_property(self) -> bool:
        return self.obj_type == ObjType.ENTITY or self.obj_type == ObjType.PROPERTY

    def get_natural_sparql_label(self, full_identifier: bool = False) -> str:
        identifier = self.alternative.get_identifier()
        if not self.alternative.has_label():
            return identifier

        label: str = self.alternative.get_label()  # type: ignore

        if full_identifier:
            label += f" ({identifier})"
        elif self.variant:
            label += f" ({self.variant})"

        if self.is_entity_or_property:
            return f"<{label}>"
        else:
            return label


def group_selections(
    selections: list[Selection],
) -> dict[ObjType, list[tuple[Alternative, list[str]]]]:
    def key(sel: Selection) -> tuple[str, str]:
        return sel.alternative.identifier, sel.obj_type.name

    grouped = {}
    for _, group in groupby(sorted(selections, key=key), key=key):
        selections = list(group)
        obj_type = selections[0].obj_type

        if obj_type not in grouped:
            grouped[obj_type] = []

        variants = sorted(
            set(selection.variant for selection in selections if selection.variant)
        )
        alt = selections[0].alternative
        grouped[obj_type].append((alt, variants))

    return grouped
