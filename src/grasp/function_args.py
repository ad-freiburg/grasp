# Arguments of functions as pydantic models: the definitions for the model
# APIs are generated from them, and the arguments of function calls are
# validated with them; annotations mark arguments that are knowledge graph
# terms or SPARQL queries

from enum import StrEnum
from typing import Annotated, Any, get_args

import openai
from pydantic import BaseModel


class ArgKind(StrEnum):
    IRI = "iri"
    # an IRI or a literal
    TERM = "term"
    SPARQL = "sparql"


Iri = Annotated[str, ArgKind.IRI]
Term = Annotated[str, ArgKind.TERM]
Sparql = Annotated[str, ArgKind.SPARQL]


def function_definition(
    name: str,
    description: str,
    args: type[BaseModel],
    kgs: list[str],
) -> dict[str, Any]:
    # a strict definition as the model APIs take it, with the kg argument
    # limited to the given knowledge graphs
    tool = openai.pydantic_function_tool(args, name=name, description=description)
    definition: dict[str, Any] = dict(tool["function"])
    kg = definition["parameters"]["properties"].get("kg")
    if kg is not None:
        kg["enum"] = kgs
    return definition


def arg_kinds(args: type[BaseModel]) -> dict[str, ArgKind]:
    # the arguments marked as knowledge graph terms or SPARQL queries; the
    # mark of an optional argument is within its type
    kinds = {}
    for name, field in args.model_fields.items():
        marks = list(field.metadata)
        for arg in get_args(field.annotation):
            marks.extend(getattr(arg, "__metadata__", ()))
        kind = next((mark for mark in marks if isinstance(mark, ArgKind)), None)
        if kind is not None:
            kinds[name] = kind
    return kinds
