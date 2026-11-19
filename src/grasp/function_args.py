# Arguments of functions as pydantic models: the definitions for the model
# APIs are generated from them, independent of their SDKs, and the arguments
# of function calls are validated with them; annotations mark arguments that
# are knowledge graph terms or SPARQL queries

from enum import StrEnum
from typing import Annotated, Any, get_args

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
    parameters = strict_json_schema(args)
    kg = parameters["properties"].get("kg")
    if kg is not None:
        kg["enum"] = kgs
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "strict": True,
    }


def strict_json_schema(args: type[BaseModel]) -> dict[str, Any]:
    # the JSON schema of the arguments for the strict mode of the model APIs:
    # all properties required and no others, nullable ones with a null type;
    # titles and defaults are left out, defaults only apply when validating
    return strict_schema(args.model_json_schema())


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = {
        key: value for key, value in schema.items() if key not in {"title", "default"}
    }
    if "properties" in schema:
        schema["properties"] = {
            name: strict_schema(prop) for name, prop in schema["properties"].items()
        }
        schema["required"] = list(schema["properties"])
        schema["additionalProperties"] = False
    if "items" in schema:
        schema["items"] = strict_schema(schema["items"])
    if "$defs" in schema:
        schema["$defs"] = {
            name: strict_schema(definition)
            for name, definition in schema["$defs"].items()
        }
    if "anyOf" in schema:
        variants = [strict_schema(variant) for variant in schema.pop("anyOf")]
        if all(variant.keys() == {"type"} for variant in variants):
            # e.g. a nullable string as ["string", "null"]
            schema["type"] = [variant["type"] for variant in variants]
        else:
            schema["anyOf"] = variants
    return schema


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
