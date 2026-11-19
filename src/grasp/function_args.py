# Function arguments as pydantic models, for definitions independent of model
# SDKs and validated calls; annotations mark KG terms and SPARQL queries

from collections.abc import Sequence
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
    args: type[BaseModel] | None = None,
    enums: dict[str, Sequence[str | None]] | None = None,
) -> dict[str, Any]:
    # strict definition for the model APIs; enums limit arguments to values
    # known at runtime, e.g. kg, also nested ones
    if args is None:
        # an empty object schema makes some vllm models generate endlessly
        return {"name": name, "description": description, "parameters": {}}
    parameters = strict_json_schema(args)
    add_enums(parameters, enums or {})
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "strict": True,
    }


def add_enums(schema: dict[str, Any], enums: dict[str, Sequence[str | None]]) -> None:
    for name, prop in schema.get("properties", {}).items():
        if name in enums:
            prop["enum"] = list(enums[name])
        add_enums(prop, enums)
    if "items" in schema:
        add_enums(schema["items"], enums)


def strict_json_schema(args: type[BaseModel]) -> dict[str, Any]:
    # all properties required and no others, nullable ones with a null type,
    # nested models inline; no titles and defaults, those only apply when validating
    schema = args.model_json_schema()
    definitions = schema.pop("$defs", {})
    return strict_schema(inline_refs(schema, definitions))


def inline_refs(schema: Any, definitions: dict[str, Any]) -> Any:
    # replaces references to nested models with their definitions
    if isinstance(schema, list):
        return [inline_refs(item, definitions) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        name = schema["$ref"].removeprefix("#/$defs/")
        rest = {key: value for key, value in schema.items() if key != "$ref"}
        schema = {**definitions[name], **rest}
    return {key: inline_refs(value, definitions) for key, value in schema.items()}


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
    if "anyOf" in schema:
        variants = [strict_schema(variant) for variant in schema.pop("anyOf")]
        detailed = [variant for variant in variants if variant.keys() != {"type"}]
        if len(detailed) <= 1 and all("type" in variant for variant in variants):
            # e.g. a nullable string or nested model as ["string", "null"] or
            # ["object", "null"], with the properties of the model
            types = [variant["type"] for variant in variants]
            details = {
                key: value
                for variant in detailed
                for key, value in variant.items()
                if key != "type"
            }
            schema = {"type": types, **schema, **details}
        else:
            schema["anyOf"] = variants
    return schema


def arg_kinds(args: type[BaseModel], prefix: str = "") -> dict[str, ArgKind]:
    # marked arguments, nested ones with dotted keys; marks of optional
    # arguments and list items are within their type
    kinds = {}
    for name, field in args.model_fields.items():
        key = f"{prefix}.{name}" if prefix else name
        marks = list(field.metadata)
        for arg in get_args(field.annotation):
            marks.extend(getattr(arg, "__metadata__", ()))
        kind = next((mark for mark in marks if isinstance(mark, ArgKind)), None)
        if kind is not None:
            kinds[key] = kind

        for arg in [field.annotation, *get_args(field.annotation)]:
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                kinds.update(arg_kinds(arg, key))
    return kinds
