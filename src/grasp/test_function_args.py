from pydantic import BaseModel, Field

from grasp.function_args import ArgKind, Iri, Sparql, arg_kinds, function_definition
from grasp.functions import ExecuteArgs, ListArgs
from grasp.tasks.sparql_qa import AnswerArgs


def test_arg_kinds():
    assert arg_kinds(ExecuteArgs) == {"sparql": ArgKind.SPARQL}
    assert arg_kinds(ListArgs) == {
        "subject": ArgKind.IRI,
        "property": ArgKind.IRI,
        "object": ArgKind.TERM,
    }
    assert arg_kinds(AnswerArgs) == {"sparql": ArgKind.SPARQL}


def test_function_definition():
    definition = function_definition(
        "list",
        "List triples.",
        ListArgs,
        {"kg": ["wd", "fb"]},
    )
    assert definition["name"] == "list"
    assert definition["strict"] is True
    parameters = definition["parameters"]
    # strict mode: all arguments required, optional ones nullable
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == set(ListArgs.model_fields)
    assert parameters["properties"]["subject"]["type"] == ["string", "null"]
    assert parameters["properties"]["kg"]["enum"] == ["wd", "fb"]
    # titles and defaults are left out
    assert "title" not in parameters
    assert parameters["properties"]["page"] == {
        "description": ListArgs.model_fields["page"].description,
        "type": "integer",
    }


def test_args_validation():
    args = ListArgs.model_validate({"kg": "wd", "subject": "wd:Q1", "page": "2"})
    assert args.subject == "wd:Q1" and args.property is None and args.page == 2
    assert args.unclipped is False


class Attempt(BaseModel):
    kg: str = Field(description="The knowledge graph")
    sparql: Sparql = Field(description="The query")


class NestedArgs(BaseModel):
    explanation: str = Field(description="Why")
    best_attempt: Attempt | None = Field(default=None, description="The attempt")
    items: list[Iri] = Field(default=[], description="Some items")


def test_nested_args():
    assert arg_kinds(NestedArgs) == {
        "best_attempt.sparql": ArgKind.SPARQL,
        "items": ArgKind.IRI,
    }
    definition = function_definition("cancel", "Cancel.", NestedArgs, {"kg": ["wd"]})
    # nested models inline, as in hand-written definitions
    parameters = definition["parameters"]
    assert "$defs" not in parameters
    assert parameters["properties"]["best_attempt"] == {
        "type": ["object", "null"],
        "description": "The attempt",
        "properties": {
            "kg": {
                "description": "The knowledge graph",
                "type": "string",
                "enum": ["wd"],
            },
            "sparql": {"description": "The query", "type": "string"},
        },
        "required": ["kg", "sparql"],
        "additionalProperties": False,
    }
    assert parameters["properties"]["items"] == {
        "description": "Some items",
        "type": "array",
        "items": {"type": "string"},
    }
