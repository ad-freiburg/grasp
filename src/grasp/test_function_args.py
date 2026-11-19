from grasp.function_args import ArgKind, arg_kinds, function_definition
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
    definition = function_definition("list", "List triples.", ListArgs, ["wd", "fb"])
    assert definition["name"] == "list"
    assert definition["strict"] is True
    parameters = definition["parameters"]
    # strict mode: all arguments required, optional ones nullable
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == set(ListArgs.model_fields)
    assert {"type": "null"} in parameters["properties"]["subject"]["anyOf"]
    assert parameters["properties"]["kg"]["enum"] == ["wd", "fb"]


def test_args_validation():
    args = ListArgs.model_validate({"kg": "wd", "subject": "wd:Q1", "page": "2"})
    assert args.subject == "wd:Q1" and args.property is None and args.page == 2
    assert args.unclipped is False
