from typing import Any

from grasp.configs import GraspConfig
from grasp.functions import ExecutionResult
from grasp.manager import KgManager
from grasp.model import Message, Response
from grasp.sparql.utils import (
    find,
    find_all,
    parse_string,
    parse_to_string_with_whitespace,
    remove_node,
)
from grasp.tasks.base import GraspTask
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import format_list


def functions() -> list[dict]:
    fns = [
        {
            "name": "answer",
            "description": "Finalize your output and stop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sparql": {
                        "type": "string",
                        "description": "The final cleaned SPARQL query",
                    },
                    "questions": {
                        "type": "array",
                        "description": "A list of natural language questions for the SPARQL query",
                        "items": {
                            "type": "string",
                            "description": "A natural language question for the SPARQL query",
                        },
                    },
                },
                "required": ["sparql", "questions"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "cancel",
            "description": "Stop the task without producing an output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason for cancelling the task",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]
    return fns


def rules() -> list[str]:
    return [
        "You can use the cancel function at any time to stop the task without producing an output "
        "(e.g. if the SPARQL query is invalid or does not make sense).",
    ]


def system_information(config: GraspConfig) -> str:
    task_kwargs = config.task_kwargs.get("wikidata-query-logs", {})
    max_questions = task_kwargs.get("max_questions", 3)
    return f"""\
You are a Wikidata expert trying to find possible user questions for \
anonymized SPARQL queries sent to the Wikidata Query Service. \
Your task is to fix and clean the given SPARQL query, \
and generate possible natural language questions for it.

You should take a step-by-step approach to achieve this:
1. Analyze the given SPARQL query, its used entities and properties, and \
execution result. Think about what the user wanted to achieve with this query. \
If this is not clear from the provided information alone, search and query Wikidata \
to gain more context about the SPARQL query.
2. Clean the SPARQL query. This e.g. includes removing superfluous variables or other \
unnecessary parts, finding better variable names, or replacing anonymized string \
literals with sensible values.
3. Formulate your final SPARQL query and execute it over Wikidata to verify its correctness. \
It should not be too different from the original anonymous query in terms of \
intent and its execution result, but you are allowed to deviate if it would make \
the query more natural, precise, etc.
4. For the final SPARQL query, generate between 1 and {max_questions} natural \
language questions that accurately capture its intent. Ensure diversity in \
both phrasing (e.g., keyword-like, question-form, or request-style) and detail \
(e.g., referencing result columns, filters, or other query components). \
Refer to entities and properties by their natural language labels, never by their \
IRIs or ids, e.g. "Victor Hugo" and "author" instead of "wd:Q535" and "wdt:P50", \
except if the question is explicitly about a specific IRI itself.
5. Provide your final output by calling the answer function."""


def remove_service(manager: KgManager, sparql: str) -> str:
    parse, _ = parse_string(sparql, manager.sparql_parser)

    for service in find_all(parse, "ServiceGraphPattern"):
        var_or_iri = service["children"][2]

        iri = find(var_or_iri, "IRIREF")
        if iri is not None and iri["value"] == "<http://wikiba.se/ontology#label>":
            remove_node(service)
            continue

        pname = find(var_or_iri, "PNAME_LN")
        if pname is not None and pname["value"] == "wikibase:label":
            remove_node(service)
            continue

    return parse_to_string_with_whitespace(parse, sparql.encode())


def remove_unused_variables(manager: KgManager, sparql: str) -> str:
    parse, _ = parse_string(sparql, manager.sparql_parser)

    clause = find(parse, "SelectClause")
    if clause is None:
        return sparql

    used = set()
    for var in find_all(parse, "Var", skip={"SelectClause"}):
        used.add(var["children"][0]["value"])

    for var in find_all(clause, "SelectVar"):
        children = var["children"]
        if len(children) != 1:
            continue

        val = children[0]["children"][0]["value"]
        # keep Label variables from service clauses
        if val not in used and not val.endswith("Label"):
            remove_node(var)

    return parse_to_string_with_whitespace(parse, sparql.encode())


def clean_sparql(sparql: str, managers: list[KgManager]) -> str:
    assert len(managers) == 1, "Only one kg manager expected"
    manager = managers[0]
    sparql = remove_service(manager, sparql)
    sparql = remove_unused_variables(manager, sparql)
    sparql = manager.prettify(sparql)
    return sparql


def prepare_sparql(
    sparql: str,
    managers: list[KgManager],
    max_rows: int = 10,
    max_columns: int = 10,
    remove_known: bool = False,
) -> tuple[ExecutionResult, str]:
    assert len(managers) == 1, "Only one kg manager expected"
    manager = managers[0]
    result, selections = prepare_sparql_result(
        sparql,
        manager.kg,
        managers,
        max_rows,
        max_columns,
    )

    # by default the sparql comes with fix_prefixes
    # for the input we do not want that, so we re-apply prefix
    # fixing and prettifying here
    if remove_known:
        try:
            result.sparql = manager.fix_prefixes(result.sparql, remove_known=True)
        except Exception:
            pass

    return result, format_sparql_result(manager, result, selections)


def input_and_state(
    sparql: str,
    managers: list[KgManager],
    max_rows: int,
    max_columns: int,
) -> tuple[str, None]:
    sparql = clean_sparql(sparql, managers)
    _, formatted = prepare_sparql(
        sparql,
        managers,
        max_rows,
        max_columns,
        remove_known=True,
    )
    return formatted, None


def output(
    messages: list[Message],
    managers: list[KgManager],
    max_rows: int = 10,
    max_columns: int = 10,
) -> Any | None:
    try:
        last = messages[-1]
        assert isinstance(last.content, Response)
        tool_call = last.content.tool_calls[0]
        output: dict[str, str] = {"formatted": "No output", **tool_call.args}
        if tool_call.name == "answer":
            output["type"] = "answer"
            questions = tool_call.args["questions"]
            output["formatted"] = f"Questions:\n{format_list(questions)}\n\n"

            result, formatted = prepare_sparql(
                tool_call.args["sparql"],
                managers,
                max_rows,
                max_columns,
            )

            output["sparql_fixed"] = result.sparql
            output["sparql_result"] = result.formatted

            output["formatted"] += formatted

        elif tool_call.name == "cancel":
            output["type"] = "cancel"
            output["reason"] = tool_call.args["reason"]
            output["formatted"] = f"Cancelled:\n{tool_call.args['reason']}"

        else:
            raise ValueError(f"Unknown output tool call {tool_call.name}")

        return output

    except Exception:
        return None


class WdqlTask(GraspTask):
    name = "wikidata-query-logs"

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        return functions()

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        if fn_name == "answer" or fn_name == "cancel":
            return "Stopping"

        else:
            raise ValueError(f"Unknown function {fn_name}")

    def done(self, fn_name: str) -> bool:
        return fn_name in {"answer", "cancel"}

    def setup(self, input: Any) -> str:
        assert isinstance(input, str), (
            "Input for wikidata-query-logs must be a string (SPARQL query)"
        )
        formatted, _ = input_and_state(
            input,
            self.managers,
            self.config.result_max_rows,
            self.config.result_max_columns,
        )
        return formatted

    def output(self, messages: list[Message]) -> dict | None:
        return output(
            messages,
            self.managers,
            self.config.result_max_rows,
            self.config.result_max_columns,
        )

    @property
    def default_input_field(self) -> str | None:
        return "sparql"
