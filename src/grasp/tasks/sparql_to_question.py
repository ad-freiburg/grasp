from typing import Any

from grasp.configs import GraspConfig
from grasp.functions import ExecutionResult
from grasp.manager import KgManager
from grasp.model import Message, Response
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
        "If there is only one knowledge graph available, assume the SPARQL query is meant for that "
        "particular knowledge graph and do not mention it explicitly in the generated questions.",
        # "If a given SPARQL query times out, it does not necessarily mean it should be altered "
        # "to finish within the time limit, especially if it would require significant changes to do so.",
    ]


def system_information(config: GraspConfig) -> str:
    task_kwargs = config.task_kwargs.get("sparql-to-question", {})
    max_questions = task_kwargs.get("max_questions", 3)
    level_of_detail = task_kwargs.get(
        "level_of_detail",
        "high level and concise, assume a non-expert user",
    )
    phrasing = task_kwargs.get(
        "phrasing",
        "mixed (e.g., keyword-like, interrogative, prosaic, bullet points, etc.)",
    )
    language = task_kwargs.get("language", "English")
    return f"""\
You are a SPARQL expert trying to find possible user questions for \
a given SPARQL query. Your task is to fix and clean the given SPARQL query \
if needed, and generate possible natural language questions for it.

Make sure that generated questions respect the following requirements:
  Maximum number of questions: {max_questions}
  Level of detail: {level_of_detail}
  Phrasing: {phrasing}
  Language: {language}

You should take a step-by-step approach to achieve this:
1. Analyze the given SPARQL query, its used entities and properties, and \
execution result. Think about what the user wanted to achieve with this query. \
If this is not clear from the provided information alone, search and query available \
knowledge graphs to gain more context about the SPARQL query.
2. Clean the SPARQL query if needed. For example, remove superfluous variables or \
other unnecessary parts, find better variable names, etc.
3. Formulate your final SPARQL query and execute it to verify its correctness. \
It should not be too different from the original query in terms of \
intent and its execution result, but you are allowed to deviate if it would make \
the query more natural, precise, etc.
4. For the final SPARQL query, generate natural language questions that accurately \
capture its intent. Make sure they follow the requirements mentioned above.
5. Provide your final output by calling the answer function."""


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


class SparqlToQuestionTask(GraspTask):
    name = "sparql-to-question"

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
            f"Input for {self.name} must be a string (SPARQL query)"
        )
        _, formatted = prepare_sparql(
            input,
            self.managers,
            self.config.result_max_rows,
            self.config.result_max_columns,
            remove_known=True,
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
