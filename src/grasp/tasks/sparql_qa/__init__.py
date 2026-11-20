import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from grasp.configs import GraspConfig
from grasp.function_args import FunctionDefinition, Sparql
from grasp.functions import (
    ExecuteArgs,
    check_known,
    selection_groups,
)
from grasp.manager import KgManager, find_manager, format_kgs
from grasp.model import Message, ToolCallResult
from grasp.model.base import ResponseMessage
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.tasks.sparql_qa.examples import (
    SparqlQaSample,
)
from grasp.tasks.sparql_qa.examples import (
    call_function as call_example_function,
)
from grasp.tasks.sparql_qa.examples import (
    functions as example_functions,
)
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import (
    FunctionCallException,
    format_enumerate,
    format_notes,
    format_section,
)


def system_information() -> str:
    return """\
You are a question answering assistant. \
Your job is to generate a SPARQL query to answer a given user question.

You should follow a step-by-step approach to generate the SPARQL query:
1. Determine possible entities and properties implied by the user question.
2. Search for the entities and properties in the knowledge graphs. Where \
applicable, constrain the searches with already identified entities and properties.
3. Gradually build up the SPARQL query using the identified entities \
and properties. Start with simple queries and add more complexity as needed. \
Execute intermediate queries to get feedback and to verify your assumptions. \
You may need to refine or rethink your current plan based on the query \
results and go back to step 2 if needed, possibly multiple times.
4. Use the answer or cancel function to finalize your answer and stop the \
generation process."""


def rules() -> list[str]:
    return [
        "Always execute your final SPARQL query before giving an answer to \
make sure it returns the expected results. Check whether the SPARQL \
query and its result are plausible using your world knowledge, if you are \
familiar with the domain. If the results look incorrect or incomplete, \
go back and refine your SPARQL query.",
        "The SPARQL query should always return the actual \
identifiers / IRIs of the items in its result. It additionally may return \
labels or other human-readable information, but they are optional and should be \
put within optional clauses unless explicitly requested by the user.",
        "Do not perform additional computation (e.g. filtering, sorting, calculations) \
on the result of the SPARQL query to determine the answer. All computation should \
be done solely within SPARQL.",
        'For questions with a "True" or "False" answer the SPARQL query \
should be an ASK query.',
    ]


class AnswerArgs(BaseModel):
    kg: str = Field(
        description="The knowledge graph on which the final SPARQL query needs "
        "to be executed",
    )
    sparql: Sparql = Field(description="The final SPARQL query")
    answer: str = Field(
        description="The answer to the question based on the SPARQL query results",
    )


class BestAttempt(BaseModel):
    sparql: Sparql = Field(description="The best SPARQL query so far")
    kg: str = Field(
        description="The knowledge graph on which the SPARQL query needs to be "
        "executed",
    )


class CancelArgs(BaseModel):
    explanation: str = Field(
        description="A detailed explanation of why you could not find a "
        "satisfactory SPARQL query",
    )
    best_attempt: BestAttempt | None = Field(
        default=None,
        description="Your best attempt at a SPARQL query so far, can be omitted "
        "if there is none",
    )


def functions(
    managers: list[KgManager],
    config: GraspConfig,
    example_indices: dict,
) -> list[FunctionDefinition]:
    kgs = [manager.kg for manager in managers]
    fns = [
        FunctionDefinition(
            "answer",
            """\
Provide your final SPARQL query and a concise answer to the user question \
based on the query results. Only include the direct answer, no reasoning \
or step summaries. This function will stop the generation process.""",
            AnswerArgs,
            {"kg": kgs},
        ),
        FunctionDefinition(
            "cancel",
            """\
If you are unable to find a SPARQL query that answers the question well, \
you can call this function instead of the answer function. This function will \
stop the generation process.""",
            CancelArgs,
            {"kg": kgs},
        ),
    ]

    fns.extend(example_functions(config, example_indices))

    return fns


def validate_args[T: BaseModel](args: type[T], data: dict | str) -> T | None:
    # None if the data are no valid arguments, e.g. of a failed call
    try:
        if isinstance(data, str):
            return args.model_validate_json(data)
        return args.model_validate(data)
    except ValidationError:
        return None


def get_raw_json_from_message(message: str) -> str | None:
    stripped = message.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    json_match = re.search(
        r"```json\s*(.*?)\s*```",
        message,
        re.IGNORECASE | re.DOTALL,
    )
    if json_match is not None:
        return json_match.group(1).strip()

    return None


def get_sparql_from_message(message: str) -> str | None:
    # Check for SPARQL code blocks
    sparql_match = re.search(
        r"```sparql\s*(.*?)\s*```",
        message,
        re.IGNORECASE | re.DOTALL,
    )
    if sparql_match is None:
        return None
    return sparql_match.group(1).strip()


def get_answer_or_cancel(
    messages: list[Message],
    default_kg: str,
) -> AnswerArgs | CancelArgs | None:
    # the last answer or cancel call, otherwise parsed from the last message or
    # from the last execute call; queries without kg are for the default one
    last_message: str | None = None
    last: AnswerArgs | CancelArgs | None = None
    last_execute: ExecuteArgs | None = None
    assert messages[0].role == "system", "First message should be system"
    assert messages[1].role == "user", "Second message should be user"
    for message in messages[2:]:
        is_intermediate_feedback = (
            message.role == "feedback" and message != messages[-1]
        )
        is_user_message = message.role == "user"
        if is_intermediate_feedback or is_user_message:
            # reset stuff after intermediate feedback or user message
            last = None
            last_message = None
            last_execute = None

        if isinstance(message.content, str):
            # not assistant message
            continue

        if isinstance(message.content.message, ResponseMessage):
            last_message = message.content.message.content
        else:
            last_message = message.content.message

        for tool_call in message.content.tool_calls:
            if tool_call.name == "answer":
                last = validate_args(AnswerArgs, tool_call.args) or last

            elif tool_call.name == "cancel":
                last = validate_args(CancelArgs, tool_call.args) or last

            elif tool_call.name == "execute":
                last_execute = (
                    validate_args(ExecuteArgs, tool_call.args) or last_execute
                )

    if last is not None:
        return last

    if last_message is not None:
        raw_json = get_raw_json_from_message(last_message)
        if raw_json is not None:
            answer = validate_args(AnswerArgs, raw_json)
            last = answer or validate_args(CancelArgs, raw_json)

    if last is None and last_message is not None:
        sparql = get_sparql_from_message(last_message)
        if sparql is not None:
            last = AnswerArgs(kg=default_kg, sparql=sparql, answer=last_message)

    # try last execute function call for SPARQL QA
    if last is None and last_execute is not None:
        last = AnswerArgs(
            kg=last_execute.kg,
            sparql=last_execute.sparql,
            answer=last_message or "No answer provided",
        )

    return last


def prepare_formatted_output(
    sparql: str,
    kg: str | None,
    managers: list[KgManager],
    max_rows: int = 10,
    max_cols: int = 10,
    request_timeout: float | tuple[float, float] = (6.0, 30.0),
    read_timeout: float = 10.0,
    sparql_result_max_rows: int | None = None,
) -> dict:
    if kg is None:
        kg = managers[0].kg

    result, selections = prepare_sparql_result(
        sparql,
        kg,
        managers,
        max_rows,
        max_cols,
        request_timeout=request_timeout,
        read_timeout=read_timeout,
        sparql_result_max_rows=sparql_result_max_rows,
    )
    manager = find_manager(managers, kg)

    groups = selection_groups(selections, manager, result.sparql)
    return {
        "sparql": result.sparql,
        "selections": [group.model_dump() for group in groups],
        "result": result.tool_result().model_dump(),
        "endpoint": manager.endpoint,
        "formatted": format_sparql_result(kg, result, groups),
    }


def output(
    messages: list[Message],
    managers: list[KgManager],
    max_rows: int,
    max_cols: int,
    request_timeout: tuple[float, float],
    read_timeout: float,
) -> dict | None:
    final = get_answer_or_cancel(messages, managers[0].kg)
    if final is None:
        return None

    output: dict[str, Any] = {
        "sparql": None,
        "kg": None,
        "selections": None,
        "result": None,
        "endpoint": None,
    }

    if isinstance(final, AnswerArgs):
        output["type"] = "answer"
        output["answer"] = final.answer.strip()
        output["sparql"] = final.sparql
        output["kg"] = final.kg
        formatted = output["answer"]

    else:
        output["type"] = "cancel"
        output["explanation"] = final.explanation.strip()

        if final.best_attempt is not None:
            output["sparql"] = final.best_attempt.sparql
            output["kg"] = final.best_attempt.kg

        formatted = output["explanation"]

    if output["sparql"] is None:
        output["formatted"] = formatted
        return output

    formatted_output = prepare_formatted_output(
        output["sparql"],
        output["kg"],
        managers,
        max_rows,
        max_cols,
        request_timeout,
        read_timeout,
    )
    # prepend answer or explanation to formatted output
    formatted_output["formatted"] = formatted + "\n\n" + formatted_output["formatted"]
    output.update(formatted_output)
    return output


def feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    cases = """\
There are two possible cases:

1) The system was able to find an answer
You are given the final SPARQL query, the knowledge graph it has to be executed \
against, and a human-readable answer to the question. You are also given some \
additional information about the SPARQL query, like the entities and properties \
it uses, and its execution result.

2) The system failed to find an answer
You are given the system's explanation for why it failed to find an answer. \
Optionally, you are provided with the system's best attempt at a SPARQL query \
so far including the same additional information as in case 1."""

    return "\n\n".join(
        [
            "You are a question answering assistant providing feedback on the "
            "output of a SPARQL-based question answering system for a given user "
            "question.",
            format_section(
                "Available knowledge graphs",
                format_kgs(managers, kg_notes),
            ),
            format_section(
                "General notes across knowledge graphs",
                format_notes(notes, enumerated=True),
            ),
            format_section(
                "Rules to follow",
                format_enumerate(rules()) if rules() else "None",
            ),
            cases,
        ]
    )


def feedback_instructions(questions: list[str], output: dict) -> str:
    assert questions, "At least one question is required for feedback"

    sections = []
    if len(questions) > 1:
        sections.append(
            format_section(
                "Previous questions",
                "\n\n".join(q.strip() for q in questions[:-1]),
            )
        )

    sections.append(format_section("Question", questions[-1].strip()))

    if output["type"] == "answer":
        # terminated with answer call
        sections.append(
            format_section(
                "The system was able to find an answer",
                f"Answer:\n{output['formatted']}",
            )
        )

    else:
        sections.append(
            format_section(
                "The system failed to find an answer",
                f"Explanation:\n{output['formatted']}",
            )
        )

    return "\n\n".join(sections)


class SparqlQaTask(GraspTask, FeedbackTask):
    name = "sparql-qa"

    # set to True when a stopping call (answer/cancel) was rejected because it
    # used IRIs that are not yet known from the trace; keeps the task from
    # stopping so the model gets another round to verify them
    answer_rejected: bool = False

    def system_information(self) -> str:
        return system_information()

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[FunctionDefinition]:
        return functions(self.managers, self.config, self.example_indices)

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str | ToolCallResult:
        if fn_name != "answer" and fn_name != "cancel":
            return call_example_function(
                self.config,
                self.managers,
                fn_name,
                fn_args,
                known,
                example_indices,
            )

        # fresh attempt to stop; assume it is accepted unless the
        # know-before-answer check below rejects it
        self.answer_rejected = False
        task_kwargs = self.config.task_kwargs.get("sparql-qa", {})
        know_before_answer = task_kwargs.get("know_before_answer", True)
        result = "Stopping"

        if not know_before_answer or self.last_step:
            # no check, or no step left to fix unknown IRIs; accept
            return result

        # locate the final query to verify: the answer's query, or a
        # cancel's best attempt (a concrete {sparql, kg} object)
        if fn_name == "answer":
            answer_args = AnswerArgs.model_validate(fn_args)
            kg = answer_args.kg
            sparql = answer_args.sparql
        else:
            best_attempt = CancelArgs.model_validate(fn_args).best_attempt
            if best_attempt is None:
                return result

            kg = best_attempt.kg
            sparql = best_attempt.sparql

        assert kg is not None and sparql is not None, "kg and sparql must be provided"

        manager = find_manager(self.managers, kg)
        try:
            # all non-common-prefix IRIs must be known from the trace
            check_known(manager, sparql, known)
        except FunctionCallException:
            # do not stop; force another round so the model can
            # verify or replace the unknown IRIs
            self.answer_rejected = True
            raise

        return result

    def done(self, fn_name: str) -> bool:
        return fn_name in {"answer", "cancel"} and not self.answer_rejected

    def limit_functions(self) -> list[str] | None:
        return ["answer", "cancel"]

    def output(self, messages: list[Message]) -> dict | None:
        return output(
            messages,
            self.managers,
            self.config.result_max_rows,
            self.config.result_max_columns,
            self.config.sparql_request_timeout,
            self.config.sparql_read_timeout,
        )

    @property
    def default_input_field(self) -> str | None:
        return "question"

    @classmethod
    def sample_cls(cls) -> type[SparqlQaSample] | None:
        return SparqlQaSample

    def feedback_system_message(
        self, kg_notes: dict[str, list[str]], notes: list[str]
    ) -> str:
        return feedback_system_message(self.managers, kg_notes, notes)

    def feedback_instructions(self, inputs: list[str], output: dict) -> str:
        return feedback_instructions(inputs, output)
