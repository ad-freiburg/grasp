from typing import Any

from pydantic import BaseModel

from grasp.configs import NoteTakingConfig
from grasp.model import Message, Response
from grasp.tasks.base import GraspTask
from grasp.tasks.exploration.functions import (
    call_function as call_note_function,
)
from grasp.tasks.exploration.functions import (
    note_function_definitions,
)
from grasp.tasks.sparql_qa.examples import (
    call_function as call_example_function,
)
from grasp.tasks.sparql_qa.examples import (
    functions as example_functions,
)
from grasp.utils import format_kg_notes, format_notes, format_section

# function names handled by the note-taking functions; everything else the
# note-taker calls (e.g. search_example) is routed to the example functions
NOTE_FUNCTIONS = {"add_note", "delete_note", "update_note", "show_notes", "stop"}


def format_value(value: Any, indent: int = 0) -> str:
    if isinstance(value, str):
        if "\n" not in value:
            return f'"{value}"'

        # multi-line string: wrap across lines as a triple-quoted block, with the
        # body indented one step deeper and the closing quotes at the base indent
        pad = " " * indent
        body = "\n".join(f"{pad}  {line}" for line in value.splitlines())
        return f'"""\n{body}\n{pad}"""'
    elif isinstance(value, list):
        return "[" + ", ".join(format_value(v, indent) for v in value) + "]"
    elif isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{k}: {format_value(v, indent)}" for k, v in value.items())
            + "}"
        )
    else:
        # None, bool, int, float, ...
        return str(value)


def format_call(name: str, args: dict[str, Any]) -> str:
    if not args:
        return f"{name}()"

    rendered = {k: format_value(v, indent=2) for k, v in args.items()}
    if not any("\n" in v for v in rendered.values()):
        return f"{name}(" + ", ".join(f"{k}={v}" for k, v in rendered.items()) + ")"

    # at least one argument spans multiple lines: put each on its own line
    lines = [f"{name}("]
    lines.extend(f"  {k}={v}," for k, v in rendered.items())
    lines.append(")")
    return "\n".join(lines)


def format_output(
    output: Any | None,
    messages: list[Message],
    level: int = 4,
) -> str:
    fmt = []
    step = 1
    for message in messages[2:]:
        if message.role == "feedback":
            fmt.append(format_section("Feedback", message.content, level))  # type: ignore
            continue

        elif message.role == "user":
            fmt.append(format_section("User", message.content, level))  # type: ignore
            continue

        assert isinstance(message.content, Response)

        assistant = message.content
        ass_content = assistant.get_content()

        blocks = []
        if "reasoning" in ass_content:
            blocks.append(ass_content["reasoning"])

        if "content" in ass_content:
            blocks.append(ass_content["content"])

        for tool_call in assistant.tool_calls:
            call = format_call(tool_call.name, tool_call.args)
            blocks.append(f"{call}:\n{tool_call.result}")

        fmt.append(format_section(f"Step {step}", "\n\n".join(blocks), level))
        step += 1

    if output is not None and "formatted" in output:
        fmt.append(
            format_section(f"Output after {step} steps", output["formatted"], level)
        )

    return "\n\n".join(fmt)


class NotesFromTracesState(BaseModel):
    notes: list[str] = []
    kg_notes: dict[str, list[str]] = {}
    # traces of a knowledge graph agent to take notes from
    # or a (trace, ground truth) tuple if ground truth is available
    outputs: list[dict] = []
    ground_truths: list[str] | None = None


def rules() -> list[str]:
    return [
        "Do not take notes on things that are already handled well by the agent.",
        "As you hit the limits on the number of notes and their length, \
gradually merge and generalize your notes, discard unnecessary details, and move \
notes that can be useful across knowledge graphs to the general section.",
    ]


def system_information(config: NoteTakingConfig) -> str:
    return f"""\
You are a note-taking assistant. Your task is to inspect and take notes on the \
traces of a knowledge graph agent performing a certain task.

Your notes should help the agent to better understand and navigate the task \
and knowledge graphs in the future. You are limited to a maximum of {config.max_notes} notes \
per knowledge graph, plus {config.max_notes} general notes for insights that apply \
across knowledge graphs or to the task in general. Each note is limited to a maximum of \
{config.max_note_length} characters to ensure it is concise and to the point.

The notes should generalize to new unseen task inputs, rather than being specific to \
the task traces and outputs at hand.

In addition to the note-taking functions, you also have access to the same knowledge \
graph functions the agent used (e.g. running SPARQL queries, listing triples, and \
search). Use them to explore the knowledge graphs further \
whenever a trace reveals something worth checking: verify suspected failure modes, \
confirm whether the agent's outputs were actually correct, explore alternative \
reasoning paths, find ways to improve the agent's strategies and efficiency, etc.

Before calling a note-taking function, provide reasoning for what you are doing and why. \
Before stopping, make sure to check all notes (not only the ones touched in this iteration) \
for the above mentioned criteria and clean them if needed.

Examples of potentially useful types of notes to include:
- overall structure, domain coverage, and schema of the knowledge graphs
- peculiarities of the knowledge graphs
- strategies when encountering certain types of questions or errors
- tips for when and how to use certain functions"""


def note_taking_instructions(
    outputs: list[dict],
    ground_truths: list[str] | None = None,
) -> str:
    formatted = []
    for i, output in enumerate(outputs):
        messages = [Message(**msg) for msg in output["messages"]]

        if i == 0:
            assert messages[0].role == "system"
            formatted.append(
                format_section(
                    "Task instructions for the agent",
                    messages[0].content,  # type: ignore
                )
            )

        assert messages[1].role == "user"
        input = messages[1].content

        parts = []
        if ground_truths is not None:
            parts.append(format_section("Reference output", ground_truths[i], level=3))

        parts.append(
            format_section(
                "Agent trace",
                format_output(output["output"], messages),
                level=3,
            )
        )

        body = "\n\n".join([str(input), *parts])
        formatted.append(format_section(f"Input {i + 1}", body))

    fmt = "\n\n".join(formatted)

    return f"""\
Look at the current notes (which might be the same notes provided \
to the agent). Then add to, delete from, or update them based on the \
given agent traces below. Explore the knowledge graphs with the provided \
functions where it helps you to verify or refine a note.

{fmt}"""


def output(state: NotesFromTracesState) -> dict:
    formatted = "\n\n".join(
        [
            "Note taking completed.",
            format_section(
                "Notes for knowledge graphs",
                format_kg_notes(state.kg_notes),
            ),
            format_section("General notes", format_notes(state.notes)),
        ]
    )

    return {
        "type": "output",
        "notes": state.notes,
        "kg_notes": state.kg_notes,
        "formatted": formatted,
    }


class NotesFromTracesTask(GraspTask):
    name = "notes-from-traces"

    def system_information(self) -> str:
        assert isinstance(self.config, NoteTakingConfig)
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        # note-taking functions plus the example functions the agent had access
        # to, so the note-taker can retrace and verify example retrievals
        fns = note_function_definitions(self.managers)
        fns.extend(example_functions(self.config))
        return fns

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert isinstance(self.config, NoteTakingConfig)
        assert isinstance(self.state, NotesFromTracesState)
        if fn_name not in NOTE_FUNCTIONS:
            # example functions (search_example / get_random_examples)
            return call_example_function(
                self.config,
                self.managers,
                fn_name,
                fn_args,
                known,
                example_indices,
            )
        return call_note_function(
            self.state.kg_notes,
            self.state.notes,
            fn_name,
            fn_args,
            self.config.max_notes,
            self.config.max_note_length,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> str:
        assert isinstance(input, NotesFromTracesState), (
            "Input for notes-from-traces must be a NotesFromTracesState"
        )
        self.state = input
        return note_taking_instructions(input.outputs, input.ground_truths)

    def output(self, messages: list[Message]) -> dict:
        assert isinstance(self.state, NotesFromTracesState)
        return output(self.state)
