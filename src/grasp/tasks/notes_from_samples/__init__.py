from typing import Any

from grasp.configs import NoteTakingConfig
from grasp.model import Message
from grasp.tasks.notes_from_traces import (
    NotesFromTracesState,
    NotesFromTracesTask,
)
from grasp.utils import format_section


class NotesFromSamplesState(NotesFromTracesState):
    # task samples to take notes from; each sample is a dict with a "kg", an
    # "input" (the task input, e.g. a question), and an optional "reference"
    # (a formatted reference output). No agent trace is involved -- the
    # note-taker explores the knowledge graphs itself on top of the samples.
    samples: list[dict] = []
    # instructions of the underlying task the samples belong to, so the
    # note-taker knows what "solving" a sample means
    task_instructions: str | None = None


def system_information(config: NoteTakingConfig) -> str:
    return f"""\
You are a note-taking assistant. Your task is to inspect task samples (inputs, \
and reference outputs where available) for a knowledge graph agent and take notes \
that help the agent solve such tasks in the future.

No agent has attempted these samples yet - it is up to you to explore the \
knowledge graphs with the provided functions (running SPARQL queries, listing \
triples, searching, and retrieving examples) to work out how each sample could be \
solved, uncover the relevant schema and peculiarities, and capture reusable insights.

Your notes should help the agent to better understand and navigate the task and \
knowledge graphs in the future. You are limited to a maximum of {config.max_notes} notes \
per knowledge graph, plus {config.max_notes} general notes for insights that apply \
across knowledge graphs or to the task in general. Each note is limited to a maximum of \
{config.max_note_length} characters to ensure it is concise and to the point.

The notes should generalize to new unseen task inputs, rather than being specific to \
the samples at hand.

Before calling a note-taking function, provide reasoning for what you are doing and why. \
Before stopping, make sure to check all notes (not only the ones touched in this iteration) \
for the above mentioned criteria and clean them if needed.

Examples of potentially useful types of notes to include:
- overall structure, domain coverage, and schema of the knowledge graphs
- peculiarities of the knowledge graphs
- strategies when encountering certain types of questions or errors
- tips for when and how to use certain functions"""


def note_taking_instructions(
    samples: list[dict],
    task_instructions: str | None = None,
) -> str:
    formatted = []
    if task_instructions is not None:
        formatted.append(
            format_section("Task instructions for the agent", task_instructions)
        )

    for i, sample in enumerate(samples):
        parts = [str(sample["input"])]
        if sample.get("reference") is not None:
            parts.append(
                format_section("Reference output", sample["reference"], level=3)
            )

        formatted.append(format_section(f"Sample {i + 1}", "\n\n".join(parts)))

    fmt = "\n\n".join(formatted)

    return f"""\
Look at the current notes (which might be the same notes provided to the agent). \
Then add to, delete from, or update them based on the task samples below. Explore \
the knowledge graphs with the provided functions to figure out how each sample could \
be solved and to verify or refine a note.

{fmt}"""


class NotesFromSamplesTask(NotesFromTracesTask):
    name = "notes-from-samples"

    def system_information(self) -> str:
        assert isinstance(self.config, NoteTakingConfig)
        return system_information(self.config)

    def setup(self, input: Any) -> str:
        assert isinstance(input, NotesFromSamplesState), (
            "Input for notes-from-samples must be a NotesFromSamplesState"
        )
        self.state = input
        return note_taking_instructions(input.samples, input.task_instructions)

    def output(self, messages: list[Message]) -> dict:
        assert isinstance(self.state, NotesFromSamplesState)
        return super().output(messages)
