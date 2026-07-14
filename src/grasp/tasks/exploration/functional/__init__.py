from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig, NotesConfig, NotesFromExplorationConfig
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.exploration import shared_rules
from grasp.tasks.exploration.functions import call_function as call_note_function
from grasp.tasks.exploration.functions import note_function_definitions
from grasp.tasks.sparql_qa.examples import (
    EXAMPLE_FUNCTIONS,
    call_function as call_example_function,
    functions as example_function_definitions,
)
from grasp.utils import format_notes


class FunctionalExplorationState(BaseModel):
    notes: list[str] = []


def rules() -> list[str]:
    return shared_rules() + [
        "Do not take notes on the structure and content of the knowledge \
graphs themselves. Instead, focus on functional insights: how the \
provided functions behave, when they are most useful, and what their \
limitations are.",
        "Do not take notes on the note-taking functions themselves, as \
they will not be available in downstream tasks.",
    ]


def system_information(config: GraspConfig) -> str:
    assert isinstance(config, NotesFromExplorationConfig)
    return f"""\
You are a note-taking assistant. Your task is to \
explore the provided functions and take notes about them using the \
provided note-taking functions.

You are limited to a maximum of {config.max_notes} general notes \
that apply across knowledge graphs. Each note is limited to a maximum of \
{config.max_note_length} characters to ensure it is concise and to the point.

Your notes should help you to better understand and use the provided \
functions in the future. The notes should generalize to a wide range of \
tasks and knowledge graphs, rather than being specific to the particular \
calls you make during the exploration.

You should follow a step-by-step approach to take notes:
1. Look at the current notes to figure out which functions and use cases \
are well-covered and which are underexplored or missing entirely.
2. Pick a function or a specific aspect of a function that is underexplored. \
Exercise it systematically - vary its arguments, explore edge cases, and \
observe how its output changes - and take notes about your findings along \
the way.
3. Repeat step 2, working through different functions and aspects, until \
you have thoroughly covered all provided functions or you run out of new \
insights to capture.
4. Before stopping, make sure to check all notes (not only the ones touched \
in this exploration) for the above-mentioned criteria and clean them if needed.

Examples of potentially useful types of notes:
- when and how to use each function effectively
- argument combinations that are particularly revealing or commonly needed
- limitations, failure modes, and surprising behaviors of each function"""


def output(state: FunctionalExplorationState) -> dict:
    formatted = f"""\
Exploration completed.

General notes across knowledge graphs:
{format_notes(state.notes)}"""

    return {
        "type": "output",
        "notes": state.notes,
        "formatted": formatted,
    }


class FunctionalExplorationTask(GraspTask):
    name = "exploration_functional"

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        return [
            *note_function_definitions(self.managers, kg_specific=False),
            *example_function_definitions(self.config),
        ]

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert isinstance(self.config, NotesConfig)
        assert isinstance(self.state, FunctionalExplorationState)
        if fn_name in EXAMPLE_FUNCTIONS:
            return call_example_function(
                self.config,
                self.managers,
                fn_name,
                fn_args,
                known,
                example_indices=example_indices,
            )

        return call_note_function(
            {},  # functional task does not write KG-specific notes
            self.state.notes,
            fn_name,
            fn_args,
            self.config.max_notes,
            self.config.max_note_length,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> str:
        assert isinstance(input, FunctionalExplorationState), (
            "Input for exploration must already be a FunctionalExplorationState"
        )
        self.state = input
        return "Explore the available functions. Add to, delete from, or \
update the current notes along the way."

    def output(self, messages: list[Message]) -> dict:
        return output(self.state)
