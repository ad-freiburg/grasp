from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig, NotesConfig, NotesFromExplorationConfig
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.exploration import shared_rules
from grasp.tasks.exploration.functions import call_function as call_note_function
from grasp.tasks.exploration.structural.functions import (
    mark_explored,
    note_function_definitions,
    show_explored,
)
from grasp.tasks.functions import find_frequent, find_frequent_function_definition
from grasp.tasks.sparql_qa.examples import (
    EXAMPLE_FUNCTIONS,
    call_function as call_example_function,
    functions as example_function_definitions,
)
from grasp.utils import format_kg_notes, format_section


class StructuralExplorationState(BaseModel):
    kg_notes: dict[str, list[str]] = {}
    explored: dict[str, list[str]] = {}


def rules() -> list[str]:
    return shared_rules() + [
        "For entities, prefer class-like nodes as seeds over instance-like nodes.",
        "Do not take notes on the provided functions and their usage. Instead, \
focus on structural and modeling insights about the knowledge graphs.",
    ]


def system_information(config: GraspConfig) -> str:
    assert isinstance(config, NotesFromExplorationConfig)
    return f"""\
You are a note-taking assistant. Your task is to \
explore knowledge graphs around selected entities or properties \
(called seeds) and take notes about them using the provided functions.

You are limited to a maximum of {config.max_notes} notes \
per knowledge graph. Each note is limited to a maximum of \
{config.max_note_length} characters to ensure it is concise and to the point.

Your notes should help you to better understand and navigate the \
knowledge graphs in the future. The notes should generalize and be useful for \
answering all kinds of questions about the knowledge graphs, rather than \
being specific to the seed nodes you explore.

You should follow a step-by-step approach to take notes:
1. Look at the current notes and already explored seeds in \
all knowledge graphs to figure out well-covered and underexplored areas.
2. Determine a seed in an underexplored area of one of the knowledge graphs. \
Avoid previously explored seeds or those very similar to them.
3. Thoroughly explore the seed's connections and use in the graph, \
and take notes about your findings along the way.
4. If there are no more insights to be gained from exploring \
the seed, mark it as explored. Before stopping, check all notes (not only \
the ones touched in this exploration) for the above-mentioned criteria and clean \
them if needed.

Examples of potentially useful types of notes:
- overall structure and domain coverage
- quirks and peculiarities
- type taxonomy and class hierarchy
- notable or unusual modeling patterns"""


def output(state: StructuralExplorationState) -> dict:
    formatted = "\n\n".join(
        [
            "Exploration completed.",
            format_section(
                "Knowledge graph specific notes",
                format_kg_notes(state.kg_notes),
                level=2,
            ),
        ]
    )

    return {
        "type": "output",
        "kg_notes": state.kg_notes,
        "explored": state.explored,
        "formatted": formatted,
    }


class StructuralExplorationTask(GraspTask):
    name = "exploration_structural"

    def system_information(self) -> str:
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        kgs = [m.kg for m in self.managers]
        functions = note_function_definitions(self.managers)
        functions.append(find_frequent_function_definition(kgs, self.config.list_k))
        functions.extend(example_function_definitions(self.config))
        return functions

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert isinstance(self.config, NotesConfig)
        assert isinstance(self.state, StructuralExplorationState)
        if fn_name == "mark_explored":
            result = mark_explored(
                self.managers,
                fn_args["kg"],
                self.state.explored,
                fn_args["iri"],
                self.explored_this_round,
            )
            self.explored_this_round = True
            return result

        elif fn_name == "show_explored":
            return show_explored(
                self.managers,
                fn_args["kg"],
                self.state.explored,
                fn_args["page"],
                self.config.list_k,
            )

        elif fn_name == "find_frequent":
            return find_frequent(
                self.managers,
                fn_args["kg"],
                fn_args["position"],
                fn_args.get("subject"),
                fn_args.get("property"),
                fn_args.get("object"),
                fn_args.get("page", 1),
                self.config.list_k,
                known,
                self.config.sparql_request_timeout,
                self.config.sparql_read_timeout,
            )

        elif fn_name in EXAMPLE_FUNCTIONS:
            return call_example_function(
                self.config,
                self.managers,
                fn_name,
                fn_args,
                known,
                example_indices=example_indices,
            )

        return call_note_function(
            self.state.kg_notes,
            [],  # cannot be used by structural exploration task
            fn_name,
            fn_args,
            self.config.max_notes,
            self.config.max_note_length,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> str:
        assert isinstance(input, StructuralExplorationState), (
            "Input for exploration must already be a StructuralExplorationState"
        )
        self.state = input
        self.explored_this_round = False
        return "Choose a seed and start the exploration. \
Add to, delete from, or update the current notes along the way."

    def output(self, messages: list[Message]) -> dict:
        return output(self.state)
