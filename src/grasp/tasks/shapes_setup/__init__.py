from typing import Any

from grammar_utils.parse import LR1Parser  # type: ignore
from pydantic import BaseModel

from grasp.model import Message
from grasp.shapes import Shapes
from grasp.sparql.utils import find_all, parse_string
from grasp.tasks.base import GraspTask
from grasp.tasks.functions import find_frequent, find_frequent_function_definition
from grasp.utils import FunctionCallException

REFERENCE_SETUP = {
    "pattern": "?instance a {CLASS}",
    "description": "All standard RDF classes via the rdf:type (a) property",
}


class ShapesSetupState(BaseModel):
    pattern: str | None = None
    description: str | None = None


def validate_pattern_format(pattern: str, parser: LR1Parser) -> None:
    if "{CLASS}" not in pattern:
        raise FunctionCallException(
            "Pattern must contain '{CLASS}' as the class placeholder."
        )
    class_pattern = pattern.replace("{CLASS}", "?class")
    test_query = f"SELECT DISTINCT ?class WHERE {{\n  {class_pattern}\n}} LIMIT 1"
    parse, _ = parse_string(test_query, parser)
    variables = {
        child["value"]
        for var in find_all(parse, "Var")
        for child in var.get("children", [])
    }
    if "?instance" not in variables:
        raise FunctionCallException(
            "Pattern must contain '?instance' as the instance variable."
        )


class ShapesSetupTask(GraspTask):
    name = "shapes-setup"

    def system_information(self) -> str:
        manager = self.managers[0]
        return f"""\
You are a knowledge graph setup assistant. Your task is to \
explore the "{manager.kg}" knowledge graph and come up with or improve \
the setup - a SPARQL graph pattern relating instances to classes and a \
description - of the shape index.

The SPARQL graph pattern should relate an instance variable ?instance to \
a class placeholder {{CLASS}}, using suitable properties and SPARQL constructs. \
The pattern will then be used in two ways:
1. To determine all classes in the knowledge graph by replacing {{CLASS}} with \
a ?class variable and embedding it in a SPARQL query of the form "SELECT DISTINCT \
?class WHERE {{ <pattern> }}".
2. To determine the shape of a class by replacing {{CLASS}} with a \
specific IRI and embedding it in various profiling queries.

The description should be a concise summary of the classes captured \
by the pattern.

You should follow a step-by-step approach:
1. Explore the knowledge graph using provided functions to understand how \
instances and classes are modeled.
2. Come up with the class graph pattern and validate it against the knowledge graph by \
executing SPARQL queries. The pattern can contain any SPARQL graph pattern constructs needed \
to capture all relevant classes. For example, if classes are modeled via more than one property \
you should use a UNION pattern, or if classes are modeled via more than one hop you should use \
property paths or multiple triple patterns.
3. Set the pattern and get the shape of a few exemplary classes \
to verify that it works as intended. If not, go back to step 1 or 2 and adjust the pattern. \
Note that you can also set the pattern to null if the knowledge graph does not \
contain any meaningful instance-class relationships.
4. Once you are satisfied with the pattern, write the corresponding description \
and stop.

Below is a reference setup you can use as a starting point if \
no shape setup is available yet. It is generic and thus \
may not be optimal for the knowledge graph at hand.

Reference graph pattern:
{REFERENCE_SETUP["pattern"]}

Reference description:
{REFERENCE_SETUP["description"]}"""

    def rules(self) -> list[str]:
        return [
            "If the user provides additional notes about the desired setup, make sure to follow them.",
            "The pattern should be single-hop; multi-hop instance-class relations may not yield sensible automatic shape computations.",
        ]

    def function_definitions(self) -> list[dict]:
        manager = self.managers[0]
        kgs = [m.kg for m in self.managers]
        fns: list[dict] = [find_frequent_function_definition(kgs, self.config.list_k),
            {
                "name": "show_setup",
                "description": "Show the current pattern and description for the shape index.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "set_pattern",
                "description": "Set or clear the shape index' SPARQL graph pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": ["string", "null"],
                            "description": "The SPARQL graph pattern, or null to clear / unset the current pattern.",
                        },
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "set_description",
                "description": "Set a concise description of what the shape index contains. "
                "Typically a single sentence about covered classes and any caveats is sufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "The description",
                        },
                    },
                    "required": ["description"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "stop",
                "description": "Stop the setup process.",
            },
        ]

        # Add get_shape only when kg_functions won't already include it.
        # kg_functions includes get_shape when manager.shapes is not None.
        if manager.shapes is None:
            fns.insert(
                0,
                {
                    "name": "get_shape",
                    "description": (
                        "Retrieve the pseudo-ShEx shape for a specific class by "
                        "running profiling queries using the current pattern. "
                        "Use this to test the pattern after setting it."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kg": {
                                "type": "string",
                                "enum": [manager.kg],
                                "description": "The knowledge graph",
                            },
                            "iri": {
                                "type": "string",
                                "description": "The class IRI",
                            },
                        },
                        "required": ["kg", "iri"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            )

        return fns

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert isinstance(self.state, ShapesSetupState)
        manager = self.managers[0]

        if fn_name == "find_frequent":
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

        elif fn_name == "show_setup":
            return format_setup(self.state)

        elif fn_name == "set_pattern":
            pattern = fn_args["pattern"]
            if pattern is not None:
                validate_pattern_format(pattern, manager.sparql_parser)
            self.state.pattern = pattern
            if pattern is None:
                manager.shapes = None
            elif manager.shapes is None or manager.shapes.pattern != pattern:
                # Pattern changed or newly set — existing index is no longer valid.
                manager.shapes = Shapes(pattern=pattern)
            # else: same pattern, keep existing shapes (index intact).
            return (
                "Pattern updated. Use get_shape to test it live."
                if pattern
                else "Pattern cleared."
            )

        elif fn_name == "set_description":
            self.state.description = fn_args["description"]
            return "Description updated."

        elif fn_name == "stop":
            return "Stopping"

        raise FunctionCallException(f"Unknown function {fn_name}")

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> str:
        assert isinstance(input, dict), "Input for shapes-setup must be a dict"
        self.input = input
        self.state = ShapesSetupState()
        manager = self.managers[0]
        if input.get("notes"):
            return input["notes"]
        return f'Set up the shape index for the "{manager.kg}" knowledge graph.'

    def output(self, messages: list[Message]) -> dict:
        assert isinstance(self.state, ShapesSetupState)
        if self.state.pattern:
            formatted = (
                f"Shape setup complete.\n\n"
                f"Pattern:\n{self.state.pattern}\n\n"
                f"Description:\n{self.state.description or 'None'}"
            )
        else:
            formatted = "Shape setup stopped without a pattern."

        return {
            "type": "output",
            "pattern": self.state.pattern,
            "description": self.state.description,
            "formatted": formatted,
        }


def format_setup(state: ShapesSetupState) -> str:
    return f"""\
Pattern:
{state.pattern or "None"}

Description:
{state.description or "None"}"""
