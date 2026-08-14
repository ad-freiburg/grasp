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
    "instance_pattern": "?instance a {CLASS}",
    "schema_pattern": """\
{
  ?property rdfs:domain/(owl:unionOf/rdf:rest*/rdf:first)? {CLASS} .
  OPTIONAL {
    { ?property rdfs:range ?other . FILTER(!isBlank(?other)) }
    UNION
    { ?property rdfs:range/owl:unionOf/rdf:rest*/rdf:first ?other }
  }
  BIND("out" AS ?dir)
}
UNION
{
  ?property rdfs:range/(owl:unionOf/rdf:rest*/rdf:first)? {CLASS} .
  OPTIONAL {
    { ?property rdfs:domain ?other . FILTER(!isBlank(?other)) }
    UNION
    { ?property rdfs:domain/owl:unionOf/rdf:rest*/rdf:first ?other }
  }
  BIND("in" AS ?dir)
}""",
    "description": "All classes derived via rdf:type instances and rdfs:domain/rdfs:range "
    "property declarations, in both directions",
}


class ShapesSetupState(BaseModel):
    instance_pattern: str | None = None
    schema_pattern: str | None = None
    description: str | None = None


def extract_variables(pattern: str, parser: LR1Parser) -> set[str]:
    class_pattern = pattern.replace("{CLASS}", "?class")
    test_query = f"SELECT DISTINCT ?class WHERE {{\n  {class_pattern}\n}} LIMIT 1"
    parse, _ = parse_string(test_query, parser)
    return {
        child["value"]
        for var in find_all(parse, "Var")
        for child in var.get("children", [])
    }


def validate_instance_pattern_format(pattern: str, parser: LR1Parser) -> None:
    if "{CLASS}" not in pattern:
        raise FunctionCallException(
            "Pattern must contain '{CLASS}' as the class placeholder."
        )
    if "?instance" not in extract_variables(pattern, parser):
        raise FunctionCallException(
            "Instance pattern must contain '?instance' as the instance variable."
        )


def validate_schema_pattern_format(pattern: str, parser: LR1Parser) -> None:
    if "{CLASS}" not in pattern:
        raise FunctionCallException(
            "Pattern must contain '{CLASS}' as the class placeholder."
        )
    variables = extract_variables(pattern, parser)
    if "?property" not in variables:
        raise FunctionCallException(
            "Schema pattern must bind '?property' as the property variable."
        )
    if "?dir" not in variables:
        raise FunctionCallException(
            "Schema pattern must bind '?dir' to \"out\" (properties whose subject "
            'is the class) or "in" (properties whose object is the class), e.g. via '
            'BIND("out" AS ?dir). Every branch of the pattern must bind it.'
        )
    if "?other" not in variables:
        raise FunctionCallException(
            "Schema pattern must bind '?other' as the class or datatype on the "
            'far side of the property (the target when ?dir is "out", the '
            'source when ?dir is "in").'
        )


class ShapesSetupTask(GraspTask):
    name = "shapes-setup"

    def system_information(self) -> str:
        manager = self.managers[0]
        return f"""\
You are a knowledge graph setup assistant. Your task is to \
explore the "{manager.kg}" knowledge graph and come up with or improve \
the setup - an instance pattern, a schema pattern, and a description - \
of the shapes index.

The two patterns are complementary and used to automatically derive class shapes \
from the knowledge graph. A knowledge graph may need either one or both; \
set the one(s) that apply and leave the other null. \
The shapes index will be built from the derived class shapes, and allows \
to retrieve them via semantic search or exact class IRI lookups for various \
purposes in downstream tasks.

The instance pattern links instances to their classes by relating a ?instance \
variable to a {{CLASS}} placeholder, e.g. "?instance a {{CLASS}}". \
Use it when the graph contains instances: the shape of a \
class is then profiled empirically by walking the triples of its instances. Classes \
are discovered by replacing {{CLASS}} with ?class.

The schema pattern links properties to their classes directly from the \
schema/ontology - no instances needed. Use it for ontology-only knowledge graphs \
(or to add curated structure). It must bind three variables:
- ?property (required): the property related to the class.
- ?dir (required): "out" when the class is the property's subject side, "in" when \
it is the property's object side. Bind it per branch, e.g. BIND("out" AS ?dir).
- ?other (required): the class or datatype on the far side of the property. \
Usually wrapped in OPTIONAL, since it may be undeclared.

Cover both directions, typically as a UNION of the two branches: a class that is \
only ever used as a value of other properties has no outgoing properties and would \
otherwise get an empty shape.

Where the schema points at anonymous nodes instead of class IRIs, traverse them \
with a property path so the classes behind them are found, and make sure ?other \
cannot bind to the anonymous node itself. Anonymous nodes that resolve to no class \
are reported as BNODE.

The reference pattern below covers these cases for a typical schema vocabulary; \
adapt it to whichever relations play the same roles here, and drop the parts this \
knowledge graph does not need.

When both patterns are set, their per-class properties/targets are merged.

The description should be a concise summary of what the shapes index is \
about and what classes it contains.

You should follow a step-by-step approach:
1. Explore the knowledge graph to understand how instances and/or classes and their \
properties are modeled.
2. Come up with the applicable pattern(s) and validate them against the knowledge \
graph by executing SPARQL queries. A pattern can contain any SPARQL graph pattern \
constructs needed (UNION, property paths, multiple triples, OPTIONAL, etc.).
3. Set the pattern(s) and get the shape of a few exemplary classes to verify they \
work as intended. If not, go back and adjust. You can set a pattern to null to clear it.
4. Once satisfied, write the corresponding description and stop.

Below is a reference setup you can use as a starting point. It is generic and thus \
may not be optimal for the knowledge graph at hand.

Reference instance pattern:
{REFERENCE_SETUP["instance_pattern"]}

Reference schema pattern:
{REFERENCE_SETUP["schema_pattern"]}

Reference description:
{REFERENCE_SETUP["description"]}"""

    def rules(self) -> list[str]:
        return [
            "If the user provides additional notes about the desired setup, make sure to follow them.",
            "Set at least one of the instance pattern or the schema pattern; set both only when the knowledge graph meaningfully supports both.",
            "The instance pattern should be single-hop; multi-hop instance-class relations may not yield sensible automatic shape computations.",
        ]

    def function_definitions(self) -> list[dict]:
        manager = self.managers[0]
        kgs = [m.kg for m in self.managers]
        fns: list[dict] = [
            find_frequent_function_definition(kgs, self.config.list_k),
            {
                "name": "show_setup",
                "description": "Show the current patterns and description for the shape index.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "set_instance_pattern",
                "description": "Set or clear the shape index' instance pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": ["string", "null"],
                            "description": "The SPARQL graph pattern binding ?instance for {CLASS}, or null to clear it.",
                        },
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "set_schema_pattern",
                "description": "Set or clear the shape index' schema pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": ["string", "null"],
                            "description": "The SPARQL graph pattern binding ?property, ?dir and ?other for {CLASS}, or null to clear it.",
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
                        "running profiling queries using the current pattern(s). "
                        "Use this to test the patterns after setting them."
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

        elif fn_name == "set_instance_pattern":
            pattern = fn_args["pattern"]
            if pattern is not None:
                validate_instance_pattern_format(pattern, manager.sparql_parser)
            self.state.instance_pattern = pattern
            self.apply_patterns(manager)
            return (
                "Instance pattern updated. Use get_shape to test it live."
                if pattern
                else "Instance pattern cleared."
            )

        elif fn_name == "set_schema_pattern":
            pattern = fn_args["pattern"]
            if pattern is not None:
                validate_schema_pattern_format(pattern, manager.sparql_parser)
            self.state.schema_pattern = pattern
            self.apply_patterns(manager)
            return (
                "Schema pattern updated. Use get_shape to test it live."
                if pattern
                else "Schema pattern cleared."
            )

        elif fn_name == "set_description":
            self.state.description = fn_args["description"]
            return "Description updated."

        elif fn_name == "stop":
            return "Stopping"

        raise FunctionCallException(f"Unknown function {fn_name}")

    def apply_patterns(self, manager) -> None:
        assert isinstance(self.state, ShapesSetupState)
        inst = self.state.instance_pattern
        sch = self.state.schema_pattern
        if inst is None and sch is None:
            manager.shapes = None
            return
        existing = manager.shapes
        if (
            existing is not None
            and existing.instance_pattern == inst
            and existing.schema_pattern == sch
        ):
            # patterns unchanged, keep current index
            return

        # new patterns, discard existing index
        manager.shapes = Shapes(instance_pattern=inst, schema_pattern=sch)

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
        if self.state.instance_pattern or self.state.schema_pattern:
            formatted = "Shape setup complete.\n\n" + format_setup(self.state)
        else:
            formatted = "Shape setup stopped without any pattern."

        return {
            "type": "output",
            "instance_pattern": self.state.instance_pattern,
            "schema_pattern": self.state.schema_pattern,
            "description": self.state.description,
            "formatted": formatted,
        }


def format_setup(state: ShapesSetupState) -> str:
    return f"""\
Instance pattern:
{state.instance_pattern or "None"}

Schema pattern:
{state.schema_pattern or "None"}

Description:
{state.description or "None"}"""
