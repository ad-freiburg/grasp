from typing import Any, Literal

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.construct_gen.coverage import (
    compute_source_coverage,
    format_source_coverage,
)
from grasp.utils import (
    FunctionCallException,
    format_enumerate,
    format_section,
)

_REVIEW_SYSTEM_BASE = """\
You are an independent validator reviewing a set of one or more proposed SPARQL CONSTRUCT queries. \
Your job is to verify whether this set correctly transforms the source knowledge \
graph into a well-formed semantically correct RDF knowledge graph. Once you have finished \
your investigation, call give_feedback with your findings. The goal is the best possible \
semantic model of the input data.\
"""

_ONTOLOGY_KG_NOTE = """\
The following knowledge graphs are ontology/vocabulary references. \
Use them to verify whether the standard ontology terms used in the query are \
semantically correct and compatible with the mapped values. \
"""

_REVIEW_CHECKLIST_COMMON = """\
Validate the following aspects of the proposed CONSTRUCT queries:
- SOURCE COVERAGE: Verify that all data from the "{kg_name}" knowledge graph is \
accounted for in the CONSTRUCT queries. Confirm that nothing significant is missing.
- VALUE TYPES AND RELATIONSHIPS: For each mapped predicate, check that the output value \
type is appropriate — literals where a literal is expected, IRIs where a relationship to another \
entity is intended.
- RELATIONSHIP MODELING: Query the "{kg_name}" knowledge graph to identify what relationships \
exist between entities in the data. Verify that the CONSTRUCT output models these relationships \
appropriately. Any field that semantically references another entity — whether that entity \
is minted within this same query or by a different query in the set — must be resolved as an IRI \
link to that entity, not left as a raw literal value.
- CONSISTENCY: Check that the same real-world entity is minted using the same IRI pattern \
everywhere it's referenced, whether within a single query or across the query set.\
"""

_REVIEW_CHECKLIST_GUIDED = (
    _REVIEW_CHECKLIST_COMMON
    + """
- NAMESPACE CONSISTENCY: For any predicates or entity IRIs that fall back to \
a custom namespace, verify that this namespace reflects the intended domain and semantics, and use that namespace consistently throughout \
the queries.
- ONTOLOGY PREDICATE CORRECTNESS: For every predicate chosen from a standard ontology, \
confirm that the value assigned is compatible with what that predicate expects. \
If you find a mismatch — for example an enumeration property receiving a string literal \
instead of an enumeration IRI, or a predicate used to express something other than \
what it is defined for — flag the predicate and the expected value type.
- ONTOLOGY CLASS CORRECTNESS: For every entity assigned an rdf:type from a standard ontology, \
verify that class correctly and specifically matches what the entity represents, and that \
its use respects any constraints the ontology places on it — such as expected domain and \
range. Flag cases where a class is used to represent something other than what it is defined \
for, an overly generic class is used when a more specific one exists, or the class conflicts \
with how the ontology defines it should be used.\
"""
)

_REVIEW_CHECKLIST_CUSTOM = (
    _REVIEW_CHECKLIST_COMMON
    + """
- NAMESPACE CONSISTENCY: Verify that all custom predicates and entity IRIs \
use the same namespace consistently throughout the queries and that this namespace reflects \
the intended domain and semantics. Judge this from the expanded IRI, not the prefix \
alias.\
"""
)

_REVIEW_CHECKLIST_CQ = """
- COMPETENCY QUESTION COVERAGE: For each competency question listed, write the SPARQL query \
that would answer it against the knowledge graph the proposed CONSTRUCT queries would produce. \
You cannot execute it, since that graph does not exist yet, but writing it out is a usability \
test of the model: if the query answering it is overly complex and requires additional work \
to produce the correct result, that \
points to a gap in the semantic modelling itself and should be flagged. Flag any question \
that could not be answered at all, and say which part of the model is the reason.
- COMPETENCY QUESTION SCOPE: Competency questions are a minimum, not the full scope. Confirm \
the model has not been narrowed or made redundant to accommodate the questions while sacrificing \
SOURCE COVERAGE.\
"""

_REVIEW_CHECKLIST_CROSS_QUERY = """
- CROSS-QUERY CONSISTENCY: When multiple queries are submitted together, verify \
that they form a coherent graph. Check that the same real-world entity is named and \
modelled consistently across queries, the same predicate names and namespace \
are used for the same relationship across queries, and that any entity type referenced by more than \
one query is minted using the same IRI pattern so the same real-world entity always \
resolves to the same IRI.\
"""


class ConstructReviewTask(GraspTask):
    """
    Independent agentic reviewer for SPARQL CONSTRUCT queries.
    managers[0] is the data KG, managers[1:] are ontology KGs —
    same convention as ConstructGenTask and cli.py.
    """

    name = "construct-review"

    @property
    def include_common_prefixes(self) -> bool:
        return False

    def __init__(
        self,
        managers: list[KgManager],
        config: GraspConfig,
        known: set[str] | None = None,
        example_indices: dict | None = None,
    ) -> None:
        super().__init__(managers, config, known, example_indices)
        self.data_manager = self.managers[0]
        self.ontology_managers = self.managers[1:]
        self.ontology_mode: Literal["guided", "custom"] = "custom"
        self.queries: list[dict] = []
        self.competency_questions: list[str] = []
        # computed in setup() and shown to the reviewer as data
        self.coverage: dict = {}

    def system_information(self) -> str:
        checklist = (
            _REVIEW_CHECKLIST_GUIDED
            if self.ontology_mode == "guided"
            else _REVIEW_CHECKLIST_CUSTOM
        ).format(kg_name=self.data_manager.kg)
        if self.competency_questions:
            checklist += _REVIEW_CHECKLIST_CQ.format(kg_name=self.data_manager.kg)
        if len(self.queries) > 1:
            checklist += _REVIEW_CHECKLIST_CROSS_QUERY

        parts = [_REVIEW_SYSTEM_BASE, checklist]
        if self.ontology_mode == "guided" and self.ontology_managers:
            parts.append(
                _ONTOLOGY_KG_NOTE
                + "\nOntology KGs: "
                + ", ".join(f'"{m.kg}"' for m in self.ontology_managers)
            )
        if self.competency_questions:
            parts.append(
                format_section(
                    "Competency questions",
                    format_enumerate(self.competency_questions),
                )
            )
        return "\n\n".join(parts)

    def rules(self) -> list[str]:
        return [
            "Do not trust the query as written — independently verify each aspect "
            "by querying the source knowledge graph directly.",
            "Only report issues you have actually verified through knowledge graph exploration. "
            "Only call give_feedback once you have finished your full investigation.",
        ]

    def function_definitions(self) -> list[dict]:
        return [
            {
                "name": "give_feedback",
                "description": """\
Provide your final feedback on the proposed CONSTRUCT query (or queries) after \
completing your investigation.

The feedback status can be one of:
1. done: The query is correct and complete — no issues found.
2. refine: The query is mostly correct but has specific issues that need targeted fixes.
3. retry: The overall approach or structure of the query is wrong and it needs to be reworked.

The feedback message should describe any issues found and what you \
observed in the knowledge graph that confirms them. When reviewing multiple \
queries, reference the relevant query name in the feedback so it's clear which \
query an issue applies to. If status is done, briefly summarise what was verified.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["done", "refine", "retry"],
                            "description": "The feedback status",
                        },
                        "feedback": {
                            "type": "string",
                            "description": "The detailed feedback message",
                        },
                    },
                    "required": ["status", "feedback"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        if fn_name == "give_feedback":
            return "feedback recorded."
        raise FunctionCallException(f"Unknown function: {fn_name}")

    def done(self, fn_name: str) -> bool:
        return fn_name == "give_feedback"

    def setup(self, input: Any) -> str:
        assert isinstance(input, dict), "Input for construct-review must be a dict"
        self.ontology_mode = input.get("ontology_mode", "custom")
        self.queries = input.get("queries", [])
        self.competency_questions = input.get("competency_questions", []) or []

        if len(self.queries) == 1:
            sparql = self.queries[0]["sparql"]
            prompt = (
                f"Review the following SPARQL CONSTRUCT query targeting the "
                f'"{self.data_manager.kg}" knowledge graph:\n\n'
                f"```sparql\n{sparql}\n```"
            )
        else:
            blocks = "\n\n".join(
                f"### {q['name']}\n```sparql\n{q['sparql']}\n```" for q in self.queries
            )
            prompt = (
                f"Review the following {len(self.queries)} SPARQL CONSTRUCT queries "
                f'targeting the "{self.data_manager.kg}" knowledge graph. Together '
                f"they are intended to produce one coherent semantic graph:\n\n{blocks}"
            )

        self.coverage = compute_source_coverage(
            self.data_manager,
            self.queries,
            self.config.sparql_request_timeout,
            self.config.sparql_read_timeout,
        )
        return prompt + "\n\n" + format_source_coverage(self.coverage)

    def output(self, messages: list[Message]) -> dict | None:
        for message in reversed(messages):
            if not hasattr(message.content, "tool_calls"):
                continue
            for tool_call in message.content.tool_calls:
                if tool_call.name == "give_feedback":
                    return tool_call.args
        return None
