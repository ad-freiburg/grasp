import re
from typing import Any, Literal

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.utils import FunctionCallException, format_enumerate, format_section

import requests

def _construct_request(
    sparql: str,
    endpoint: str,
    headers: dict,
    params: dict,
    timeout: float = 30.0,
) -> str:
    req_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
        **headers,
    }
    response = requests.post(
        endpoint,
        data={**params, "query": sparql},
        headers=req_headers,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise FunctionCallException(
            f"CONSTRUCT query failed: HTTP {response.status_code}\n"
            f"{response.text[:500]}"
        )
    return response.text

_ONTOLOGY_KG_NOTE = """\
The following knowledge graphs are ontology/vocabulary references. \
When deciding how to model entities and their predicates in the output, \
search these knowledge graphs to find well-defined classes and properties whose \
semantics and expected value types match your intended mapping. Do not generate \
CONSTRUCT queries targeting these KGs.\
"""

_STEPS_GUIDED_DESIGN = """\
You should follow a step-by-step approach:
1. Explore the knowledge graph — use the available notes (and shape profile \
if available) as a starting point, and run targeted queries where needed to \
fill gaps. Get a clear picture of what classes and entities/predicates exist, \
what the values look like, and what the data is about, and where the existing \
modelling is already sound versus where it needs improvement.
2. Design the mapping:
- Decide what entities to model, what type to assign each, and what fields represent \
inter-entity relationships.
- Decide how to map them: mint IRIs from a stable identifier of the entity itself, in a \
namespace that reflects the domain — replacing the source's if it does not.
- Identify what null looks like in the data and how to handle it in the mapping.
- Decide whether one CONSTRUCT query is enough to cover the data completely, \
or whether the relationships in the knowledge graph require multiple queries. \
There is no fixed number, but together the queries must cover all the data without loss.
- For each entity type, find a matching class for its rdf:type, and for each source \
predicate, find a matching property, by searching the ontology knowledge graphs — \
search_entity for classes and search_property for predicates are the most direct \
approach, but notes and other functions are available too. A term is a good match \
when its semantics and expected value type/domain/range align with your intended \
mapping. Only fall back to the custom namespace if no ontology term fits after a \
genuine search."""

_STEPS_CUSTOM_DESIGN = """\
You should follow a step-by-step approach:
1. Explore the knowledge graph — use the available notes (and shape profile \
if available) as a starting point, and run targeted queries where needed to \
fill gaps. Get a clear picture of what classes and entities/predicates exist, \
what the values look like, and what the data is about, and where the existing \
modelling is already sound versus where it needs improvement.
2. Design the mapping:
- Decide what entities to model, what type to assign each, and what fields represent \
inter-entity relationships.
- Decide how to map them: mint IRIs from a stable identifier of the entity itself, in a \
namespace that reflects the domain — replacing the source's if it does not.
- Identify what null looks like in the data and how to handle it in the mapping.
- Identify whether there are unit informations like currency, scale or language metadata etc. \
associated with the data and how to handle them in the mapping.\
- Decide whether one CONSTRUCT query is enough to cover the data completely, or whether the \
relationships in the knowledge graph require multiple queries. There is no fixed number — let \
what you find in the knowledge graph decide, but together the queries must cover \
all the data without loss."""

_STEPS_WRITE_TEST = """\
3. Write each CONSTRUCT query and test it independently using propose_construct under a \
short, descriptive name. Every test saves the query text under that name; testing the same \
name again replaces the saved text. Check that it runs and produces non-empty, well-formed \
output. Do not add a LIMIT yourself — the test applies its own, and the saved query must \
cover all the data."""

# inserted before submission, only when competency questions are given
_STEPS_COMPETENCY_CHECK = """\
{n}. Before submitting, check the queries you wrote against each competency question by \
sketching the SPARQL query that would answer it against the graph produced \
by your queries. Use it as a usability test to see if the model is semantically well-formed \
and easy to query. If the question cannot be answered against the model at all, the design is \
missing something that has to be fixed. Do not narrow or make the model redundant \
to make the competency questions easier to answer."""

_STEPS_SUBMIT = """\
{n}. Once every query you planned is ready, call submit_construct with the names of all \
queries that make up the set. The saved text of each named query is what gets submitted.
"""


def _steps(
    ontology_mode: Literal["guided", "custom"],
    competency_questions: list[str],
) -> str:
    parts = [
        _STEPS_GUIDED_DESIGN
        if ontology_mode == "guided"
        else _STEPS_CUSTOM_DESIGN
    ]
    parts.append(_STEPS_WRITE_TEST)
    n = 4
    if competency_questions:
        parts.append(_STEPS_COMPETENCY_CHECK.format(n=n))
        n += 1
    parts.append(_STEPS_SUBMIT.format(n=n))
    return "\n".join(parts)

_COMPETENCY_QUESTIONS_NOTE = """\
Treat each competency question listed below as a query someone may later write against your \
output knowledge graph, and use it as a semantic usability test of the model that you propose: if \
the query answering it is overly complex - that verbosity of the SPARQL query may point \
to a gap in the semantic modelling itself that can be improved unless the underlying \
data requires such modeling.

The list of competency questions is not exhaustive — it states the minimum the \
output knowledge graph must be able to answer, not the full scope of it. \
Modelling all of the source data completely still applies.
"""


def _system_base(kg_name: str) -> str:
    return (
        f'You are a semantic RDF generation assistant. Your task is to analyse the '
        f'"{kg_name}" knowledge graph and produce one or more SPARQL CONSTRUCT '
        f'queries that model it as a well-formed, semantically correct knowledge graph. '
        f'The knowledge graph may already model some of its data well — where it does, '
        f'preserve or refine that modelling; where it doesn\'t, remodel it properly. The goal '
        f'is the best possible semantic model of the input data.'
    )

def system_information(
    ontology_mode: Literal["guided", "custom"],
    data_kg_name: str,
    ontology_kg_names: list[str],
    competency_questions: list[str] | None = None,
) -> str:
    competency_questions = competency_questions or []

    parts = [_system_base(data_kg_name)]
    if ontology_mode == "guided" and ontology_kg_names:
        parts.append(
            _ONTOLOGY_KG_NOTE
            + "\nOntology KGs: "
            + ", ".join(f'"{n}"' for n in ontology_kg_names)
        )
    parts.append(_steps(ontology_mode, competency_questions))
    if competency_questions:
        parts.append(_COMPETENCY_QUESTIONS_NOTE)
        parts.append(
            format_section(
                "Competency questions",
                format_enumerate(competency_questions),
            )
        )
    return "\n\n".join(parts)

_RULES_COMMON = [
"Always add rdfs:label to every entity in the CONSTRUCT but never make the label \
a precondition for the entity's other triples — a missing label shouldn't drop \
the entity from the query.",
"Where a property may be missing for a particular entity, generate it independently (e.g. with \
OPTIONAL) so a missing value doesn't suppress triples for the other \
properties of the same entity — but only where the data is actually \
missing, not as a blanket pattern for every property.",
"Never emit a triple with a null marker itself as the value.",
"If the same entity type is referenced by more than one query, mint its \
IRIs using the exact same pattern in every query that touches it, and reuse the same predicate \
names and namespace for the same concept across queries, so the queries \
combine into one coherent, non-conflicting graph.",
"A field name may carry information the value itself doesn't — a unit, a currency, a scale. \
Keep that information in the model rather than losing it with the name.",
"When resubmitting after reviewer feedback, change a query by testing the new \
version with propose_construct under the same name, then call submit_construct \
with the names of all queries in the set. The names you send are the whole set: \
a name you leave out is dropped, and its reason goes in the `removed` argument. \
Use list_constructs and get_construct to check the saved names and texts instead \
of relying on earlier messages."
]

# trailing LIMIT / OFFSET modifiers, in either order, at the end of a query
_TRAILING_LIMIT = re.compile(r"(\s*\b(?:LIMIT|OFFSET)\s+\d+)+\s*$", re.IGNORECASE)

class ConstructGenTask(GraspTask):
    name = "construct-gen"

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
        self.competency_questions: list[str] = []

        # latest tested version of every query, by name; written by
        # propose_construct, "ok" is whether that test produced triples
        self.drafts: dict[str, dict] = {}
        # output state: the current set of submitted queries, by name,
        # copied from drafts by submit_construct
        self.queries: dict[str, dict] = {}
        # whether the most recent submit_construct call was accepted
        self.submit_ok = False
        # queries the model explicitly withdrew, with its reasons
        self.removed_queries: list[dict] = []
        # verdict of the last review round
        self.review: dict | None = None


    def system_information(self) -> str:
        return system_information(
            self.ontology_mode,
            self.data_manager.kg,
            [m.kg for m in self.ontology_managers],
            self.competency_questions,
        )

    def rules(self) -> list[str]:
        return _RULES_COMMON

    def function_definitions(self) -> list[dict]:
        return [
            {
                "name": "propose_construct",
                "description": (
                    "Test a SPARQL CONSTRUCT query by running it with LIMIT 10. "
                    "Returns the first triples produced so you can verify the output "
                    "looks semantically correct. The query text is saved under the "
                    "given name, replacing any earlier text saved under that name; "
                    "submit_construct submits saved queries by name. Use this to "
                    "iteratively refine your query before submitting."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sparql": {
                            "type": "string",
                            "description": "The SPARQL CONSTRUCT query to test",
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Short, descriptive identifier for this query. Testing "
                                "again under the same name replaces the saved text."
                            ),
                        },
                    },
                    "required": ["sparql", "name"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "list_constructs",
                "description": (
                    "List the names of all saved queries, with whether each is in "
                    "the submitted set and whether its last test succeeded."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "get_construct",
                "description": (
                    "Return the exact saved text of a query, i.e. the version from "
                    "its latest propose_construct test."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the saved query.",
                        },
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "submit_construct",
                "description": (
                    "Submit the set of SPARQL CONSTRUCT queries by name. The saved "
                    "text of each named query (its latest propose_construct test) is "
                    "executed to produce the semantic RDF output. Only call this once "
                    "you are satisfied with the queries after testing them."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "description": (
                                "Names of all queries in the set, every time you call "
                                "this. A previously submitted query whose name is not "
                                "listed is dropped from the set."
                            ),
                            "items": {"type": "string"},
                        },
                        "removed": {
                            "type": ["array", "null"],
                            "description": (
                                "For previously submitted queries you are dropping, "
                                "the reason each is no longer needed. Use null or an "
                                "empty list when nothing is dropped."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": (
                                            "Name of the previously submitted query "
                                            "being dropped."
                                        ),
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": (
                                            "Why this query is no longer part of the "
                                            "set, and where its data is covered now "
                                            "if it still is."
                                        ),
                                    },
                                },
                                "required": ["name", "reason"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["names", "removed"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        if fn_name == "propose_construct":
            name = fn_args["name"].strip()
            if not name:
                return "The query needs a non-empty name. Please test it again with a name."

            # the saved text becomes the submitted query, so a LIMIT/OFFSET the
            # model added for testing would truncate the full output
            sparql = fn_args["sparql"].strip()
            saved = _TRAILING_LIMIT.sub("", sparql).strip()
            note = (
                "\nNote: the trailing LIMIT/OFFSET was removed from the saved "
                "query, since the saved query must cover all the data."
                if saved != sparql else ""
            )

            # save before running, so the latest test is always what is saved,
            # also when it fails
            self.drafts[name] = {"name": name, "sparql": saved, "ok": False}

            manager = self.data_manager

            try:
                result = _construct_request(
                    saved + "\nLIMIT 10",
                    manager.endpoint,
                    manager.headers,
                    manager.params,
                    timeout=self.config.sparql_query_timeout,
                )
            except FunctionCallException as e:
                return f"{e}\nSaved as '{name}', but it cannot be submitted until a test succeeds.{note}"
            except Exception as e:
                return (
                    f"Error executing CONSTRUCT query:\n{e}\nSaved as '{name}', "
                    f"but it cannot be submitted until a test succeeds.{note}"
                )

            lines = [ln for ln in result.strip().splitlines() if ln.strip()]
            if not lines:
                return (
                    f"Query '{name}' returned no triples. "
                    "Check your WHERE clause, table filter, and NULL handling. "
                    f"Saved as '{name}', but it cannot be submitted until a test "
                    f"succeeds.{note}"
                )

            self.drafts[name]["ok"] = True
            preview = "\n".join(lines[:20]) # at most 20 triples
            return (
                f"Preview for '{name}' ({len(lines)} triples with LIMIT 10):\n{preview}\n"
                f"Saved as '{name}'.{note}"
            )

        elif fn_name == "list_constructs":
            if not self.drafts:
                return "No queries saved yet. Test a query with propose_construct to save it."

            rows = []
            for name, draft in self.drafts.items():
                submitted = self.queries.get(name)
                if not draft["ok"]:
                    status = "last test failed"
                    if submitted is not None:
                        status += "; the submitted set holds an earlier working version"
                elif submitted is None:
                    status = "not submitted"
                elif submitted["sparql"] != draft["sparql"]:
                    status = "submitted, but tested again since; resubmit to send the new version"
                else:
                    status = "submitted"
                rows.append(f"{name}: {status}")
            return (
                f"Saved queries ({len(rows)}), submitted set has {len(self.queries)}:\n"
                + format_enumerate(rows)
            )

        elif fn_name == "get_construct":
            name = fn_args["name"].strip()
            draft = self.drafts.get(name)
            if draft is None:
                return (
                    f"No saved query named '{name}'. Saved names: "
                    f"{', '.join(self.drafts) or 'none'}."
                )
            status = "last test succeeded" if draft["ok"] else "last test failed"
            return f"Saved text of '{name}' ({status}):\n{draft['sparql']}"

        elif fn_name == "submit_construct":
            self.submit_ok = False
            names = [n.strip() for n in fn_args.get("names") or []]
            removed = fn_args.get("removed") or []

            if not names:
                return (
                    "Your call contained no query names, so nothing was changed. "
                    "Call submit_construct again with the names of all queries in the set."
                )

            if any(not n for n in names):
                return "Every name must be non-empty. Nothing was changed."

            seen: set[str] = set()
            duplicates: set[str] = set()
            for n in names:
                if n in seen:
                    duplicates.add(n)
                seen.add(n)
            if duplicates:
                return (
                    f"Duplicate name(s): {', '.join(sorted(duplicates))}. Each "
                    "query is listed once. Nothing was changed."
                )

            unknown = [n for n in names if n not in self.drafts]
            if unknown:
                return (
                    f"No saved query named {', '.join(unknown)}. Test each query with "
                    "propose_construct first. Saved names: "
                    f"{', '.join(self.drafts) or 'none'}. Nothing was changed."
                )

            failed = [n for n in names if not self.drafts[n]["ok"]]
            if failed:
                return (
                    f"The latest test of {', '.join(failed)} failed or returned no "
                    "triples. Fix and test again with propose_construct before "
                    "submitting. Nothing was changed."
                )

            removed_names = [r["name"].strip() for r in removed]
            bad_removed = [
                n for n in removed_names
                if n not in self.queries or n in seen
            ]
            if bad_removed:
                return (
                    f"Cannot give a removal reason for {', '.join(bad_removed)}: "
                    "`removed` is only for previously submitted queries that are not "
                    "in `names`. Nothing was changed."
                )

            old = self.queries
            new = {
                n: {"name": n, "sparql": self.drafts[n]["sparql"]}
                for n in names
            }
            added = [n for n in new if n not in old]
            updated = [n for n in new if n in old and old[n]["sparql"] != new[n]["sparql"]]
            dropped = [n for n in old if n not in new]
            not_submitted = [n for n, d in self.drafts.items() if d["ok"] and n not in new]

            self.queries = new
            self.removed_queries.extend( # accumulated reasons across rounds
                {"name": r["name"].strip(), "reason": r["reason"].strip()}
                for r in removed
            )
            self.submit_ok = True
            return (
                f"CONSTRUCT queries saved. Added: {', '.join(added) or 'none'}. "
                f"Updated: {', '.join(updated) or 'none'}. "
                f"Dropped: {', '.join(dropped) or 'none'}. "
                f"Current set ({len(new)}): {', '.join(new)}. "
                f"Tested but not in the set: {', '.join(not_submitted) or 'none'}."
            )

        raise FunctionCallException(f"Unknown function: {fn_name}")
 
    def done(self, fn_name: str) -> bool:
        return fn_name == "submit_construct" and self.submit_ok

    def setup(self, input: Any) -> str:
        assert isinstance(input, dict), "Input for construct-gen must be a dict"
        self.ontology_mode = input.get("ontology_mode", "custom")
        self.competency_questions = input.get("competency_questions", []) or []
        return (
            f'Analyse the "{self.data_manager.kg}" knowledge graph and generate '
            f"one or more SPARQL CONSTRUCT queries that model it as a well-formed, "
            f"semantically correct knowledge graph."
        )

    def output(self, messages: list[Message]) -> dict | None:
        if not self.queries:
            return None

        queries = list(self.queries.values())
        blocks = "\n\n".join(
            f"### {q['name']}\n```sparql\n{q['sparql']}\n```"
            for q in queries
        )
        return {
            "type": "output",
            "queries": queries,
            "removed": self.removed_queries,
            "kg": self.data_manager.kg,
            "review": self.review,
            "formatted": (
                f"CONSTRUCT queries for {self.data_manager.kg}:\n\n{blocks}"
            ),
        }

