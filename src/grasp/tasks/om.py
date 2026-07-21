import random
from typing import Any
from enum import StrEnum

import requests
from pydantic import BaseModel
from universal_ml_utils.table import generate_table

from grasp.configs import GraspConfig
from grasp.functions import find_manager
from grasp.manager import KgManager, format_kgs
from grasp.model import Message
from grasp.sparql.types import AskResult
from grasp.sparql.utils import prepare_identifier_for_sparql
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.utils import FunctionCallException, format_list, format_notes
from grasp.tasks.cea import prepare_annotation, Annotation


# TODO: If we keep this design, the class hierarchy should
# obviously be refactored...
class Entity(Annotation):
    def format(self) -> str:
        output = f"Full IRI: {self.identifier}, Shortened IRI: {self.entity}"
        if self.label is not None:
            output += f", label: {self.label}"
        if self.aliases is not None and self.aliases != []:
            output += f", aliases: {self.aliases}"
        if self.infos is not None and self.infos != []:
            output += f", infos: {self.infos}"
        return output

    @staticmethod
    def from_annotation_object(annotation: Annotation) -> 'Entity':
        return Entity(
            identifier=annotation.identifier,
            entity=annotation.entity,
            label=annotation.label,
            aliases=annotation.aliases,
            infos=annotation.infos
        )


class Relation(StrEnum):
    EQUIVALENCE = "="
    SUBSUMPTION = "<="
    COMPLEX = "complex"


class Correspondence(BaseModel):
    # TODO: Enxtendable to handle an entity from
    # one ontology and a corresponding complex
    # pattern from another onology?
    """
    A singe correspondence between two entities
    from different ontologies.
    """
    entity_source: Entity
    entity_target: Entity

    # OAEI standard
    measure: float = 1.0
    relation: Relation = Relation.EQUIVALENCE


class PotentialCorrespondences(BaseModel):
    source_entity: Entity
    candidates: list[Entity]


class LogMapValidationResult(BaseModel):
    valid: bool
    message: str


class AlignmentTaskInput(BaseModel):
    unmatched_entities: list[Entity] = []
    potential_correspondences: list[PotentialCorrespondences] = []
    input_alignment: list[Correspondence] = []

    source_kg: str
    target_kg: str

    description: str | None = None


class AlignmentState:

    __global_correspondences: dict[str, Correspondence] = {}

    def __init__(self):
        self.task_input: AlignmentTaskInput | None = None
        # mapping from each processed entity IRI to its correspondence object
        self.correspondences: dict[str, Correspondence] = AlignmentState.__global_correspondences
        # self.proccessed: set[str] = set()

    def add_1_to_1_correspondence(self, correspondence: Correspondence):
        self.correspondences[correspondence.entity_source.identifier] = correspondence
        self.correspondences[correspondence.entity_target.identifier] = correspondence

    def get_correspondence(self, entity: str) -> Correspondence | None:
        """
        returns a `Correspondence`-object in which the full IRI `entity`
        is mapped to another entity or `None`, if `entity` is not
        matched yet.
        """
        return self.correspondences.get(entity)

    def remove_correspondence(self, entity_source: str, entity_target: str):
        self.correspondences.pop(entity_source)
        self.correspondences.pop(entity_target)

    def to_dict(self) -> dict:
        unique_correspondences = {id(c): c for c in self.correspondences.values()}
        return {
            "formatted": self.format(),
            "correspondences": [c.model_dump() for c in unique_correspondences.values()]
        }

    def format(self, num: int | None = None) -> str:
        unique_correspondences = list({id(c): c for c in self.correspondences.values()}.values())

        if not unique_correspondences:
            return "No correspondences established yet."

        if num is not None and num < len(unique_correspondences):
            unique_correspondences = random.sample(unique_correspondences, num)

        data = [
            [c.entity_source.identifier, c.entity_target.identifier, str(c.relation.value)]
            for c in unique_correspondences
        ]
        headers = [["Source Entity (Ontology 1)", "Target Entity (Ontology 2)", "Relation"]]

        return generate_table(data=data, headers=headers)


def functions(managers: list[KgManager]) -> list[dict]:
    fns = [
        {
            "name": "set_correspondence",
            "description": """\
Establishes a 1:1 semantic correspondence between an entity from the source \
ontology and one from the target ontology.

The system automatically enforces 1:1 mapping constraints. If either the source \
or target entity is already involved in an existing mapping:
1. If 'overwrite' is False (default), the function returns a conflict warning \
detailing the existing match to prompt a manual review.
2. If 'overwrite' is True, the previous mapping is silently replaced by the new one.
Use this to lock in your alignment decisions.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "The exact IRI of the source entity",
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "The exact IRI of the target entity",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "If True, resolves mapping collisions by overwriting existing "
                        "correspondences involving these entities.",
                    },
                },
                "required": ["source_entity", "target_entity", "overwrite"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "delete_correspondence",
            "description": "Delete an existing correspondence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "The full IRI of the source entity whose correspondence should be removed",
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "The full IRI of the target entity whose correspondence should be removed",
                    },
                },
                "required": ["source_entity", "target_entity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_correspondences",
            "description": "Displays the current state of established mappings. If 'num' is provided, "
            "it shows a random sample; otherwise, it lists all correspondences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num": {
                        "type": "integer",
                        "description": "The number of random mappings to display for review",
                        },
                    },
                "required": [],
                "additionalProperties": False,
                },
            "strict": False,
        },
        {
            "name": "stop",
            "description": "Finalize your mappings and stop the alignment process.",
        },
    ]
    return fns


def system_information() -> str:
    # TODO: extend for other matching tasks
    return """\
You are an expert ontology alignment assistant. Your mission is to establish precise \
T-Box (schema) correspondences between a source and a target ontology.
You will process the source ontology in batches and must identify equivalent classes and properties \
within the target ontology. Follow this step-by-step approach:
1. Analyze the source and target ontology to understand their core domain \
and conceptual scope. Consider how concepts might be represented, structured or labeled \
differently in the target ontology.
2. Identify equivalent classes or properties in the target ontology. Start \
by establishing high-confidence matches, regardless of their position in the hierarchy, \
with obvious matches based on strict lexical similarities (e.g., exact label matches, \
identical local names, or unambiguous synonyms). These initial 1:1 correspondences will \
serve as your structural anchors for the rest of the process. \
3. Leverage the established anchors to explore their structural neighborhood. \
Use the provided functions to compare the superclasses, subclasses, domains, and ranges \
of your anchored entities. Use this structural context to discover new, less obvious mappings \
and to validate the semantic correctness of your baseline alignment.
4. Once all significant correspondences have been established and \
verified, use the stop function to finalize the alignment and terminate \
the process."""


def rules() -> list[str]:
    return [
        "Injectivity: Each entity from the source graph must map to exactly one entity in the target graph. "
        "Every new correspondence will be automatically validated by the system. If a collision "
        "occurs (an entity is already mapped), the system will present the conflicting mappings to "
        "you. You must then review both the previous and the new match to determine which is "
        "conceptually superior and resolve the conflict.",
        "Strict Equivalence: Do not set correspondences between source and target entities if "
        "they are not truly equivalent. Mappings must not cause logical contradictions.",
        "Precision over Recall: If you cannot find a suitable reference entity in the target graph,"
        "leave the source entity unmatched. It is better to set fewer but fully logically "
        "consistent correspondences than more but less robust equivalences.",
        "Batch over Iteration: To improve efficiency and avoid hitting context limits, "
        "favor batch-retrieval SPARQL queries over repetitive individual searches when "
        "executing manual SPARQL queries for identifying patterns across multiple entities",
        "Built-in functions: Before writing custom SPARQL queries, prioritize using the built-in "
        "search, list and shapes functions to explore the target ontology quickly.",
        "Completeness: Perform a final comprehensive review of all established correspondences "
        "before concluding the task. You must explicitly evaluate every single entity provided "
        "in the source list. Do not stop until you have attempted to match all of them. "
        "If an entity truly has no match, explain briefly why, but do not simply skip it."
    ]


def check_alignment_with_logmap(state: AlignmentState, server_url: str) -> str | None:
    """
    Sends the current, deduplicated alignment to the LogMap server for a
    consistency/conflict check. Returns feedback text to surface to the LLM if
    the check failed or could not be performed, or None if it passed cleanly.
    """
    unique_correspondences = list({id(c): c for c in state.correspondences.values()}.values())

    try:
        response = requests.post(
            f"{server_url}/validate",
            json=[c.model_dump() for c in unique_correspondences],
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return f"LogMap check could not be performed: {e}"

    result = LogMapValidationResult(**response.json())
    if not result.valid:
        return f"LogMap check failed: {result.message}"

    else:
        return result.message


def entity_exists_in_kg(manager: KgManager, iri: str) -> bool:
    # subject and object position cover classes/instances; predicate position
    # is needed too since a property may appear only there (e.g. used in
    # triples without a separate rdf:type owl:ObjectProperty/DatatypeProperty
    # declaration triple of its own)
    wrapped = prepare_identifier_for_sparql(iri, manager.iri_literal_parser)
    sparql = (
        f"ASK {{ {{ {wrapped} ?p ?o }} UNION {{ ?s ?p {wrapped} }} "
        f"UNION {{ ?s {wrapped} ?o }} }}"
    )
    result = manager.execute_sparql(sparql)
    assert isinstance(result, AskResult)
    return result.boolean


def ensure_known(manager: KgManager, iri: str, known: set[str]) -> None:
    """
    Fast path: skip the SPARQL round-trip if the IRI was already surfaced by
    an earlier function call result during this conversation. Otherwise fall
    back to an actual live existence check against the KG, and only report
    the IRI as invalid to the LLM once that check has genuinely failed.
    """
    if iri in known:
        return

    if entity_exists_in_kg(manager, iri):
        known.add(iri)
        return

    raise FunctionCallException(
        f"The entity {iri} does not seem to exist in the knowledge graph. "
        "Double check the IRI."
    )


# TODO: Refactor to eliminate code duplicates to `annotate`-method from cea.py
def add_1_to_1_correspondence(
        managers: list[KgManager],
        kg_source: str,
        kg_target: str,
        entity_source: str,
        entity_target: str,
        state: AlignmentState,
        known: set[str],
        know_before_use: bool = True,
        overwrite: bool = False,
        logmap_server_url: str | None = None,
        ) -> str:

    try:
        manager_source, _ = find_manager(managers, kg_source)
        manager_target, _ = find_manager(managers, kg_target)
        entity_object_source = Entity.from_annotation_object(prepare_annotation(manager_source, entity_source))
        entity_object_target = Entity.from_annotation_object(prepare_annotation(manager_target, entity_target))
        full_iri_source = entity_object_source.identifier
        full_iri_target = entity_object_target.identifier

        if know_before_use:
            ensure_known(manager_source, full_iri_source, known)
            ensure_known(manager_target, full_iri_target, known)

        if not overwrite:
            existing_source_corr = state.get_correspondence(full_iri_source)
            existing_target_corr = state.get_correspondence(full_iri_target)

            if existing_source_corr is not None:
                return (
                    f"Collision Error: The source entity {entity_source} is already mapped to"
                    f"{existing_source_corr.entity_target.entity}. If you want to change this mapping, "
                    "you must set 'overwrite' to True."
                    )
            if existing_target_corr is not None:
                return (
                    f"Collision Error: The target entity {entity_target} is already mapped to"
                    f"{existing_target_corr.entity_source.entity}. If you want to change this mapping, "
                    "you must set 'overwrite' to True."
                    )

        correspondence = Correspondence(entity_source=entity_object_source, entity_target=entity_object_target)
        state.add_1_to_1_correspondence(correspondence)
        message = f"Aligned {entity_source} from {kg_source} with {entity_target} from {kg_target}."

        if logmap_server_url is not None:
            feedback = check_alignment_with_logmap(state, logmap_server_url)
            if feedback is not None:
                message += f"\n{feedback}"

        return message

    except ValueError as e:
        raise FunctionCallException(str(e)) from e


def delete_correspondence(entity1: str, entity2: str, state: AlignmentState) -> str:
    correspondence1 = state.get_correspondence(entity1)
    correspondence2 = state.get_correspondence(entity2)
    if correspondence1 is None or correspondence1 is not correspondence2:
        raise FunctionCallException(
            f"There is no Correspondence aligning {entity1} to {entity2} yet. "
            "Notice you have to use the full IRI as entity reference."
            )
    state.remove_correspondence(entity1, entity2)
    return f"Deleted correspondence between {entity1} and {entity2}"


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: AlignmentState | None = None,
    example_indices: dict | None = None,
) -> str:
    assert isinstance(state, AlignmentState), (
        "Alignments must be provided as state for OM task"
    )
    assert not example_indices, "Example indices are not supported for OM task"

    if fn_name == "set_correspondence":
        if state is None:
            return "No AlignmentState was provided. Cannot set correspondences without AlignmentState"
        kg1 = state.task_input.source_kg  # type: ignore
        kg2 = state.task_input.target_kg  # type: ignore
        overwrite = fn_args.get("overwrite", False)

        ent1, ent2 = fn_args["source_entity"], fn_args["target_entity"]
        om_kwargs = config.task_kwargs.get("om", {})
        logmap_server_url = om_kwargs.get("logmap_server_url")
        know_before_use = om_kwargs.get("know_before_use", True)

        return add_1_to_1_correspondence(
            managers, kg1, kg2, ent1, ent2, state, known,
            know_before_use, overwrite, logmap_server_url,
        )

    elif fn_name == "delete_correspondence":
        return delete_correspondence(fn_args["source_entity"], fn_args["target_entity"], state)

    elif fn_name == "show_correspondences":
        return state.format(num=fn_args.get("num"))

    elif fn_name == "stop":
        return "Stopping"

    else:
        raise ValueError(f"Unknown function {fn_name}")


def input_instructions(task_input: AlignmentTaskInput, state: AlignmentState) -> str:
    instructions = ""
    if len(task_input.unmatched_entities) != 0:
        instructions += f"""\
Align the following entities from the source ontology {task_input.source_kg} \
with entities from the target ontology {task_input.target_kg}:
"""
        if task_input.description:
            instructions += f"Context: {task_input.description}\n\n"

        for entity in task_input.unmatched_entities:
            instructions += f"- {entity.format()}\n"

    if len(task_input.potential_correspondences) != 0:
        instructions += f"""\
You are given a list of potential correspondences found by simple string matching. \
Verify them and set the correspondences using the built-in functions for pairs you truely \
find to represent an equivalnce. Remeber the injectivity rule.
"""
        for corr in task_input.potential_correspondences:
            instructions += f"- source entity: {corr.source_entity.format()}; potential matches: "
            instructions += "; ".join((c.format() for c in corr.candidates)) + "\n"

    return instructions


def input_and_state(input: Any, config: GraspConfig) -> tuple[str, AlignmentState]:
    try:
        task_input = AlignmentTaskInput(**input)
    except Exception as e:
        raise ValueError("OM task input must match AlignmentTaskInput schema") from e

    state = AlignmentState()
    state.task_input = task_input
    for correspondence in state.task_input.input_alignment:
        state.add_1_to_1_correspondence(correspondence)

    instructions = input_instructions(task_input, state)
    return instructions, state


def feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    return f"""\
You are an expert ontology matching judge. Your task is to evaluate if the assistant has correctly identified and set correspondences between the entities.

The system has access to the following knowledge graphs:
{format_kgs(managers, kg_notes)}

The system was provided the following notes across all knowledge graphs:
{format_notes(notes)}

The system was provided the following rules to follow:
{format_list(rules()) if rules() else "None"}

Provide your feedback with the give_feedback function.\
"""


def feedback_instructions(inputs: list[str], output: dict) -> str:
    assert inputs, "At least one input is required for feedback"

    if len(inputs) > 1:
        prompt = (
            "Previous inputs:\n" + "\n\n".join(i.strip() for i in inputs[:-1]) + "\n\n"
        )
    else:
        prompt = ""

    prompt += f"Input:\n{inputs[-1].strip()}"
    prompt += f"Current Alignment State:\n{output['formatted']}\n\n"
    prompt += (
        "Does the current state fulfill the task? If the system has" +
        "not finished the matching process yet, provide prompts describing" +
        "the concrete next step for the system to continue and fininalize the matching.")
    return prompt


class OmTask(GraspTask, FeedbackTask):
    name = "om"

    def setup(self, input: Any) -> str:
        instructions, self.state = input_and_state(input, self.config)
        return instructions

    def system_information(self) -> str:
        return system_information()

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        return functions(self.managers)

    def output(self, messages: list[Message]) -> dict:
        return self.state.to_dict()

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
            ) -> str:
        return call_function(
            self.config,
            self.managers,
            fn_name,
            fn_args,
            known,
            self.state,
            example_indices,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    @property
    def default_input_field(self) -> str | None:
        return None

    def feedback_system_message(
        self, kg_notes: dict[str, list[str]], notes: list[str]
    ) -> str:
        return feedback_system_message(self.managers, kg_notes, notes)

    def feedback_instructions(self, inputs: list[str], output: dict) -> str:
        return feedback_instructions(inputs, output)
