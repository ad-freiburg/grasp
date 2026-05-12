import random
from typing import Any
from enum import StrEnum

from pydantic import BaseModel
from universal_ml_utils.table import generate_table

from grasp.configs import GraspConfig
from grasp.functions import find_manager
from grasp.manager import KgManager, format_kgs
from grasp.model import Message
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.utils import FunctionCallException, format_list, format_notes
from grasp.tasks.cea import prepare_annotation as prepare_entity, Annotation


# TODO: If we keep this design, the class hierarchy should
# obviously be refactored...
class Entity(Annotation):
    def format(self) -> str:
        output = f"IRI: {self.entity}"
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
    entity1: Entity
    entity2: Entity

    # OAEI standard
    measure: float = 1.0
    relation: Relation = Relation.EQUIVALENCE


class AlignmentTaskInput(BaseModel):
    source_data: list[Entity]

    source_kg: str
    target_kg: str

    description: str | None = None


class AlignmentState:
    def __init__(self):
        self.task_input: AlignmentTaskInput | None = None
        # mapping from each processed entity IRI to its correspondence object
        self.correspondences: dict[str, Correspondence] = {}
        # self.proccessed: set[str] = set()

    def add_1_to_1_correspondence(self, correspondence: Correspondence):
        self.correspondences[correspondence.entity1.entity] = correspondence
        self.correspondences[correspondence.entity2.entity] = correspondence

    def get_correspondence(self, entity: str) -> Correspondence | None:
        """
        returns a `Correspondence`-object in which the IRI `entity` 
        is mapped to antoher entity or `None`, if `entity` is not
        matched yet.
        """
        return self.correspondences.get(entity)

    def remove_correspondence(self, entity1: str, entity2: str):
        self.correspondences.pop(entity1)
        self.correspondences.pop(entity2)

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
            [c.entity1.identifier, c.entity2.identifier, str(c.relation.value)]
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
                        "description": "The IRI of the source entity whose correspondence should be removed",
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "The IRI of the target entity whose correspondence should be removed",
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
You will process the source ontology in batches and must identify equivalent classes \
within the target ontology. Follow this step-by-step approach:
1. Analyze the source ontology to understand its core domain \
and conceptual scope. Consider how these concepts might be represented or labeled \
differently in the target ontology.
2. Identify equivalent classes in the target ontology. Start \
with high-level, foundational concepts or obvious matches to anchor the alignment. \
Leverage the provided search and exploration functions to validate semantic and \
structural similarities. Refine your mappings as you gain deeper insights into \
both knowledge graphs. You may need to adapt your correspondences based on new \
insights along the way.
3. Once all significant correspondences have been established and \
verified, use the stop function to finalize the alignment and terminate \
the process."""


def rules() -> list[str]:
    return [
        "Each entity from the source graph must map to exactly one entity in the target graph."
        "Every new correspondence will be automatically validated by the system. If a collision"
        "occurs (an entity is already mapped), the system will present the conflicting mappings to"
        "yout. You must then review both the previous and the new match to determine which is "
        "conceptually superior and resolve the conflict.",
        "If you cannot find a suitable reference entity in the target graph, "
        "leave the source entity unmatched.",
        "When multiple target candidates exist, select the one that best fits the structural context "
        "(hierarchy, properties) or represents the most accurate level of abstraction.",
        "Leverage the semantic context of both graphs. To improve efficiency, favor batch-retrieval "
        "SPARQL queries over repetitive individual searches when identifying patterns across "
        "multiple entities",
        "Before writing custom SPARQL queries, prioritize using the built-in search_entities and list_entities functions to explore the target ontology quickly."
        "Perform a final comprehensive review of all established correspondences before concluding the task.",
        "You must explicitly evaluate EVERY SINGLE entity provided in the source list. Do not stop until you" "have attempted to match all of them. If an entity truly has no match, explain briefly why,"
        "but do not simply skip it."
    ]


# TODO: If we keep this design, the class hierarchy and inheritance should
# obviously be refactored...
def prepare_correspondence(manager1: KgManager, manager2: KgManager, entity1: str, entity2: str) -> Correspondence:
    entity_object_1 = Entity.from_annotation_object(prepare_entity(manager1, entity1))
    entity_object_2 = Entity.from_annotation_object(prepare_entity(manager2, entity2))
    return Correspondence(
        entity1=entity_object_1,
        entity2=entity_object_2
        )


# TODO: Refactor to eliminate code duplicates to `annotate`-method from cea.py
def add_1_to_1_correspondence(
        managers: list[KgManager],
        kg1: str,
        kg2: str,
        entity1: str,
        entity2: str,
        state: AlignmentState,
        known: set[str],
        know_before_use: bool = True,
        overwrite: bool = False
        ) -> str:
    manager1, _ = find_manager(managers, kg1)
    manager2, _ = find_manager(managers, kg2)
    try:
        correspondence = prepare_correspondence(manager1, manager2, entity1, entity2)
        # TODO: Handle know_before_use
        state.add_1_to_1_correspondence(correspondence)
    except ValueError as e:
        raise FunctionCallException(str(e)) from e
    # TODO: Handle overwriting feedback to LLM message
    return f"Aligned {entity1} from {kg1} with {entity2} from {kg2}"


def delete_correspondence(entity1: str, entity2: str, state: AlignmentState) -> str:
    correspondence1 = state.get_correspondence(entity1)
    correspondence2 = state.get_correspondence(entity2)
    if correspondence1 is None or correspondence1 is not correspondence2:
        raise FunctionCallException(
            f"There is no Correspondence aligning {entity1} to {entity2} yet"
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

        return add_1_to_1_correspondence(
            managers, kg1, kg2, ent1, ent2, state, known, config.know_before_use, overwrite
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
    instructions = f"""\
Align the following entities from the source ontology {task_input.source_kg} \
with entities from the target ontology {task_input.target_kg}:

"""
    if task_input.description:
        instructions += f"Context: {task_input.description}\n\n"

    for entity in task_input.source_data:
        instructions += f"- {entity.format()}\n"

    return instructions


def input_and_state(input: Any, config: GraspConfig) -> tuple[str, AlignmentState]:
    try:
        task_input = AlignmentTaskInput(**input)
    except Exception as e:
        raise ValueError("OM task input must match AlignmentTaskInput schema") from e

    state = AlignmentState()
    state.task_input = task_input

    instructions = input_instructions(task_input, state)
    return instructions, state


def feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    # Der Kritiker braucht den Kontext der Ontologien
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
