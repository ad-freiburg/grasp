from enum import StrEnum

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.tasks.auto_setup import AutoSetupTask
from grasp.tasks.base import GraspTask
from grasp.tasks.cea import CeaTask
from grasp.tasks.exploration import (
    FunctionalExplorationTask,
    StructuralExplorationTask,
)
from grasp.tasks.entity_linking import EntityLinkingTask
from grasp.tasks.general_qa import GeneralQaTask
from grasp.tasks.notes_from_samples import NotesFromSamplesTask
from grasp.tasks.notes_from_traces import NotesFromTracesTask
from grasp.tasks.question_generation import QuestionGenerationTask
from grasp.tasks.shapes_setup import ShapesSetupTask
from grasp.tasks.sparql_qa import SparqlQaTask
from grasp.tasks.sparql_to_question import SparqlToQuestionTask
from grasp.tasks.wikidata_query_logs import WdqlTask


# official tasks supported by GRASP, excluding exploration
# and auto-setp which are special tasks
class Task(StrEnum):
    SPARQL_QA = "sparql-qa"
    GENERAL_QA = "general-qa"
    CEA = "cea"
    WDQL = "wikidata-query-logs"
    S2Q = "sparql-to-question"
    ENTITY_LINKING = "entity-linking"


_REGISTRY: dict[str, type[GraspTask]] = {
    cls.name: cls
    for cls in [
        SparqlQaTask,
        GeneralQaTask,
        CeaTask,
        WdqlTask,
        SparqlToQuestionTask,
        EntityLinkingTask,
        FunctionalExplorationTask,
        StructuralExplorationTask,
        QuestionGenerationTask,
        NotesFromTracesTask,
        NotesFromSamplesTask,
        AutoSetupTask,
        ShapesSetupTask,
    ]
}


def get_task(
    task: str,
    managers: list[KgManager],
    config: GraspConfig,
    known: set[str] | None = None,
) -> GraspTask:
    if task not in _REGISTRY:
        raise ValueError(f"Unknown task {task}")
    return _REGISTRY[task](managers, config, known)


def rules() -> list[str]:
    return [
        "Explain your thought process before each step and function call.",
        "Do not ask the user for clarification, neither on the initial input nor on \
follow-up inputs or feedback. When the task input is incomplete or \
ambiguous, proceed based on reasonable assumptions.",
        "Use IRIs returned in function call results as is in subsequent function calls. \
Shortening them to their prefixed form, and escaping or encoding special characters might \
lead to errors and unexpected or empty results.",
        'Do not use "SERVICE wikibase:label { bd:serviceParam wikibase:language ..." \
in SPARQL queries. It is not SPARQL standard and unsupported by most SPARQL endpoints. \
Use rdfs:label or similar properties to get labels instead.',
        "If example or shape indices are available, using them early on to quickly find \
relevant information to solve the task is recommended. \
    ",
    ]


def multimodal_rules() -> list[str]:
    return [
        """You MUST assume that you do not have direct access to image or audio content. \
You MAY only process non-text media through the available multimodal tools. \
When a user provides an image or audio file and the task depends on visual or auditory evidence, you MUST use `analyze(...)` to inspect it. \
You MUST use `load(...)` only when the media needs to be prepared or normalized before analysis. \
You MUST NOT use multimodal tools when text or structured data is sufficient. \
If a visually observable attribute is requested and text or structured sources do not answer it, you MUST use `analyze(...)` instead of refusing. \
You MUST NOT claim to see the media directly; any conclusion about the media must be based only on multimodal tool output."""
    ]
