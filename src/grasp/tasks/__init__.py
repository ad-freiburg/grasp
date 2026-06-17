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
    WDQL = "wikidata-query-logs"


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
relevant information to solve the task is recommended.",
    ]
