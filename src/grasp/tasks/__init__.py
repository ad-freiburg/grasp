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


def multimodal_rules(isMultimodal: bool) -> list[str]:
    rules = [
        "You MUST NOT use multimodal tool calls when text or structured data is sufficient.",
        "Reuse prior inspection or analysis results, do not analyze the same media twice.",
        "You MUST use ALL user provided inputs before canceling the task.",
        "When the Answer can not be provided by structured text data alone, try to use images or other data queried from the KG"
    ]
    if (isMultimodal):
        rules.append("The current conversation includes directly accessible image input.")
        rules.append("When a current-message image is relevant, first inspect it yourself using your\
   built-in visual understanding. Do not call any tool for this initial inspection.")
        rules.append("Use load(...) only to retrieve media not directly accessible.")
        rules.append("Use analyze(...) only audio inputs.")
        rules.append("Never invent input IDs, file handles, URLs, vision models, or media references.")
    else:
        rules.append("You MUST assume that you do not have direct access to image or audio content.")
        rules.append("If a visually observable attribute is requested and text or structured sources \
do not answer it, you MUST use `analyze(...)` instead of refusing.")
        rules.append("When a user provides an audio or image file and the task depends on visual \
or auditory evidence, you MUST use `analyze(...)` to inspect it.")
        rules.append("When structured Data does not suffice for your answer and a visual or accoustic analysis could help, \
you can use analyze() to inspect referenced data from the Database like image-urls or links to media")
    return rules
