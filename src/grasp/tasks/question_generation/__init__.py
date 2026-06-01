from typing import Any

from pydantic import BaseModel

from grasp.configs import NotesGenerateQuestionsConfig
from grasp.model import Message
from grasp.tasks.base import GraspTask
from grasp.tasks.functions import find_frequent, find_frequent_function_definition
from grasp.tasks.question_generation.functions import (
    function_definitions,
    show_questions,
    submit_question,
)
from grasp.utils import format_list


class QuestionGenerationState(BaseModel):
    questions: dict[str, list[str]] = {}


def rules() -> list[str]:
    return [
        "You may submit questions without verification or questions you know "
        "are unanswerable or unrelated to the knowledge graph, since actual users "
        "might do the same.",
        "Aim for a roughly equal distribution of questions across the available "
        "knowledge graphs. Where appropriate, you may also submit questions "
        "that span multiple knowledge graphs.",
    ]


def system_information(config: NotesGenerateQuestionsConfig) -> str:
    return f"""Your task is to generate a diverse pool of questions \
that real users might ask over the available knowledge graphs.

Examples for types of questions to consider (not exhaustive):
- Factual (e.g., "President of the United States")
- Exploratory (e.g., "Tell me about the works of Vincent van Gogh")
- Aggregative (e.g., "How many people live in the largest city in Brazil?")
- Low-level / Technical (e.g., "All properties of the entity Q42")
- High-level (e.g., "What are the ethical implications of AI?")
- Detailed (e.g. "List all movies directed by Christopher Nolan, \
each with a comma-separated list of its main actors, and their release years \
relative to 2010, the one with the most actors born before 1970 first")

You should follow a step-by-step approach:
1. Look at the existing questions to identify well-covered \
areas and gaps regarding topic, type, difficulty, etc.
2. Come up with a potential user question that targets a new or \
underrepresented angle. If needed, explore the knowledge graph using the provided \
functions to gain inspiration or context. Optionally, soft-verify \
that your question is answerable by drafting a corresponding \
SPARQL query or checking the existence of relevant entities or properties.
3. Once you are satisfied with your question, finalize and submit it. \
Otherwise, you can discard it and start over from step 1.
4. Repeat the process until you have submitted roughly {config.questions_per_round} \
questions for this round, then stop."""


def output(state: QuestionGenerationState) -> dict:
    kg_questions = []
    for kg, questions in state.questions.items():
        kg_questions.append(f'Questions for "{kg}":\n' + format_list(questions))

    formatted = "\n\n".join(kg_questions)

    return {
        "type": "output",
        "questions": {kg: list(qs) for kg, qs in state.questions.items()},
        "formatted": formatted,
    }


class QuestionGenerationTask(GraspTask):
    name = "question-generation"

    def system_information(self) -> str:
        assert isinstance(self.config, NotesGenerateQuestionsConfig)
        return system_information(self.config)

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        kgs = [m.kg for m in self.managers]
        functions = function_definitions(kgs)
        functions.append(find_frequent_function_definition(kgs, self.config.list_k))
        return functions

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert isinstance(self.config, NotesGenerateQuestionsConfig)
        assert self.state is not None, (
            "State must be provided for question generation task"
        )

        if fn_name == "stop":
            return "Stopping"

        elif fn_name == "submit_question":
            return submit_question(
                self.state.questions,
                fn_args["kg"],
                fn_args["question"],
            )

        elif fn_name == "show_questions":
            return show_questions(
                self.state.questions,
                fn_args["kg"],
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

        raise ValueError(f"Unknown function {fn_name}")

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> str:
        assert isinstance(input, QuestionGenerationState), (
            "Input for question generation must be a QuestionGenerationState"
        )
        self.state = input
        return "Start generating questions."

    def output(self, messages: list[Message]) -> dict:
        return output(self.state)
