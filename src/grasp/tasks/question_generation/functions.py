from pydantic import BaseModel, Field

from grasp.function_args import function_definition
from grasp.utils import FunctionCallException, clip, format_enumerate


class SubmitQuestionArgs(BaseModel):
    kg: str = Field(
        description="The knowledge graph the question is intended for",
    )
    question: str = Field(
        description="The question as a real user would phrase it",
    )


class ShowQuestionsArgs(BaseModel):
    kg: str = Field(
        description="The knowledge graph for which to show submitted questions",
    )
    page: int = Field(
        description="Page number (1-indexed) for paginating results "
        "(default should be 1)",
    )


def function_definitions(kgs: list[str]) -> list[dict]:
    return [
        function_definition(
            "submit_question",
            "Submit a generated user question for the given "
            "knowledge graph. The question is added to the pool for the round.",
            SubmitQuestionArgs,
            {"kg": kgs},
        ),
        function_definition(
            "show_questions",
            "Show previously submitted questions for the given "
            "knowledge graph, most recent first.",
            ShowQuestionsArgs,
            {"kg": kgs},
        ),
        function_definition(
            "stop",
            "Stop the question generation process.",
        ),
    ]


def submit_question(
    questions: dict[str, list[str]],
    kg: str,
    question: str,
) -> str:
    question = question.strip()
    if not question:
        raise FunctionCallException("Question must not be empty")

    kg_questions = questions.setdefault(kg, [])
    kg_questions.append(question)
    return f'Submitted question {len(kg_questions)} for "{kg}": {clip(question, 64)}'


def show_questions(
    questions: dict[str, list[str]],
    kg: str,
    page: int,
    k: int,
) -> str:
    if page < 1:
        raise FunctionCallException("Page number must be at least 1")

    kg_questions = questions.get(kg, [])
    if not kg_questions:
        return "None"

    total_pages = (len(kg_questions) + k - 1) // k
    if page > total_pages:
        raise FunctionCallException(f"Page number exceeds maximum page {total_pages}")

    # most recent first
    ordered = list(reversed(kg_questions))

    start = (page - 1) * k
    end = page * k
    page_items = ordered[start:end]

    header = f"Most recently submitted questions (page {page} of {total_pages}):\n"
    return header + format_enumerate(page_items, start=start)
