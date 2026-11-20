from logging import Logger
from typing import Any, Literal

from pydantic import BaseModel, Field
from universal_ml_utils.logging import get_logger

from grasp.function_args import FunctionDefinition
from grasp.model import Message, Model
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.utils import format_message, format_response


def format_feedback(feedback: dict) -> str:
    status = feedback["status"]
    return f"Feedback (status={status}):\n{feedback['feedback']}"


class GiveFeedbackArgs(BaseModel):
    status: Literal["done", "refine", "retry"] = Field(description="The feedback type")
    feedback: str = Field(description="The feedback message")


def functions() -> list[FunctionDefinition]:
    return [
        FunctionDefinition(
            "give_feedback",
            """\
Provide feedback on the output of the system for the \
specified task.

The feedback status can be one of:
1. done: The output is correct and complete in its current form
2. refine: The output is sensible, but needs some refinement
3. retry: The output is incorrect and needs to be reworked

The feedback message should describe the reasoning behind the chosen status \
and provide suggestions for improving the output if applicable.""",
            GiveFeedbackArgs,
        )
    ]


def generate_feedback(
    model: Model,
    task: GraspTask,
    kg_notes: dict[str, list[str]],
    notes: list[str],
    inputs: list[str],
    output: Any,
    logger: Logger = get_logger("GRASP FEEDBACK"),
) -> dict | None:
    if not isinstance(task, FeedbackTask):
        return None

    messages: list[Message] = [
        Message.system(content=task.feedback_system_message(kg_notes, notes)),
        Message.user(content=task.feedback_instructions(inputs, output)),
    ]

    for msg in messages:
        logger.debug(format_message(msg))

    config = task.config.model_copy(deep=True)
    tool_choice = config.tool_choice
    if tool_choice != "required":
        config.tool_choice = "required"
        logger.debug(
            f"Setting tool choice to 'required' for feedback generation (original: '{tool_choice}')"
        )

    # dont catch errors here, if feedback gen fails
    # we want to know and handle it outside for potential retries
    response = model(messages, functions(), config)
    logger.debug(format_response(response))

    # model failing to follow protocol is a model issue,, no retry
    assert len(response.tool_calls) == 1, "No tool call found"  # type: ignore
    tool_call = response.tool_calls[0]  # type: ignore
    assert tool_call.name == "give_feedback", "Feedback function not called"
    return GiveFeedbackArgs.model_validate(tool_call.args).model_dump()
