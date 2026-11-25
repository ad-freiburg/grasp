from typing import Literal

from pydantic import BaseModel, Field

from grasp.model import FunctionDefinition, Message, Response
from grasp.tasks.base import GraspTask


class GiveFeedbackArgs(BaseModel):
    status: Literal["done", "refine", "retry"] = Field(description="The feedback type")
    feedback: str = Field(description="The feedback message")


class Feedback(GiveFeedbackArgs):
    # messages of the feedback task that gave the feedback
    trace: list[Message] = []


def format_feedback(feedback: Feedback) -> str:
    return f"Feedback (status={feedback.status}):\n{feedback.feedback}"


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


class FeedbackTask(GraspTask):
    # feedback on another task's output, with KG functions for more than one step

    @property
    def kg_functions(self) -> bool:
        return self.config.max_steps > 1

    def function_definitions(self) -> list[FunctionDefinition]:
        return functions()

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        if fn_name != "give_feedback":
            raise ValueError(f"Unknown function {fn_name}")

        GiveFeedbackArgs.model_validate(fn_args)
        return "Stopping"

    def done(self, fn_name: str) -> bool:
        # also with invalid arguments, the run then ends without feedback
        return fn_name == "give_feedback"

    def limit_message(self) -> str:
        return (
            "The step or time limit is reached. You have one last step, "
            "give your feedback now."
        )

    def limit_functions(self) -> list[str] | None:
        return ["give_feedback"]

    def output(self, messages: list[Message]) -> dict | None:
        # the feedback given, if its arguments were valid
        for message in reversed(messages):
            if not isinstance(message.content, Response):
                continue

            for tool_call in reversed(message.content.tool_calls):
                if tool_call.name == "give_feedback" and tool_call.error is None:
                    return GiveFeedbackArgs.model_validate(tool_call.args).model_dump()

        return None
