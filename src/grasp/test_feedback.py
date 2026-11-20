import uuid
from typing import Any, Callable

import grasp.core as core
from grasp.configs import GraspConfig
from grasp.core import generate
from grasp.model import FunctionDefinition, Message, Model, Response, ToolCall
from grasp.tasks.base import FeedbackTask
from grasp.tasks.feedback import ReviewTask
from grasp.test_core import DummyTask, fn


class SingleStepTask(DummyTask, FeedbackTask):
    def feedback_system_message(self, kg_notes, notes) -> str:
        return "Feedback system."

    def feedback_instructions(self, inputs, output) -> str:
        return f"Feedback on {output} for {inputs}."


class ReviewedTask(DummyTask):
    def reviewer(self, inputs, output) -> tuple[str, Any] | None:
        return "dummy-review", {"inputs": inputs, "output": output}


class DummyReviewer(ReviewTask):
    name = "dummy-review"

    def system_information(self) -> str:
        return "Dummy reviewer."

    def rules(self) -> list[str]:
        return []

    def function_definitions(self) -> list[FunctionDefinition]:
        return [fn("check"), *super().function_definitions()]

    def call_function(self, fn_name, fn_args, known, example_indices) -> str:
        if fn_name == "check":
            return "checked"
        return super().call_function(fn_name, fn_args, known, example_indices)

    def setup(self, input: Any) -> str:
        return f"Review {input['output']} for {input['inputs']}."


class ScriptedModel(Model):
    # answers each call with the tool call or text the script returns
    def __init__(self, config, script: Callable[[list[Message], list[str]], Any]):
        super().__init__(config)
        self.script = script
        self.calls: list[tuple[list[Message], list[str], Any]] = []

    def call(self, messages, fns, config=None) -> Response:
        names = [f["name"] for f in fns]
        self.calls.append((list(messages), names, config))
        step = self.script(messages, names)
        if isinstance(step, str):
            return Response(id=uuid.uuid4().hex, message=step)

        name, args = step
        return Response(
            id=uuid.uuid4().hex,
            tool_calls=[ToolCall(id=uuid.uuid4().hex, name=name, args=args)],
        )


def run(task_cls, script, **config_kwargs):
    config = GraspConfig(feedback=True, **config_kwargs)
    model = ScriptedModel(config, script)
    tasks = {"dummy": task_cls, "dummy-review": DummyReviewer}
    original = core.get_task
    core.get_task = lambda name, managers, config, *args, **kwargs: tasks[name](
        managers,
        config,
    )
    try:
        events = list(
            generate("dummy", "go", config, [], custom_model=model, yield_output=True)
        )
    finally:
        core.get_task = original
    return model, events


def is_review(messages: list[Message]) -> bool:
    system = messages[0].content
    return isinstance(system, str) and "Dummy reviewer." in system


def feedback_messages(output: dict) -> list[dict]:
    return [m for m in output["messages"] if "trace" in m]


def test_message_dump():
    assert Message.user("hi").model_dump() == {"role": "user", "content": "hi"}
    message = Message(role="user", content="hi", trace=[Message.system("sys")])
    assert message.model_dump() == {
        "role": "user",
        "content": "hi",
        "trace": [{"role": "system", "content": "sys"}],
    }
    assert Message.model_validate(message.model_dump()) == message


def test_single_step_feedback():
    def script(messages, names):
        if names == ["give_feedback"]:
            return "give_feedback", {"status": "done", "feedback": "Fine."}
        return "submit", {}

    model, events = run(SingleStepTask, script)
    # one call for the task, one for the feedback
    assert len(model.calls) == 2
    messages, names, config = model.calls[1]
    assert names == ["give_feedback"]
    assert config.tool_choice == "required"
    assert [(m.role, m.content) for m in messages] == [
        ("system", "Feedback system."),
        ("user", "Feedback on {'submitted': True} for ['go']."),
    ]

    feedback = [e for e in events if e["type"] == "feedback"]
    assert feedback == [{"type": "feedback", "status": "done", "feedback": "Fine."}]

    output = events[-1]
    assert output["error"] is None
    [message] = feedback_messages(output)
    assert message["content"] == "Feedback (status=done):\nFine."
    # system, instructions and the response with the feedback
    assert [m["role"] for m in message["trace"]] == ["system", "user", "assistant"]


def test_agentic_feedback():
    statuses = ["refine", "done"]

    def script(messages, names):
        if not is_review(messages):
            return "submit", {"round": len(messages)}

        assert "give_feedback" in names and "submit" not in names
        if isinstance(messages[-1].content, str):
            # first step of a review
            return "check", {}
        status = statuses.pop(0)
        return "give_feedback", {"status": status, "feedback": f"It is {status}."}

    model, events = run(ReviewedTask, script)
    # two rounds of submit, check and feedback
    assert len(model.calls) == 6
    review_input = model.calls[1][0][1]
    assert review_input.content == "Review {'submitted': True} for ['go']."

    # only the feedback of the reviewer reaches the events
    assert [e["type"] for e in events] == [
        "input",
        "system",
        "tool",
        "feedback",
        "tool",
        "feedback",
        "output",
    ]
    feedback = [e for e in events if e["type"] == "feedback"]
    assert [e["status"] for e in feedback] == ["refine", "done"]
    assert all(set(e) == {"type", "status", "feedback"} for e in feedback)

    output = events[-1]
    assert output["task"] == "dummy"
    assert output["error"] is None
    first, second = feedback_messages(output)
    assert first["content"] == "Feedback (status=refine):\nIt is refine."
    assert second["content"] == "Feedback (status=done):\nIt is done."
    for message in (first, second):
        trace = message["trace"]
        assert [m["role"] for m in trace] == [
            "system",
            "user",
            "assistant",
            "assistant",
        ]
        assert "Dummy reviewer." in trace[0]["content"]
        calls = [m["content"]["tool_calls"][0]["name"] for m in trace[2:]]
        assert calls == ["check", "give_feedback"]

    # the task sees the feedback, but not the trace behind it
    task_messages = model.calls[3][0]
    assert task_messages[-1].content == first["content"]
    assert not is_review(task_messages)


def test_agentic_feedback_at_limit():
    def script(messages, names):
        if not is_review(messages):
            return "submit", {}
        if "check" in names:
            return "check", {"step": len(messages)}
        return "give_feedback", {"status": "done", "feedback": "Out of steps."}

    model, events = run(ReviewedTask, script, max_steps=2)
    # the last step of the reviewer only offers give_feedback
    assert [names[-2:] for _, names, _ in model.calls[1:]] == [
        ["check", "give_feedback"],
        ["check", "give_feedback"],
        ["give_feedback"],
    ]
    output = events[-1]
    assert output["error"] is None
    [message] = feedback_messages(output)
    assert message["content"] == "Feedback (status=done):\nOut of steps."


def test_agentic_feedback_without_verdict():
    def script(messages, names):
        if not is_review(messages):
            return "submit", {}
        return "Looks good to me."

    _, events = run(ReviewedTask, script)
    assert not [e for e in events if e["type"] == "feedback"]
    output = events[-1]
    # the output of the task is kept
    assert output["output"] == {"submitted": True}
    assert output["error"]["reason"] == "feedback"
    assert "no feedback given" in output["error"]["content"]
    assert not feedback_messages(output)
