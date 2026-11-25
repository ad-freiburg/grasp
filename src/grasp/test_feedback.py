import uuid
from typing import Any, Callable

import pytest

import grasp.core as core
from grasp.configs import GraspConfig
from grasp.core import generate, system_instructions
from grasp.model import Message, Model, Response, ToolCall
from grasp.tasks import get_task
from grasp.tasks.feedback import FeedbackTask
from grasp.test_core import DummyTask, fn


class ReviewedTask(DummyTask):
    feedback_task = "dummy-feedback"


class DummyFeedbackTask(FeedbackTask):
    name = "dummy-feedback"

    def system_information(self) -> str:
        return "Dummy feedback."

    def rules(self) -> list[str]:
        return []

    def setup(self, input: dict) -> str:
        return f"Feedback on {input['output']} for {input['inputs']}."


class ScriptedModel(Model):
    # answers each call with the tool call or text the script returns
    def __init__(self, config, script: Callable[[list[Message], list[str]], Any]):
        super().__init__(config)
        self.script = script
        self.calls: list[tuple[list[Message], list[str]]] = []

    def call(self, messages, fns, config=None) -> Response:
        names = [f["name"] for f in fns]
        self.calls.append((list(messages), names))
        step = self.script(messages, names)
        if isinstance(step, str):
            return Response(id=uuid.uuid4().hex, message=step)

        name, args = step
        return Response(
            id=uuid.uuid4().hex,
            tool_calls=[ToolCall(id=uuid.uuid4().hex, name=name, args=args)],
        )


def run(script, **config_kwargs):
    config = GraspConfig(feedback=True, **config_kwargs)
    model = ScriptedModel(config, script)
    tasks = {"dummy": ReviewedTask, "dummy-feedback": DummyFeedbackTask}
    configs = {}

    def get_task(name, managers, config, *args, **kwargs):
        configs[name] = config
        return tasks[name](managers, config)

    originals = core.get_task, core.kg_functions
    core.get_task = get_task
    # stands in for the knowledge graph functions
    core.kg_functions = lambda *args: [fn("execute")]
    try:
        events = list(
            generate("dummy", "go", config, [], custom_model=model, yield_output=True)
        )
    finally:
        core.get_task, core.kg_functions = originals
    return model, events, configs


def is_feedback(messages: list[Message]) -> bool:
    system = messages[0].content
    return isinstance(system, str) and "Dummy feedback." in system


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
    # as sent to the model
    assert message.model_dump(exclude={"trace"}) == {"role": "user", "content": "hi"}


def test_single_step_feedback():
    def script(messages, names):
        if is_feedback(messages):
            return "give_feedback", {"status": "done", "feedback": "Fine."}
        return "submit", {}

    model, events, configs = run(script)
    # one call for the task, one for the feedback
    assert len(model.calls) == 2
    messages, names = model.calls[1]
    # no knowledge graph functions
    assert names == ["give_feedback"]
    assert configs["dummy-feedback"].max_steps == 1
    assert not configs["dummy-feedback"].feedback
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[1].content == "Feedback on {'submitted': True} for ['go']."

    feedback = [e for e in events if e["type"] == "feedback"]
    assert feedback == [{"type": "feedback", "status": "done", "feedback": "Fine."}]

    output = events[-1]
    assert output["task"] == "dummy"
    assert output["error"] is None
    [message] = feedback_messages(output)
    assert message["content"] == "Feedback (status=done):\nFine."
    assert [m["role"] for m in message["trace"]] == ["system", "user", "assistant"]


def test_multi_step_feedback():
    statuses = ["refine", "done"]

    def script(messages, names):
        if not is_feedback(messages):
            assert names == ["execute", "work", "submit"]
            return "submit", {"round": len(messages)}

        assert names == ["execute", "give_feedback"]
        if isinstance(messages[-1].content, str):
            # first step of a feedback run, its error does not end the run
            return "execute", {}
        status = statuses.pop(0)
        return "give_feedback", {"status": status, "feedback": f"It is {status}."}

    model, events, configs = run(script, feedback_max_steps=5)
    assert configs["dummy-feedback"].max_steps == 5
    # two rounds of submit, execute and feedback
    assert len(model.calls) == 6

    # only the feedback itself reaches the events
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
        calls = [m["content"]["tool_calls"][0]["name"] for m in trace[2:]]
        assert calls == ["execute", "give_feedback"]

    # the task sees the feedback, but not the run behind it
    task_messages = model.calls[3][0]
    assert task_messages[-1].content == first["content"]
    assert not is_feedback(task_messages)


def test_feedback_at_limit():
    def script(messages, names):
        if not is_feedback(messages):
            return "submit", {}
        if "execute" in names:
            return "execute", {"step": len(messages)}
        return "give_feedback", {"status": "done", "feedback": "Out of steps."}

    model, events, _ = run(script, feedback_max_steps=2, show_limits=True)
    # the last step of the feedback run only offers give_feedback
    assert [names for _, names in model.calls[1:]] == [
        ["execute", "give_feedback"],
        ["execute", "give_feedback"],
        ["give_feedback"],
    ]
    output = events[-1]
    assert output["error"] is None
    [message] = feedback_messages(output)
    assert message["content"] == "Feedback (status=done):\nOut of steps."
    # the feedback run shows its own limits
    results = [
        m["content"]["tool_calls"][0]["result"]["formatted"]
        for m in message["trace"]
        if m["role"] == "assistant"
    ]
    assert [r.splitlines()[-1] for r in results] == [
        "Used 1 of 2 steps (1 left).",
        "Used 2 of 2 steps (0 left).",
        "Used 2 of 2 steps (0 left).",
    ]


@pytest.mark.parametrize("max_steps", [1, 5])
def test_invalid_feedback_ends_run(max_steps: int):
    attempts = []

    def script(messages, names):
        if not is_feedback(messages):
            return "submit", {}
        attempts.append(names)
        return "give_feedback", {"status": "good", "feedback": "Fine."}

    _, events, _ = run(script, feedback_max_steps=max_steps)
    # no further step for the feedback task
    assert len(attempts) == 1
    assert not [e for e in events if e["type"] == "feedback"]
    output = events[-1]
    assert output["output"] == {"submitted": True}
    assert output["error"]["reason"] == "feedback"
    assert "no valid feedback given" in output["error"]["content"]
    assert not feedback_messages(output)


@pytest.mark.parametrize("max_steps", [1, 5])
def test_no_feedback_given(max_steps: int):
    def script(messages, names):
        if not is_feedback(messages):
            return "submit", {}
        return "Looks good to me."

    _, events, _ = run(script, feedback_max_steps=max_steps)
    assert not [e for e in events if e["type"] == "feedback"]
    output = events[-1]
    # the output of the task is kept
    assert output["output"] == {"submitted": True}
    assert output["error"]["reason"] == "feedback"
    assert "no valid feedback given" in output["error"]["content"]
    assert not feedback_messages(output)


@pytest.mark.parametrize("task", ["sparql-qa", "cea", "entity-linking"])
def test_feedback_tasks(task: str):
    config = GraspConfig(max_steps=1)
    name = f"{task}-feedback"
    assert get_task(task, [], config).feedback_task == name
    feedback_task = get_task(name, [], config)
    assert isinstance(feedback_task, FeedbackTask)
    assert not feedback_task.kg_functions
    assert [f.name for f in feedback_task.function_definitions()] == ["give_feedback"]
    # the rules of the task are part of the system message
    system = system_instructions(feedback_task, [], {}, [])
    assert "providing feedback" in system
    assert all(rule in system for rule in get_task(task, [], config).rules())
    output = {"type": "answer", "formatted": "the result"}
    inputs = feedback_task.setup({"inputs": ["in"], "output": output})
    assert "in" in inputs and "the result" in inputs
