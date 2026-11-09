import time
import uuid

import pytest

import grasp.core as core
from grasp.configs import GraspConfig
from grasp.core import Limits, format_duration, generate
from grasp.model import Model, Response, ToolCall
from grasp.tasks.base import GraspTask

TASK_FNS = {"work", "submit"}


def fn(name: str) -> dict:
    return {
        "name": name,
        "description": f"Call {name}.",
        "parameters": {"type": "object", "properties": {}},
    }


class DummyTask(GraspTask):
    name = "dummy"

    def __init__(self, *args, restrict: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.restrict = restrict
        self.submitted = False

    def system_information(self) -> str:
        return "Dummy task."

    def rules(self) -> list[str]:
        return []

    def function_definitions(self) -> list[dict]:
        return [fn("work"), fn("submit")]

    def call_function(self, fn_name, fn_args, known, example_indices) -> str:
        if fn_name == "submit":
            self.submitted = True
        return "ok"

    def done(self, fn_name: str) -> bool:
        return fn_name == "submit"

    def output(self, messages) -> dict | None:
        return {"submitted": self.submitted}

    def limit_message(self) -> str:
        return "Dummy limit message."

    def limit_functions(self) -> list[str] | None:
        return ["submit"] if self.restrict else None


class FakeModel(Model):
    # calls work whenever offered, so it never finishes by itself
    def __init__(self, config, delay: float = 0.0):
        super().__init__(config)
        self.delay = delay
        self.offered: list[list[str]] = []

    def call(self, messages, fns, config=None) -> Response:
        time.sleep(self.delay)
        names = [f["name"] for f in fns if f["name"] in TASK_FNS]
        self.offered.append(names)
        name = "work" if "work" in names else names[0]
        # distinct arguments, otherwise the loop detection kicks in
        return Response(
            id=uuid.uuid4().hex,
            tool_calls=[
                ToolCall(
                    id=uuid.uuid4().hex,
                    name=name,
                    args={"step": len(self.offered)},
                )
            ],
        )


def run(restrict: bool = True, delay: float = 0.0, **config_kwargs):
    config = GraspConfig(**config_kwargs)
    task = DummyTask([], config, restrict=restrict)
    model = FakeModel(config, delay)
    original = core.get_task
    core.get_task = lambda *args, **kwargs: task
    try:
        events = list(
            generate("dummy", "go", config, [], custom_model=model, yield_output=True)
        )
    finally:
        core.get_task = original
    return model, events


def test_limits():
    limits = Limits(3, None)
    assert not limits.reached
    limits.steps = 3
    assert limits.reached
    assert limits.format() == "Used 3 of 3 steps (0 left)."

    limits = Limits(10, 5.0, start=time.monotonic() - 10)
    assert limits.reached
    assert limits.format() == "Used 0 of 10 steps (10 left), 5s of 5s (0s left)."

    limits = Limits(10, 600.0, start=time.monotonic() - 125)
    assert (
        limits.format() == "Used 0 of 10 steps (10 left), 2m 5s of 10m (7m 55s left)."
    )


def test_format_duration():
    assert format_duration(0) == "0s"
    assert format_duration(59) == "59s"
    assert format_duration(60) == "1m"
    assert format_duration(580) == "9m 40s"


def test_step_limit_restricts_last_step():
    model, events = run(max_steps=3)
    # three regular steps plus one last step with only the limit functions
    assert model.offered == [["work", "submit"]] * 3 + [["submit"]]
    limit_events = [e for e in events if e["type"] == "limit"]
    assert [e["content"] for e in limit_events] == ["Dummy limit message."]
    output = events[-1]
    assert output["output"] == {"submitted": True}
    assert output["error"] is None


def test_step_limit_without_restriction():
    model, events = run(restrict=False, max_steps=2)
    # the model keeps working in its last step, so the run ends with an error
    assert model.offered == [["work", "submit"]] * 3
    output = events[-1]
    assert output["output"] == {"submitted": False}
    assert output["error"]["reason"] == "limit"


def test_time_limit_restricts_last_step():
    model, events = run(delay=0.2, max_steps=100, max_time=0.3)
    assert len(model.offered) < 100
    assert model.offered[-1] == ["submit"]
    assert all(names == ["work", "submit"] for names in model.offered[:-1])
    assert events[-1]["output"] == {"submitted": True}


@pytest.mark.parametrize("show", [False, True])
def test_show_limits(show: bool):
    _, events = run(max_steps=2, show_limits=show)
    output = events[-1]
    results = [
        tool_call["result"]
        for message in output["messages"]
        if message["role"] == "assistant"
        for tool_call in message["content"]["tool_calls"]
    ]
    assert len(results) == 3
    assert all(("Used " in result) == show for result in results)
    if show:
        assert results[0].endswith("Used 1 of 2 steps (1 left).")
        assert results[-1].endswith("Used 2 of 2 steps (0 left).")
