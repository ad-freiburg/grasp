"""Step budget in grasp.core.generate, without an LLM: a fake model records the
functions and tool choice of every call, a dummy task offers a 'work' and a
'submit' function."""
import uuid

import pytest

import grasp.core as core
from grasp.configs import GraspConfig
from grasp.core import StepBudget, generate
from grasp.model import Model, Response, ToolCall
from grasp.tasks.base import GraspTask

WORK = {
    "name": "work",
    "description": "Do some work.",
    "parameters": {"type": "object", "properties": {}},
}
SUBMIT = {
    "name": "submit",
    "description": "Submit the result.",
    "parameters": {"type": "object", "properties": {}},
}
TASK_FNS = {"work", "submit"}


class DummyTask(GraspTask):
    name = "dummy"

    def __init__(self, *args, final: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.final = final
        self.submitted = False

    def system_information(self) -> str:
        return "Dummy task."

    def rules(self) -> list[str]:
        return []

    def function_definitions(self) -> list[dict]:
        return [WORK, SUBMIT]

    def call_function(self, fn_name, fn_args, known, example_indices) -> str:
        if fn_name == "submit":
            self.submitted = True
        return "ok"

    def done(self, fn_name: str) -> bool:
        return fn_name == "submit"

    def output(self, messages) -> dict | None:
        return {"submitted": self.submitted}

    def final_step_functions(self) -> list[dict] | None:
        return [SUBMIT] if self.final else None


class FakeModel(Model):
    """Always calls 'work' if offered (a model that never finishes by itself),
    otherwise the first offered function; records the offered task functions
    and the tool choice of every call."""

    def __init__(self, config):
        super().__init__(config)
        self.calls: list[tuple[list[str], str]] = []

    def call(self, messages, fns, config=None) -> Response:
        config = config or self.config
        names = [fn["name"] for fn in fns]
        self.calls.append(([n for n in names if n in TASK_FNS], config.tool_choice))
        name = "work" if "work" in names else names[0]
        # distinct arguments, otherwise the loop detection would intervene
        return Response(
            id=uuid.uuid4().hex,
            tool_calls=[
                ToolCall(id=uuid.uuid4().hex, name=name, args={"step": len(self.calls)})
            ],
        )


def run(max_steps: int, fraction: float | None, final: bool = True):
    config = GraspConfig(max_steps=max_steps, budget_warning_fraction=fraction)
    task = DummyTask([], config, final=final)
    model = FakeModel(config)
    original = core.get_task
    core.get_task = lambda *args, **kwargs: task
    try:
        events = list(
            generate("dummy", "go", config, [], custom_model=model, yield_output=True)
        )
    finally:
        core.get_task = original
    budget_events = [e for e in events if e["type"] == "budget"]
    return model, events[-1], budget_events


def test_step_budget_arithmetic():
    budget = StepBudget(8, 0.3)
    states = []
    for _ in range(8):
        states.append((budget.left, budget.warn, budget.final))
        budget.used += 1
    # ceil(0.3 * 8) = 3: warnings before the last three steps, the last is final
    assert [s for s in states if s[1]] == [(3, True, False), (2, True, False), (1, True, True)]
    assert StepBudget(8, None).warn is False


def test_warnings_and_forced_final_step():
    model, output, budget_events = run(max_steps=8, fraction=0.3)
    assert len(model.calls) == 8  # max_steps counts model calls
    assert [e["steps_left"] for e in budget_events] == [3, 2, 1]
    assert [e["final"] for e in budget_events] == [False, False, True]
    # steps 1-7: all functions, tool choice of the config; step 8: only submit, required
    assert all(fns == ["work", "submit"] and tc == "auto" for fns, tc in model.calls[:7])
    assert model.calls[7] == (["submit"], "required")
    assert output["output"] == {"submitted": True}


def test_budget_messages_in_history():
    _, output, _ = run(max_steps=5, fraction=0.3)
    users = [m for m in output["messages"] if m["role"] == "user"]
    # input plus ceil(0.3 * 5) = 2 budget messages
    assert len(users) == 3
    assert "last step" in users[-1]["content"]


def test_final_step_forced_without_warnings():
    model, output, budget_events = run(max_steps=4, fraction=None)
    assert [e["final"] for e in budget_events] == [True]  # only the final step message
    assert model.calls[-1] == (["submit"], "required")
    assert output["output"] == {"submitted": True}


def test_unchanged_without_fraction_and_final_functions():
    model, output, budget_events = run(max_steps=4, fraction=None, final=False)
    assert budget_events == []
    assert len(model.calls) == 4
    assert all(fns == ["work", "submit"] and tc == "auto" for fns, tc in model.calls)
    assert output["output"] == {"submitted": False}  # ran out of steps, as before


@pytest.mark.parametrize("max_steps", [1, 2])
def test_tiny_budgets(max_steps):
    model, output, _ = run(max_steps=max_steps, fraction=0.3)
    assert model.calls[-1] == (["submit"], "required")
    assert output["output"] == {"submitted": True}
