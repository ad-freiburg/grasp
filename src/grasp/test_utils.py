import pytest

from grasp.utils import is_invalid_output


def output(reason: str | None, out: dict | None = None) -> dict:
    error = None if reason is None else {"content": "...", "reason": reason}
    return {"output": out, "error": error, "messages": []}


@pytest.mark.parametrize("reason", ["timeout", "api", "empty", "feedback"])
def test_infrastructure_errors_always_invalid(reason: str) -> None:
    assert is_invalid_output(output(reason))
    assert is_invalid_output(output(reason), agent_error_invalid=False)


@pytest.mark.parametrize("reason", ["limit", "loop"])
def test_agent_errors(reason: str) -> None:
    assert is_invalid_output(output(reason))
    assert not is_invalid_output(output(reason), agent_error_invalid=False)
    # none outputs still count when asked for
    assert is_invalid_output(
        output(reason),
        none_output_invalid=True,
        agent_error_invalid=False,
    )


def test_valid_output() -> None:
    assert not is_invalid_output(output(None, {"sparql": "SELECT"}))
    assert is_invalid_output(None)
