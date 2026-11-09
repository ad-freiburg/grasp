import json
import os

import pytest

import grasp.notes as notes_module
from grasp.configs import (
    NotesFromExplorationConfig,
    NotesFromOutputsConfig,
    NotesGenerateQuestionsConfig,
)
from grasp.utils import is_model_error


def error_output(reason: str) -> dict:
    return {
        "type": "output",
        "output": None,
        "error": {"content": f"{reason} error", "reason": reason},
        "messages": [],
    }


def test_is_model_error():
    assert is_model_error(error_output("api"))
    assert is_model_error(error_output("timeout"))
    assert is_model_error(error_output("empty"))
    # limits and loops come from the agent, not from the model API
    assert not is_model_error(error_output("limit"))
    assert not is_model_error(error_output("loop"))
    assert not is_model_error({"output": {}, "error": None})


def run_exploration(monkeypatch, tmp_path, outputs: list[dict]) -> str:
    monkeypatch.setattr(notes_module, "setup", lambda config: ([], {}))
    monkeypatch.setattr(notes_module, "load_example_indices", lambda *args: {})
    monkeypatch.setattr(notes_module, "load_notes", lambda config: ([], {}))
    monkeypatch.setattr(notes_module, "generate", lambda *args, **kwargs: None)
    rounds = iter(outputs)
    monkeypatch.setattr(notes_module, "consume_generator", lambda gen: next(rounds))

    out_dir = str(tmp_path / "exploration")
    config = NotesFromExplorationConfig(mode="functional", num_rounds=len(outputs))
    notes_module.take_notes_from_exploration(config, out_dir)
    return out_dir


def test_exploration_stops_on_model_error(monkeypatch, tmp_path):
    ok = {"type": "output", "output": {}, "error": None, "messages": []}
    with pytest.raises(RuntimeError, match="round 1 failed with a model error"):
        run_exploration(monkeypatch, tmp_path, [ok, error_output("api"), ok])

    out_dir = str(tmp_path / "exploration")
    # the failed round keeps its trace but writes no notes, the third never runs
    assert os.path.exists(os.path.join(out_dir, "output.exploration.round_1.json"))
    assert not os.path.exists(os.path.join(out_dir, "notes.exploration.round_1.json"))
    assert not os.path.exists(os.path.join(out_dir, "output.exploration.round_2.json"))


def test_exploration_continues_after_limit(monkeypatch, tmp_path):
    ok = {"type": "output", "output": {}, "error": None, "messages": []}
    out_dir = run_exploration(monkeypatch, tmp_path, [error_output("limit"), ok])
    assert os.path.exists(os.path.join(out_dir, "notes.exploration.round_1.json"))


def patch_rounds(monkeypatch, outputs: list[dict]) -> None:
    monkeypatch.setattr(notes_module, "setup", lambda config: ([], {}))
    monkeypatch.setattr(notes_module, "load_notes", lambda config: ([], {}))
    monkeypatch.setattr(notes_module, "generate", lambda *args, **kwargs: None)
    rounds = iter(outputs)
    monkeypatch.setattr(notes_module, "consume_generator", lambda gen: next(rounds))


def test_question_generation_stops_on_model_error(monkeypatch, tmp_path):
    ok = {"type": "output", "output": {}, "error": None, "messages": []}
    patch_rounds(monkeypatch, [error_output("timeout"), ok])
    config = NotesGenerateQuestionsConfig(num_rounds=2)
    with pytest.raises(RuntimeError, match="Question generation round 0 failed"):
        notes_module.generate_questions(config, str(tmp_path / "questions"))


def test_notes_from_outputs_stops_on_model_error(monkeypatch, tmp_path):
    outputs_file = tmp_path / "outputs.jsonl"
    outputs_file.write_text(json.dumps({"task": "sparql-qa", "id": "1"}) + "\n")
    patch_rounds(monkeypatch, [error_output("empty")])
    config = NotesFromOutputsConfig(outputs=[str(outputs_file)], num_rounds=1, seed=22)
    out_dir = tmp_path / "notes"
    with pytest.raises(RuntimeError, match="Note-taking round 0 failed"):
        notes_module.take_notes_from_outputs("sparql-qa", config, str(out_dir))
    assert (out_dir / "output.sparql-qa.round_0.json").exists()
    assert not (out_dir / "notes.sparql-qa.round_0.json").exists()


def ok_output(**output) -> dict:
    return {"type": "output", "output": output, "error": None, "messages": []}


def patch_stateful_rounds(monkeypatch, step) -> list:
    # generate hands the state through, consume_generator lets `step` change it
    # like the agent's function calls would; `step` returns the round output
    states = []
    monkeypatch.setattr(notes_module, "setup", lambda config: ([], {}))
    monkeypatch.setattr(notes_module, "load_example_indices", lambda *args: {})
    monkeypatch.setattr(notes_module, "load_notes", lambda config: ([], {}))
    monkeypatch.setattr(notes_module, "generate", lambda task, state, *a, **k: state)

    def consume(state):
        states.append(state)
        return step(len(states) - 1, state)

    monkeypatch.setattr(notes_module, "consume_generator", consume)
    return states


def test_functional_exploration_resumes(monkeypatch, tmp_path):
    out_dir = str(tmp_path / "functional")

    def step(i, state):
        if i == 1:
            return error_output("api")
        state.notes.append(f"note {i}")
        return ok_output()

    patch_stateful_rounds(monkeypatch, step)
    config = NotesFromExplorationConfig(mode="functional", num_rounds=3)
    with pytest.raises(RuntimeError, match="round 1 failed"):
        notes_module.take_notes_from_exploration(config, out_dir)

    states = patch_stateful_rounds(
        monkeypatch, lambda i, state: state.notes.append(f"resumed {i}") or ok_output()
    )
    notes_module.take_notes_from_exploration(config, out_dir, start_round=1)
    assert states[0].notes[0] == "note 0"
    with open(os.path.join(out_dir, "notes.exploration.round_2.json")) as f:
        assert json.load(f) == ["note 0", "resumed 0", "resumed 1"]
    assert os.path.exists(os.path.join(out_dir, "config.yaml"))
    assert os.path.exists(os.path.join(out_dir, "config.start_round_1.yaml"))


def test_structural_exploration_resumes_with_explored_seeds(monkeypatch, tmp_path):
    out_dir = str(tmp_path / "structural")

    def step(i, state):
        state.kg_notes.setdefault("wikidata", []).append(f"note {i}")
        state.explored.setdefault("wikidata", []).append(f"seed {i}")
        return ok_output(explored=state.explored)

    patch_stateful_rounds(monkeypatch, step)
    notes_module.take_notes_from_exploration(
        NotesFromExplorationConfig(mode="structural", num_rounds=2), out_dir
    )

    states = patch_stateful_rounds(monkeypatch, step)
    notes_module.take_notes_from_exploration(
        NotesFromExplorationConfig(mode="structural", num_rounds=3),
        out_dir,
        start_round=2,
    )
    assert states[0].explored == {"wikidata": ["seed 0", "seed 1", "seed 0"]}
    with open(os.path.join(out_dir, "notes.exploration.wikidata.round_2.json")) as f:
        assert json.load(f) == ["note 0", "note 1", "note 0"]


def test_question_generation_resumes(monkeypatch, tmp_path):
    out_dir = str(tmp_path / "questions")

    def step(i, state):
        state.questions.setdefault("wikidata", []).append(f"question {i}")
        return ok_output()

    patch_stateful_rounds(monkeypatch, step)
    notes_module.generate_questions(NotesGenerateQuestionsConfig(num_rounds=1), out_dir)
    # samples of a KG that is not configured are ignored
    with open(os.path.join(out_dir, "samples.other.round_0.jsonl"), "w") as f:
        f.write(json.dumps({"question": "other"}) + "\n")
    states = patch_stateful_rounds(monkeypatch, step)
    notes_module.generate_questions(
        NotesGenerateQuestionsConfig(num_rounds=2), out_dir, start_round=1
    )
    assert states[0].questions == {"wikidata": ["question 0", "question 0"]}


def test_notes_from_outputs_resumes(monkeypatch, tmp_path):
    outputs_file = tmp_path / "outputs.jsonl"
    outputs_file.write_text(json.dumps({"task": "sparql-qa", "id": "1"}) + "\n")
    out_dir = str(tmp_path / "notes")

    def step(i, state):
        state.notes.append(f"note {i}")
        state.kg_notes.setdefault("wikidata", []).append(f"kg note {i}")
        return ok_output()

    config = NotesFromOutputsConfig(outputs=[str(outputs_file)], num_rounds=1, seed=22)
    patch_stateful_rounds(monkeypatch, step)
    notes_module.take_notes_from_outputs("sparql-qa", config, out_dir)

    config = NotesFromOutputsConfig(outputs=[str(outputs_file)], num_rounds=2, seed=22)
    states = patch_stateful_rounds(monkeypatch, step)
    notes_module.take_notes_from_outputs("sparql-qa", config, out_dir, start_round=1)
    assert states[0].notes[0] == "note 0"
    assert states[0].kg_notes == {"wikidata": ["kg note 0", "kg note 0"]}


def test_resume_errors(monkeypatch, tmp_path):
    patch_stateful_rounds(monkeypatch, lambda i, state: ok_output())
    config = NotesFromExplorationConfig(mode="functional", num_rounds=2)
    missing = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        notes_module.take_notes_from_exploration(config, missing, start_round=1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        notes_module.take_notes_from_exploration(config, missing, start_round=2)

    # the previous round never finished
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="previous round is missing"):
        notes_module.take_notes_from_exploration(config, str(empty), start_round=1)
