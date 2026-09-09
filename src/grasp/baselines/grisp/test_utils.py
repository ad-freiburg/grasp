import json
import os

from grasp.baselines.grisp.utils import (
    find_best_checkpoint,
    load_sparql_parser,
    normalize_skeleton_parse,
    skeleton_wording_key,
)
from grasp.sparql.utils import find_all, parse_to_string

SPARQL_PARSER = load_sparql_parser()


def normalize(skeleton: str, blank_placeholders: bool = True) -> str:
    normalized = normalize_skeleton_parse(
        SPARQL_PARSER.parse(skeleton),
        {},
        blank_placeholders,
    )
    return parse_to_string(normalized)


class TestNormalizeSkeletonParse:
    def test_variables_renamed_by_first_appearance(self):
        assert normalize("SELECT ?x WHERE { ?x <iri>a</iri> ?y }") == normalize(
            "SELECT ?s WHERE { ?s <iri>a</iri> ?o }"
        )

    def test_repeated_variable_keeps_one_name(self):
        normalized = normalize("SELECT ?x WHERE { ?x <iri>a</iri> ?x }")
        assert normalized.count("?var0") == 3
        assert "?var1" not in normalized

    def test_placeholders_blanked(self):
        normalized = normalize("SELECT ?x WHERE { ?x <iri>born in</iri> ?y }")
        assert "<IRI>" in normalized
        assert "born in" not in normalized

    def test_placeholders_kept_when_not_blanked(self):
        normalized = normalize(
            "SELECT ?x WHERE { ?x <iri>born in</iri> ?y }",
            blank_placeholders=False,
        )
        assert "born in" in normalized

    def test_original_parse_not_modified(self):
        parse = SPARQL_PARSER.parse("SELECT ?x WHERE { ?x <iri>a</iri> ?y }")
        normalize_skeleton_parse(parse, {})
        variables = [var["value"] for var in find_all(parse, "VAR1")]
        assert variables == ["?x", "?x", "?y"]


class TestSkeletonWordingKey:
    def wording_key(self, skeleton: str) -> str:
        return skeleton_wording_key(SPARQL_PARSER.parse(skeleton))

    def test_same_key_for_different_wording(self):
        assert self.wording_key(
            "SELECT ?x WHERE { ?x <iri>born in</iri> ?y }"
        ) == self.wording_key("SELECT ?s WHERE { ?s <iri>place of birth</iri> ?o }")

    def test_different_key_for_reordered_triples(self):
        # merging lines placeholders up by document order, so the order of
        # the triples has to match as well
        assert self.wording_key(
            "SELECT ?x WHERE { ?x <iri>a</iri> ?y . ?y <iri>b</iri> <iri>c</iri> }"
        ) != self.wording_key(
            "SELECT ?x WHERE { ?y <iri>b</iri> <iri>c</iri> . ?x <iri>a</iri> ?y }"
        )

    def test_different_key_for_extra_placeholder(self):
        assert self.wording_key(
            "SELECT ?x WHERE { ?x <iri>a</iri> ?y }"
        ) != self.wording_key("SELECT ?x WHERE { ?x <iri>a</iri> <iri>b</iri> }")


class TestFindBestCheckpoint:
    def write(self, directory: str, step: int, metrics: dict) -> None:
        checkpoint = os.path.join(directory, f"checkpoint-{step}")
        os.makedirs(checkpoint)
        state = {"global_step": step, "log_history": [{"step": step, **metrics}]}
        with open(os.path.join(checkpoint, "trainer_state.json"), "w") as f:
            json.dump(state, f)

    def test_selects_the_lowest_balanced_loss(self, tmp_path):
        directory = str(tmp_path)
        self.write(directory, 10, {"eval_balanced_loss": 0.4, "eval_loss": 0.1})
        self.write(directory, 20, {"eval_balanced_loss": 0.2, "eval_loss": 0.9})
        assert find_best_checkpoint(directory) == os.path.join(
            directory, "checkpoint-20"
        )

    def test_falls_back_to_eval_loss_for_older_runs(self, tmp_path):
        directory = str(tmp_path)
        self.write(directory, 10, {"eval_loss": 0.4})
        self.write(directory, 20, {"eval_loss": 0.2})
        assert find_best_checkpoint(directory) == os.path.join(
            directory, "checkpoint-20"
        )

    def test_entries_from_other_steps_are_ignored(self, tmp_path):
        directory = str(tmp_path)
        checkpoint = os.path.join(directory, "checkpoint-20")
        os.makedirs(checkpoint)
        state = {
            "global_step": 20,
            "log_history": [
                {"step": 10, "eval_balanced_loss": 0.9},
                {"step": 20, "loss": 0.5},
                {"step": 20, "eval_balanced_loss": 0.2},
            ],
        }
        with open(os.path.join(checkpoint, "trainer_state.json"), "w") as f:
            json.dump(state, f)
        assert find_best_checkpoint(directory) == checkpoint
