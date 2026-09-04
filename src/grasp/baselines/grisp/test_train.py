import pytest
import torch

from grasp.baselines.grisp.data import IGNORE_INDEX
from grasp.baselines.grisp.train import (
    EVAL_SUM_KEYS,
    GRISPTrainer,
    eval_loss_metrics,
)


def sums(
    ntp_loss: float = 0.0,
    ntp_n: float = 0.0,
    rce_loss: float = 0.0,
    rce_n: float = 0.0,
) -> dict[str, float]:
    return {
        "ntp_loss": ntp_loss,
        "ntp_n": ntp_n,
        "rce_loss": rce_loss,
        "rce_n": rce_n,
    }


class TestEvalLossMetrics:
    def test_loss_is_normalized_over_the_whole_eval_set(self):
        metrics = eval_loss_metrics(
            sums(ntp_loss=200.0, ntp_n=100, rce_loss=8.0, rce_n=4)
        )
        assert metrics["loss"] == (200.0 + 8.0) / (100 + 4)

    def test_per_task_losses_use_their_own_denominators(self):
        metrics = eval_loss_metrics(
            sums(ntp_loss=200.0, ntp_n=100, rce_loss=8.0, rce_n=4)
        )
        assert metrics["ntp_loss"] == 2.0
        assert metrics["rce_loss"] == 2.0

    def test_balanced_loss_weights_both_tasks_equally(self):
        metrics = eval_loss_metrics(
            sums(ntp_loss=200.0, ntp_n=100, rce_loss=8.0, rce_n=2)
        )
        # 2.0 per ntp token and 4.0 per rce row
        assert metrics["balanced_loss"] == 3.0

    def test_token_weighting_lets_the_rce_rows_be_drowned_out(self):
        # the reason balanced_loss exists: one row per rce sample against
        # hundreds of tokens per ntp sample
        metrics = eval_loss_metrics(
            sums(ntp_loss=100.0, ntp_n=1000, rce_loss=50.0, rce_n=10)
        )
        assert metrics["loss"] < 0.15
        assert metrics["rce_loss"] == 5.0
        assert metrics["balanced_loss"] > 2.5

    def test_missing_task_is_left_out_rather_than_counted_as_zero(self):
        metrics = eval_loss_metrics(sums(ntp_loss=200.0, ntp_n=100))
        assert "rce_loss" not in metrics
        assert metrics["loss"] == 2.0
        assert metrics["balanced_loss"] == 2.0

    def test_rce_only_eval_set(self):
        metrics = eval_loss_metrics(sums(rce_loss=8.0, rce_n=4))
        assert "ntp_loss" not in metrics
        assert metrics["loss"] == 2.0
        assert metrics["balanced_loss"] == 2.0

    def test_empty_eval_set_reports_nothing(self):
        assert eval_loss_metrics(sums()) == {}


class FakeModel:
    # returns logits that put all mass on token 0, so every loss is a finite
    # constant we can predict without running a real model
    def __init__(self, vocab: int = 8) -> None:
        self.vocab = vocab

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> "FakeModel":
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab)
        logits[:, :, 0] = 1.0
        self.logits = logits
        return self


class FakeTrainer:
    # GRISPTrainer.compute_loss only reaches back into self for the eval sums,
    # so the accumulation can be exercised without building a real Trainer
    def __init__(self, collect: bool) -> None:
        self.eval_sums = (
            {key: torch.zeros((), dtype=torch.float64) for key in EVAL_SUM_KEYS}
            if collect
            else None
        )

    add_eval_sums = GRISPTrainer.add_eval_sums

    def loss(self, inputs: dict) -> float:
        loss = GRISPTrainer.compute_loss(self, FakeModel(), inputs)  # type: ignore
        return loss.item()  # type: ignore

    def collected(self) -> dict[str, float]:
        assert self.eval_sums is not None
        return {key: value.item() for key, value in self.eval_sums.items()}


def batch(n_ntp: int, n_rce: int, length: int = 6, labelled: int = 3) -> dict:
    total = n_ntp + n_rce
    labels = torch.full((total, length), IGNORE_INDEX)
    # ntp rows carry `labelled` supervised tokens each, rce rows carry none
    labels[:n_ntp, -labelled - 1 : -1] = 1
    answer_pos = torch.full((total,), -1, dtype=torch.long)
    answer_pos[n_ntp:] = length - 2
    is_rce = torch.zeros(total, dtype=torch.bool)
    is_rce[n_ntp:] = True
    return {
        "input_ids": torch.ones(total, length, dtype=torch.long),
        "labels": labels,
        "answer_pos": answer_pos,
        "is_rce": is_rce,
        "option_token_ids": torch.zeros(total, 2, dtype=torch.long),
        "option_mask": torch.ones(total, 2, dtype=torch.bool),
        "target_dist": torch.tensor([[1.0, 0.0]] * total),
    }


class TestEvalSumAccumulation:
    def test_nothing_is_collected_while_training(self):
        trainer = FakeTrainer(collect=False)
        trainer.loss(batch(2, 2))
        assert trainer.eval_sums is None

    def test_denominators_count_ntp_tokens_and_rce_rows(self):
        trainer = FakeTrainer(collect=True)
        trainer.loss(batch(2, 3, labelled=4))
        assert trainer.collected()["ntp_n"] == 2 * 4
        assert trainer.collected()["rce_n"] == 3

    def test_sums_accumulate_across_batches(self):
        trainer = FakeTrainer(collect=True)
        trainer.loss(batch(2, 0))
        trainer.loss(batch(0, 5))
        assert trainer.collected()["ntp_n"] == 2 * 3
        assert trainer.collected()["rce_n"] == 5

    def test_global_normalization_differs_from_averaging_the_batches(self):
        # the homogeneous-batch case that made eval_loss a per-task mean: one
        # ntp batch and one rce batch, averaged, is not the token-weighted mean
        trainer = FakeTrainer(collect=True)
        per_batch_mean = (trainer.loss(batch(4, 0)) + trainer.loss(batch(0, 4))) / 2
        metrics = eval_loss_metrics(trainer.collected())
        assert metrics["balanced_loss"] == pytest.approx(per_batch_mean)
        assert metrics["loss"] != pytest.approx(per_batch_mean)

    def test_rows_with_a_truncated_answer_are_left_out_of_the_denominator(self):
        trainer = FakeTrainer(collect=True)
        inputs = batch(0, 3, length=6)
        # push one answer past the end of the sequence, as truncation would
        inputs["answer_pos"][0] = 99
        trainer.loss(inputs)
        assert trainer.collected()["rce_n"] == 2
