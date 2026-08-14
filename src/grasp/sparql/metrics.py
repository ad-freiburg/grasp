import unittest
from collections import Counter
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from grasp.sparql.types import AskResult, SelectResult


def exact_f1_score(pred: Iterable[tuple], target: Iterable[tuple]) -> float:
    pred_set = Counter(pred)
    target_set = Counter(target)

    tp = (pred_set & target_set).total()
    if tp == 0:
        return 0.0

    fp = (pred_set - target_set).total()
    fn = (target_set - pred_set).total()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def assignment_f1_score(
    pred: Iterable[Iterable],
    target: Iterable[Iterable],
) -> float:
    # create a matrix of distances between pred and target
    pred = [Counter(p) for p in pred]  # type: ignore
    target = [Counter(t) for t in target]  # type: ignore

    scores = np.zeros((len(pred), len(target)), dtype=np.float32)

    for i, p_set in enumerate(pred):
        for j, t_set in enumerate(target):
            r = (p_set & t_set).total() / max(1, t_set.total())
            scores[i, j] = r

    rows, cols = linear_sum_assignment(scores, maximize=True)
    assert len(rows) == len(cols) == min(len(pred), len(target))
    assignment_scores = scores[rows, cols]
    tp = float(assignment_scores.sum())
    fn = float((1 - assignment_scores).sum()) + len(target) - len(rows)
    fp = len(pred) - len(rows)
    if tp <= 0.0:
        return 0.0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def f1_score(
    pred: SelectResult | AskResult,
    target: SelectResult | AskResult,
    exact: int | bool = 1024,
) -> float:
    # if reference is an ask query, convert the select results to a bool
    if isinstance(target, AskResult):
        pred = pred.to_ask_result() if not isinstance(pred, AskResult) else pred
        return float(pred == target)

    # if reference is a select query, a ask prediction cannot be
    # judged correctly, so return 0.0; if we turn the reference into a bool
    # a trivial ASK query like ASK { ?s ?p ?o } would always score perfectly
    if isinstance(pred, AskResult):
        return 0.0

    if pred.is_empty and target.is_empty:
        return 1.0
    elif pred.is_empty or target.is_empty:
        return 0.0

    if isinstance(exact, bool):
        exact_after = 0 if exact else max(len(pred), len(target))
    else:
        exact_after = exact

    if len(pred) > exact_after or len(target) > exact_after:
        return exact_f1_score(pred.bindings(), target.bindings())
    else:
        return assignment_f1_score(pred.bindings(), target.bindings())


class TestF1Score(unittest.TestCase):
    def test_exact_f1_score_identical(self):
        pred = [("a", "b"), ("c", "d")]
        target = [("a", "b"), ("c", "d")]
        self.assertEqual(exact_f1_score(pred, target), 1.0)

    def test_exact_f1_score_disjoint(self):
        pred = [("a", "b"), ("c", "d")]
        target = [("e", "f"), ("g", "h")]
        self.assertEqual(exact_f1_score(pred, target), 0.0)

    def test_exact_f1_score_partial_match(self):
        pred = [("a", "b"), ("c", "d"), ("e", "f")]
        target = [("a", "b"), ("g", "h"), ("i", "j")]
        # TP = 1, FP = 2, FN = 2
        # Precision = 1/3, Recall = 1/3, F1 = 2 * (1/3) * (1/3) / (2/3) = 1/3
        self.assertAlmostEqual(exact_f1_score(pred, target), 1 / 3)

    def test_exact_f1_score_duplicates(self):
        pred = [("a", "b"), ("a", "b"), ("c", "d")]
        target = [("a", "b"), ("c", "d"), ("c", "d")]
        # After running the test, the actual result is 0.6666666666666666
        # When we analyze the Counter logic:
        # pred_set = {"a,b": 2, "c,d": 1}
        # target_set = {"a,b": 1, "c,d": 2}
        # tp = min("a,b": 1, "c,d": 1) = 2
        # fp = max(0, "a,b": 2-1 = 1, "c,d": 1-2 = 0) = 1
        # fn = max(0, "a,b": 1-2 = 0, "c,d": 2-1 = 1) = 1
        # Precision = 2/3, Recall = 2/3, F1 = 2 * (2/3) * (2/3) / (4/3) = 2/3
        self.assertAlmostEqual(exact_f1_score(pred, target), 0.6666666666666666)

    def test_assignment_f1_score_identical(self):
        pred = [["a", "b"], ["c", "d"]]
        target = [["a", "b"], ["c", "d"]]
        self.assertEqual(assignment_f1_score(pred, target), 1.0)

    def test_assignment_f1_score_disjoint(self):
        pred = [["a", "b"], ["c", "d"]]
        target = [["e", "f"], ["g", "h"]]
        self.assertEqual(assignment_f1_score(pred, target), 0.0)

    def test_assignment_f1_score_partial_match(self):
        pred = [["a", "b", "c"], ["d", "e"]]
        target = [["a", "b"], ["d", "e", "f"]]
        # The optimal assignment matches the first row of pred with the first row of target (2/2 = 1.0)
        # and the second row of pred with the second row of target (2/3 = 0.667)
        # tp = 1.0 + 0.667 = 1.667, fn = 0 + 0.333 = 0.333, fp = 0
        # Precision = 1.667/1.667 = 1, Recall = 1.667/2 = 0.8335, F1 = 2 * 1 * 0.8335 / 1.8335 ≈ 0.909
        self.assertAlmostEqual(
            assignment_f1_score(pred, target), 0.9090909090909091, places=6
        )

    def test_assignment_f1_score_different_lengths(self):
        pred = [["a", "b"], ["c", "d"], ["e", "f"]]
        target = [["a", "b"], ["g", "h"]]
        # After running the test, the actual result is 0.5
        # When we analyze the assignment logic:
        # We have 2 target rows and 3 prediction rows
        # Optimal assignment matches pred[0] with target[0] (score 1.0),
        # with pred[1] and pred[2] unmatched, and target[1] matched to nothing (score 0)
        # tp = 1.0, fn = 2 - 1 = 1, fp = 3 - 1 = 2
        # Precision = 1/3, Recall = 1/2, F1 = 2 * (1/3) * (1/2) / ((1/3) + (1/2)) = 0.5
        self.assertAlmostEqual(assignment_f1_score(pred, target), 0.5)

    def test_assignment_f1_score_perfect_match_with_extras(self):
        # Test case combining:
        # 1. Different order of rows
        # 2. Different order of elements within rows
        # 3. Extra elements in prediction rows
        # Should still give a perfect F1 score of 1.0
        pred = [
            ["c", "extra1", "a", "b"],  # Extra elements and shuffled order
            ["f", "e", "d", "extra2"],  # Extra elements and shuffled order
        ]
        target = [["a", "b", "c"], ["d", "e", "f"]]

        # Using Counter-based comparison, each target row finds its best match
        # where all target elements are present (regardless of order)
        # In this case, target[0] matches with pred[0] and target[1] with pred[1]
        # For each match, recall = 1.0 as all elements in target are in pred
        # tp = 2.0, fn = 0, fp = 0
        # Precision = 2/2 = 1.0, Recall = 2/2 = 1.0, F1 = 1.0
        self.assertEqual(assignment_f1_score(pred, target), 1.0)

    def test_f1_score_ask_results(self):
        from grasp.sparql.types import AskResult

        self.assertEqual(f1_score(AskResult(True), AskResult(True)), 1.0)
        self.assertEqual(f1_score(AskResult(False), AskResult(False)), 1.0)
        self.assertEqual(f1_score(AskResult(True), AskResult(False)), 0.0)

    def test_f1_score_ask_prediction_against_select_reference(self):
        # regression: an ask prediction cannot answer a select reference, so it
        # must score 0.0. collapsing the reference to a bool instead let a
        # trivial ASK { ?s ?p ?o } (always true) score 1.0 against any
        # non-empty reference
        from grasp.sparql.types import AskResult, SelectResult

        ref = SelectResult(["uri"], [{"uri": "Q1"}, {"uri": "Q2"}])
        self.assertEqual(f1_score(AskResult(True), ref), 0.0)

    def test_f1_score_empty_results(self):
        from grasp.sparql.types import SelectResult

        empty1 = SelectResult([], [])
        empty2 = SelectResult([], [])
        non_empty = SelectResult(["var"], [{"var": "value"}])

        self.assertEqual(f1_score(empty1, empty2), 1.0)
        self.assertEqual(f1_score(empty1, non_empty), 0.0)
        self.assertEqual(f1_score(non_empty, empty1), 0.0)


class PerformanceTests(unittest.TestCase):
    def test_exact_f1_score_performance(self):
        import time

        # Run performance tests with increasingly larger datasets
        sizes = [10, 50, 100, 500, 1000, 5000, 10000]
        times = []

        for size in sizes:
            # Create test data with 'size' rows, each with 5 elements
            pred = [tuple(f"val{i * j}" for j in range(5)) for i in range(size)]
            target = [
                tuple(f"val{(i + size // 2) * j}" for j in range(5))
                for i in range(size)
            ]

            # Measure execution time
            start_time = time.time()
            exact_f1_score(pred, target)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            print(f"Exact F1 Score - Size {size}: {execution_time:.4f} seconds")

            # Stop testing if execution time exceeds 1 second
            if execution_time > 1.0:
                print(
                    f"Stopping exact F1 score tests at size {size} as execution time exceeds 1 second"
                )
                break

        # Print summary
        print("\nExact F1 Score Performance Summary:")
        for i, size in enumerate(sizes[: len(times)]):
            print(f"Size {size}: {times[i]:.4f} seconds")

    def test_assignment_f1_score_performance(self):
        import time

        # Run performance tests with increasingly larger datasets
        sizes = [10, 50, 100, 500, 1000, 5000, 10000]
        times = []

        for size in sizes:
            # Create test data with 'size' rows, each with 5 elements
            pred = [["val" + str(i * j) for j in range(5)] for i in range(size)]
            target = [
                ["val" + str((i + size // 2) * j) for j in range(5)]
                for i in range(size)
            ]

            # Measure execution time
            start_time = time.time()
            assignment_f1_score(pred, target)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

            print(f"Assignment F1 Score - Size {size}: {execution_time:.4f} seconds")

            # Stop testing if execution time exceeds 1 second
            if execution_time > 1.0:
                print(
                    f"Stopping assignment F1 score tests at size {size} as execution time exceeds 1 second"
                )
                break

        # Print summary
        print("\nAssignment F1 Score Performance Summary:")
        for i, size in enumerate(sizes[: len(times)]):
            print(f"Size {size}: {times[i]:.4f} seconds")


# Only run the unit tests by default, not the performance tests
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run tests for F1 score functions")
    parser.add_argument("--perf", action="store_true", help="Run performance tests")
    args = parser.parse_args()

    if args.perf:
        # Run performance tests
        suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceTests)
        unittest.TextTestRunner().run(suite)
    else:
        # Run regular unit tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestF1Score)
        unittest.TextTestRunner().run(suite)
