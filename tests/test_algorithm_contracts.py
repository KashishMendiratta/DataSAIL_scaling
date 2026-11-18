"""Algorithm-level regression tests for split sampling and assignment."""
from collections import Counter
import unittest

import numpy as np
import pandas as pd

from src.assignment.balanced_knn import balanced_knn_assign
from src.assignment.knn_assignment import evaluate_assignment_quality, knn_assign
from src.downsampling.strategies import random_downsample, stratified_downsample
try:
    from src.pipeline_engine import choose_anchors, verify_assignments
except ModuleNotFoundError as exc:
    if exc.name != "rdkit":
        raise
    choose_anchors = verify_assignments = None


class DownsamplingTests(unittest.TestCase):
    def test_random_downsample_is_reproducible_and_partitions_original_rows(self):
        frame = pd.DataFrame({"value": range(10)}, index=np.arange(100, 110))

        sampled_a, remaining_a = random_downsample(frame, ratio=0.4, seed=23)
        sampled_b, remaining_b = random_downsample(frame, ratio=0.4, seed=23)

        self.assertEqual(sampled_a.index.tolist(), sampled_b.index.tolist())
        self.assertEqual(remaining_a.index.tolist(), remaining_b.index.tolist())
        self.assertEqual(len(sampled_a), 4)
        self.assertTrue(set(sampled_a.index).isdisjoint(remaining_a.index))
        self.assertEqual(set(sampled_a.index) | set(remaining_a.index), set(frame.index))
        pd.testing.assert_frame_equal(frame.loc[sampled_a.index], sampled_a)

    def test_stratified_downsample_preserves_class_proportions_and_rows(self):
        frame = pd.DataFrame({"value": range(12)}, index=np.arange(200, 212))
        labels = pd.Series(["a"] * 8 + ["b"] * 4, index=frame.index)

        sampled, remaining = stratified_downsample(frame, labels, ratio=0.5, seed=7)

        self.assertEqual(len(sampled), 6)
        self.assertEqual(Counter(labels.loc[sampled.index]), {"a": 4, "b": 2})
        self.assertTrue(set(sampled.index).isdisjoint(remaining.index))
        self.assertEqual(set(sampled.index) | set(remaining.index), set(frame.index))


class AssignmentTests(unittest.TestCase):
    @staticmethod
    def small_fingerprint_fixture():
        # Query exactly matches anchor 0; anchor 1 is farther away.
        fingerprints = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
        query = np.asarray([[1, 0, 0]], dtype=np.uint8)
        splits = pd.Series(["train", "test"])
        return fingerprints, splits, query

    def test_knn_assignment_methods_choose_closest_split(self):
        train_fps, train_splits, query = self.small_fingerprint_fixture()

        for method in ("majority", "weighted", "closest"):
            with self.subTest(method=method):
                assignments, confidence = knn_assign(
                    train_fps, train_splits, query, k=1, method=method
                )
                self.assertEqual(assignments.tolist(), ["train"])
                self.assertEqual(confidence.shape, (1,))
                self.assertTrue(np.all((confidence >= 0) & (confidence <= 1)))
                self.assertAlmostEqual(float(confidence[0]), 1.0)

    def test_knn_assign_can_return_assignments_without_confidence(self):
        train_fps, train_splits, query = self.small_fingerprint_fixture()
        assignments = knn_assign(
            train_fps, train_splits, query, k=1, method="closest", return_confidence=False
        )
        self.assertEqual(assignments.tolist(), ["train"])

    def test_knn_assign_rejects_unknown_voting_method(self):
        train_fps, train_splits, query = self.small_fingerprint_fixture()
        with self.assertRaisesRegex(ValueError, "Unknown method"):
            knn_assign(train_fps, train_splits, query, k=1, method="not-a-method")

    def test_balanced_knn_assign_returns_valid_split_and_finite_confidence(self):
        train_fps, train_splits, query = self.small_fingerprint_fixture()
        assignments, confidence = balanced_knn_assign(
            train_fps,
            train_splits,
            query,
            k=1,
            target_ratios={"train": 0.7, "val": 0.2, "test": 0.1},
            n_jobs=1,
        )
        self.assertEqual(assignments.tolist(), ["train"])
        self.assertEqual(confidence.shape, (1,))
        self.assertTrue(np.isfinite(confidence).all())
        self.assertTrue(np.all((confidence >= 0) & (confidence <= 1)))

    def test_assignment_quality_reports_distribution_and_confidence(self):
        metrics = evaluate_assignment_quality(
            np.asarray(["train", "train", "test"]), np.asarray([0.9, 0.4, 0.8])
        )
        self.assertEqual(metrics["total_assigned"], 3)
        self.assertEqual(metrics["split_distribution"], {"train": 2, "test": 1})
        self.assertAlmostEqual(metrics["mean_confidence"], 0.7)
        self.assertEqual(metrics["low_confidence_count"], 1)


@unittest.skipIf(choose_anchors is None, "pipeline-engine tests require RDKit")
class PipelineInvariantTests(unittest.TestCase):
    def test_random_anchor_selection_is_seed_reproducible_and_exact_sized(self):
        frame = pd.DataFrame({"label": ["a", "b"] * 20})
        spec = {"stratify_column": "label", "task_type": "classification", "path": "fixture.csv"}
        first = choose_anchors(frame, spec, ratio=0.25, strategy="random", seed=31)
        repeated = choose_anchors(frame, spec, ratio=0.25, strategy="random", seed=31)
        self.assertEqual(len(first), 10)
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(len(np.unique(first)), len(first))

    def test_stratified_anchor_selection_maintains_balanced_classes(self):
        frame = pd.DataFrame({"label": ["a"] * 20 + ["b"] * 20})
        spec = {"stratify_column": "label", "task_type": "classification", "path": "fixture.csv"}
        anchors = choose_anchors(frame, spec, ratio=0.25, strategy="stratified", seed=31)
        self.assertEqual(len(anchors), 10)
        self.assertEqual(Counter(frame.iloc[anchors]["label"]), {"a": 5, "b": 5})

    def test_assignment_validation_rejects_missing_split_and_duplicate_ids(self):
        frame = pd.DataFrame({"_datasail_id": ["id-1", "id-2"]})
        with self.assertRaisesRegex(ValueError, "Expected all split labels"):
            verify_assignments(frame, np.asarray(["train", "test"]), ["train", "val", "test"])

        frame["_datasail_id"] = ["id-1", "id-1"]
        with self.assertRaisesRegex(ValueError, "cardinality or uniqueness"):
            verify_assignments(frame, np.asarray(["train", "test"]), ["train", "val", "test"])


if __name__ == "__main__":
    unittest.main()
