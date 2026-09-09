import numpy as np
import pandas as pd
import pytest

from src.assignment.balanced_knn import balanced_knn_assign
from src.assignment.knn_assignment import evaluate_assignment_quality, knn_assign


def test_knn_assigns_nearest_reference_split() -> None:
    reference = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
    splits = pd.Series(["train", "test"])
    queries = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)

    assignments, confidence = knn_assign(reference, splits, queries, k=1)

    assert assignments.tolist() == ["train", "test"]
    assert confidence.tolist() == [1.0, 1.0]


def test_balanced_assignment_respects_requested_splits() -> None:
    reference = np.eye(6, dtype=bool)
    splits = pd.Series(["train", "train", "train", "val", "test", "test"])
    queries = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0]], dtype=bool)

    assignments, confidence = balanced_knn_assign(
        reference,
        splits,
        queries,
        k=2,
        target_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
    )

    assert set(assignments).issubset({"train", "val", "test"})
    assert np.all((confidence >= 0) & (confidence <= 1))


def test_balanced_assignment_rejects_invalid_configuration() -> None:
    reference = np.eye(2, dtype=bool)
    splits = pd.Series(["train", "test"])

    with pytest.raises(ValueError, match="sum to 1"):
        balanced_knn_assign(
            reference,
            splits,
            reference,
            k=1,
            target_ratios={"train": 0.8, "test": 0.3},
        )


def test_assignment_quality_summary() -> None:
    metrics = evaluate_assignment_quality(
        np.array(["train", "train", "test"]),
        np.array([0.9, 0.4, 1.0]),
    )

    assert metrics["total_assigned"] == 3
    assert metrics["split_distribution"] == {"train": 2, "test": 1}
    assert metrics["low_confidence_count"] == 1
