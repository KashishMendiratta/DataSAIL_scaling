"""
k-Nearest Neighbors assignment for remaining samples.

After down-sampling and splitting, assign remaining samples to splits
based on their similarity to samples in each split.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple
from collections import Counter


def knn_assign(
    train_fps: np.ndarray,
    train_splits: pd.Series,
    remaining_fps: np.ndarray,
    k: int = 5,
    method: str = "majority",
    return_confidence: bool = True
):
    """
    Assign remaining samples to splits using k-NN.

    Args:
        train_fps:
            Fingerprints of down-sampled data

        train_splits:
            Split assignments
            for down-sampled data

        remaining_fps:
            Fingerprints to assign

        k:
            Number of neighbours

        method:
            Assignment method
            ('majority',
             'weighted',
             'closest')

        return_confidence:
            Whether to return
            confidence scores

    Returns:

        If return_confidence=False:

            assignments

        If return_confidence=True:

            (assignments, confidences)
    """

    # Fit kNN model

    knn = NearestNeighbors(
        n_neighbors=k,
        metric="jaccard",
        n_jobs=-1
    )

    knn.fit(train_fps)

    # Find neighbours

    distances, indices = knn.kneighbors(
        remaining_fps
    )

    assignments = []
    confidences = []

    for i in range(
        len(remaining_fps)
    ):

        neighbor_indices = indices[i]

        neighbor_splits = (
            train_splits
            .iloc[neighbor_indices]
            .values
        )

        neighbor_distances = distances[i]

        # -----------------
        # Majority vote
        # -----------------

        if method == "majority":

            vote_counts = Counter(
                neighbor_splits
            )

            most_common = (
                vote_counts
                .most_common(1)[0]
            )

            assignment = (
                most_common[0]
            )

            confidence = (
                most_common[1] / k
            )

        # -----------------
        # Weighted vote
        # -----------------

        elif method == "weighted":

            weights = (
                1 /
                (neighbor_distances + 1e-10)
            )

            split_weights = {}

            for split, weight in zip(
                neighbor_splits,
                weights
            ):

                split_weights[
                    split
                ] = (
                    split_weights
                    .get(split,0)
                    + weight
                )

            assignment = max(
                split_weights,
                key=split_weights.get
            )

            best_score = (
                split_weights[
                    assignment
                ]
            )

            confidence = (
                best_score /
                sum(
                    split_weights.values()
                )
            )

        # -----------------
        # Closest neighbour
        # -----------------

        elif method == "closest":

            assignment = (
                neighbor_splits[0]
            )

            confidence = 1.0

        else:

            raise ValueError(
                f"Unknown method: "
                f"{method}"
            )

        assignments.append(
            assignment
        )

        confidences.append(
            confidence
        )

    assignments = np.array(
        assignments
    )

    confidences = np.array(
        confidences
    )

    if return_confidence:

        return (
            assignments,
            confidences
        )

    return assignments


def evaluate_assignment_quality(
    assignments: np.ndarray,
    confidences: np.ndarray = None
) -> Dict:
    """
    Evaluate assignment quality.
    """

    metrics = {

        "total_assigned":
            len(assignments),

        "split_distribution":
            dict(
                Counter(assignments)
            ),
    }

    if confidences is not None:

        metrics[
            "mean_confidence"
        ] = float(
            np.mean(confidences)
        )

        metrics[
            "min_confidence"
        ] = float(
            np.min(confidences)
        )

        metrics[
            "low_confidence_count"
        ] = int(
            np.sum(
                confidences < 0.5
            )
        )

    return metrics