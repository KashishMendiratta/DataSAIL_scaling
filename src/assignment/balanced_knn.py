"""
Balanced kNN assignment: combines similarity with split size balancing.

This addresses the majority class bias in naive kNN assignment by
considering both similarity to neighbors AND current split sizes.
"""
from collections import Counter
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def balanced_knn_assign(
    train_fps: np.ndarray,
    train_splits: pd.Series,
    remaining_fps: np.ndarray,
    k: int = 5,
    target_ratios: Mapping[str, float] | None = None,
    balance_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign remaining samples using balanced kNN.
    
    Combines similarity-based assignment with split-size balancing to
    prevent majority class bias.
    
    Args:
        train_fps: Fingerprints of down-sampled data
        train_splits: Split assignments for down-sampled data  
        remaining_fps: Fingerprints of remaining data
        k: Number of neighbors
        target_ratios: Target split ratios (default: {train:0.7, val:0.2, test:0.1})
        balance_weight: Weight for balance factor (higher = more balancing)
        
    Returns:
        Tuple of (assignments, confidence_scores)
    """
    if target_ratios is None:
        target_ratios = {"train": 0.7, "val": 0.2, "test": 0.1}

    if len(train_fps) != len(train_splits):
        raise ValueError("train_fps and train_splits must have equal lengths")
    if not 1 <= k <= len(train_fps):
        raise ValueError("k must be between 1 and the number of reference samples")
    if balance_weight < 0:
        raise ValueError("balance_weight must be non-negative")
    if not target_ratios or any(ratio <= 0 for ratio in target_ratios.values()):
        raise ValueError("target_ratios must contain positive values")
    if not np.isclose(sum(target_ratios.values()), 1.0):
        raise ValueError("target_ratios must sum to 1")
    unknown_splits = set(train_splits) - set(target_ratios)
    if unknown_splits:
        raise ValueError(f"Reference data contains unknown splits: {sorted(unknown_splits)}")
    
    # Fit kNN
    knn = NearestNeighbors(n_neighbors=k, metric='jaccard', n_jobs=-1)
    knn.fit(train_fps)
    
    # Find neighbors
    distances, indices = knn.kneighbors(remaining_fps)
    
    assignments = []
    confidences = []
    
    # Track current split sizes (start with down-sampled splits)
    current_counts = Counter(train_splits)
    total_assigned = len(train_splits)
    
    for i in range(len(remaining_fps)):
        neighbor_indices = indices[i]
        neighbor_splits = train_splits.iloc[neighbor_indices].values
        neighbor_distances = distances[i]
        
        # Compute scores for each split
        split_scores = {}
        
        for split_name in target_ratios.keys():
            # Similarity score: how many neighbors are in this split?
            # Weight by distance (closer = higher weight)
            split_mask = (neighbor_splits == split_name)
            if split_mask.any():
                split_distances = neighbor_distances[split_mask]
                # Convert distances to similarities (0=identical, 1=completely different)
                similarities = 1 - split_distances
                similarity_score = similarities.mean()
            else:
                similarity_score = 0.0
            
            # Balance factor: prefer under-represented splits
            current_ratio = current_counts[split_name] / total_assigned
            target_ratio = target_ratios[split_name]
            
            # If split is under-represented, boost its score
            # If over-represented, reduce its score
            balance_factor = target_ratio / (current_ratio + 1e-10)
            
            # Combined score
            score = similarity_score * (balance_factor ** balance_weight)
            split_scores[split_name] = score
        
        # Assign to split with highest score
        best_split = max(split_scores, key=split_scores.get)
        assignments.append(best_split)
        
        # Update counts
        current_counts[best_split] += 1
        total_assigned += 1
        
        # Confidence = how much better was best split vs others?
        scores = list(split_scores.values())
        scores.sort(reverse=True)
        if len(scores) > 1 and scores[0] > 0:
            confidence = (scores[0] - scores[1]) / scores[0]
        else:
            confidence = 0.5
        confidences.append(confidence)
    
    return np.array(assignments), np.array(confidences)
