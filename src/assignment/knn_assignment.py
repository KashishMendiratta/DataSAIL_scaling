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
    method: str = 'majority'
) -> np.ndarray:
    """
    Assign remaining samples to splits using k-NN.
    
    Args:
        train_fps: Fingerprints of training (down-sampled) data
        train_splits: Split assignments for training data
        remaining_fps: Fingerprints of remaining data to assign
        k: Number of neighbors to consider
        method: Assignment method ('majority', 'weighted', 'closest')
        
    Returns:
        Array of split assignments for remaining samples
    """
    # Fit k-NN model on training data
    knn = NearestNeighbors(n_neighbors=k, metric='jaccard', n_jobs=-1)
    knn.fit(train_fps)
    
    # Find k nearest neighbors for each remaining sample
    distances, indices = knn.kneighbors(remaining_fps)
    
    # Assign based on method
    assignments = []
    
    for i in range(len(remaining_fps)):
        neighbor_indices = indices[i]
        neighbor_splits = train_splits.iloc[neighbor_indices].values
        neighbor_distances = distances[i]
        
        if method == 'majority':
            # Simple majority vote
            assignment = Counter(neighbor_splits).most_common(1)[0][0]
            
        elif method == 'weighted':
            # Distance-weighted voting
            # Convert distances to weights (closer = higher weight)
            weights = 1 / (neighbor_distances + 1e-10)
            
            split_weights = {}
            for split, weight in zip(neighbor_splits, weights):
                split_weights[split] = split_weights.get(split, 0) + weight
            
            assignment = max(split_weights, key=split_weights.get)
            
        elif method == 'closest':
            # Assign to split of closest neighbor
            assignment = neighbor_splits[0]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        assignments.append(assignment)
    
    return np.array(assignments)


def assign_with_confidence(
    train_fps: np.ndarray,
    train_splits: pd.Series,
    remaining_fps: np.ndarray,
    k: int = 5,
    confidence_threshold: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign samples with confidence scores.
    
    Args:
        train_fps: Training fingerprints
        train_splits: Training split assignments
        remaining_fps: Remaining fingerprints
        k: Number of neighbors
        confidence_threshold: Minimum confidence for assignment
        
    Returns:
        Tuple of (assignments, confidence_scores)
    """
    knn = NearestNeighbors(n_neighbors=k, metric='jaccard', n_jobs=-1)
    knn.fit(train_fps)
    
    distances, indices = knn.kneighbors(remaining_fps)
    
    assignments = []
    confidences = []
    
    for i in range(len(remaining_fps)):
        neighbor_splits = train_splits.iloc[indices[i]].values
        
        # Count votes for each split
        vote_counts = Counter(neighbor_splits)
        most_common = vote_counts.most_common(1)[0]
        
        assignment = most_common[0]
        confidence = most_common[1] / k  # Proportion of neighbors voting for this split
        
        assignments.append(assignment)
        confidences.append(confidence)
    
    return np.array(assignments), np.array(confidences)


def evaluate_assignment_quality(
    assignments: np.ndarray,
    confidences: np.ndarray = None
) -> Dict:
    """
    Evaluate quality of assignments.
    
    Args:
        assignments: Assigned splits
        confidences: Confidence scores (optional)
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'total_assigned': len(assignments),
        'split_distribution': dict(Counter(assignments)),
    }
    
    if confidences is not None:
        metrics['mean_confidence'] = float(np.mean(confidences))
        metrics['min_confidence'] = float(np.min(confidences))
        metrics['low_confidence_count'] = int(np.sum(confidences < 0.5))
    
    return metrics
