"""
Down-sampling strategies for large datasets.

This module implements various strategies to reduce dataset size
before running DataSAIL clustering.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def random_downsample(data: pd.DataFrame, ratio: float, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random down-sampling: randomly select a fraction of samples.
    
    Args:
        data: DataFrame with molecular data
        ratio: Fraction of data to keep (e.g., 0.1 for 10%)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_data, remaining_data)
    """
    np.random.seed(seed)
    n_samples = int(len(data) * ratio)
    
    # Randomly select indices
    sampled_indices = np.random.choice(len(data), size=n_samples, replace=False)
    remaining_indices = np.array([i for i in range(len(data)) if i not in sampled_indices])
    
    sampled_data = data.iloc[sampled_indices].reset_index(drop=True)
    remaining_data = data.iloc[remaining_indices].reset_index(drop=True)
    
    return sampled_data, remaining_data


def stratified_downsample(data: pd.DataFrame, labels: pd.Series, ratio: float, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified down-sampling: maintain class distribution in the sample.
    
    Args:
        data: DataFrame with molecular data
        labels: Series with class labels
        ratio: Fraction of data to keep
        seed: Random seed
        
    Returns:
        Tuple of (sampled_data, remaining_data)
    """
    np.random.seed(seed)
    
    sampled_indices = []
    for label in labels.unique():
        # Get indices for this class
        class_indices = np.where(labels == label)[0]
        n_class_samples = int(len(class_indices) * ratio)
        
        # Sample from this class
        class_sampled = np.random.choice(class_indices, size=n_class_samples, replace=False)
        sampled_indices.extend(class_sampled)
    
    sampled_indices = np.array(sampled_indices)
    remaining_indices = np.array([i for i in range(len(data)) if i not in sampled_indices])
    
    sampled_data = data.iloc[sampled_indices].reset_index(drop=True)
    remaining_data = data.iloc[remaining_indices].reset_index(drop=True)
    
    return sampled_data, remaining_data


def diversity_downsample(fingerprints: np.ndarray, data: pd.DataFrame, ratio: float, 
                        method: str = 'kmeans', seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Diversity-based down-sampling: select diverse representatives.
    
    Uses k-means clustering and selects samples closest to cluster centers
    to ensure chemical diversity.
    
    Args:
        fingerprints: Molecular fingerprints (n_samples, n_features)
        data: DataFrame with molecular data
        ratio: Fraction of data to keep
        method: Diversity method ('kmeans' or 'maxmin')
        seed: Random seed
        
    Returns:
        Tuple of (sampled_data, remaining_data)
    """
    n_samples = int(len(data) * ratio)
    
    if method == 'kmeans':
        # Use k-means to find diverse representatives
        kmeans = KMeans(n_clusters=n_samples, random_state=seed, n_init=10)
        kmeans.fit(fingerprints)
        
        # Find closest sample to each cluster center
        sampled_indices = []
        for center in kmeans.cluster_centers_:
            distances = euclidean_distances([center], fingerprints)[0]
            closest_idx = np.argmin(distances)
            if closest_idx not in sampled_indices:
                sampled_indices.append(closest_idx)
        
        # If we don't have enough (due to duplicates), add random samples
        if len(sampled_indices) < n_samples:
            remaining_pool = [i for i in range(len(data)) if i not in sampled_indices]
            additional = np.random.choice(remaining_pool, 
                                         size=n_samples - len(sampled_indices), 
                                         replace=False)
            sampled_indices.extend(additional)
    
    elif method == 'maxmin':
        # MaxMin algorithm: iteratively select most diverse samples
        sampled_indices = []
        
        # Start with random sample
        np.random.seed(seed)
        first_idx = np.random.randint(len(fingerprints))
        sampled_indices.append(first_idx)
        
        # Iteratively add most distant sample
        for _ in range(n_samples - 1):
            # Compute minimum distance to already selected samples
            selected_fps = fingerprints[sampled_indices]
            min_distances = euclidean_distances(fingerprints, selected_fps).min(axis=1)
            
            # Don't select already sampled points
            min_distances[sampled_indices] = -1
            
            # Select point with maximum minimum distance
            next_idx = np.argmax(min_distances)
            sampled_indices.append(next_idx)
    
    else:
        raise ValueError(f"Unknown diversity method: {method}")
    
    sampled_indices = np.array(sampled_indices)
    remaining_indices = np.array([i for i in range(len(data)) if i not in sampled_indices])
    
    sampled_data = data.iloc[sampled_indices].reset_index(drop=True)
    remaining_data = data.iloc[remaining_indices].reset_index(drop=True)
    
    return sampled_data, remaining_data


def get_downsampling_method(method: str):
    """
    Get down-sampling function by name.
    
    Args:
        method: Name of method ('random', 'stratified', 'diversity_kmeans', 'diversity_maxmin')
        
    Returns:
        Down-sampling function
    """
    methods = {
        'random': random_downsample,
        'stratified': stratified_downsample,
        'diversity_kmeans': lambda fps, data, ratio, seed: diversity_downsample(fps, data, ratio, 'kmeans', seed),
        'diversity_maxmin': lambda fps, data, ratio, seed: diversity_downsample(fps, data, ratio, 'maxmin', seed),
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
    
    return methods[method]
