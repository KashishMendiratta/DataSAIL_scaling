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
        Original indices are preserved!
    """
    np.random.seed(seed)
    n_samples = int(len(data) * ratio)
    
    # Get original indices
    all_indices = data.index.tolist()
    
    # Randomly select indices
    sampled_indices = np.random.choice(all_indices, size=n_samples, replace=False)
    remaining_indices = [idx for idx in all_indices if idx not in sampled_indices]
    
    # Use loc to preserve original indices
    sampled_data = data.loc[sampled_indices].copy()
    remaining_data = data.loc[remaining_indices].copy()
    
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
        Original indices are preserved!
    """
    np.random.seed(seed)
    
    sampled_indices = []
    for label in labels.unique():
        # Get indices for this class
        class_indices = labels[labels == label].index.tolist()
        n_class_samples = int(len(class_indices) * ratio)
        
        # Sample from this class
        class_sampled = np.random.choice(class_indices, size=n_class_samples, replace=False)
        sampled_indices.extend(class_sampled)
    
    remaining_indices = [idx for idx in data.index if idx not in sampled_indices]
    
    sampled_data = data.loc[sampled_indices].copy()
    remaining_data = data.loc[remaining_indices].copy()
    
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
        Original indices are preserved!
    """
    n_samples = int(len(data) * ratio)
    
    # Map between array positions and dataframe indices
    index_to_position = {idx: pos for pos, idx in enumerate(data.index)}
    position_to_index = {pos: idx for idx, pos in index_to_position.items()}
    
    if method == 'kmeans':
        # Use k-means to find diverse representatives
        kmeans = KMeans(n_clusters=n_samples, random_state=seed, n_init=10)
        kmeans.fit(fingerprints)
        
        # Find closest sample to each cluster center
        sampled_positions = []
        for center in kmeans.cluster_centers_:
            distances = euclidean_distances([center], fingerprints)[0]
            closest_pos = np.argmin(distances)
            if closest_pos not in sampled_positions:
                sampled_positions.append(closest_pos)
        
        # If we don't have enough (due to duplicates), add random samples
        if len(sampled_positions) < n_samples:
            remaining_positions = [p for p in range(len(fingerprints)) if p not in sampled_positions]
            additional = np.random.choice(remaining_positions, 
                                         size=n_samples - len(sampled_positions), 
                                         replace=False)
            sampled_positions.extend(additional)
    
    elif method == 'maxmin':
        # MaxMin algorithm: iteratively select most diverse samples
        sampled_positions = []
        
        # Start with random sample
        np.random.seed(seed)
        first_pos = np.random.randint(len(fingerprints))
        sampled_positions.append(first_pos)
        
        # Iteratively add most distant sample
        for _ in range(n_samples - 1):
            # Compute minimum distance to already selected samples
            selected_fps = fingerprints[sampled_positions]
            min_distances = euclidean_distances(fingerprints, selected_fps).min(axis=1)
            
            # Don't select already sampled points
            min_distances[sampled_positions] = -1
            
            # Select point with maximum minimum distance
            next_pos = np.argmax(min_distances)
            sampled_positions.append(next_pos)
    
    else:
        raise ValueError(f"Unknown diversity method: {method}")
    
    # Convert positions back to original indices
    sampled_indices = [position_to_index[pos] for pos in sampled_positions]
    remaining_indices = [idx for idx in data.index if idx not in sampled_indices]
    
    sampled_data = data.loc[sampled_indices].copy()
    remaining_data = data.loc[remaining_indices].copy()
    
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
