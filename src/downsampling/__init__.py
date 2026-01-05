"""Down-sampling strategies for dataset reduction."""
from .strategies import (
    random_downsample,
    stratified_downsample,
    diversity_downsample,
    get_downsampling_method
)

__all__ = [
    'random_downsample',
    'stratified_downsample',
    'diversity_downsample',
    'get_downsampling_method'
]
