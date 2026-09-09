import pandas as pd
import pytest

from src.downsampling.strategies import random_downsample, stratified_downsample


def test_random_downsample_is_disjoint_complete_and_reproducible() -> None:
    data = pd.DataFrame({"value": range(20)})

    sampled, remaining = random_downsample(data, ratio=0.25, seed=7)
    repeated, _ = random_downsample(data, ratio=0.25, seed=7)

    assert len(sampled) == 5
    assert sampled.index.tolist() == repeated.index.tolist()
    assert set(sampled.index).isdisjoint(remaining.index)
    assert set(sampled.index) | set(remaining.index) == set(data.index)


def test_stratified_downsample_preserves_each_class() -> None:
    data = pd.DataFrame({"value": range(20)})
    labels = pd.Series([0] * 10 + [1] * 10)

    sampled, remaining = stratified_downsample(data, labels, ratio=0.2, seed=3)

    assert labels.loc[sampled.index].value_counts().to_dict() == {0: 2, 1: 2}
    assert set(sampled.index).isdisjoint(remaining.index)


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1])
def test_downsampling_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio"):
        random_downsample(pd.DataFrame({"value": range(5)}), ratio=ratio)
