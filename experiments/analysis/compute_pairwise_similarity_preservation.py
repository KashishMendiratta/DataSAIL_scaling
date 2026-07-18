import pandas as pd
import numpy as np
import os
from itertools import combinations
from sklearn.metrics import pairwise_distances

from src.utils import compute_fingerprints
from src.downsampling import (
    random_downsample,
    stratified_downsample
)

DATASETS = ["bace", "bbbp", "tox21", "hiv"]

LABEL_COLUMNS = {
    "bace": "Class",
    "bbbp": "p_np",
    "tox21": "NR-AR",
    "hiv": "HIV_active"
}

SMILES_COLUMNS = {
    "bace": "mol",
    "bbbp": "smiles",
    "tox21": "smiles",
    "hiv": "smiles"
}

DATA_PATHS = {
    "bace": "data/raw/moleculenet/bace.csv",
    "bbbp": "data/raw/moleculenet/bbbp.csv",
    "tox21": "data/raw/moleculenet/tox21.csv",
    "hiv": "data/raw/moleculenet/hiv.csv"
}

RATIO = 0.25
SEED = 42
HIV_SAMPLE_PAIRS = 100000

results = []


def exact_mean_similarity(fps):

    distances = pairwise_distances(
        fps,
        metric="jaccard"
    )

    similarities = 1 - distances

    upper = np.triu_indices_from(
        similarities,
        k=1
    )

    return similarities[upper].mean()


def sampled_mean_similarity(
    fps,
    n_pairs=100000,
    seed=42
):

    rng = np.random.default_rng(seed)

    n = len(fps)

    idx1 = rng.integers(
        0,
        n,
        n_pairs
    )

    idx2 = rng.integers(
        0,
        n,
        n_pairs
    )

    valid = idx1 != idx2

    idx1 = idx1[valid]
    idx2 = idx2[valid]

    sims = []

    for i, j in zip(idx1, idx2):

        sim = (
            1
            -
            pairwise_distances(
                fps[i:i+1],
                fps[j:j+1],
                metric="jaccard"
            )[0, 0]
        )

        sims.append(sim)

    return np.mean(sims)


for dataset in DATASETS:

    print(f"\nProcessing {dataset}")

    df = pd.read_csv(
        DATA_PATHS[dataset]
    )

    smiles_col = SMILES_COLUMNS[
        dataset
    ]

    label_col = LABEL_COLUMNS[
        dataset
    ]

    fps, valid_idx = compute_fingerprints(
        df[smiles_col].tolist()
    )

    df = (
        df.iloc[valid_idx]
        .reset_index(drop=True)
    )

    # recreate subsets

    rand_sampled, _ = random_downsample(
        df,
        RATIO,
        seed=SEED
    )

    strat_sampled, _ = (
        stratified_downsample(
            df,
            df[label_col],
            RATIO,
            seed=SEED
        )
    )

    rand_fps = fps[
        rand_sampled.index
    ]

    strat_fps = fps[
        strat_sampled.index
    ]

    if dataset == "hiv":

        full_sim = sampled_mean_similarity(
            fps,
            HIV_SAMPLE_PAIRS
        )

        rand_sim = sampled_mean_similarity(
            rand_fps,
            HIV_SAMPLE_PAIRS
        )

        strat_sim = sampled_mean_similarity(
            strat_fps,
            HIV_SAMPLE_PAIRS
        )

    else:

        full_sim = exact_mean_similarity(
            fps
        )

        rand_sim = exact_mean_similarity(
            rand_fps
        )

        strat_sim = exact_mean_similarity(
            strat_fps
        )

    results.extend([

        {
            "dataset": dataset,
            "method": "random",
            "full_similarity":
                round(full_sim, 3),
            "subset_similarity":
                round(rand_sim, 3),
            "delta_similarity":
                round(
                    abs(
                        full_sim
                        -
                        rand_sim
                    ),
                    3
                )
        },

        {
            "dataset": dataset,
            "method": "stratified",
            "full_similarity":
                round(full_sim, 3),
            "subset_similarity":
                round(strat_sim, 3),
            "delta_similarity":
                round(
                    abs(
                        full_sim
                        -
                        strat_sim
                    ),
                    3
                )
        }

    ])


results_df = pd.DataFrame(
    results
)

print("\nMean Pairwise Similarity Preservation")
print(results_df)

os.makedirs(
    "results/analysis",
    exist_ok=True
)

results_df.to_csv(
    "results/analysis/pairwise_similarity_preservation.csv",
    index=False
)

print(
    "\nSaved to:"
)

print(
    "results/analysis/pairwise_similarity_preservation.csv"
)