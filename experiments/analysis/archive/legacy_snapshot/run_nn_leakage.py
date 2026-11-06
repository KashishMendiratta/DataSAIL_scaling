import os
import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances
from src.utils import compute_fingerprints


BASE_DIR = "results/experiments/splits"
OUTPUT_FILE = "results/analysis/nn_leakage_results.csv"
'''
DATASETS = ["tox21", "hiv"]

SPLIT_TYPES = [
    "datasail_scaled_stratified_lambda_0",
    "datasail_scaled_stratified_lambda_0.5",
    "datasail_scaled_stratified_lambda_1",
    "datasail_scaled_stratified_lambda_2",
    "datasail_scaled_stratified_lambda_5"
]
'''
DATASETS = ["bace", "bbbp", "tox21", "hiv"]

SPLIT_TYPES = [

    # Baselines
    "random",
    "datasail_full",

    # Scaled methods
    "datasail_scaled_random",
    "datasail_scaled_stratified",

]

TRAIN_SAMPLE = None
TEST_SAMPLE = None
BATCH_SIZE = 200

SIM_THRESHOLD = 0.7

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def load_split(dataset, split_type):
    path = os.path.join(BASE_DIR, dataset, split_type)

    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Missing split files in {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def get_smiles_column(df):
    return "smiles" if "smiles" in df.columns else "mol"


def sample_df(df, n):
    return df.sample(n=n, random_state=42) if len(df) > n else df


def prepare_fps(df):
    smiles_col = get_smiles_column(df)
    smiles = df[smiles_col].tolist()
    X, valid_idx = compute_fingerprints(smiles)
    return X


def compute_nn_similarity(X_train, X_test):

    max_similarities = []

    for i in range(0, len(X_test), BATCH_SIZE):
        batch = X_test[i:i+BATCH_SIZE]

        distances = pairwise_distances(batch, X_train, metric="jaccard")
        similarities = 1 - distances

        max_similarities.extend(similarities.max(axis=1))

    return np.array(max_similarities)


def run():

    results = []

    for dataset in DATASETS:
        for split_type in SPLIT_TYPES:

            path = os.path.join(BASE_DIR, dataset, split_type)

            if not os.path.exists(path):
                print(f"Skipping {dataset} - {split_type}")
                continue

            print(f"\nDataset: {dataset} | Split: {split_type}")

            try:
                train_df, test_df = load_split(dataset, split_type)

                if TRAIN_SAMPLE is not None:
                    train_df = sample_df(train_df, min(TRAIN_SAMPLE, len(train_df)))

                if TEST_SAMPLE is not None:
                    test_df = sample_df(test_df, min(TEST_SAMPLE, len(test_df)))

                X_train = prepare_fps(train_df)
                X_test = prepare_fps(test_df)

                print("  → Computing NN similarity...")

                nn_sim = compute_nn_similarity(X_train, X_test)

                results.append({
                    "dataset": dataset,
                    "split": split_type,
                    "mean_max_similarity": nn_sim.mean(),
                    "std_max_similarity": nn_sim.std(),
                    "leakage_fraction": np.mean(nn_sim > SIM_THRESHOLD),
                    "threshold": SIM_THRESHOLD,
                    "train_sample_size": len(X_train),
                    "test_sample_size": len(X_test)
                })

                print(f"  → Mean NN similarity: {nn_sim.mean():.4f}")
                print(f"  → Leakage fraction: {np.mean(nn_sim > SIM_THRESHOLD):.4f}")

            except Exception as e:
                print(f"Failed: {e}")

    df = pd.DataFrame(results)
    df = df.round(3)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved NN leakage results to:", OUTPUT_FILE)
    print(df)


if __name__ == "__main__":
    run()