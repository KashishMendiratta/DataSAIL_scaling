import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler

from src.utils import compute_fingerprints

# CONFIG


BASE_DIR = "results/experiments/splits"
OUTPUT_DIR = "results/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["bace", "bbbp", "tox21", "hiv"]


SPLIT_TYPES = [

    # Baselines
    "random",
    "datasail_full",

    # Scaled methods
    "datasail_scaled_random",
    "datasail_scaled_stratified",

]

LABEL_COLUMNS = {
    "bace": ["Class"],
    "bbbp": ["p_np"],
    "tox21": [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
        "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
        "SR-HSE", "SR-MMP", "SR-p53"
    ],
    "hiv": ["HIV_active"]
}


# HELPERS

def load_split(dataset, split_type):
    path = os.path.join(BASE_DIR, dataset, split_type)

    train_path = os.path.join(path, "train.csv")
    val_path = os.path.join(path, "val.csv")
    test_path = os.path.join(path, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Missing split files in {path}")

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    return train, val, test


def get_smiles_column(df):
    if "smiles" in df.columns:
        return "smiles"
    elif "mol" in df.columns:
        return "mol"
    else:
        raise ValueError("No SMILES column found")


def compute_class_distribution(y):
    total = len(y)
    pos = np.sum(y)
    neg = total - pos
    ratio = pos / total if total > 0 else 0
    return total, pos, neg, ratio


def prepare_features(df, label_col):

    if label_col not in df.columns:
        raise ValueError(f"{label_col} not found")

    df = df.dropna(subset=[label_col]).copy()

    smiles_col = get_smiles_column(df)
    smiles_list = df[smiles_col].tolist()

    fps, valid_idx = compute_fingerprints(smiles_list)

    df = df.iloc[valid_idx].reset_index(drop=True)

    y = df[label_col].values
    X = fps

    return X, y


def evaluate_model(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # SAFE METRICS
    def safe_auc():
        try:
            return roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        except:
            return np.nan

    def safe_pr_auc():
        try:
            return average_precision_score(y_test, y_prob) if y_prob is not None else np.nan
        except:
            return np.nan

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "auc": safe_auc(),

        # Extended metrics
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "pr_auc": safe_pr_auc()
    }

    return results


# MAIN

def run():

    all_results = []

    for dataset in DATASETS:
        for split_type in SPLIT_TYPES:

            path = os.path.join(BASE_DIR, dataset, split_type)

            if not os.path.exists(path):
                print(f"Skipping {dataset} - {split_type}")
                continue

            print(f"\nDataset: {dataset} | Split: {split_type}")

            try:
                train_df, val_df, test_df = load_split(dataset, split_type)
            except Exception as e:
                print(f"Skipping {dataset}-{split_type}: {e}")
                continue

            label_list = LABEL_COLUMNS[dataset]

            for label_col in label_list:

                print(f"  → Label: {label_col}")

                try:
                    X_train, y_train = prepare_features(train_df, label_col)
                    X_test, y_test = prepare_features(test_df, label_col)
                except Exception as e:
                    print(f"Skipping label {label_col}: {e}")
                    continue

                # Class distribution tracking
                train_total, _, _, train_ratio = compute_class_distribution(y_train)
                test_total, _, _, test_ratio = compute_class_distribution(y_test)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                models = {
                    "random_forest": RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1
                    ),
                    "logistic_regression": LogisticRegression(
                        max_iter=1000
                    )
                }

                for name, model in models.items():

                    if name == "logistic_regression":
                        metrics = evaluate_model(
                            model,
                            X_train_scaled,
                            y_train,
                            X_test_scaled,
                            y_test
                        )
                    else:
                        metrics = evaluate_model(
                            model,
                            X_train,
                            y_train,
                            X_test,
                            y_test
                        )

                    result = {
                        "dataset": dataset,
                        "split": split_type,
                        "label": label_col,
                        "model": name,
                        **metrics,
                        "train_pos_ratio": train_ratio,
                        "test_pos_ratio": test_ratio,
                        "train_size": train_total,
                        "test_size": test_total
                    }

                    all_results.append(result)

    df = pd.DataFrame(all_results)

    output_file = os.path.join(OUTPUT_DIR, "ml_results_extended.csv")
    df = df.round(3)
    df.to_csv(output_file, index=False)

    print("\nSaved results to:", output_file)
    print(df.head())


if __name__ == "__main__":
    run()