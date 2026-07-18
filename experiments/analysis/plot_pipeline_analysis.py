import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = "results/experiments"
LEAKAGE_FILE = "results/analysis/nn_leakage_results.csv"
ML_FILE = "results/analysis/ml_results_extended.csv"
PLOT_DIR = "results/analysis/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


PALETTE = {
    "random": "#1f77b4",
    "datasail_scaled_stratified": "#ff7f0e",
    "datasail_full": "#2ca02c",
    #"datasail_scaled_random": "#ff0000" 
}

DATASETS = ["bace", "bbbp", "tox21", "hiv"]
SPLIT_ORDER = [
    # Baselines
    "random",
    "datasail_full",

    # Scaled methods
    #"datasail_scaled_random",
    "datasail_scaled_stratified"
    ]


# HELPER

def read_runtime(path):
    if not os.path.exists(path):
        return None

    with open(path) as f:
        for line in f:
            if line.startswith("total_runtime_seconds"):
                try:
                    return float(line.split(":")[1].strip())
                except:
                    return None
    return None


# LOAD RUNTIME

runtime_data = []

for dataset in DATASETS:
    dataset_dir = os.path.join(BASE_DIR, dataset)

    runtime_files = {
        #"random": "runtime_random.txt",
        "datasail_scaled": "runtime_scaled_stratified.txt",
        "datasail_full": "runtime_full.txt"
    }

    for split, filename in runtime_files.items():
        path = os.path.join(dataset_dir, filename)
        runtime = read_runtime(path)

        if runtime is None:
            continue

        runtime_data.append({
            "dataset": dataset,
            "split": split,
            "runtime": runtime
        })

runtime_df = pd.DataFrame(runtime_data)

runtime_df["split"] = pd.Categorical(
    runtime_df["split"],
    categories=SPLIT_ORDER,
    ordered=True
)


# 1. RUNTIME PLOT

plt.figure()
sns.barplot(
    data=runtime_df,
    x="dataset",
    y="runtime",
    hue="split",
    palette=PALETTE
)
plt.title("Runtime Comparison Across Pipelines")
plt.ylabel("Runtime (seconds)")
plt.xlabel("Dataset")
plt.legend(title="Pipeline")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/runtime_comparison.png")
plt.close()


# LOAD LEAKAGE

leak_df = pd.read_csv(LEAKAGE_FILE)

leak_df["split"] = pd.Categorical(
    leak_df["split"],
    categories=SPLIT_ORDER,
    ordered=True
)


# 2. LEAKAGE vs RUNTIME

merged_df = pd.merge(runtime_df, leak_df, on=["dataset", "split"])

plt.figure()
sns.scatterplot(
    data=merged_df,
    x="runtime",
    y="mean_max_similarity",
    hue="split",
    style="dataset",
    s=120,
    palette=PALETTE
)
plt.title("Leakage vs Runtime Trade-off")
plt.xlabel("Runtime (seconds)")
plt.ylabel("Mean Max Similarity (Leakage)")
plt.legend(title="Pipeline")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/leakage_vs_runtime.png")
plt.close()


# 3. SPLIT QUALITY

ml_df = pd.read_csv(ML_FILE)

ml_df["split"] = pd.Categorical(
    ml_df["split"],
    categories=SPLIT_ORDER,
    ordered=True
)

ml_df["balance_gap"] = abs(
    ml_df["train_pos_ratio"] - ml_df["test_pos_ratio"]
)

quality_df = (
    ml_df.groupby(["dataset", "split"])
    .mean(numeric_only=True)
    .reset_index()
)

plt.figure()
sns.barplot(
    data=quality_df,
    x="dataset",
    y="balance_gap",
    hue="split",
    palette=PALETTE
)
plt.title("Split Quality: Train-Test Class Distribution Gap")
plt.ylabel("|Train - Test Positive Ratio|")
plt.xlabel("Dataset")
plt.legend(title="Pipeline")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/split_quality_balance_gap.png")
plt.close()

print("\nPipeline analysis plots saved successfully!")