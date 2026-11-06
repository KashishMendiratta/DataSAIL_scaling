import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "results/analysis/ml_results_extended.csv"
PLOT_DIR = "results/analysis/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


PALETTE = {
    "random": "#1f77b4",          # blue
    "datasail_scaled_stratified": "#ff7f0e", # orange
    "datasail_full": "#2ca02c",    # green
    #"datasail_scaled_random": "#ff0000" #red
}

SPLIT_ORDER = [
    # Baselines
    "random",
    "datasail_full",

    # Scaled methods
    #"datasail_scaled_random",
    "datasail_scaled_stratified"
    ]

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_FILE)

df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)

# -----------------------------
# HELPER
# -----------------------------
def save_plot(name):
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

# =============================
# AGGREGATED DATA
# =============================
agg_df = df.groupby(["dataset", "split", "model"]).mean(numeric_only=True).reset_index()

# -----------------------------
# 1. MCC vs Split
# -----------------------------
plt.figure()
sns.boxplot(data=agg_df, x="split", y="mcc", hue="model")
plt.title("MCC across Splits (Performance Comparison)")
save_plot("agg_mcc_split.png")

# -----------------------------
# 2. ROC-AUC vs PR-AUC
# -----------------------------
plt.figure()
sns.scatterplot(data=agg_df, x="auc", y="pr_auc", hue="dataset", style="model")
plt.title("ROC-AUC vs PR-AUC")
save_plot("agg_auc_vs_pr_auc.png")

# -----------------------------
# 3. Accuracy vs F1
# -----------------------------
plt.figure()
sns.scatterplot(data=agg_df, x="accuracy", y="f1", hue="dataset")
plt.title("Accuracy vs F1 (Detecting Misleading Performance)")
save_plot("agg_accuracy_vs_f1.png")

# -----------------------------
# 4. Class Distribution
# -----------------------------
plt.figure()
sns.barplot(
    data=agg_df,
    x="dataset",
    y="test_pos_ratio",
    hue="split",
    palette=PALETTE
)
plt.title("Test Set Class Distribution Across Splits")
save_plot("agg_class_distribution.png")

# -----------------------------
# 5. Model Comparison
# -----------------------------
plt.figure()
sns.boxplot(data=agg_df, x="model", y="mcc")
plt.title("Model Comparison (MCC)")
save_plot("agg_model_mcc.png")

# =============================
# PER DATASET PLOTS
# =============================

# -----------------------------
# 6. MCC per Dataset
# -----------------------------
plt.figure()
sns.boxplot(
    data=df,
    x="dataset",
    y="mcc",
    hue="split",
    palette=PALETTE
)
plt.title("MCC per Dataset (Detailed)")
save_plot("detailed_mcc_dataset.png")

# -----------------------------
# 7. Tox21 Failure Plot
# -----------------------------
tox21_df = df[df["dataset"] == "tox21"]

plt.figure()
sns.scatterplot(
    data=tox21_df,
    x="accuracy",
    y="f1",
    hue="split",
    palette=PALETTE
)
plt.title("Tox21: Accuracy vs F1 (Failure Case)")
save_plot("tox21_failure_accuracy_vs_f1.png")

# -----------------------------
# 8. PR-AUC Distribution
# -----------------------------
plt.figure()
sns.boxplot(
    data=tox21_df,
    x="split",
    y="pr_auc",
    palette=PALETTE
)
plt.title("Tox21 PR-AUC Distribution")
save_plot("tox21_pr_auc_distribution.png")

# -----------------------------
# 9. Precision vs Recall
# -----------------------------
plt.figure()
sns.scatterplot(data=df, x="recall", y="precision", hue="dataset")
plt.title("Precision vs Recall")
save_plot("precision_vs_recall.png")

# -----------------------------
# 10. Dataset-wise MCC by Model
# -----------------------------
plt.figure()
sns.boxplot(data=df, x="dataset", y="mcc", hue="model")
plt.title("Dataset-wise MCC by Model")
save_plot("dataset_model_mcc.png")

print("\nAll plots generated successfully!")