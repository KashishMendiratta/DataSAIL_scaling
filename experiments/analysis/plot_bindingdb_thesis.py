"""Create the accepted BindingDB thesis tables and figures from consolidated CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "linear_100k": "Linear 100k", "linear_500k": "Linear 500k",
    "hashed_dual_encoder_100k": "Hashed dual encoder", "sequence_cnn_100k": "Sequence CNN",
    "esm2_8m_100k": "ESM-2 8M",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(args.table_dir / "split_summary.csv")
    models = pd.read_csv(args.table_dir / "model_test_summary.csv")
    leakage = pd.read_csv(args.table_dir / "leakage_summary.csv")

    splits.to_csv(args.output_dir / "table_split_scale_retention.csv", index=False)
    models.to_csv(args.output_dir / "table_model_by_regime.csv", index=False)

    plot = leakage.copy()
    plot["series"] = plot["regime"] + " · " + plot["entity"] + " · " + plot["query_split"]
    ligand = plot["entity"].eq("ligand")
    plot.loc[ligand, "series"] += plot.loc[ligand, "metric"].map(
        lambda value: " · chiral" if str(value).endswith("_chiral") else " · non-chiral"
    )
    plot = plot.sort_values(["regime", "entity", "query_split"])
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(plot["series"], plot["mean"], color="#2878B5")
    ax.scatter(plot["max"], plot["series"], color="#C82423", label="maximum", zorder=3)
    ax.set(xlabel="Nearest-training similarity", xlim=(0, 1.02), title="BindingDB cross-split leakage")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(args.output_dir / "figure_leakage_distributions.pdf")
    fig.savefig(args.output_dir / "figure_leakage_distributions.png", dpi=220)
    plt.close(fig)

    runtime = models.drop_duplicates(["model", "regime"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    for ax, regime in zip(axes, ("C1e", "C1f", "C2")):
        data = runtime[runtime.regime == regime]
        for _, row in data.iterrows():
            ax.scatter(row["regime_runtime_seconds"], row["rmse"], s=55)
            ax.annotate(MODEL_LABELS.get(row["model"], row["model"]),
                        (row["regime_runtime_seconds"], row["rmse"]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set(title=regime, xlabel="Regime runtime (s)", ylabel="Test RMSE")
    fig.suptitle("BindingDB predictive performance versus runtime")
    fig.tight_layout(); fig.savefig(args.output_dir / "figure_performance_vs_runtime.pdf")
    fig.savefig(args.output_dir / "figure_performance_vs_runtime.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
