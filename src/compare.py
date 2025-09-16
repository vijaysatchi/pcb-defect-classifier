# Builds a comparison table + chart across every run that's been evaluated.
#
# python -m src.compare

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import PROJECT_ROOT


def build_comparison(comparison_csv, out_csv, out_plot, training_runs_csv=None):
    if not comparison_csv.exists():
        raise FileNotFoundError(
            f"{comparison_csv} not found. Run `python -m src.evaluate --run-name ...` first."
        )
    df = pd.read_csv(comparison_csv).sort_values("test_accuracy", ascending=False)

    training_runs_csv = training_runs_csv or (comparison_csv.parent / "training_runs.csv")
    if training_runs_csv.exists():
        hyperparams = pd.read_csv(training_runs_csv)[
            ["run_name", "lr", "batch_size", "epochs_run", "seed"]
        ].drop_duplicates("run_name", keep="last")
        df = df.merge(hyperparams, on="run_name", how="left")

    columns = [
        "run_name", "model", "lr", "batch_size", "epochs_run",
        "best_val_accuracy", "test_accuracy", "test_precision_macro",
        "test_recall_macro", "test_f1_macro",
    ]
    display_df = df[[c for c in columns if c in df.columns]].round(4)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    display_df.to_csv(out_csv, index=False)

    metrics = ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=row["run_name"])
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the held-out test set")
    ax.legend()
    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    return display_df


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, default=PROJECT_ROOT / "outputs" / "metrics" / "comparison.csv")
    parser.add_argument("--out-csv", type=Path, default=PROJECT_ROOT / "outputs" / "metrics" / "model_comparison.csv")
    parser.add_argument("--out-plot", type=Path, default=PROJECT_ROOT / "outputs" / "plots" / "model_comparison.png")
    return parser


def main():
    args = build_arg_parser().parse_args()
    table = build_comparison(args.comparison_csv, args.out_csv, args.out_plot)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
