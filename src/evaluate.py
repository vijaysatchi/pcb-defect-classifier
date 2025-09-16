# Evaluate a trained checkpoint on the test set.
#
# python -m src.evaluate --run-name baseline_lr0.001_bs32

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.dataset import get_dataloaders
from src.models import build_model
from src.utils import PROJECT_ROOT, get_device, upsert_row_csv


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def plot_confusion_matrix(cm, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(epoch_log, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epoch_log["epoch"], epoch_log["train_loss"], label="train")
    axes[0].plot(epoch_log["epoch"], epoch_log["val_loss"], label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - loss"); axes[0].legend()

    axes[1].plot(epoch_log["epoch"], epoch_log["train_accuracy"], label="train")
    axes[1].plot(epoch_log["epoch"], epoch_log["val_accuracy"], label="val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} - accuracy"); axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_run(run_name, splits_dir, batch_size):
    device = get_device()
    checkpoint_path = PROJECT_ROOT / "outputs" / "models" / f"{run_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train this run first, e.g. "
            f"`python -m src.train --model baseline --run-name {run_name}`."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        checkpoint["model_name"],
        num_classes=len(checkpoint["class_names"]),
        in_channels=checkpoint["in_channels"],
        image_size=checkpoint["image_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loaders = get_dataloaders(
        batch_size=batch_size,
        image_size=checkpoint["image_size"],
        grayscale=checkpoint["in_channels"] == 1,
        splits_dir=splits_dir,
        class_names=checkpoint["class_names"],
    )

    labels, preds = collect_predictions(model, loaders["test"], device)
    class_names = checkpoint["class_names"]

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(report).transpose().to_csv(metrics_dir / f"{run_name}_test_report.csv")
    plot_confusion_matrix(cm, class_names, plots_dir / f"{run_name}_confusion_matrix.png",
                           title=f"Confusion matrix - {run_name}")

    epoch_log_path = metrics_dir / f"{run_name}_epoch_log.csv"
    if epoch_log_path.exists():
        epoch_log = pd.read_csv(epoch_log_path)
        plot_training_curves(epoch_log, plots_dir / f"{run_name}_training_curves.png", title=run_name)

    result = {
        "run_name": run_name,
        "model": checkpoint["model_name"],
        "test_accuracy": accuracy,
        "test_precision_macro": precision,
        "test_recall_macro": recall,
        "test_f1_macro": f1,
        "best_val_accuracy": checkpoint["val_accuracy"],
        "best_epoch": checkpoint["epoch"],
    }
    upsert_row_csv(metrics_dir / "comparison.csv", result, key_columns=["run_name"])

    print(f"Test results for '{run_name}':")
    print(f"  accuracy:  {accuracy:.4f}")
    print(f"  precision: {precision:.4f} (macro)")
    print(f"  recall:    {recall:.4f} (macro)")
    print(f"  f1:        {f1:.4f} (macro)")
    print(f"  confusion matrix saved to outputs/plots/{run_name}_confusion_matrix.png")
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main():
    args = build_arg_parser().parse_args()
    evaluate_run(args.run_name, args.splits_dir, args.batch_size)


if __name__ == "__main__":
    main()
