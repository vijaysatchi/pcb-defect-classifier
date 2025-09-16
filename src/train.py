# Train a model.
#
# python -m src.train --model baseline --epochs 25 --lr 0.001
# python -m src.train --model deeper --epochs 25 --lr 0.001

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.models import build_model
from src.utils import CLASS_NAMES, PROJECT_ROOT, EarlyStopping, append_row_csv, get_device, set_seed


def run_one_epoch(model, loader, criterion, optimizer, device, train):
    model.train(mode=train)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    loaders = get_dataloaders(batch_size=args.batch_size, image_size=args.image_size)
    in_channels = 3 if args.rgb else 1
    model = build_model(args.model, num_classes=len(CLASS_NAMES), in_channels=in_channels, image_size=args.image_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_name = args.run_name or f"{args.model}_lr{args.lr}_bs{args.batch_size}"
    epoch_log_path = PROJECT_ROOT / "outputs" / "metrics" / f"{run_name}_epoch_log.csv"
    if epoch_log_path.exists():
        epoch_log_path.unlink()

    models_dir = PROJECT_ROOT / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"{run_name}_best.pt"
    early_stopping = EarlyStopping(patience=args.patience) if args.patience > 0 else None
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_one_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s"
        )
        append_row_csv(
            epoch_log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "epoch_time": elapsed,
            },
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "in_channels": in_channels,
                    "image_size": args.image_size,
                    "class_names": CLASS_NAMES,
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                },
                best_model_path,
            )

        if early_stopping is not None and early_stopping.step(val_loss):
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    print(f"Best val_accuracy={best_val_acc:.4f}. Checkpoint saved to {best_model_path}")

    append_row_csv(
        PROJECT_ROOT / "outputs" / "metrics" / "training_runs.csv",
        {
            "run_name": run_name,
            "model": args.model,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "epochs_run": epoch,
            "seed": args.seed,
            "best_val_accuracy": best_val_acc,
        },
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "deeper"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
