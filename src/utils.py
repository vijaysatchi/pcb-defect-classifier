import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = ["missing_hole", "spur", "short", "open_circuit"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    # use GPU if we have one, otherwise fall back to CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    # stop training if val_loss hasn't improved in `patience` epochs
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def append_row_csv(csv_path, row):
    # add one row to a csv, creating it (with header) if it doesn't exist yet
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    write_header = not csv_path.exists()
    row_df.to_csv(csv_path, mode="a", header=write_header, index=False)


def upsert_row_csv(csv_path, row, key_columns):
    # replace the row matching key_columns (if any) and add the new one
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        mask = pd.Series(True, index=existing.index)
        for col in key_columns:
            mask &= existing[col].astype(str) == str(row[col])
        existing = existing[~mask]
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(csv_path, index=False)
