# PyTorch Dataset for the processed crops. Normalization + augmentation are
# done with numpy on the raw pixel arrays before converting to a tensor.
# Augmentation is only applied to the train split.

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils import CLASS_NAMES, PROJECT_ROOT

DATASET_MEAN = 0.5
DATASET_STD = 0.5


class PCBDefectDataset(Dataset):
    def __init__(self, split, splits_dir=PROJECT_ROOT / "data" / "splits",
                 image_size=128, grayscale=True, augment=None, class_names=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val' or 'test', got '{split}'")

        self.split = split
        self.splits_dir = Path(splits_dir)
        self.image_size = image_size
        self.grayscale = grayscale
        self.augment = augment if augment is not None else (split == "train")
        self.class_names = class_names or CLASS_NAMES
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        csv_path = self.splits_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Could not find {csv_path}. Run `python -m src.preprocessing` first."
            )
        manifest = pd.read_csv(csv_path)
        manifest = manifest[manifest["class_name"].isin(self.class_names)].reset_index(drop=True)
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def _load_array(self, path):
        full_path = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
        image = Image.open(full_path)
        image = image.convert("L") if self.grayscale else image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(image, dtype=np.uint8)

    def _augment(self, array):
        rng = np.random.default_rng()

        if rng.random() < 0.5:
            array = np.fliplr(array)
        if rng.random() < 0.5:
            array = np.flipud(array)
        if rng.random() < 0.5:
            k = int(rng.integers(1, 4))
            array = np.rot90(array, k=k)

        array = array.astype(np.float32)

        brightness = rng.uniform(-20, 20)
        array = array + brightness

        contrast = rng.uniform(0.85, 1.15)
        mean = array.mean()
        array = (array - mean) * contrast + mean

        noise_std = rng.uniform(0, 6)
        array = array + rng.normal(0, noise_std, size=array.shape)

        array = np.clip(array, 0, 255)
        return array.astype(np.uint8)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        array = self._load_array(row["crop_path"])

        if self.augment:
            array = self._augment(array)

        array = array.astype(np.float32) / 255.0
        array = (array - DATASET_MEAN) / DATASET_STD

        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        else:
            array = array.transpose(2, 0, 1)

        tensor = torch.from_numpy(array.copy()).float()
        label = self.class_to_idx[row["class_name"]]
        return tensor, label


def get_dataloaders(batch_size=32, image_size=128, grayscale=True, num_workers=0,
                     splits_dir=PROJECT_ROOT / "data" / "splits", class_names=None):
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = PCBDefectDataset(
            split=split, splits_dir=splits_dir, image_size=image_size,
            grayscale=grayscale, class_names=class_names,
        )
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, drop_last=False,
        )
    return loaders
