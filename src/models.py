# two CNNs: a small baseline and a deeper one, to compare

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    # 2 conv blocks -> fc classifier
    def __init__(self, num_classes=4, in_channels=1, image_size=128, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        reduced = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32 * reduced * reduced, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class DeeperCNN(nn.Module):
    # 4 conv blocks, channels double each time, batchnorm to keep training stable
    def __init__(self, num_classes=4, in_channels=1, image_size=128, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        reduced = image_size // 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * reduced * reduced, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(name, num_classes=4, in_channels=1, image_size=128):
    name = name.lower()
    if name == "baseline":
        return BaselineCNN(num_classes=num_classes, in_channels=in_channels, image_size=image_size)
    if name == "deeper":
        return DeeperCNN(num_classes=num_classes, in_channels=in_channels, image_size=image_size)
    raise ValueError(f"Unknown model '{name}'. Choose 'baseline' or 'deeper'.")
