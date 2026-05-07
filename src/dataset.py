from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class ECGDataset(Dataset):
    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path, header=None)
        self.X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_class_weights(dataset: ECGDataset) -> torch.Tensor:
    counts = torch.bincount(dataset.y)
    weights = 1.0 / counts.float()
    return weights / weights.sum()


def make_weighted_sampler(dataset: ECGDataset) -> WeightedRandomSampler:
    class_weights = get_class_weights(dataset)
    sample_weights = class_weights[dataset.y]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_loaders(
    train_path: str | Path,
    test_path: str | Path,
    batch_size: int = 256,
    val_split: float = 0.1,
    num_workers: int = 0,
):
    full_train = ECGDataset(train_path)
    test_ds = ECGDataset(test_path)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_train, [train_size, val_size])

    # Weighted sampler on the train split to handle class imbalance
    # Build a temporary dataset to get labels for the train indices
    train_labels = torch.tensor([full_train.y[i].item() for i in train_ds.indices])
    counts = torch.bincount(train_labels)
    w = 1.0 / counts.float()
    sample_weights = w[train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
