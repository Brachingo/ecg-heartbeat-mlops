import torch
import torch.nn as nn


class ECGClassifier(nn.Module):
    """1D-CNN for ECG heartbeat classification (MIT-BIH, 5 classes)."""

    def __init__(self, num_classes: int = 5, input_size: int = 187):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        flat_size = 128 * (input_size // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 187)  →  add channel dim  →  (B, 1, 187)
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)
