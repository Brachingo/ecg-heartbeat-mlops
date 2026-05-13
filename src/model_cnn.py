import torch
import torch.nn as nn


class ClasificadorCNN(nn.Module):
    """CNN-1D para clasificación de latidos ECG (MIT-BIH, 5 clases)."""

    def __init__(self, num_clases: int = 5, tamano_entrada: int = 187):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            # Bloque 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            # Bloque 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )

        tamano_plano = 128 * (tamano_entrada // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(tamano_plano, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)
