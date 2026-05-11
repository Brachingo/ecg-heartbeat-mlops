# Modelo GRU + CNN para clasificación de latidos ECG (MIT-BIH, 5 clases).
# Basado en el mejor modelo de la asignatura de DL para Series Temporales.

import torch
import torch.nn as nn


class CNNGRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            #nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(46),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            #nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), Usado para el modelo de 3 capas
            nn.Linear(64, 5),
        )

    def forward(self, x):
        if x.dim() == 2:          # (B, 187) → (B, 1, 187)
            x = x.unsqueeze(1)
        x = self.cnn(x)           # (B, 64, 46)
        x = x.permute(0, 2, 1)   # (B, 46, 64)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])