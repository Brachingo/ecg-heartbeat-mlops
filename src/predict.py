from pathlib import Path

import numpy as np
import torch

from config import CLASS_NAMES, TrainConfig
from model import ECGClassifier


def load_model(model_path: str | Path | None = None, device: str | None = None) -> ECGClassifier:
    cfg = TrainConfig()
    path = Path(model_path) if model_path else cfg.model_dir / cfg.model_name
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = ECGClassifier(num_classes=cfg.num_classes, input_size=cfg.input_size)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.to(dev)
    model.eval()
    return model


def predict(signal: list[float], model: ECGClassifier | None = None) -> dict:
    """
    Classify a single ECG heartbeat.

    Parameters
    ----------
    signal : list of 187 float values (one heartbeat window)
    model  : optional pre-loaded model (avoids reloading on every call)

    Returns
    -------
    dict with keys: predicted_class (int), label (str), probabilities (list[float])
    """
    if model is None:
        model = load_model()

    device = next(model.parameters()).device
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 187)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    predicted_class = int(np.argmax(probs))
    return {
        "predicted_class": predicted_class,
        "label": CLASS_NAMES[predicted_class],
        "probabilities": {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
    }
