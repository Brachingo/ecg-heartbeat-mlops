from pathlib import Path

import numpy as np
import torch

from config import NOMBRES_CLASES, ConfigEntrenamiento
from src.model_cnn import ClasificadorCNN
from src.model_gru import ClasificadorGRU


def cargar_modelo(ruta_modelo: str | Path | None = None, dispositivo: str | None = None):
    cfg = ConfigEntrenamiento()
    ruta = Path(ruta_modelo) if ruta_modelo else cfg.directorio_modelos / cfg.nombre_modelo
    dev = torch.device(dispositivo or ("cuda" if torch.cuda.is_available() else "cpu"))

    modelo = ClasificadorCNN(num_clases=cfg.num_clases, tamano_entrada=cfg.tamano_entrada) if cfg.arq == "CNN" else ClasificadorGRU(num_clases=cfg.num_clases, tamano_entrada=cfg.tamano_entrada)
    modelo.load_state_dict(torch.load(ruta, map_location=dev))
    modelo.to(dev)
    modelo.eval()
    return modelo


def predecir(senal: list[float], modelo) -> dict:
    """
    Clasifica un único latido ECG.

    Parámetros
    ----------
    senal  : lista de 187 valores float (una ventana de un latido normalizado)
    modelo : modelo precargado opcional (evita recargarlo en cada llamada)

    Devuelve
    --------
    dict con claves: clase_predicha (int), etiqueta (str), probabilidades (dict)
    """
    if modelo is None:
        modelo = cargar_modelo()

    dispositivo = next(modelo.parameters()).device
    x = torch.tensor(senal, dtype=torch.float32).unsqueeze(0).to(dispositivo)  # (1, 187)

    with torch.no_grad():
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    clase_predicha = int(np.argmax(probs))
    return {
        "clase_predicha": clase_predicha,
        "etiqueta": NOMBRES_CLASES[clase_predicha],
        "probabilidades": {NOMBRES_CLASES[i]: round(p, 4) for i, p in enumerate(probs)},
    }
