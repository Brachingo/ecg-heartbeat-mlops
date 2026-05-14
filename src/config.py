from dataclasses import dataclass
from pathlib import Path

DIR_RAIZ = Path(__file__).parent.parent

@dataclass
class ConfigEntrenamiento:
    # Datos
    directorio_datos: Path = DIR_RAIZ / "data"
    archivo_entrenamiento: str = "mitbih_train.csv"
    archivo_test: str = "mitbih_test.csv"

    # Modelo
    num_clases: int = 5
    tamano_entrada: int = 187

    # Entrenamiento
    arq: str = "CNN"  # o "GRU"
    epocas: int = 30
    batch_size: int = 256
    tasa_aprendizaje: float = 1e-3
    weight_decay: float = 1e-4
    proporcion_validacion: float = 0.1

    # Guardado del modelo
    directorio_modelos: Path = DIR_RAIZ / "models"
    nombre_modelo: str = "ecg_clasificador.pt"

    # W&B
    wandb_proyecto: str = "ecg-heartbeat-mlops"
    wandb_entidad: str  = "brachi-upm"
    nombre_run: str = "run"
    artefacto_dataset: str = "mitbih:latest"
    artefacto_modelo: str = "ecg_model:v0"

NOMBRES_CLASES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Desconocido",
}
