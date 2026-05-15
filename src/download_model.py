import argparse
from pathlib import Path
import wandb
import warnings
from config import ConfigEntrenamiento

DIR_RAIZ = Path(__file__).parent
cfg = ConfigEntrenamiento()


def descargar_modelo(model: str) -> Path:
    ruta_destino = cfg.directorio_modelos / cfg.nombre_modelo

    cfg.directorio_modelos.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()

    # Prueba si el modelo existe en W&B y si no usar la versión por defecto mostrando una advertencia
    nombre_artifact = model if model else cfg.artefacto_modelo
    try:
        artifact = api.artifact(f"{cfg.wandb_entidad}/{cfg.wandb_proyecto}/{nombre_artifact}", type="model")
    except wandb.errors.CommError:
        warnings.warn(
            f"No se encontró el artefacto '{nombre_artifact}' en W&B. "
            f"Se usará la primera versión disponible: '{cfg.artefacto_modelo}'.",
            UserWarning,
            stacklevel=2,
        )

        artifact = api.artifact(f"{cfg.wandb_entidad}/{cfg.wandb_proyecto}/{cfg.artefacto_modelo}", type="model")

    # Descarga en un directorio temporal y copia al destino final
    directorio_descarga = Path(artifact.download())
    archivos = list(directorio_descarga.glob("*.pt"))
    if not archivos:
        raise FileNotFoundError(f"No se encontró ningún archivo .pt en el artefacto descargado: {directorio_descarga}")

    archivos[0].replace(ruta_destino)
    return ruta_destino


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga un modelo ECG desde W&B")
    parser.add_argument(
        "--model",
        default = None,
        help="Versión del modelo a descargar según el dataset usado",
    )
    args = parser.parse_args()
    descargar_modelo(args.model)