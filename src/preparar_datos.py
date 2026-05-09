"""
Prepara y sube a W&B dos variantes del dataset MIT-BIH como Artifacts:

  - mitbih-original   : CSV originales de Kaggle (desbalanceado)
  - mitbih-balanceado : train con oversampling de clases minoritarias
                        (el test se mantiene igual en ambos)

Uso:
    cd src
    python preparar_datos.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).parent))
from config import ConfigEntrenamiento, NOMBRES_CLASES

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

COL_ETIQUETA = 187  # última columna del CSV


def cargar_csvs(cfg: ConfigEntrenamiento) -> tuple[pd.DataFrame, pd.DataFrame]:
    ruta_entren = cfg.directorio_datos / cfg.archivo_entrenamiento
    ruta_test = cfg.directorio_datos / cfg.archivo_test

    for ruta in (ruta_entren, ruta_test):
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró {ruta}.\n"
                "Descarga el dataset desde https://www.kaggle.com/datasets/shayanfazeli/heartbeat "
                "y coloca los CSV en la carpeta data/"
            )

    train = pd.read_csv(ruta_entren, header=None)
    test = pd.read_csv(ruta_test, header=None)
    return train, test


def distribucion(df: pd.DataFrame) -> dict:
    conteos = df[COL_ETIQUETA].value_counts().sort_index()
    total = len(df)
    return {
        NOMBRES_CLASES[int(clase)]: {
            "muestras": int(conteos[clase]),
            "porcentaje": round(100 * conteos[clase] / total, 2),
        }
        for clase in conteos.index
    }


def tabla_wandb(df: pd.DataFrame, nombre: str) -> wandb.Table:
    dist = distribucion(df)
    tabla = wandb.Table(columns=["Clase", "Muestras", "Porcentaje (%)"])
    for etiqueta, stats in dist.items():
        tabla.add_data(etiqueta, stats["muestras"], stats["porcentaje"])
    return tabla


def balancear(train: pd.DataFrame, semilla: int = 42) -> pd.DataFrame:
    """
    Oversampling con reemplazo de todas las clases minoritarias
    hasta igualar el tamaño de la clase mayoritaria (Normal).
    El test nunca se toca para mantener la evaluación realista.
    """
    n_objetivo = int(train[COL_ETIQUETA].value_counts().max())
    fragmentos = []

    for clase in sorted(train[COL_ETIQUETA].unique()):
        subconjunto = train[train[COL_ETIQUETA] == clase]
        if len(subconjunto) < n_objetivo:
            subconjunto = resample(
                subconjunto,
                replace=True,
                n_samples=n_objetivo,
                random_state=semilla,
            )
        fragmentos.append(subconjunto)

    return pd.concat(fragmentos).sample(frac=1, random_state=semilla).reset_index(drop=True)


def subir_artifact(
    run: wandb.run,
    nombre: str,
    descripcion: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    directorio_tmp: Path,
) -> None:
    directorio_tmp.mkdir(parents=True, exist_ok=True)

    ruta_train = directorio_tmp / f"{nombre}_train.csv"
    ruta_test = directorio_tmp / f"{nombre}_test.csv"
    df_train.to_csv(ruta_train, index=False, header=False)
    df_test.to_csv(ruta_test, index=False, header=False)

    artifact = wandb.Artifact(
        name=nombre,
        type="dataset",
        description=descripcion,
        metadata={
            "muestras_train": len(df_train),
            "muestras_test": len(df_test),
            "num_clases": 5,
            "caracteristicas": 187,
            "fuente": "MIT-BIH Arrhythmia Database (Kaggle)",
        },
    )
    artifact.add_file(str(ruta_train), name="train.csv")
    artifact.add_file(str(ruta_test), name="test.csv")
    run.log_artifact(artifact)
    print(f"  Artifact '{nombre}' subido ({len(df_train):,} train | {len(df_test):,} test)")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    cfg = ConfigEntrenamiento()
    directorio_tmp = cfg.directorio_datos / "tmp_artifacts"

    print("Cargando CSVs...")
    train_orig, test = cargar_csvs(cfg)

    print("\nDistribución original (train):")
    for clase, stats in distribucion(train_orig).items():
        print(f"  {clase:20s}: {stats['muestras']:6,}  ({stats['porcentaje']}%)")

    print("\nBalanceando dataset...")
    train_bal = balancear(train_orig)

    print("\nDistribución balanceada (train):")
    for clase, stats in distribucion(train_bal).items():
        print(f"  {clase:20s}: {stats['muestras']:6,}  ({stats['porcentaje']}%)")

    print("\nIniciando W&B...")
    run = wandb.init(
        project=cfg.wandb_proyecto,
        entity=cfg.wandb_entidad,
        job_type="preparacion-datos",
        name="preparar-datasets",
    )

    # Tablas de distribución para visualizar en W&B
    run.log({
        "distribucion/original": tabla_wandb(train_orig, "original"),
        "distribucion/balanceado": tabla_wandb(train_bal, "balanceado"),
    })

    print("\nSubiendo artifacts a W&B...")

    subir_artifact(
        run=run,
        nombre="mitbih-original",
        descripcion="Dataset MIT-BIH original de Kaggle. Train desbalanceado (~83% Normal).",
        df_train=train_orig,
        df_test=test,
        directorio_tmp=directorio_tmp,
    )

    subir_artifact(
        run=run,
        nombre="mitbih-balanceado",
        descripcion="Dataset MIT-BIH con oversampling en train. Todas las clases igualadas a la mayoritaria.",
        df_train=train_bal,
        df_test=test,
        directorio_tmp=directorio_tmp,
    )

    run.finish()

    # Limpiar CSVs temporales
    for f in directorio_tmp.glob("*.csv"):
        f.unlink()
    directorio_tmp.rmdir()

    print("\nListo. Revisa los artifacts en tu proyecto de W&B.")


if __name__ == "__main__":
    main()
