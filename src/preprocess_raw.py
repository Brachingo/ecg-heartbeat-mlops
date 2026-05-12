import sys
from pathlib import Path
import os

import pandas as pd
import numpy as np
import wandb
import tempfile

sys.path.insert(0, str(Path(__file__).parent))
from config import ConfigEntrenamiento, NOMBRES_CLASES

np.random.seed(42)

COL_NAMES = [f"t{i}" for i in range(187)] + ["label"]

def cargar_datos(cfg: ConfigEntrenamiento):
    ruta_entren = cfg.directorio_datos / cfg.archivo_entrenamiento
    ruta_test = cfg.directorio_datos / cfg.archivo_test
    df_train = pd.read_csv(ruta_entren, header=None, names=COL_NAMES)
    df_test  = pd.read_csv(ruta_test,  header=None, names=COL_NAMES)
    df_train["label"] = df_train["label"].astype(int)
    df_test["label"]  = df_test["label"].astype(int)
    return df_train, df_test

df_train, df_test = cargar_datos(ConfigEntrenamiento)

run = wandb.init(project="ecg-heartbeat-mlops")

artifact = wandb.Artifact(
    name="mitbih-raw",
    type="dataset",
    description="Dataset ECG sin balancear, con distribución original de clases para balanceo de pesos",
    metadata={
        "muestras_train": len(df_train),
        "muestras_test":  len(df_test),
        "num_clases": 5,
        "caracteristicas": 187
    }
)

with tempfile.TemporaryDirectory() as tmpdir:
    train_path = os.path.join(tmpdir, "train.csv")
    test_path  = os.path.join(tmpdir, "test.csv")
    df_train.to_csv(train_path, index=False, header=False)
    df_test.to_csv(test_path,  index=False, header=False)
    artifact.add_file(train_path, name="train.csv")
    artifact.add_file(test_path,  name="test.csv")
    run.log_artifact(artifact)

run.finish()

