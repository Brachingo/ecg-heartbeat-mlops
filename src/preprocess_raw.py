import pandas as pd
import numpy as np
import wandb
import tempfile
import os

COL_NAMES = [f"t{i}" for i in range(187)] + ["label"]


def cargar_datos(ruta_train, ruta_test):
    df_train = pd.read_csv(ruta_train, header=None, names=COL_NAMES)
    df_test  = pd.read_csv(ruta_test,  header=None, names=COL_NAMES)
    df_train["label"] = df_train["label"].astype(int)
    df_test["label"]  = df_test["label"].astype(int)
    return df_train, df_test

df_train, df_test = cargar_datos("data/mitbih_train.csv", "data/mitbih_test.csv")

run = wandb.init(project="ecg-classification")

artifact = wandb.Artifact(
    name="ecg-raw",
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
    train_path = os.path.join(tmpdir, "train_raw.csv")
    test_path  = os.path.join(tmpdir, "test_raw.csv")
    df_train.to_csv(train_path, index=False, header=False)
    df_test.to_csv(test_path,  index=False, header=False)
    artifact.add_file(train_path, name="train_raw.csv")
    artifact.add_file(test_path,  name="test_raw.csv")
    run.log_artifact(artifact)

run.finish()

