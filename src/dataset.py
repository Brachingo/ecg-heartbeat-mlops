from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class ConjuntoDatosECG(Dataset):
    def __init__(self, ruta_csv: str | Path):
        df = pd.read_csv(ruta_csv, header=None)
        self.X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def obtener_pesos_clases(dataset: ConjuntoDatosECG) -> torch.Tensor:
    conteos = torch.bincount(dataset.y)
    pesos = 1.0 / conteos.float()
    return pesos / pesos.sum()


def obtener_cargadores(
    ruta_entrenamiento: str | Path,
    ruta_test: str | Path,
    batch_size: int = 256,
    proporcion_validacion: float = 0.1,
    num_workers: int = 0,
):
    conjunto_completo = ConjuntoDatosECG(ruta_entrenamiento)
    conjunto_test = ConjuntoDatosECG(ruta_test)

    tamano_val = int(len(conjunto_completo) * proporcion_validacion)
    tamano_entren = len(conjunto_completo) - tamano_val
    conjunto_entren, conjunto_val = torch.utils.data.random_split(
        conjunto_completo, [tamano_entren, tamano_val]
    )

    # Muestreador ponderado para compensar el desbalance de clases
    etiquetas_entren = torch.tensor([conjunto_completo.y[i].item() for i in conjunto_entren.indices])
    conteos = torch.bincount(etiquetas_entren)
    pesos = 1.0 / conteos.float()
    pesos_muestra = pesos[etiquetas_entren]
    muestreador = WeightedRandomSampler(pesos_muestra, num_samples=len(pesos_muestra), replacement=True)

    cargador_entren = DataLoader(conjunto_entren, batch_size=batch_size, sampler=muestreador, num_workers=num_workers)
    cargador_val = DataLoader(conjunto_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cargador_test = DataLoader(conjunto_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return cargador_entren, cargador_val, cargador_test
