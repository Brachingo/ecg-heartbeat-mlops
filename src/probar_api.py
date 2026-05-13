import requests
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

df = pd.read_csv("/data/mitbih_test.csv", header=None)
fila = df.iloc[0]
senal = fila.iloc[:187].tolist()
clase_real = int(fila.iloc[187])

respuesta = requests.post(
    "http://localhost:8000/predecir",
    json={"senal": senal},
)

resultado = respuesta.json()
print(f"Clase real:      {clase_real}")
print(f"Clase predicha:  {resultado['clase_predicha']}")
print(f"Etiqueta:        {resultado['etiqueta']}")
print(f"Probabilidades:")
for clase, prob in resultado["probabilidades"].items():
    print(f"  {clase}: {prob:.4f}")
