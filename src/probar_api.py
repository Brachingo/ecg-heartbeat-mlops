import requests
import pandas as pd
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

df = pd.read_csv(Path(__file__).parent.parent / "data" / "mitbih_test.csv", header=None)
entry = random.randint(0, 21982)
print(f"Fila: {entry} de test dataset\n")
fila = df.iloc[entry]
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
