# Clasificador de Latidos ECG — Proyecto Final MLOps

**Autor:** Lucas Pérez  
**Máster:** Deep Learning — UPM  
**Asignatura:** MLOps

Clasifica latidos ECG en 5 categorías de arritmia usando una CNN-1D entrenada sobre el dataset MIT-BIH Arrhythmia.

| Clase | Etiqueta |
|-------|----------|
| 0 | Normal |
| 1 | Latido ectópico supraventricular |
| 2 | Latido ectópico ventricular |
| 3 | Latido de fusión |
| 4 | Desconocido / marcapasos |

---

## Inicio rápido (local)

### 1. Clonar el repositorio y crear el entorno

```bash
git clone <url-del-repo>
cd ecg-heartbeat-mlops
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Descargar el dataset

Descarga el **ECG Heartbeat Categorization Dataset** de Kaggle:  
<https://www.kaggle.com/datasets/shayanfazeli/heartbeat>

Coloca `mitbih_train.csv` y `mitbih_test.csv` dentro de la carpeta `data/`.

### 3. Entrenar el modelo

```bash
cd src
python train.py --epocas 30 --batch_size 256 --lr 1e-3
```

El mejor checkpoint se guarda en `models/ecg_clasificador.pt`.  
Las métricas y la matriz de confusión se registran automáticamente en W&B.

### 4. Lanzar la API

```bash
uvicorn api.main:app --reload
```

Documentación interactiva disponible en <http://localhost:8000/docs>

#### Ejemplo de petición

```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{"senal": [0.1, 0.2, ...]}'   # 187 valores
```

### 5. Ejecutar los tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
# Construir e iniciar
docker compose up --build

# La API estará disponible en http://localhost:8000
```

---

## W&B

- Proyecto: `ecg-heartbeat-mlops`  
- Enlace: _<añadir URL del proyecto W&B tras el primer entrenamiento>_

---

## Estructura del proyecto

```
├── api/           Servicio FastAPI
├── data/          Datasets CSV (no versionados en git)
├── models/        Checkpoints entrenados (no versionados en git)
├── notebooks/     Exploración y análisis
├── src/           Entrenamiento, dataset, modelo, predicción
├── tests/         Tests unitarios e integración con pytest
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
