# ECG Heartbeat Classifier — MLOps Final Project

**Author:** Lucas Pérez  
**Master:** Deep Learning — UPM  
**Subject:** MLOps

Classifies ECG heartbeats into 5 arrhythmia categories using a 1D-CNN trained on the MIT-BIH Arrhythmia dataset.

| Class | Label |
|-------|-------|
| 0 | Normal |
| 1 | Supraventricular ectopic beat |
| 2 | Ventricular ectopic beat |
| 3 | Fusion beat |
| 4 | Unknown / paced beat |

---

## Quickstart (local)

### 1. Clone & create environment

```bash
git clone <repo-url>
cd ecg-heartbeat-mlops
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download dataset

Download the **ECG Heartbeat Categorization Dataset** from Kaggle:  
<https://www.kaggle.com/datasets/shayanfazeli/heartbeat>

Place `mitbih_train.csv` and `mitbih_test.csv` inside the `data/` folder.

### 3. Train

```bash
cd src
python train.py --epochs 30 --batch_size 256 --lr 1e-3
```

The best checkpoint is saved to `models/ecg_classifier.pt`.  
Training metrics and the confusion matrix are logged to W&B automatically.

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

Docs available at <http://localhost:8000/docs>

#### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.2, ...]}'   # 187 values
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
# Build & start
docker compose up --build

# API available at http://localhost:8000
```

---

## W&B

- Project: `ecg-heartbeat-mlops`  
- Link: _<add W&B project URL after first run>_

---

## Project structure

```
├── api/           FastAPI service
├── data/          CSV datasets (not tracked by git)
├── models/        Trained checkpoints (not tracked by git)
├── notebooks/     EDA & exploration
├── src/           Training, dataset, model, predict
├── tests/         pytest unit & integration tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
