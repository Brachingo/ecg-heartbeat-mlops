import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient

# Patch load_model so tests don't need a trained checkpoint
with patch("api.main.load_model", return_value=MagicMock()):
    from api.main import app

client = TestClient(app)

VALID_SIGNAL = [0.0] * 187


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_classes_endpoint():
    r = client.get("/classes")
    assert r.status_code == 200
    data = r.json()
    assert "0" in data or 0 in data


def test_predict_wrong_length():
    r = client.post("/predict", json={"signal": [0.0] * 100})
    assert r.status_code == 422


def test_predict_empty_signal():
    r = client.post("/predict", json={"signal": []})
    assert r.status_code == 422


def test_predict_valid_signal():
    fake_result = {
        "predicted_class": 0,
        "label": "Normal",
        "probabilities": {"Normal": 0.9, "Supraventricular": 0.05,
                          "Ventricular": 0.02, "Fusion": 0.02, "Unknown": 0.01},
    }
    with patch("api.main.classify_signal", return_value=fake_result):
        r = client.post("/predict", json={"signal": VALID_SIGNAL})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Normal"
    assert "probabilities" in body
