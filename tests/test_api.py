import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient

# Se parchea cargar_modelo para que los tests no necesiten un checkpoint entrenado
with patch("api.main.cargar_modelo", return_value=MagicMock()):
    from api.main import app

cliente = TestClient(app)

SENAL_VALIDA = [0.0] * 187


def test_health( ):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_raiz():
    r = cliente.get("/")
    assert r.status_code == 200


def test_endpoint_clases():
    r = cliente.get("/clases")
    assert r.status_code == 200
    datos = r.json()
    assert "0" in datos or 0 in datos


def test_predecir_longitud_incorrecta():
    r = cliente.post("/predecir", json={"senal": [0.0] * 100})
    assert r.status_code == 422


def test_predecir_senal_vacia():
    r = cliente.post("/predecir", json={"senal": []})
    assert r.status_code == 422


def test_predecir_senal_valida():
    resultado_falso = {
        "clase_predicha": 0,
        "etiqueta": "Normal",
        "probabilidades": {
            "Normal": 0.9,
            "Supraventricular": 0.05,
            "Ventricular": 0.02,
            "Fusion": 0.02,
            "Desconocido": 0.01,
        },
    }
    with patch("api.main.clasificar_senal", return_value=resultado_falso), \
         patch("api.main._modelo", MagicMock()):
        r = cliente.post("/predecir", json={"senal": SENAL_VALIDA})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["etiqueta"] == "Normal"
    assert "probabilidades" in cuerpo
