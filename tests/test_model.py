import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from model import ClasificadorECG


@pytest.fixture
def modelo():
    return ClasificadorECG(num_clases=5, tamano_entrada=187)


def test_forma_salida(modelo):
    x = torch.randn(8, 187)
    salida = modelo(x)
    assert salida.shape == (8, 5), f"Se esperaba (8,5), se obtuvo {salida.shape}"


def test_muestra_individual(modelo):
    x = torch.randn(1, 187)
    salida = modelo(x)
    assert salida.shape == (1, 5)


def test_salida_son_logits(modelo):
    x = torch.randn(4, 187)
    salida = modelo(x)
    assert salida.dtype == torch.float32


def test_diferentes_tamanos_batch(modelo):
    for bs in [1, 16, 64]:
        x = torch.randn(bs, 187)
        salida = modelo(x)
        assert salida.shape == (bs, 5)


def test_flujo_gradientes(modelo):
    x = torch.randn(4, 187)
    salida = modelo(x)
    perdida = salida.sum()
    perdida.backward()
    for nombre, param in modelo.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Sin gradiente en {nombre}"
