import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
print(f"Ruta de importación: {sys.path[0]}")
import pytest
import torch

from src.model_cnn import ClasificadorCNN
from src.model_gru import ClasificadorGRU


@pytest.fixture
def modelo_cnn():
    return ClasificadorCNN(num_clases=5, tamano_entrada=187)


@pytest.fixture
def modelo_gru():
    return ClasificadorGRU(num_clases=5, tamano_entrada=187)


@pytest.mark.parametrize("modelo", ["modelo_cnn", "modelo_gru"])
def test_forma_salida(modelo, request):
    modelo = request.getfixturevalue(modelo)
    x = torch.randn(8, 187)
    salida = modelo(x)
    assert salida.shape == (8, 5), f"Se esperaba (8,5), se obtuvo {salida.shape}"


@pytest.mark.parametrize("modelo", ["modelo_cnn", "modelo_gru"])
def test_muestra_individual(modelo, request):
    modelo = request.getfixturevalue(modelo)
    x = torch.randn(1, 187)
    salida = modelo(x)
    assert salida.shape == (1, 5)


@pytest.mark.parametrize("modelo", ["modelo_cnn", "modelo_gru"])
def test_salida_son_logits(modelo, request):
    modelo = request.getfixturevalue(modelo)
    x = torch.randn(4, 187)
    salida = modelo(x)
    assert salida.dtype == torch.float32


@pytest.mark.parametrize("modelo", ["modelo_cnn", "modelo_gru"])
def test_diferentes_tamanos_batch(modelo, request):
    modelo = request.getfixturevalue(modelo)
    for bs in [1, 16, 64]:
        x = torch.randn(bs, 187)
        salida = modelo(x)
        assert salida.shape == (bs, 5)


@pytest.mark.parametrize("modelo", ["modelo_cnn", "modelo_gru"])
def test_flujo_gradientes(modelo, request):
    modelo = request.getfixturevalue(modelo)
    x = torch.randn(4, 187)
    salida = modelo(x)
    perdida = salida.sum()
    perdida.backward()
    for nombre, param in modelo.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Sin gradiente en {nombre}"

# Si todos los tests pasan, se imprimirá un mensaje de éxito
if __name__ == "__main__":
    pytest.main([__file__])
    print("Todos los tests de modelo pasaron exitosamente")