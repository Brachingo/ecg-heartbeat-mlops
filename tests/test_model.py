import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from model import ECGClassifier


@pytest.fixture
def model():
    return ECGClassifier(num_classes=5, input_size=187)


def test_output_shape(model):
    x = torch.randn(8, 187)
    out = model(x)
    assert out.shape == (8, 5), f"Expected (8,5), got {out.shape}"


def test_single_sample(model):
    x = torch.randn(1, 187)
    out = model(x)
    assert out.shape == (1, 5)


def test_output_is_logits(model):
    x = torch.randn(4, 187)
    out = model(x)
    # Logits are not constrained to [0,1]
    assert out.dtype == torch.float32


def test_different_batch_sizes(model):
    for bs in [1, 16, 64]:
        x = torch.randn(bs, 187)
        out = model(x)
        assert out.shape == (bs, 5)


def test_grad_flows(model):
    x = torch.randn(4, 187, requires_grad=False)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
