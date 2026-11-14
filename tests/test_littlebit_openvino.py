import torch
import openvino as ov
from torch import nn

from quantization.functions import STEBinary
from quantization.modules import LittleBitLinear


def make_tiny_littlebit_linear():
    torch.manual_seed(7)
    layer = nn.Linear(16, 12, bias=True)
    layer.__class__ = LittleBitLinear
    layer.__quant_convert__(
        do_train=True,
        quant_func=STEBinary,
        split_dim=8,
        residual=False,
        min_split_dim=8,
    )
    layer.eval()
    return layer


def test_official_littlebit_factorised_layer_runs_on_cpu():
    layer = make_tiny_littlebit_linear()
    x = torch.randn(2, 3, 16)
    with torch.inference_mode():
        y = layer(x)
    assert y.shape == (2, 3, 12)
    assert torch.isfinite(y).all()


def test_official_littlebit_layer_converts_and_runs_in_openvino_cpu():
    layer = make_tiny_littlebit_linear()
    x = torch.randn(2, 3, 16)
    with torch.inference_mode():
        expected = layer(x).detach().cpu().numpy()

    # Exercise the actual upstream LittleBitLinear graph, not a reimplementation.
    model = ov.convert_model(layer, example_input=x)
    compiled = ov.Core().compile_model(model, "CPU")
    actual = compiled([x.numpy()])[compiled.output(0)]

    assert actual.shape == expected.shape
    assert torch.isfinite(torch.from_numpy(actual)).all()
    max_abs_error = float(abs(actual - expected).max())
    # OpenVINO's CPU graph fusion/reordering differs slightly from eager PyTorch.
    # Keep an explicit absolute error ceiling as well as a modest relative tolerance.
    assert max_abs_error <= 5e-3, f"max absolute error {max_abs_error:.6g} exceeds 5e-3"
    torch.testing.assert_close(
        torch.from_numpy(actual),
        torch.from_numpy(expected),
        rtol=1e-2,
        atol=5e-3,
    )
