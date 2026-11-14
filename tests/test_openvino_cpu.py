"""CPU plugin smoke test; no model download or CUDA required."""
import numpy as np
import pytest

def test_openvino_cpu_can_compile_and_infer_small_matmul():
    ov = pytest.importorskip("openvino")
    core = ov.Core()
    assert "CPU" in core.available_devices, core.available_devices

    ops = ov.opset13
    x = ops.parameter([2, 3], ov.Type.f32, name="input")
    weights = ops.constant(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
    result = ops.matmul(x, weights, False, False)
    model = ov.Model([result], [x], "cpu_smoke")
    compiled = core.compile_model(model, "CPU")
    output = compiled([np.array([[1, 1, 1], [2, 0, 1]], dtype=np.float32)])[compiled.output(0)]
    np.testing.assert_allclose(output, [[9, 12], [7, 10]])
