import numpy as np
import pytest
from scripts.littlebit_reference import xnor_popcount_dot

def test_signed_binary_dot_matches_float_reference():
    rng = np.random.default_rng(7)
    a = rng.integers(0, 2, size=(11, 128), dtype=np.int8).astype(bool)
    b = rng.integers(0, 2, size=(11, 128), dtype=np.int8).astype(bool)
    actual = xnor_popcount_dot(a, b)
    expected = ((a * 2 - 1) * (b * 2 - 1)).sum(axis=-1)
    np.testing.assert_array_equal(actual, expected)

def test_rejects_mismatched_inner_dimension():
    with pytest.raises(ValueError):
        xnor_popcount_dot(np.zeros((2, 8), bool), np.zeros((2, 7), bool))

def test_known_values():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert xnor_popcount_dot(a, b) == 0
