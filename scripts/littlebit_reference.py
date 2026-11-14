"""Tiny NumPy reference for signed binary dot products.

This is a mathematical microbenchmark/reference, NOT Samsung's implementation,
a checkpoint loader, or a complete LittleBit inference engine.
"""
from __future__ import annotations
import numpy as np

def xnor_popcount_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot product of vectors encoded as booleans (+1=True, -1=False).

    Returns the exact integer dot product for the corresponding +/-1 vectors.
    Uses NumPy equality counting as a portable correctness reference; it does
    not claim to use hardware POPCNT or an optimised packed-bit kernel.
    """
    a = np.asarray(a, dtype=np.bool_)
    b = np.asarray(b, dtype=np.bool_)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("last dimensions must match")
    matches = np.count_nonzero(a == b, axis=-1)
    return 2 * matches - a.shape[-1]
