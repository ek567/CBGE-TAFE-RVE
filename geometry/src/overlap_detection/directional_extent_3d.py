from __future__ import annotations
import numpy as np

def spheroid_directional_extent(a: float, b: float, p: np.ndarray, u: np.ndarray) -> float:
    s = float(np.dot(p, u))
    val = b * b + (a * a - b * b) * (s * s)
    if val <= 0.0:
        val = max(val, 0.0)
    return float(np.sqrt(val))

def definitely_separated_by_directional_extent(a_i: float, b_i: float, p_i: np.ndarray, a_j: float, b_j: float, p_j: np.ndarray, r_ij: np.ndarray, *, sep_tol_abs: float=0.0) -> bool:
    d2 = float(np.dot(r_ij, r_ij))
    if d2 <= 0.0:
        return False
    d = float(np.sqrt(d2))
    u = r_ij / d
    e_i = spheroid_directional_extent(a_i, b_i, p_i, u)
    e_j = spheroid_directional_extent(a_j, b_j, p_j, u)
    return d > e_i + e_j + float(sep_tol_abs)
