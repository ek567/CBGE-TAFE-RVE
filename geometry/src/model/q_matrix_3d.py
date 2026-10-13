from __future__ import annotations
from typing import Any
import numpy as np
from src.utils.math_utils import mat3_to_sym6, normalize

def build_q_matrix_closed_form(a: float, b: float, p: np.ndarray) -> np.ndarray:
    a_value = float(a)
    b_value = float(b)
    if not a_value >= b_value > 0.0:
        raise ValueError(f'Expected a >= b > 0, got a={a_value}, b={b_value}.')
    p_unit = normalize(np.asarray(p, dtype=float))
    identity = np.eye(3, dtype=float)
    return b_value ** 2 * identity + (a_value ** 2 - b_value ** 2) * np.outer(p_unit, p_unit)

def build_q_sym_closed_form(a: float, b: float, p: np.ndarray) -> np.ndarray:
    Q = build_q_matrix_closed_form(a=a, b=b, p=p)
    return mat3_to_sym6(Q)

def build_q_sym_pw85(a: float, b: float, p: np.ndarray) -> np.ndarray:
    a_value = float(a)
    b_value = float(b)
    if not a_value >= b_value > 0.0:
        raise ValueError(f'Expected a >= b > 0, got a={a_value}, b={b_value}.')
    p_unit = normalize(np.asarray(p, dtype=float))
    q = np.zeros(6, dtype=float)
    try:
        import pypw85
    except ImportError as exc:
        raise ImportError('pypw85 is required for build_q_sym_pw85 but is not available.') from exc
    try:
        pypw85.spheroid(b_value, a_value, p_unit, q)
    except Exception as exc:
        raise RuntimeError('pypw85.spheroid failed to build q.') from exc
    if q.shape != (6,):
        raise RuntimeError(f'Unexpected q shape returned by pypw85: {q.shape}.')
    return q

def compare_closed_form_with_pw85(a: float, b: float, p: np.ndarray) -> dict[str, Any]:
    q_closed_form = build_q_sym_closed_form(a=a, b=b, p=p)
    q_pw85 = build_q_sym_pw85(a=a, b=b, p=p)
    difference = q_closed_form - q_pw85
    return {'q_closed_form': q_closed_form, 'q_pw85': q_pw85, 'difference': difference, 'max_abs_diff': float(np.max(np.abs(difference)))}
