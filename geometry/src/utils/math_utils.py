from __future__ import annotations
import numpy as np

def normalize(v: np.ndarray) -> np.ndarray:
    array = np.asarray(v, dtype=float)
    if array.shape != (3,):
        raise ValueError(f'Expected shape (3,), got {array.shape}.')
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError('Cannot normalize a zero vector.')
    return array / norm

def tangent_projector(p: np.ndarray) -> np.ndarray:
    p_unit = normalize(p)
    identity = np.eye(3, dtype=float)
    return identity - np.outer(p_unit, p_unit)

def sym6_to_mat3(q: np.ndarray) -> np.ndarray:
    array = np.asarray(q, dtype=float)
    if array.shape != (6,):
        raise ValueError(f'Expected shape (6,), got {array.shape}.')
    q11, q12, q13, q22, q23, q33 = array
    return np.array([[q11, q12, q13], [q12, q22, q23], [q13, q23, q33]], dtype=float)

def mat3_to_sym6(Q: np.ndarray) -> np.ndarray:
    array = np.asarray(Q, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f'Expected shape (3, 3), got {array.shape}.')
    return np.array([array[0, 0], array[0, 1], array[0, 2], array[1, 1], array[1, 2], array[2, 2]], dtype=float)

def safe_dot(a: np.ndarray, b: np.ndarray) -> float:
    array_a = np.asarray(a, dtype=float)
    array_b = np.asarray(b, dtype=float)
    if array_a.shape != (3,):
        raise ValueError(f'Expected a.shape == (3,), got {array_a.shape}.')
    if array_b.shape != (3,):
        raise ValueError(f'Expected b.shape == (3,), got {array_b.shape}.')
    return float(np.dot(array_a, array_b))
