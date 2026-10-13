from __future__ import annotations
import math
from types import MappingProxyType
import numpy as np
ORIENTATION_TARGET_NAMES = ('1d-x', '2d-xz', '3d-isotropic')
DEFAULT_ORIENTATION_TARGET = '3d-isotropic'
ORIENTATION_TOLERANCE_DEFAULTS = MappingProxyType({'1d-x': 0.05, '2d-xz': 0.05, '3d-isotropic': 0.015})

def validate_orientation_target(orientation_target: str) -> str:
    if not isinstance(orientation_target, str) or orientation_target not in ORIENTATION_TARGET_NAMES:
        raise ValueError(f'orientation_target must be one of {ORIENTATION_TARGET_NAMES}, got {orientation_target!r}.')
    return orientation_target

def resolve_orientation_tolerance(orientation_target: str, orientation_tol: float | None=None) -> float:
    name = validate_orientation_target(orientation_target)
    tolerance = float(ORIENTATION_TOLERANCE_DEFAULTS[name] if orientation_tol is None else orientation_tol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f'orientation_tol must be finite and >= 0, got {tolerance}.')
    return tolerance

def orientation_target_tensor_3d(orientation_target: str=DEFAULT_ORIENTATION_TARGET) -> np.ndarray:
    name = validate_orientation_target(orientation_target)
    if name == '1d-x':
        return np.diag([1.0, 0.0, 0.0])
    if name == '2d-xz':
        return np.diag([0.5, 0.0, 0.5])
    return 1.0 / 3.0 * np.eye(3, dtype=float)
