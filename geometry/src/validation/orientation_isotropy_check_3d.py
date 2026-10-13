from __future__ import annotations
from typing import Any
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.model.orientation_targets_3d import orientation_target_tensor_3d, resolve_orientation_tolerance
from src.utils.math_utils import normalize

def _validate_particles(particles: list[EllipsoidParticle3D]) -> None:
    if not isinstance(particles, list):
        raise TypeError('particles must be a list of EllipsoidParticle3D.')
    if len(particles) == 0:
        raise ValueError('particles must be a non-empty list.')
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise TypeError(f'particles[{idx}] must be an EllipsoidParticle3D, got {type(particle).__name__}.')

def compute_orientation_tensor(particles: list[EllipsoidParticle3D]) -> np.ndarray:
    _validate_particles(particles)
    tensor = np.zeros((3, 3), dtype=float)
    for particle in particles:
        p_unit = normalize(particle.p)
        tensor += np.outer(p_unit, p_unit)
    tensor /= float(len(particles))
    if tensor.shape != (3, 3):
        raise RuntimeError('Orientation tensor must have shape (3, 3).')
    return tensor

def orientation_tensor_eigenvalues(orientation_tensor: np.ndarray) -> np.ndarray:
    A = np.asarray(orientation_tensor, dtype=float)
    if A.shape != (3, 3):
        raise ValueError('orientation_tensor must have shape (3, 3).')
    A = 0.5 * (A + A.T)
    eigvals = np.linalg.eigvalsh(A)
    if eigvals.shape != (3,):
        raise RuntimeError('Eigenvalues must have shape (3,).')
    return eigvals

def check_orientation_isotropy(particles: list[EllipsoidParticle3D], tolerance: float | None=None, *, orientation_target: str='3d-isotropic') -> dict[str, Any]:
    _validate_particles(particles)
    tol = resolve_orientation_tolerance(orientation_target, tolerance)
    A = compute_orientation_tensor(particles)
    eigvals = orientation_tensor_eigenvalues(A)
    target = 1.0 / 3.0
    deviations = np.abs(eigvals - target)
    isotropic_max_abs_deviation = float(np.max(deviations))
    is_isotropic = bool(np.all(deviations < tol))
    A_star = orientation_target_tensor_3d(orientation_target)
    residual = A - A_star
    if orientation_target == '3d-isotropic':
        max_abs_deviation = isotropic_max_abs_deviation
        is_target_aligned = is_isotropic
    else:
        residual_eigenvalues = orientation_tensor_eigenvalues(residual)
        max_abs_deviation = float(np.max(np.abs(residual_eigenvalues)))
        is_target_aligned = bool(max_abs_deviation < tol)
    return {'num_particles': int(len(particles)), 'orientation_tensor': A, 'eigenvalues': eigvals, 'target_eigenvalue': target, 'orientation_target': orientation_target, 'target_tensor': A_star, 'deviation_tensor': residual, 'spectral_norm': max_abs_deviation, 'frobenius_norm': float(np.linalg.norm(residual, ord='fro')), 'max_abs_deviation': max_abs_deviation, 'isotropic_max_abs_deviation': isotropic_max_abs_deviation, 'is_isotropic': is_isotropic, 'is_target_aligned': is_target_aligned, 'tolerance': tol}

def format_orientation_isotropy_report(result: dict[str, Any]) -> str:
    A = np.asarray(result['orientation_tensor'], dtype=float)
    eigvals = np.asarray(result['eigenvalues'], dtype=float)
    lines = ['=== Orientation Target / Isotropy Check ===', f"num_particles      : {int(result['num_particles'])}", f"target_eigenvalue  : {float(result['target_eigenvalue']):.12f}", f"tolerance          : {float(result['tolerance']):.12f}", 'orientation_tensor :', np.array2string(A, precision=8, suppress_small=False), f'eigenvalues        : {np.array2string(eigvals, precision=8, suppress_small=False)}', f"max_abs_deviation  : {float(result['max_abs_deviation']):.12f}", f"is_isotropic       : {bool(result['is_isotropic'])}", f"orientation_target : {result.get('orientation_target', '3d-isotropic')}", f"target_spectral_norm: {float(result['max_abs_deviation']):.12f}", f"target_frobenius_norm: {float(result.get('frobenius_norm', np.nan)):.12f}", f"is_target_aligned  : {bool(result.get('is_target_aligned', result['is_isotropic']))}"]
    return '\n'.join(lines)
