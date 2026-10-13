from __future__ import annotations
from typing import Any
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.model.orientation_targets_3d import orientation_target_tensor_3d
from src.utils.math_utils import normalize

def _validate_particles(particles: list[EllipsoidParticle3D]) -> None:
    if not isinstance(particles, list):
        raise TypeError('particles must be a list of EllipsoidParticle3D.')
    if len(particles) == 0:
        raise ValueError('particles must be a non-empty list.')
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise TypeError(f'particles[{idx}] must be an EllipsoidParticle3D, got {type(particle).__name__}.')

def target_isotropic_orientation_tensor_3d() -> np.ndarray:
    return orientation_target_tensor_3d()

def compute_orientation_tensor_3d(particles: list[EllipsoidParticle3D]) -> np.ndarray:
    _validate_particles(particles)
    tensor = np.zeros((3, 3), dtype=float)
    n_particles = len(particles)
    for particle in particles:
        p_i = normalize(particle.p)
        tensor += np.outer(p_i, p_i)
    tensor /= float(n_particles)
    if tensor.shape != (3, 3):
        raise RuntimeError(f'Orientation tensor must have shape (3, 3), got {tensor.shape}.')
    return tensor

def compute_orientation_deviation_tensor_3d(particles: list[EllipsoidParticle3D], *, orientation_target: str='3d-isotropic') -> np.ndarray:
    A_N = compute_orientation_tensor_3d(particles)
    A_star = orientation_target_tensor_3d(orientation_target)
    deviation = A_N - A_star
    if deviation.shape != (3, 3):
        raise RuntimeError(f'Deviation tensor must have shape (3, 3), got {deviation.shape}.')
    return deviation

def compute_orientation_regularization_energy(particles: list[EllipsoidParticle3D], *, orientation_target: str='3d-isotropic') -> float:
    deviation = compute_orientation_deviation_tensor_3d(particles, orientation_target=orientation_target)
    energy = 0.5 * float(np.sum(deviation * deviation))
    return float(energy)

def compute_orientation_regularization_gradients(particles: list[EllipsoidParticle3D], *, orientation_target: str='3d-isotropic') -> list[np.ndarray]:
    _validate_particles(particles)
    n_particles = len(particles)
    deviation = compute_orientation_deviation_tensor_3d(particles, orientation_target=orientation_target)
    gradients: list[np.ndarray] = []
    for particle in particles:
        p_i = normalize(particle.p)
        projector = np.eye(3, dtype=float) - np.outer(p_i, p_i)
        grad_i = projector @ (2.0 / float(n_particles) * (deviation @ p_i))
        grad_i = np.asarray(grad_i, dtype=float)
        if grad_i.shape != (3,):
            raise RuntimeError(f'Orientation regularization gradient must have shape (3,), got {grad_i.shape}.')
        gradients.append(grad_i)
    return gradients

def orientation_regularization_summary(particles: list[EllipsoidParticle3D], *, orientation_target: str='3d-isotropic') -> dict[str, Any]:
    _validate_particles(particles)
    A_N = compute_orientation_tensor_3d(particles)
    A_star = orientation_target_tensor_3d(orientation_target)
    deviation = A_N - A_star
    energy = compute_orientation_regularization_energy(particles, orientation_target=orientation_target)
    eigenvalues = np.linalg.eigvalsh(0.5 * (A_N + A_N.T))
    return {'num_particles': int(len(particles)), 'orientation_target': orientation_target, 'orientation_tensor': A_N, 'target_tensor': A_star, 'deviation_tensor': deviation, 'frobenius_norm': float(np.linalg.norm(deviation, ord='fro')), 'energy': float(energy), 'eigenvalues': np.asarray(eigenvalues, dtype=float)}
