from __future__ import annotations
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D

def _validate_vec3(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f'Expected {name}.shape == (3,), got {array.shape}.')
    return array

def _validate_box(box: np.ndarray) -> np.ndarray:
    box_array = _validate_vec3('box', box)
    if np.any(box_array <= 0.0):
        raise ValueError(f'All box lengths must be positive, got {box_array}.')
    return box_array

def minimum_image_vector(c_i: np.ndarray, c_j: np.ndarray, box: np.ndarray) -> np.ndarray:
    c_i_array = _validate_vec3('c_i', c_i)
    c_j_array = _validate_vec3('c_j', c_j)
    box_array = _validate_box(box)
    delta = c_j_array - c_i_array
    r_ij = delta - box_array * np.rint(delta / box_array)
    if r_ij.shape != (3,):
        raise ValueError(f'Expected minimum-image vector shape (3,), got {r_ij.shape}.')
    return np.asarray(r_ij, dtype=float)

def wrap_position(c: np.ndarray, box: np.ndarray) -> np.ndarray:
    c_array = _validate_vec3('c', c)
    box_array = _validate_box(box)
    wrapped = np.mod(c_array, box_array)
    if wrapped.shape != (3,):
        raise ValueError(f'Expected wrapped position shape (3,), got {wrapped.shape}.')
    return np.asarray(wrapped, dtype=float)

def wrap_particles_in_box(particles: list[EllipsoidParticle3D], box: np.ndarray) -> list[EllipsoidParticle3D]:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    box_array = _validate_box(box)
    wrapped_particles: list[EllipsoidParticle3D] = []
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise ValueError(f'particles[{idx}] must be an EllipsoidParticle3D, got {type(particle)!r}.')
        particle_copy = particle.clone()
        particle_copy.c = wrap_position(particle_copy.c, box_array)
        wrapped_particles.append(particle_copy)
    return wrapped_particles
