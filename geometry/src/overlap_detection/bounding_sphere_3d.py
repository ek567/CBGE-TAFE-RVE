from __future__ import annotations
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.periodic.mic_3d import minimum_image_vector

def candidate_by_bounding_sphere(a_i: float, a_j: float, r_ij: np.ndarray, padding: float=0.0) -> bool:
    a_i_value = float(a_i)
    a_j_value = float(a_j)
    padding_value = float(padding)
    r_ij_array = np.asarray(r_ij, dtype=float)
    if a_i_value <= 0.0:
        raise ValueError(f'Expected a_i > 0, got {a_i_value}.')
    if a_j_value <= 0.0:
        raise ValueError(f'Expected a_j > 0, got {a_j_value}.')
    if padding_value < 0.0:
        raise ValueError(f'Expected padding >= 0, got {padding_value}.')
    if r_ij_array.shape != (3,):
        raise ValueError(f'Expected r_ij.shape == (3,), got {r_ij_array.shape}.')
    radius_sum = a_i_value + a_j_value + padding_value
    distance_sq = float(np.dot(r_ij_array, r_ij_array))
    return bool(distance_sq <= radius_sum * radius_sum)

def bounding_sphere_intersection(particle_i: EllipsoidParticle3D, particle_j: EllipsoidParticle3D, box: np.ndarray | None=None, use_mic: bool=False, padding: float=0.0) -> bool:
    if not isinstance(particle_i, EllipsoidParticle3D):
        raise ValueError(f'particle_i must be EllipsoidParticle3D, got {type(particle_i)!r}.')
    if not isinstance(particle_j, EllipsoidParticle3D):
        raise ValueError(f'particle_j must be EllipsoidParticle3D, got {type(particle_j)!r}.')
    if use_mic:
        if box is None:
            raise ValueError('box must be provided when use_mic=True.')
        r_ij = minimum_image_vector(c_i=particle_i.c, c_j=particle_j.c, box=np.asarray(box, dtype=float))
    else:
        r_ij = np.asarray(particle_j.c - particle_i.c, dtype=float)
    if r_ij.shape != (3,):
        raise ValueError(f'Expected r_ij.shape == (3,), got {r_ij.shape}.')
    return candidate_by_bounding_sphere(a_i=particle_i.a, a_j=particle_j.a, r_ij=r_ij, padding=padding)
