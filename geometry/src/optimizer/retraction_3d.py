from __future__ import annotations
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.periodic.mic_3d import wrap_position
from src.utils.math_utils import normalize

def retract_orientation(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    p_array = np.asarray(p, dtype=float)
    dp_array = np.asarray(dp, dtype=float)
    if p_array.shape != (3,):
        raise ValueError(f'Expected p.shape == (3,), got {p_array.shape}.')
    if dp_array.shape != (3,):
        raise ValueError(f'Expected dp.shape == (3,), got {dp_array.shape}.')
    return normalize(p_array + dp_array)

def apply_particle_trial_step(particle: EllipsoidParticle3D, direction_c: np.ndarray, direction_p: np.ndarray, alpha: float, box: np.ndarray | None=None, wrap_positions: bool=False) -> EllipsoidParticle3D:
    if not isinstance(particle, EllipsoidParticle3D):
        raise ValueError(f'particle must be EllipsoidParticle3D, got {type(particle)!r}.')
    direction_c_array = np.asarray(direction_c, dtype=float)
    direction_p_array = np.asarray(direction_p, dtype=float)
    if direction_c_array.shape != (3,):
        raise ValueError(f'Expected direction_c.shape == (3,), got {direction_c_array.shape}.')
    if direction_p_array.shape != (3,):
        raise ValueError(f'Expected direction_p.shape == (3,), got {direction_p_array.shape}.')
    alpha_value = float(alpha)
    new_particle = particle.clone()
    new_c = np.asarray(new_particle.c + alpha_value * direction_c_array, dtype=float)
    new_p = retract_orientation(new_particle.p, alpha_value * direction_p_array)
    if wrap_positions:
        if box is None:
            raise ValueError('box must be provided when wrap_positions=True.')
        new_c = wrap_position(new_c, np.asarray(box, dtype=float))
    new_particle.c = new_c
    new_particle.p = new_p
    return new_particle

def apply_trial_step(particles: list[EllipsoidParticle3D], direction_cs: list[np.ndarray], direction_ps: list[np.ndarray], alpha: float, box: np.ndarray | None=None, wrap_positions: bool=False) -> list[EllipsoidParticle3D]:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    if not isinstance(direction_cs, list):
        raise ValueError('direction_cs must be a list.')
    if not isinstance(direction_ps, list):
        raise ValueError('direction_ps must be a list.')
    if len(direction_cs) != len(particles):
        raise ValueError(f'Expected len(direction_cs) == len(particles), got {len(direction_cs)} != {len(particles)}.')
    if len(direction_ps) != len(particles):
        raise ValueError(f'Expected len(direction_ps) == len(particles), got {len(direction_ps)} != {len(particles)}.')
    updated_particles: list[EllipsoidParticle3D] = []
    for idx, (particle, direction_c, direction_p) in enumerate(zip(particles, direction_cs, direction_ps)):
        if not isinstance(particle, EllipsoidParticle3D):
            raise ValueError(f'particles[{idx}] must be EllipsoidParticle3D, got {type(particle)!r}.')
        updated_particles.append(apply_particle_trial_step(particle=particle, direction_c=direction_c, direction_p=direction_p, alpha=alpha, box=box, wrap_positions=wrap_positions))
    return updated_particles
