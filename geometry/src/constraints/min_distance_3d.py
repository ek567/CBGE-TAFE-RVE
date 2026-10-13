from __future__ import annotations
from typing import List
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D

def is_min_distance_enabled(delta_min: float) -> bool:
    return float(delta_min) > 0.0

def inflate_particles_min_distance_3d(particles_physical: List[EllipsoidParticle3D], delta_min: float) -> List[EllipsoidParticle3D]:
    if not isinstance(particles_physical, list):
        raise TypeError('particles_physical must be a list of EllipsoidParticle3D.')
    delta = float(delta_min)
    if delta < 0.0:
        raise ValueError(f'delta_min must be >= 0, got {delta}.')
    out: List[EllipsoidParticle3D] = []
    da = 0.5 * delta
    for idx, pt in enumerate(particles_physical):
        if not isinstance(pt, EllipsoidParticle3D):
            raise TypeError(f'particles_physical[{idx}] must be EllipsoidParticle3D, got {type(pt).__name__}.')
        a_opt = float(pt.a) + da
        b_opt = float(pt.b) + da
        if b_opt <= 0.0:
            raise ValueError(f'Inflated b becomes non-positive for particle[{idx}]: b={pt.b}, delta_min={delta}.')
        if a_opt < b_opt:
            raise ValueError(f'Inflated a < b for particle[{idx}]: a_opt={a_opt}, b_opt={b_opt}.')
        out.append(EllipsoidParticle3D(a=a_opt, b=b_opt, c=np.asarray(pt.c, dtype=float), p=np.asarray(pt.p, dtype=float), id=pt.id))
    return out

def restore_physical_particles_after_optimization_3d(particles_physical: List[EllipsoidParticle3D], particles_opt_final: List[EllipsoidParticle3D]) -> List[EllipsoidParticle3D]:
    if not isinstance(particles_physical, list):
        raise TypeError('particles_physical must be a list.')
    if not isinstance(particles_opt_final, list):
        raise TypeError('particles_opt_final must be a list.')
    if len(particles_physical) != len(particles_opt_final):
        raise ValueError(f'particles_physical and particles_opt_final must have same length, got {len(particles_physical)} vs {len(particles_opt_final)}.')
    out: List[EllipsoidParticle3D] = []
    for i, (pp, po) in enumerate(zip(particles_physical, particles_opt_final)):
        if not isinstance(pp, EllipsoidParticle3D):
            raise TypeError(f'particles_physical[{i}] must be EllipsoidParticle3D, got {type(pp).__name__}.')
        if not isinstance(po, EllipsoidParticle3D):
            raise TypeError(f'particles_opt_final[{i}] must be EllipsoidParticle3D, got {type(po).__name__}.')
        out.append(EllipsoidParticle3D(a=float(pp.a), b=float(pp.b), c=np.asarray(po.c, dtype=float), p=np.asarray(po.p, dtype=float), id=pp.id if pp.id is not None else po.id))
    return out
