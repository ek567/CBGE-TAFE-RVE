from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.model.q_matrix_3d import build_q_matrix_closed_form, build_q_sym_closed_form, build_q_sym_pw85
from src.overlap_detection.candidate_pairs_3d import CandidatePairConfig3D, generate_candidate_pairs

@dataclass
class ParticleGeomCache3D:
    q_sym: np.ndarray
    Q_full: Optional[np.ndarray]
    anisotropy: np.ndarray
    p: np.ndarray

@dataclass
class StepContext3D:
    particles_ref_id: int
    box: Optional[np.ndarray]
    use_mic: bool
    pair_config: CandidatePairConfig3D
    candidate_pairs: list[tuple[int, int]]
    particle_cache: ParticleGeomCache3D

def build_particle_geom_cache_3d(particles, *, use_pw85_q: bool=True, cache_Q_full: bool=True) -> ParticleGeomCache3D:
    n = len(particles)
    q_sym = np.zeros((n, 6), dtype=float)
    Q_full = np.zeros((n, 3, 3), dtype=float) if cache_Q_full else None
    anisotropy = np.zeros(n, dtype=float)
    p_arr = np.zeros((n, 3), dtype=float)
    for i, part in enumerate(particles):
        p_arr[i, :] = part.p
        anisotropy[i] = part.a * part.a - part.b * part.b
        if use_pw85_q:
            q_sym[i, :] = build_q_sym_pw85(a=part.a, b=part.b, p=part.p)
        else:
            q_sym[i, :] = build_q_sym_closed_form(a=part.a, b=part.b, p=part.p)
        if cache_Q_full:
            Q_full[i, :, :] = build_q_matrix_closed_form(a=part.a, b=part.b, p=part.p)
    return ParticleGeomCache3D(q_sym=q_sym, Q_full=Q_full, anisotropy=anisotropy, p=p_arr)

def build_step_context_3d(particles, *, box: np.ndarray | None=None, use_mic: bool=False, pair_config: CandidatePairConfig3D | None=None, use_pw85_q: bool=True, cache_Q_full: bool=True, metrics=None) -> StepContext3D:
    if pair_config is None:
        pair_config = CandidatePairConfig3D()
    candidate_pairs = generate_candidate_pairs(particles=particles, box=box, use_mic=bool(use_mic), config=pair_config, metrics=metrics)
    particle_cache = build_particle_geom_cache_3d(particles, use_pw85_q=bool(use_pw85_q) and len(candidate_pairs) > 0, cache_Q_full=cache_Q_full)
    return StepContext3D(particles_ref_id=id(particles), box=box, use_mic=bool(use_mic), pair_config=pair_config, candidate_pairs=candidate_pairs, particle_cache=particle_cache)
