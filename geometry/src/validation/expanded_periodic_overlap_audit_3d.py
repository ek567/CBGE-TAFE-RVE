from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.model.pair_contact_measure_pw_3d import compute_pair_contact_measure
from src.model.pair_potential_3d import pair_overlap_gap
from src.model.q_matrix_3d import build_q_sym_pw85
from src.periodic.mic_3d import _validate_box

@dataclass
class ExpandedParticle3D:
    source_index: int
    source_id: int | None
    image_shift: np.ndarray
    a: float
    b: float
    c: np.ndarray
    p: np.ndarray

    def __post_init__(self) -> None:
        self.source_index = int(self.source_index)
        self.source_id = None if self.source_id is None else int(self.source_id)
        self.image_shift = np.asarray(self.image_shift, dtype=float)
        self.c = np.asarray(self.c, dtype=float)
        self.p = np.asarray(self.p, dtype=float)
        self.a = float(self.a)
        self.b = float(self.b)
        if self.source_index < 0:
            raise ValueError(f'Expected source_index >= 0, got {self.source_index}.')
        if self.image_shift.shape != (3,):
            raise ValueError(f'Expected image_shift.shape == (3,), got {self.image_shift.shape}.')
        if self.c.shape != (3,):
            raise ValueError(f'Expected c.shape == (3,), got {self.c.shape}.')
        if self.p.shape != (3,):
            raise ValueError(f'Expected p.shape == (3,), got {self.p.shape}.')
        if not self.a >= self.b > 0.0:
            raise ValueError(f'Expected a >= b > 0, got a={self.a}, b={self.b}.')

def _ensure_pypw85_available() -> None:
    try:
        import pypw85
    except ImportError as exc:
        raise ImportError('pypw85 is required for expanded_periodic_overlap_audit_3d but is not available.') from exc

def _image_shifts_3x3x3() -> list[np.ndarray]:
    shifts: list[np.ndarray] = []
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                shifts.append(np.array([sx, sy, sz], dtype=float))
    return shifts

def build_expanded_particles(particles: list[EllipsoidParticle3D], box: np.ndarray) -> list[ExpandedParticle3D]:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    box_array = _validate_box(box)
    shifts = _image_shifts_3x3x3()
    expanded: list[ExpandedParticle3D] = []
    for source_index, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise ValueError(f'particles[{source_index}] must be EllipsoidParticle3D, got {type(particle)!r}.')
        for shift in shifts:
            expanded_center = np.asarray(particle.c + shift * box_array, dtype=float)
            expanded.append(ExpandedParticle3D(source_index=source_index, source_id=particle.id, image_shift=shift, a=particle.a, b=particle.b, c=expanded_center, p=particle.p.copy()))
    return expanded

def build_expanded_audit_pairs(expanded_particles: list[ExpandedParticle3D]) -> list[tuple[int, int]]:
    if not isinstance(expanded_particles, list):
        raise ValueError('expanded_particles must be a list.')
    pairs: list[tuple[int, int]] = []
    n = len(expanded_particles)
    for i in range(n):
        particle_i = expanded_particles[i]
        if not isinstance(particle_i, ExpandedParticle3D):
            raise ValueError(f'expanded_particles[{i}] must be ExpandedParticle3D, got {type(particle_i)!r}.')
        for j in range(i + 1, n):
            particle_j = expanded_particles[j]
            if not isinstance(particle_j, ExpandedParticle3D):
                raise ValueError(f'expanded_particles[{j}] must be ExpandedParticle3D, got {type(particle_j)!r}.')
            if particle_i.source_index == particle_j.source_index:
                continue
            pairs.append((i, j))
    return pairs

def audit_periodic_system_overlap(particles: list[EllipsoidParticle3D], box: np.ndarray, overlap_tol: float=1e-06) -> dict[str, Any]:
    _ensure_pypw85_available()
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    box_array = _validate_box(box)
    overlap_tol_value = float(overlap_tol)
    if overlap_tol_value < 0.0:
        raise ValueError(f'Expected overlap_tol >= 0, got {overlap_tol_value}.')
    expanded_particles = build_expanded_particles(particles=particles, box=box_array)
    audit_pairs = build_expanded_audit_pairs(expanded_particles=expanded_particles)
    q_cache: dict[int, np.ndarray] = {}
    for source_index, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise ValueError(f'particles[{source_index}] must be EllipsoidParticle3D, got {type(particle)!r}.')
        q_cache[source_index] = build_q_sym_pw85(a=particle.a, b=particle.b, p=particle.p)
    max_overlap = -1.0
    worst_pair_sources: tuple[int, int] | None = None
    worst_pair_source_ids: tuple[int | None, int | None] | None = None
    worst_pair_images: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    worst_F_ij: float | None = None
    worst_lambda_star: float | None = None
    overlap_records: list[dict[str, Any]] = []
    for i_idx, j_idx in audit_pairs:
        particle_i = expanded_particles[i_idx]
        particle_j = expanded_particles[j_idx]
        r_ij = np.asarray(particle_j.c - particle_i.c, dtype=float)
        if r_ij.shape != (3,):
            raise ValueError(f'Expected r_ij.shape == (3,), got {r_ij.shape}.')
        q_i = q_cache[particle_i.source_index]
        q_j = q_cache[particle_j.source_index]
        F_ij, lambda_star = compute_pair_contact_measure(r_ij=r_ij, q_i=q_i, q_j=q_j)
        overlap = pair_overlap_gap(F_ij=F_ij, threshold=1.0)
        if overlap > max_overlap:
            max_overlap = float(overlap)
            worst_pair_sources = (int(particle_i.source_index), int(particle_j.source_index))
            worst_pair_source_ids = (particle_i.source_id, particle_j.source_id)
            worst_pair_images = (tuple((int(v) for v in particle_i.image_shift)), tuple((int(v) for v in particle_j.image_shift)))
            worst_F_ij = float(F_ij)
            worst_lambda_star = float(lambda_star)
        if overlap > overlap_tol_value:
            overlap_records.append({'pair': (int(particle_i.source_index), int(particle_j.source_index)), 'pair_source_ids': (particle_i.source_id, particle_j.source_id), 'overlap': float(overlap), 'shift_i': tuple((int(v) for v in particle_i.image_shift)), 'shift_j': tuple((int(v) for v in particle_j.image_shift)), 'F_ij': float(F_ij), 'lambda_star': float(lambda_star), 'r_ij': r_ij})
    has_overlap = bool(len(overlap_records) > 0)
    return {'has_overlap': has_overlap, 'overlap_free': not has_overlap, 'num_expanded_particles': int(len(expanded_particles)), 'num_checked_pairs': int(len(audit_pairs)), 'num_overlapping_pairs': int(len(overlap_records)), 'max_overlap': float(max_overlap if max_overlap >= 0.0 else 0.0), 'max_overlap_pair': worst_pair_sources, 'max_overlap_pair_source_ids': worst_pair_source_ids, 'worst_pair_sources': worst_pair_sources, 'worst_pair_source_ids': worst_pair_source_ids, 'worst_pair_images': worst_pair_images, 'worst_F_ij': worst_F_ij, 'worst_lambda_star': worst_lambda_star, 'overlap_records': overlap_records}
