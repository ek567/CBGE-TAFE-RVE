from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.monitoring.metrics_3d import StepMetrics3D
from src.monitoring.timers_3d import timed_field
from src.overlap_detection.bounding_sphere_3d import candidate_by_bounding_sphere
from src.overlap_detection.spatial_grid_3d import build_uniform_grid, build_uniform_grid_spec, generate_candidate_pairs_from_uniform_grid
from src.periodic.mic_3d import minimum_image_vector

@dataclass
class CandidatePairConfig3D:
    method: str = 'all_pairs'
    use_bounding_sphere: bool = True
    sphere_padding: float = 0.0
    cell_size: float | None = None
    auto_cell_size: bool = True

    def __post_init__(self) -> None:
        self.method = str(self.method)
        self.use_bounding_sphere = bool(self.use_bounding_sphere)
        self.sphere_padding = float(self.sphere_padding)
        self.cell_size = None if self.cell_size is None else float(self.cell_size)
        self.auto_cell_size = bool(self.auto_cell_size)
        if self.method not in {'all_pairs', 'uniform_grid'}:
            raise ValueError(f"Unsupported method={self.method!r}. Expected one of {{'all_pairs', 'uniform_grid'}}.")
        if self.sphere_padding < 0.0:
            raise ValueError(f'sphere_padding must be >= 0, got {self.sphere_padding}.')
        if self.cell_size is not None and self.cell_size <= 0.0:
            raise ValueError(f'cell_size must be > 0, got {self.cell_size}.')

def generate_all_pairs(n_particles: int) -> list[tuple[int, int]]:
    n_value = int(n_particles)
    if n_value < 0:
        raise ValueError(f'Expected n_particles >= 0, got {n_value}.')
    return [(i, j) for i in range(n_value) for j in range(i + 1, n_value)]

def filter_candidate_pairs_by_bounding_sphere(particles: list[EllipsoidParticle3D], pair_indices: list[tuple[int, int]], box: np.ndarray | None, use_mic: bool, padding: float=0.0, metrics: StepMetrics3D | None=None) -> list[tuple[int, int]]:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    if not isinstance(pair_indices, list):
        raise ValueError('pair_indices must be a list.')
    box_array = None if box is None else np.asarray(box, dtype=float)
    if box_array is not None:
        if box_array.shape != (3,):
            raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
        if np.any(box_array <= 0.0):
            raise ValueError(f'Box lengths must be positive, got {box_array}.')
    filtered: list[tuple[int, int]] = []
    with timed_field(metrics, 'time_broadphase'):
        for idx, pair in enumerate(pair_indices):
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(f'pair_indices[{idx}] must be tuple[int, int], got {pair!r}.')
            i = int(pair[0])
            j = int(pair[1])
            if not 0 <= i < len(particles):
                raise ValueError(f'Invalid i={i} for len(particles)={len(particles)}.')
            if not 0 <= j < len(particles):
                raise ValueError(f'Invalid j={j} for len(particles)={len(particles)}.')
            if not i < j:
                raise ValueError(f'Expected pair with i < j, got ({i}, {j}).')
            particle_i = particles[i]
            particle_j = particles[j]
            if use_mic:
                if box_array is None:
                    raise ValueError('box must be provided when use_mic=True.')
                r_ij = minimum_image_vector(c_i=particle_i.c, c_j=particle_j.c, box=box_array)
            else:
                r_ij = np.asarray(particle_j.c - particle_i.c, dtype=float)
            keep = candidate_by_bounding_sphere(a_i=particle_i.a, a_j=particle_j.a, r_ij=r_ij, padding=padding)
            if keep:
                filtered.append((i, j))
    return filtered

def generate_candidate_pairs(particles: list[EllipsoidParticle3D], box: np.ndarray | None, use_mic: bool, config: CandidatePairConfig3D | None=None, metrics: StepMetrics3D | None=None) -> list[tuple[int, int]]:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise ValueError(f'particles[{idx}] must be EllipsoidParticle3D, got {type(particle)!r}.')
    if config is None:
        config = CandidatePairConfig3D()
    elif not isinstance(config, CandidatePairConfig3D):
        raise ValueError(f'config must be CandidatePairConfig3D or None, got {type(config)!r}.')
    if metrics is not None:
        if not isinstance(metrics, StepMetrics3D):
            raise ValueError(f'metrics must be StepMetrics3D or None, got {type(metrics)!r}.')
        metrics.num_particles = int(len(particles))
        metrics.num_all_pairs = int(len(particles) * (len(particles) - 1) // 2)
    raw_pairs: list[tuple[int, int]]
    if config.method == 'all_pairs':
        with timed_field(metrics, 'time_pair_generation'):
            raw_pairs = generate_all_pairs(len(particles))
    elif config.method == 'uniform_grid':
        with timed_field(metrics, 'time_neighbor_build'):
            spec = build_uniform_grid_spec(particles=particles, box=box, periodic=bool(use_mic), cell_size=config.cell_size, auto_cell_size=config.auto_cell_size)
            grid = build_uniform_grid(particles=particles, spec=spec)
        with timed_field(metrics, 'time_pair_generation'):
            raw_pairs = generate_candidate_pairs_from_uniform_grid(particles=particles, grid=grid)
    else:
        raise RuntimeError(f'Unhandled candidate-pair method: {config.method!r}.')
    if config.use_bounding_sphere:
        filtered_pairs = filter_candidate_pairs_by_bounding_sphere(particles=particles, pair_indices=raw_pairs, box=box, use_mic=use_mic, padding=config.sphere_padding, metrics=metrics)
    else:
        filtered_pairs = raw_pairs
    if metrics is not None:
        metrics.num_candidate_pairs = int(len(filtered_pairs))
    return filtered_pairs
