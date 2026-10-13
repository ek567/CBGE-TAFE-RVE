from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D

@dataclass
class UniformGridSpec3D:
    cell_size: float
    periodic: bool
    box: np.ndarray | None
    n_cells: np.ndarray | None
    domain_min: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.cell_size = float(self.cell_size)
        self.periodic = bool(self.periodic)
        if self.cell_size <= 0.0:
            raise ValueError(f'cell_size must be > 0, got {self.cell_size}.')
        if self.box is not None:
            self.box = np.asarray(self.box, dtype=float)
            if self.box.shape != (3,):
                raise ValueError(f'Expected box.shape == (3,), got {self.box.shape}.')
            if np.any(self.box <= 0.0):
                raise ValueError(f'Box lengths must be positive, got {self.box}.')
        if self.n_cells is not None:
            self.n_cells = np.asarray(self.n_cells, dtype=int)
            if self.n_cells.shape != (3,):
                raise ValueError(f'Expected n_cells.shape == (3,), got {self.n_cells.shape}.')
            if np.any(self.n_cells <= 0):
                raise ValueError(f'n_cells must be positive, got {self.n_cells}.')
        if self.domain_min is not None:
            self.domain_min = np.asarray(self.domain_min, dtype=float)
            if self.domain_min.shape != (3,):
                raise ValueError(f'Expected domain_min.shape == (3,), got {self.domain_min.shape}.')
        if self.periodic and self.box is None:
            raise ValueError('box must be provided when periodic=True.')
        if self.periodic and self.n_cells is None:
            raise ValueError('n_cells must be provided when periodic=True.')
        if not self.periodic and self.domain_min is None:
            raise ValueError('domain_min must be provided when periodic=False.')

@dataclass
class UniformGrid3D:
    spec: UniformGridSpec3D
    cell_to_particles: dict[tuple[int, int, int], list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, UniformGridSpec3D):
            raise ValueError(f'spec must be UniformGridSpec3D, got {type(self.spec)!r}.')
        if not isinstance(self.cell_to_particles, dict):
            raise ValueError('cell_to_particles must be a dict.')
        for key, value in self.cell_to_particles.items():
            if not (isinstance(key, tuple) and len(key) == 3):
                raise ValueError(f'Each cell key must be a 3-tuple of ints, got key={key!r}.')
            if not isinstance(value, list):
                raise ValueError(f'Each cell value must be list[int], got {type(value)!r}.')

def estimate_uniform_grid_cell_size(particles: list[EllipsoidParticle3D]) -> float:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    if len(particles) == 0:
        raise ValueError('particles must not be empty.')
    max_a = 0.0
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise ValueError(f'particles[{idx}] must be EllipsoidParticle3D, got {type(particle)!r}.')
        max_a = max(max_a, float(particle.a))
    if max_a <= 0.0:
        raise RuntimeError(f'Invalid max_a={max_a} computed from particles.')
    return float(2.0 * max_a)

def build_uniform_grid_spec(particles: list[EllipsoidParticle3D], box: np.ndarray | None, periodic: bool, cell_size: float | None=None, auto_cell_size: bool=True) -> UniformGridSpec3D:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    if len(particles) == 0:
        raise ValueError('particles must not be empty.')
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise ValueError(f'particles[{idx}] must be EllipsoidParticle3D, got {type(particle)!r}.')
    periodic_flag = bool(periodic)
    if cell_size is None:
        if not bool(auto_cell_size):
            raise ValueError('cell_size must be provided when auto_cell_size=False.')
        cell_size_value = estimate_uniform_grid_cell_size(particles)
    else:
        cell_size_value = float(cell_size)
        if cell_size_value <= 0.0:
            raise ValueError(f'cell_size must be > 0, got {cell_size_value}.')
    if periodic_flag:
        if box is None:
            raise ValueError('box must be provided when periodic=True.')
        box_array = np.asarray(box, dtype=float)
        if box_array.shape != (3,):
            raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
        if np.any(box_array <= 0.0):
            raise ValueError(f'Box lengths must be positive, got {box_array}.')
        n_cells = np.floor(box_array / cell_size_value).astype(int)
        n_cells = np.maximum(n_cells, 1)
        return UniformGridSpec3D(cell_size=cell_size_value, periodic=True, box=box_array, n_cells=n_cells, domain_min=None)
    positions = np.array([particle.c for particle in particles], dtype=float)
    if positions.shape[1:] != (3,):
        raise ValueError(f'Expected stacked positions shape (N,3), got {positions.shape}.')
    domain_min = np.min(positions, axis=0)
    domain_max = np.max(positions, axis=0)
    extent = np.maximum(domain_max - domain_min, cell_size_value)
    n_cells = np.floor(extent / cell_size_value).astype(int) + 1
    n_cells = np.maximum(n_cells, 1)
    return UniformGridSpec3D(cell_size=cell_size_value, periodic=False, box=None, n_cells=n_cells, domain_min=domain_min)

def particle_cell_index(position: np.ndarray, spec: UniformGridSpec3D) -> tuple[int, int, int]:
    if not isinstance(spec, UniformGridSpec3D):
        raise ValueError(f'spec must be UniformGridSpec3D, got {type(spec)!r}.')
    position_array = np.asarray(position, dtype=float)
    if position_array.shape != (3,):
        raise ValueError(f'Expected position.shape == (3,), got {position_array.shape}.')
    if spec.periodic:
        if spec.box is None or spec.n_cells is None:
            raise RuntimeError('Periodic grid spec requires box and n_cells.')
        scaled = np.floor(position_array / spec.cell_size).astype(int)
        wrapped = np.mod(scaled, spec.n_cells)
        return (int(wrapped[0]), int(wrapped[1]), int(wrapped[2]))
    if spec.domain_min is None or spec.n_cells is None:
        raise RuntimeError('Non-periodic grid spec requires domain_min and n_cells.')
    shifted = position_array - spec.domain_min
    scaled = np.floor(shifted / spec.cell_size).astype(int)
    clipped = np.clip(scaled, 0, spec.n_cells - 1)
    return (int(clipped[0]), int(clipped[1]), int(clipped[2]))

def build_uniform_grid(particles: list[EllipsoidParticle3D], spec: UniformGridSpec3D) -> UniformGrid3D:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    if not isinstance(spec, UniformGridSpec3D):
        raise ValueError(f'spec must be UniformGridSpec3D, got {type(spec)!r}.')
    cell_to_particles: dict[tuple[int, int, int], list[int]] = {}
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise ValueError(f'particles[{idx}] must be EllipsoidParticle3D, got {type(particle)!r}.')
        cell_idx = particle_cell_index(particle.c, spec)
        if cell_idx not in cell_to_particles:
            cell_to_particles[cell_idx] = []
        cell_to_particles[cell_idx].append(int(idx))
    return UniformGrid3D(spec=spec, cell_to_particles=cell_to_particles)

def neighbor_cell_offsets_3d() -> list[tuple[int, int, int]]:
    offsets: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                offsets.append((dx, dy, dz))
    return offsets

def wrapped_neighbor_cell_index(cell_index: tuple[int, int, int], offset: tuple[int, int, int], spec: UniformGridSpec3D) -> tuple[int, int, int] | None:
    if not isinstance(spec, UniformGridSpec3D):
        raise ValueError(f'spec must be UniformGridSpec3D, got {type(spec)!r}.')
    if not (isinstance(cell_index, tuple) and len(cell_index) == 3):
        raise ValueError(f'cell_index must be a 3-tuple, got {cell_index!r}.')
    if not (isinstance(offset, tuple) and len(offset) == 3):
        raise ValueError(f'offset must be a 3-tuple, got {offset!r}.')
    if spec.n_cells is None:
        raise RuntimeError('spec.n_cells must not be None.')
    raw = np.array(cell_index, dtype=int) + np.array(offset, dtype=int)
    if spec.periodic:
        wrapped = np.mod(raw, spec.n_cells)
        return (int(wrapped[0]), int(wrapped[1]), int(wrapped[2]))
    if np.any(raw < 0) or np.any(raw >= spec.n_cells):
        return None
    return (int(raw[0]), int(raw[1]), int(raw[2]))

def generate_candidate_pairs_from_uniform_grid(particles: list[EllipsoidParticle3D], grid: UniformGrid3D) -> list[tuple[int, int]]:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    if not isinstance(grid, UniformGrid3D):
        raise ValueError(f'grid must be UniformGrid3D, got {type(grid)!r}.')
    pair_set: set[tuple[int, int]] = set()
    offsets = neighbor_cell_offsets_3d()
    for cell_idx, particle_indices in grid.cell_to_particles.items():
        if not (isinstance(cell_idx, tuple) and len(cell_idx) == 3):
            raise ValueError(f'Invalid cell index key: {cell_idx!r}.')
        if not isinstance(particle_indices, list):
            raise ValueError(f'Expected cell_to_particles[{cell_idx!r}] to be list, got {type(particle_indices)!r}.')
        for offset in offsets:
            neighbor_idx = wrapped_neighbor_cell_index(cell_index=cell_idx, offset=offset, spec=grid.spec)
            if neighbor_idx is None:
                continue
            neighbor_particles = grid.cell_to_particles.get(neighbor_idx, [])
            if len(neighbor_particles) == 0:
                continue
            for i in particle_indices:
                for j in neighbor_particles:
                    if i == j:
                        continue
                    pair = (int(min(i, j)), int(max(i, j)))
                    pair_set.add(pair)
    return sorted(pair_set)
