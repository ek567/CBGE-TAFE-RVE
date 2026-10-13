from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Sequence
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.model.geometry_features_3d import BoxGeometryFeatures3D, CornerFeature3D, EdgeFeature3D, FaceFeature3D, build_box_geometry_features_3d
from src.model.projected_clearance_3d import FaceClearanceResult3D, ProjectedClearanceResult3D, compute_face_signed_clearance, compute_projected_signed_clearance
from src.model.q_matrix_3d import build_q_matrix_closed_form
from src.utils.math_utils import tangent_projector
NormalizeBy3D = Literal['num_particles', 'num_terms', 'none']

@dataclass(frozen=True)
class GeometryDistortionConfig3D:
    enabled: bool = True
    include_face: bool = True
    include_edge: bool = False
    include_corner: bool = True
    d_face: float = 0.05
    d_edge: float = 0.08
    d_corner: float = 0.1
    w_face: float = 1.0
    w_edge: float = 1.0
    w_corner: float = 1.0
    lambda_geo: float = 1.0
    normalize_by: NormalizeBy3D = 'num_particles'
    use_physical_axes: bool = True
    root_tol: float = 1e-12
    root_max_iter: int = 80
    clearance_tol: float = 1e-14
    active_violation_abs_tol: float = 1e-12
    active_violation_rel_tol: float = 1e-10
    edge_tangent_filter: bool = True
    candidate_filter: bool = True
    candidate_margin_abs: float = 0.0
    candidate_margin_rel: float = 1.0
    nonsmooth_rule: Literal['inactive_at_boundary'] = 'inactive_at_boundary'

    def __post_init__(self) -> None:
        for name in ('d_face', 'd_edge', 'd_corner'):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f'{name} must be > 0, got {value}.')
        for name in ('w_face', 'w_edge', 'w_corner', 'lambda_geo'):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f'{name} must be >= 0, got {value}.')
        if self.normalize_by not in ('num_particles', 'num_terms', 'none'):
            raise ValueError(f'Unknown normalize_by={self.normalize_by!r}.')
        if self.root_tol <= 0.0:
            raise ValueError(f'root_tol must be > 0, got {self.root_tol}.')
        if int(self.root_max_iter) <= 0:
            raise ValueError(f'root_max_iter must be > 0, got {self.root_max_iter}.')
        if self.clearance_tol < 0.0:
            raise ValueError(f'clearance_tol must be >= 0, got {self.clearance_tol}.')
        if self.active_violation_abs_tol < 0.0:
            raise ValueError(f'active_violation_abs_tol must be >= 0, got {self.active_violation_abs_tol}.')
        if self.active_violation_rel_tol < 0.0:
            raise ValueError(f'active_violation_rel_tol must be >= 0, got {self.active_violation_rel_tol}.')
        if self.candidate_margin_abs < 0.0:
            raise ValueError(f'candidate_margin_abs must be >= 0, got {self.candidate_margin_abs}.')
        if self.candidate_margin_rel < 0.0:
            raise ValueError(f'candidate_margin_rel must be >= 0, got {self.candidate_margin_rel}.')

@dataclass(frozen=True)
class GeometryActiveRecord3D:
    particle_index: int
    feature_name: str
    feature_kind: str
    s: float
    violation: float
    energy: float

@dataclass(frozen=True)
class GeometryDistortionResult3D:
    total_energy: float
    face_energy: float
    edge_energy: float
    corner_energy: float
    grad_cs: list[np.ndarray]
    grad_ps: list[np.ndarray]
    num_active_face: int
    num_active_edge: int
    num_active_corner: int
    worst_violation: float
    max_face_violation: float
    max_edge_violation: float
    max_corner_violation: float
    worst_particle_index: int | None
    worst_feature_name: str | None
    worst_feature_kind: str | None
    min_abs_signed_clearance: float
    active_records: list[GeometryActiveRecord3D] = field(default_factory=list)
    candidate_stats: dict[str, int] = field(default_factory=dict)

@dataclass(frozen=True)
class GeometryCandidateCache3D:
    face_indices_by_particle: list[tuple[int, ...]]
    edge_indices_by_particle: list[tuple[int, ...]]
    corner_indices_by_particle: list[tuple[int, ...]]
    guard_margin: float
    face_candidates: list[tuple[int, int]]
    edge_candidates: list[tuple[int, int]]
    corner_candidates: list[tuple[int, int]]

def _validate_particles(particles: list[EllipsoidParticle3D]) -> None:
    if not isinstance(particles, list):
        raise TypeError('particles must be a list of EllipsoidParticle3D.')
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise TypeError(f'particles[{idx}] must be EllipsoidParticle3D, got {type(particle).__name__}.')

def _validate_box(box: np.ndarray) -> np.ndarray:
    box_array = np.asarray(box, dtype=float)
    if box_array.shape != (3,):
        raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
    if np.any(box_array <= 0.0):
        raise ValueError(f'Box lengths must be positive, got {box_array}.')
    return box_array

def _normalization_factor(n_particles: int, features: BoxGeometryFeatures3D, normalize_by: NormalizeBy3D) -> float:
    if normalize_by == 'none':
        return 1.0
    if normalize_by == 'num_particles':
        return float(max(1, int(n_particles)))
    if normalize_by == 'num_terms':
        return float(max(1, int(n_particles) * features.num_enabled_terms()))
    raise ValueError(f'Unknown normalize_by={normalize_by!r}.')

def _axes_for_particle(particle: EllipsoidParticle3D, index: int, physical_axes_override: Sequence[tuple[float, float]] | np.ndarray | None) -> tuple[float, float]:
    if physical_axes_override is None:
        return (float(particle.a), float(particle.b))
    axes = physical_axes_override[index]
    if len(axes) != 2:
        raise ValueError(f'physical_axes_override[{index}] must contain two values (a,b).')
    a_value = float(axes[0])
    b_value = float(axes[1])
    if not a_value >= b_value > 0.0:
        raise ValueError(f'physical_axes_override[{index}] must satisfy a >= b > 0, got {(a_value, b_value)}.')
    return (a_value, b_value)

def _safe_sign(s: float, gamma: float | None=None) -> float:
    if s > 0.0:
        return 1.0
    if s < 0.0:
        return -1.0
    if gamma is not None:
        if gamma > 0.0:
            return 1.0
        if gamma < 0.0:
            return -1.0
    return 1.0

def _active_violation_tolerance(*, d_safe: float, config: GeometryDistortionConfig3D) -> float:
    return max(float(config.active_violation_abs_tol), float(config.active_violation_rel_tol) * float(d_safe))

def _penalty_energy_and_coeff(*, s: float, d_safe: float, weight: float, gamma: float | None=None, active_tol: float=0.0) -> tuple[float, float, float]:
    d_value = float(d_safe)
    w_value = float(weight)
    violation_raw = d_value - abs(float(s))
    if violation_raw <= float(active_tol) or w_value == 0.0:
        return (0.0, 0.0, 0.0)
    violation = violation_raw
    energy = 0.5 * w_value * (violation / d_value) ** 2
    coeff = -(w_value / (d_value * d_value)) * violation * _safe_sign(float(s), gamma)
    return (float(energy), float(coeff), float(violation))

def _orientation_tangent_gradient_from_projected_data(*, a_axis: float, b_axis: float, p: np.ndarray, B: np.ndarray, clearance: ProjectedClearanceResult3D) -> np.ndarray:
    delta = float(a_axis) ** 2 - float(b_axis) ** 2
    if delta == 0.0:
        return np.zeros(3, dtype=float)
    p_unit = np.asarray(p, dtype=float)
    B_array = np.asarray(B, dtype=float)
    Bt_a = B_array.T @ clearance.a
    scalar = float(np.dot(p_unit, Bt_a))
    ell = float(clearance.ell)
    if ell <= 0.0:
        return np.zeros(3, dtype=float)
    grad_p = -(delta / ell) * scalar * Bt_a
    return tangent_projector(p_unit) @ grad_p

def _edge_tangent_filter_passes(*, particle: EllipsoidParticle3D, Q: np.ndarray, edge: EdgeFeature3D) -> bool:
    t = np.asarray(edge.tangent, dtype=float)
    c = np.asarray(particle.c, dtype=float)
    endpoint_A = np.asarray(edge.endpoint_A, dtype=float)
    tau = float(np.dot(t, c - endpoint_A))
    h_t_sq = float(t @ Q @ t)
    h_t = float(np.sqrt(max(h_t_sq, 0.0)))
    interval_left = tau - h_t
    interval_right = tau + h_t
    edge_left = -float(edge.d_safe)
    edge_right = float(edge.length) + float(edge.d_safe)
    return not (interval_right < edge_left or interval_left > edge_right)

def _candidate_margin(d_safe: float, config: GeometryDistortionConfig3D) -> float:
    return max(float(config.candidate_margin_abs), float(config.candidate_margin_rel) * float(d_safe))

def _face_candidate_passes(*, particle: EllipsoidParticle3D, Q: np.ndarray, face: FaceFeature3D, config: GeometryDistortionConfig3D, margin_override: float | None=None) -> bool:
    n = face.B.reshape(3)
    m = float(np.dot(n, particle.c) + float(face.r[0]))
    h_sq = float(n @ Q @ n)
    h = float(np.sqrt(max(h_sq, 0.0)))
    margin = _candidate_margin(face.d_safe, config) if margin_override is None else float(margin_override)
    return bool(abs(m) - h <= float(face.d_safe) + margin)

def _face_sphere_candidate_passes(*, particle: EllipsoidParticle3D, a_axis: float, face: FaceFeature3D, margin: float) -> bool:
    n = face.B.reshape(3)
    m = float(np.dot(n, particle.c) + float(face.r[0]))
    return bool(abs(m) - float(a_axis) <= float(face.d_safe) + float(margin))

def _edge_candidate_passes(*, particle: EllipsoidParticle3D, a_axis: float, edge: EdgeFeature3D, config: GeometryDistortionConfig3D, margin_override: float | None=None) -> bool:
    m = np.asarray(edge.B, dtype=float) @ np.asarray(particle.c, dtype=float) + np.asarray(edge.r, dtype=float)
    center_normal_dist = float(np.linalg.norm(m))
    margin = _candidate_margin(edge.d_safe, config) if margin_override is None else float(margin_override)
    return bool(center_normal_dist - float(a_axis) <= float(edge.d_safe) + margin)

def _corner_candidate_passes(*, particle: EllipsoidParticle3D, a_axis: float, corner: CornerFeature3D, config: GeometryDistortionConfig3D, margin_override: float | None=None) -> bool:
    center_dist = float(np.linalg.norm(np.asarray(particle.c, dtype=float) - np.asarray(corner.C, dtype=float)))
    margin = _candidate_margin(corner.d_safe, config) if margin_override is None else float(margin_override)
    return bool(center_dist - float(a_axis) <= float(corner.d_safe) + margin)

def build_geometry_candidate_cache_3d(*, particles: list[EllipsoidParticle3D], box: np.ndarray, config: GeometryDistortionConfig3D, physical_axes_override: Sequence[tuple[float, float]] | np.ndarray | None=None, features: BoxGeometryFeatures3D | None=None, guard_margin: float=0.0) -> GeometryCandidateCache3D:
    _validate_particles(particles)
    box_array = _validate_box(box)
    if physical_axes_override is not None and len(physical_axes_override) != len(particles):
        raise ValueError(f'physical_axes_override must have the same length as particles; got {len(physical_axes_override)} and {len(particles)}.')
    if features is None:
        features = build_box_geometry_features_3d(box_array, d_face=config.d_face, d_edge=config.d_edge, d_corner=config.d_corner, w_face=config.w_face, w_edge=config.w_edge, w_corner=config.w_corner, include_face=config.include_face, include_edge=config.include_edge, include_corner=config.include_corner)
    margin = max(0.0, float(guard_margin))
    face_by_particle: list[tuple[int, ...]] = []
    edge_by_particle: list[tuple[int, ...]] = []
    corner_by_particle: list[tuple[int, ...]] = []
    face_candidates: list[tuple[int, int]] = []
    edge_candidates: list[tuple[int, int]] = []
    corner_candidates: list[tuple[int, int]] = []
    for i, particle in enumerate(particles):
        a_axis, b_axis = _axes_for_particle(particle=particle, index=i, physical_axes_override=physical_axes_override if config.use_physical_axes else None)
        Q = build_q_matrix_closed_form(a_axis, b_axis, particle.p)
        f_idxs: list[int] = []
        for k, face in enumerate(features.faces):
            if _face_sphere_candidate_passes(particle=particle, a_axis=a_axis, face=face, margin=margin):
                f_idxs.append(k)
                face_candidates.append((i, k))
        e_idxs: list[int] = []
        for k, edge in enumerate(features.edges):
            if _edge_candidate_passes(particle=particle, a_axis=a_axis, edge=edge, config=config, margin_override=margin):
                e_idxs.append(k)
                edge_candidates.append((i, k))
        c_idxs: list[int] = []
        for k, corner in enumerate(features.corners):
            if _corner_candidate_passes(particle=particle, a_axis=a_axis, corner=corner, config=config, margin_override=margin):
                c_idxs.append(k)
                corner_candidates.append((i, k))
        face_by_particle.append(tuple(f_idxs))
        edge_by_particle.append(tuple(e_idxs))
        corner_by_particle.append(tuple(c_idxs))
    return GeometryCandidateCache3D(face_indices_by_particle=face_by_particle, edge_indices_by_particle=edge_by_particle, corner_indices_by_particle=corner_by_particle, guard_margin=float(margin), face_candidates=face_candidates, edge_candidates=edge_candidates, corner_candidates=corner_candidates)

def _accumulate_feature_contribution(*, particle_index: int, feature_name: str, feature_kind: str, s: float, gamma: float | None, d_safe: float, weight: float, active_tol: float, grad_s_c: np.ndarray, grad_s_p: np.ndarray, grad_cs: list[np.ndarray], grad_ps: list[np.ndarray], compute_grad: bool) -> tuple[float, float, GeometryActiveRecord3D | None]:
    energy, coeff, violation = _penalty_energy_and_coeff(s=float(s), d_safe=float(d_safe), weight=float(weight), gamma=gamma, active_tol=float(active_tol))
    if compute_grad and coeff != 0.0:
        grad_cs[particle_index] += coeff * np.asarray(grad_s_c, dtype=float)
        grad_ps[particle_index] += coeff * np.asarray(grad_s_p, dtype=float)
    record = None
    if violation > 0.0:
        record = GeometryActiveRecord3D(particle_index=int(particle_index), feature_name=str(feature_name), feature_kind=str(feature_kind), s=float(s), violation=float(violation), energy=float(energy))
    return (float(energy), float(violation), record)

def _evaluate_face_feature(*, particle: EllipsoidParticle3D, particle_index: int, Q: np.ndarray, a_axis: float, b_axis: float, face: FaceFeature3D, config: GeometryDistortionConfig3D, grad_cs: list[np.ndarray], grad_ps: list[np.ndarray], compute_grad: bool) -> tuple[float, float, GeometryActiveRecord3D | None, float]:
    n = face.B.reshape(3)
    m = float(np.dot(n, particle.c) + float(face.r[0]))
    h_sq = float(n @ Q @ n)
    h = float(np.sqrt(max(h_sq, 0.0)))
    if h <= 0.0:
        raise RuntimeError(f'Face {face.name}: invalid projected half-length h={h}.')
    clearance: FaceClearanceResult3D = compute_face_signed_clearance(m=m, h=h)
    s = float(clearance.s)
    if clearance.branch == 'left':
        grad_s_c = n.copy()
    elif clearance.branch == 'right':
        grad_s_c = -n.copy()
    else:
        raise RuntimeError(f'Unknown face clearance branch {clearance.branch!r}.')
    delta = float(a_axis) ** 2 - float(b_axis) ** 2
    grad_s_p = -(delta * float(np.dot(n, particle.p)) / h) * n
    grad_s_p = tangent_projector(particle.p) @ grad_s_p
    energy, violation, record = _accumulate_feature_contribution(particle_index=particle_index, feature_name=face.name, feature_kind='face', s=s, gamma=None, d_safe=face.d_safe, weight=face.weight, active_tol=_active_violation_tolerance(d_safe=face.d_safe, config=config), grad_s_c=grad_s_c, grad_s_p=grad_s_p, grad_cs=grad_cs, grad_ps=grad_ps, compute_grad=compute_grad)
    return (energy, violation, record, abs(s))

def _evaluate_projected_feature(*, particle: EllipsoidParticle3D, particle_index: int, Q: np.ndarray, a_axis: float, b_axis: float, feature: EdgeFeature3D | CornerFeature3D, config: GeometryDistortionConfig3D, grad_cs: list[np.ndarray], grad_ps: list[np.ndarray], compute_grad: bool, root_tol: float, root_max_iter: int, clearance_tol: float) -> tuple[float, float, GeometryActiveRecord3D | None, float]:
    B = np.asarray(feature.B, dtype=float)
    r = np.asarray(feature.r, dtype=float)
    m = B @ particle.c + r
    S = B @ Q @ B.T
    clearance = compute_projected_signed_clearance(m=m, S=S, tol=float(root_tol), max_iter=int(root_max_iter), center_tol=float(clearance_tol))
    s = float(clearance.s)
    grad_s_c = -B.T @ clearance.nu
    grad_s_p = _orientation_tangent_gradient_from_projected_data(a_axis=a_axis, b_axis=b_axis, p=particle.p, B=B, clearance=clearance)
    energy, violation, record = _accumulate_feature_contribution(particle_index=particle_index, feature_name=feature.name, feature_kind=feature.kind, s=s, gamma=float(clearance.gamma), d_safe=feature.d_safe, weight=feature.weight, active_tol=_active_violation_tolerance(d_safe=feature.d_safe, config=config), grad_s_c=grad_s_c, grad_s_p=grad_s_p, grad_cs=grad_cs, grad_ps=grad_ps, compute_grad=compute_grad)
    return (energy, violation, record, abs(s))

def evaluate_geometry_distortion_energy_and_gradients(particles: list[EllipsoidParticle3D], box: np.ndarray, config: GeometryDistortionConfig3D | None=None, *, physical_axes_override: Sequence[tuple[float, float]] | np.ndarray | None=None, features: BoxGeometryFeatures3D | None=None, candidate_cache: GeometryCandidateCache3D | None=None, compute_grad: bool=True, apply_lambda: bool=False, metrics: object | None=None) -> GeometryDistortionResult3D:
    _validate_particles(particles)
    box_array = _validate_box(box)
    if config is None:
        config = GeometryDistortionConfig3D()
    if not isinstance(config, GeometryDistortionConfig3D):
        raise TypeError(f'config must be GeometryDistortionConfig3D or None, got {type(config).__name__}.')
    n_particles = len(particles)
    grad_cs = [np.zeros(3, dtype=float) for _ in particles]
    grad_ps = [np.zeros(3, dtype=float) for _ in particles]
    if not config.enabled or n_particles == 0:
        return GeometryDistortionResult3D(total_energy=0.0, face_energy=0.0, edge_energy=0.0, corner_energy=0.0, grad_cs=grad_cs, grad_ps=grad_ps, num_active_face=0, num_active_edge=0, num_active_corner=0, worst_violation=0.0, max_face_violation=0.0, max_edge_violation=0.0, max_corner_violation=0.0, worst_particle_index=None, worst_feature_name=None, worst_feature_kind=None, min_abs_signed_clearance=float('inf'), active_records=[], candidate_stats={})
    if physical_axes_override is not None and len(physical_axes_override) != n_particles:
        raise ValueError(f'physical_axes_override must have the same length as particles; got {len(physical_axes_override)} and {n_particles}.')
    if features is None:
        features = build_box_geometry_features_3d(box_array, d_face=config.d_face, d_edge=config.d_edge, d_corner=config.d_corner, w_face=config.w_face, w_edge=config.w_edge, w_corner=config.w_corner, include_face=config.include_face, include_edge=config.include_edge, include_corner=config.include_corner)
    norm_factor = _normalization_factor(n_particles=n_particles, features=features, normalize_by=config.normalize_by)
    face_energy = 0.0
    edge_energy = 0.0
    corner_energy = 0.0
    num_active_face = 0
    num_active_edge = 0
    num_active_corner = 0
    worst_violation = 0.0
    max_face_violation = 0.0
    max_edge_violation = 0.0
    max_corner_violation = 0.0
    worst_particle_index: int | None = None
    worst_feature_name: str | None = None
    worst_feature_kind: str | None = None
    min_abs_s = float('inf')
    active_records: list[GeometryActiveRecord3D] = []
    stats = {'num_face_tests_total': int(n_particles * len(features.faces)), 'num_face_candidates': 0, 'num_face_skipped_by_filter': 0, 'num_edge_tests_total': int(n_particles * len(features.edges)), 'num_edge_candidates': 0, 'num_edge_skipped_by_filter': 0, 'num_corner_tests_total': int(n_particles * len(features.corners)), 'num_corner_candidates': 0, 'num_corner_skipped_by_filter': 0, 'num_corner_exact_distance_calls': 0}
    if candidate_cache is not None:
        if len(candidate_cache.face_indices_by_particle) != n_particles:
            raise ValueError('candidate_cache particle count does not match particles.')
        stats['num_face_candidates'] = int(len(candidate_cache.face_candidates))
        stats['num_edge_candidates'] = int(len(candidate_cache.edge_candidates))
        stats['num_corner_candidates'] = int(len(candidate_cache.corner_candidates))
        stats['num_face_skipped_by_filter'] = stats['num_face_tests_total'] - stats['num_face_candidates']
        stats['num_edge_skipped_by_filter'] = stats['num_edge_tests_total'] - stats['num_edge_candidates']
        stats['num_corner_skipped_by_filter'] = stats['num_corner_tests_total'] - stats['num_corner_candidates']
    for i, particle in enumerate(particles):
        a_axis, b_axis = _axes_for_particle(particle=particle, index=i, physical_axes_override=physical_axes_override if config.use_physical_axes else None)
        Q = build_q_matrix_closed_form(a_axis, b_axis, particle.p)
        if candidate_cache is not None:
            face_iter = [features.faces[k] for k in candidate_cache.face_indices_by_particle[i]]
        else:
            face_iter = []
            for face in features.faces:
                if bool(config.candidate_filter) and (not _face_candidate_passes(particle=particle, Q=Q, face=face, config=config)):
                    stats['num_face_skipped_by_filter'] += 1
                    continue
                stats['num_face_candidates'] += 1
                face_iter.append(face)
        for face in face_iter:
            energy, violation, record, abs_s = _evaluate_face_feature(particle=particle, particle_index=i, Q=Q, a_axis=a_axis, b_axis=b_axis, face=face, config=config, grad_cs=grad_cs, grad_ps=grad_ps, compute_grad=compute_grad)
            face_energy += energy
            min_abs_s = min(min_abs_s, abs_s)
            if record is not None:
                num_active_face += 1
                active_records.append(record)
            if violation > max_face_violation:
                max_face_violation = violation
            if violation > worst_violation:
                worst_violation = violation
                worst_particle_index = i
                worst_feature_name = face.name
                worst_feature_kind = 'face'
        if candidate_cache is not None:
            edge_iter = [features.edges[k] for k in candidate_cache.edge_indices_by_particle[i]]
        else:
            edge_iter = []
            for edge in features.edges:
                if bool(config.candidate_filter) and (not _edge_candidate_passes(particle=particle, a_axis=a_axis, edge=edge, config=config)):
                    stats['num_edge_skipped_by_filter'] += 1
                    continue
                stats['num_edge_candidates'] += 1
                edge_iter.append(edge)
        for edge in edge_iter:
            if config.edge_tangent_filter and (not _edge_tangent_filter_passes(particle=particle, Q=Q, edge=edge)):
                continue
            energy, violation, record, abs_s = _evaluate_projected_feature(particle=particle, particle_index=i, Q=Q, a_axis=a_axis, b_axis=b_axis, feature=edge, config=config, grad_cs=grad_cs, grad_ps=grad_ps, compute_grad=compute_grad, root_tol=config.root_tol, root_max_iter=config.root_max_iter, clearance_tol=config.clearance_tol)
            edge_energy += energy
            min_abs_s = min(min_abs_s, abs_s)
            if record is not None:
                num_active_edge += 1
                active_records.append(record)
            if violation > max_edge_violation:
                max_edge_violation = violation
            if violation > worst_violation:
                worst_violation = violation
                worst_particle_index = i
                worst_feature_name = edge.name
                worst_feature_kind = 'edge'
        if candidate_cache is not None:
            corner_iter = [features.corners[k] for k in candidate_cache.corner_indices_by_particle[i]]
        else:
            corner_iter = []
            for corner in features.corners:
                if bool(config.candidate_filter) and (not _corner_candidate_passes(particle=particle, a_axis=a_axis, corner=corner, config=config)):
                    stats['num_corner_skipped_by_filter'] += 1
                    continue
                stats['num_corner_candidates'] += 1
                corner_iter.append(corner)
        for corner in corner_iter:
            stats['num_corner_exact_distance_calls'] += 1
            energy, violation, record, abs_s = _evaluate_projected_feature(particle=particle, particle_index=i, Q=Q, a_axis=a_axis, b_axis=b_axis, feature=corner, config=config, grad_cs=grad_cs, grad_ps=grad_ps, compute_grad=compute_grad, root_tol=config.root_tol, root_max_iter=config.root_max_iter, clearance_tol=config.clearance_tol)
            corner_energy += energy
            min_abs_s = min(min_abs_s, abs_s)
            if record is not None:
                num_active_corner += 1
                active_records.append(record)
            if violation > max_corner_violation:
                max_corner_violation = violation
            if violation > worst_violation:
                worst_violation = violation
                worst_particle_index = i
                worst_feature_name = corner.name
                worst_feature_kind = 'corner'
    scale = 1.0 / norm_factor
    lambda_scale = float(config.lambda_geo) if apply_lambda else 1.0
    total_scale = scale * lambda_scale
    face_energy *= total_scale
    edge_energy *= total_scale
    corner_energy *= total_scale
    total_energy = face_energy + edge_energy + corner_energy
    if compute_grad:
        for i in range(n_particles):
            grad_cs[i] *= total_scale
            grad_ps[i] *= total_scale
    else:
        grad_cs = [np.zeros(3, dtype=float) for _ in particles]
        grad_ps = [np.zeros(3, dtype=float) for _ in particles]
    if metrics is not None:
        setattr(metrics, 'geometry_energy', float(total_energy))
        setattr(metrics, 'face_energy', float(face_energy))
        setattr(metrics, 'edge_energy', float(edge_energy))
        setattr(metrics, 'corner_energy', float(corner_energy))
        setattr(metrics, 'num_active_face_constraints', int(num_active_face))
        setattr(metrics, 'num_active_edge_constraints', int(num_active_edge))
        setattr(metrics, 'num_active_corner_constraints', int(num_active_corner))
        for key, value in stats.items():
            current = getattr(metrics, key, 0)
            setattr(metrics, key, int(current) + int(value))
        setattr(metrics, 'worst_geometry_violation', float(worst_violation))
        setattr(metrics, 'max_geometry_violation', float(worst_violation))
        setattr(metrics, 'max_face_violation', float(max_face_violation))
        setattr(metrics, 'max_edge_violation', float(max_edge_violation))
        setattr(metrics, 'max_corner_violation', float(max_corner_violation))
        setattr(metrics, 'max_face_geometry_violation', float(max_face_violation))
        setattr(metrics, 'max_edge_geometry_violation', float(max_edge_violation))
        setattr(metrics, 'max_corner_geometry_violation', float(max_corner_violation))
        setattr(metrics, 'worst_geometry_particle_id', worst_particle_index)
        setattr(metrics, 'worst_geometry_feature', worst_feature_name)
        setattr(metrics, 'min_abs_signed_clearance', float(min_abs_s))
    return GeometryDistortionResult3D(total_energy=float(total_energy), face_energy=float(face_energy), edge_energy=float(edge_energy), corner_energy=float(corner_energy), grad_cs=grad_cs, grad_ps=grad_ps, num_active_face=int(num_active_face), num_active_edge=int(num_active_edge), num_active_corner=int(num_active_corner), worst_violation=float(worst_violation), max_face_violation=float(max_face_violation), max_edge_violation=float(max_edge_violation), max_corner_violation=float(max_corner_violation), worst_particle_index=worst_particle_index, worst_feature_name=worst_feature_name, worst_feature_kind=worst_feature_kind, min_abs_signed_clearance=float(min_abs_s), active_records=active_records, candidate_stats=dict(stats))
