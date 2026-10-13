from __future__ import annotations
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.model.geometry_distortion_energy_3d import GeometryDistortionConfig3D, evaluate_geometry_distortion_energy_and_gradients
from src.model.geometry_features_3d import build_box_geometry_features_3d
from src.model.projected_clearance_3d import compute_projected_signed_clearance

def _relative_error(a: np.ndarray, b: np.ndarray, floor: float=1e-12) -> np.ndarray:
    a_array = np.asarray(a, dtype=float)
    b_array = np.asarray(b, dtype=float)
    denom = np.maximum(np.maximum(np.abs(a_array), np.abs(b_array)), float(floor))
    return np.abs(a_array - b_array) / denom

def _clone_particles(particles: list[EllipsoidParticle3D]) -> list[EllipsoidParticle3D]:
    return [particle.clone() for particle in particles]

def finite_difference_geometry_position_gradient(particles: list[EllipsoidParticle3D], box: np.ndarray, config: GeometryDistortionConfig3D, *, particle_index: int, eps: float=1e-06, physical_axes_override=None) -> np.ndarray:
    eps_value = float(eps)
    if eps_value <= 0.0:
        raise ValueError(f'eps must be > 0, got {eps_value}.')
    grad = np.zeros(3, dtype=float)
    basis = np.eye(3, dtype=float)
    for k in range(3):
        particles_plus = _clone_particles(particles)
        particles_minus = _clone_particles(particles)
        particles_plus[particle_index].c = particles_plus[particle_index].c + eps_value * basis[k]
        particles_minus[particle_index].c = particles_minus[particle_index].c - eps_value * basis[k]
        e_plus = evaluate_geometry_distortion_energy_and_gradients(particles_plus, box, config, physical_axes_override=physical_axes_override, compute_grad=False).total_energy
        e_minus = evaluate_geometry_distortion_energy_and_gradients(particles_minus, box, config, physical_axes_override=physical_axes_override, compute_grad=False).total_energy
        grad[k] = (e_plus - e_minus) / (2.0 * eps_value)
    return grad

def finite_difference_geometry_orientation_gradient(particles: list[EllipsoidParticle3D], box: np.ndarray, config: GeometryDistortionConfig3D, *, particle_index: int, eps: float=1e-06, physical_axes_override=None) -> np.ndarray:
    eps_value = float(eps)
    if eps_value <= 0.0:
        raise ValueError(f'eps must be > 0, got {eps_value}.')
    grad = np.zeros(3, dtype=float)
    basis = np.eye(3, dtype=float)
    for k in range(3):
        particles_plus = _clone_particles(particles)
        particles_minus = _clone_particles(particles)
        p0 = particles[particle_index].p
        particles_plus[particle_index].p = p0 + eps_value * basis[k]
        particles_plus[particle_index].p = particles_plus[particle_index].p / np.linalg.norm(particles_plus[particle_index].p)
        particles_minus[particle_index].p = p0 - eps_value * basis[k]
        particles_minus[particle_index].p = particles_minus[particle_index].p / np.linalg.norm(particles_minus[particle_index].p)
        e_plus = evaluate_geometry_distortion_energy_and_gradients(particles_plus, box, config, physical_axes_override=physical_axes_override, compute_grad=False).total_energy
        e_minus = evaluate_geometry_distortion_energy_and_gradients(particles_minus, box, config, physical_axes_override=physical_axes_override, compute_grad=False).total_energy
        grad[k] = (e_plus - e_minus) / (2.0 * eps_value)
    p_unit = particles[particle_index].p / np.linalg.norm(particles[particle_index].p)
    projector = np.eye(3, dtype=float) - np.outer(p_unit, p_unit)
    return projector @ grad

def check_geometry_distortion_gradient(particles: list[EllipsoidParticle3D], box: np.ndarray, config: GeometryDistortionConfig3D, *, particle_index: int=0, eps: float=1e-06, physical_axes_override=None) -> dict:
    analytic = evaluate_geometry_distortion_energy_and_gradients(particles, box, config, physical_axes_override=physical_axes_override, compute_grad=True)
    fd_c = finite_difference_geometry_position_gradient(particles, box, config, particle_index=particle_index, eps=eps, physical_axes_override=physical_axes_override)
    fd_p = finite_difference_geometry_orientation_gradient(particles, box, config, particle_index=particle_index, eps=eps, physical_axes_override=physical_axes_override)
    analytic_c = analytic.grad_cs[particle_index]
    analytic_p = analytic.grad_ps[particle_index]
    return {'energy': analytic.total_energy, 'analytic_c': analytic_c, 'fd_c': fd_c, 'abs_error_c': np.abs(analytic_c - fd_c), 'rel_error_c': _relative_error(analytic_c, fd_c), 'analytic_p': analytic_p, 'fd_p': fd_p, 'abs_error_p': np.abs(analytic_p - fd_p), 'rel_error_p': _relative_error(analytic_p, fd_p), 'num_active_face': analytic.num_active_face, 'num_active_edge': analytic.num_active_edge, 'num_active_corner': analytic.num_active_corner, 'worst_violation': analytic.worst_violation, 'worst_feature_name': analytic.worst_feature_name}

def check_single_feature_states() -> dict:
    box = np.array([1.0, 1.0, 1.0], dtype=float)
    config = GeometryDistortionConfig3D(include_face=True, include_edge=False, include_corner=True, d_face=0.05, d_edge=0.05, d_corner=0.05, normalize_by='none')
    p_safe = EllipsoidParticle3D(a=0.1, b=0.05, c=np.array([0.3, 0.5, 0.5]), p=np.array([1.0, 0.0, 0.0]))
    safe_result = evaluate_geometry_distortion_energy_and_gradients([p_safe], box, config)
    p_gap = EllipsoidParticle3D(a=0.1, b=0.05, c=np.array([0.13, 0.5, 0.5]), p=np.array([1.0, 0.0, 0.0]))
    gap_result = evaluate_geometry_distortion_energy_and_gradients([p_gap], box, config)
    p_deep = EllipsoidParticle3D(a=0.2, b=0.05, c=np.array([0.08, 0.5, 0.5]), p=np.array([1.0, 0.0, 0.0]))
    deep_result = evaluate_geometry_distortion_energy_and_gradients([p_deep], box, config)
    return {'safe_outside_energy': safe_result.total_energy, 'safe_outside_active_face': safe_result.num_active_face, 'outer_gap_energy': gap_result.total_energy, 'outer_gap_active_face': gap_result.num_active_face, 'deep_crossing_energy': deep_result.total_energy, 'deep_crossing_active_face': deep_result.num_active_face}

def check_projected_clearance_basic() -> dict:
    S2 = np.diag([4.0, 1.0])
    outside = compute_projected_signed_clearance(np.array([3.0, 0.2]), S2)
    inside = compute_projected_signed_clearance(np.array([0.2, 0.1]), S2)
    S3 = np.diag([4.0, 1.0, 1.0])
    outside3 = compute_projected_signed_clearance(np.array([3.0, 0.2, 0.1]), S3)
    inside3 = compute_projected_signed_clearance(np.array([0.2, 0.1, 0.1]), S3)
    return {'outside_2d_s': outside.s, 'outside_2d_status': outside.status, 'inside_2d_s': inside.s, 'inside_2d_status': inside.status, 'outside_3d_s': outside3.s, 'outside_3d_status': outside3.status, 'inside_3d_s': inside3.s, 'inside_3d_status': inside3.status}

def audit_geometry_distortion_3d(particles: list[EllipsoidParticle3D], box: np.ndarray, config: GeometryDistortionConfig3D, *, physical_axes_override=None, feasible_violation_tol: float | None=None, feasible_face_violation_tol: float | None=None, feasible_edge_violation_tol: float | None=None, feasible_corner_violation_tol: float | None=None) -> dict:
    result = evaluate_geometry_distortion_energy_and_gradients(particles, box, config, physical_axes_override=physical_axes_override, compute_grad=False)
    active_records = [record.__dict__.copy() for record in result.active_records]
    raw_counts = {'face': int(result.num_active_face), 'edge': int(result.num_active_edge), 'corner': int(result.num_active_corner)}
    if feasible_violation_tol is None:
        default_feasible_tol = None
    else:
        default_feasible_tol = float(feasible_violation_tol)
        if default_feasible_tol < 0.0:
            raise ValueError(f'feasible_violation_tol must be >= 0 when provided, got {feasible_violation_tol}.')

    def _tol_or_default(value: float | None, default: float | None) -> float | None:
        if value is None:
            return default
        out = float(value)
        if out < 0.0:
            raise ValueError(f'feasible per-kind violation tolerances must be >= 0, got {out}.')
        return out
    feasible_tols = {'face': _tol_or_default(feasible_face_violation_tol, default_feasible_tol), 'edge': _tol_or_default(feasible_edge_violation_tol, default_feasible_tol), 'corner': _tol_or_default(feasible_corner_violation_tol, default_feasible_tol)}
    if all((value is None for value in feasible_tols.values())):
        feasible_records = active_records
        feasible_counts = dict(raw_counts)
    else:
        feasible_records = []
        for record in active_records:
            kind = str(record.get('feature_kind'))
            tol = feasible_tols.get(kind)
            if tol is None:
                keep = True
            else:
                keep = float(record.get('violation', 0.0)) > float(tol)
            if keep:
                feasible_records.append(record)
        feasible_counts = {'face': sum((1 for record in feasible_records if record.get('feature_kind') == 'face')), 'edge': sum((1 for record in feasible_records if record.get('feature_kind') == 'edge')), 'corner': sum((1 for record in feasible_records if record.get('feature_kind') == 'corner'))}
    return {'geometry_energy': float(result.total_energy), 'face_energy': float(result.face_energy), 'edge_energy': float(result.edge_energy), 'corner_energy': float(result.corner_energy), 'max_face_violation': float(result.max_face_violation), 'max_edge_violation': float(result.max_edge_violation), 'max_corner_violation': float(result.max_corner_violation), 'max_geometry_violation': float(result.worst_violation), 'num_face_forbidden': int(raw_counts['face']), 'num_edge_forbidden': int(raw_counts['edge']), 'num_corner_forbidden': int(raw_counts['corner']), 'num_face_forbidden_raw': int(raw_counts['face']), 'num_edge_forbidden_raw': int(raw_counts['edge']), 'num_corner_forbidden_raw': int(raw_counts['corner']), 'feasible_violation_tol': default_feasible_tol, 'feasible_face_violation_tol': feasible_tols['face'], 'feasible_edge_violation_tol': feasible_tols['edge'], 'feasible_corner_violation_tol': feasible_tols['corner'], 'num_face_forbidden_feasible': int(feasible_counts['face']), 'num_edge_forbidden_feasible': int(feasible_counts['edge']), 'num_corner_forbidden_feasible': int(feasible_counts['corner']), 'worst_violation': float(result.worst_violation), 'worst_particle_index': result.worst_particle_index, 'worst_feature_name': result.worst_feature_name, 'worst_feature_kind': result.worst_feature_kind, 'min_abs_signed_clearance': float(result.min_abs_signed_clearance), 'candidate_stats': dict(getattr(result, 'candidate_stats', {})), 'active_records': active_records, 'active_records_feasible': feasible_records}
