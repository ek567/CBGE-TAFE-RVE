from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.data.pair_result_3d import PairContactResult3D
from src.model.orientation_regularization_3d import compute_orientation_regularization_energy, compute_orientation_regularization_gradients
from src.model.geometry_distortion_energy_3d import GeometryCandidateCache3D, GeometryDistortionConfig3D, evaluate_geometry_distortion_energy_and_gradients
from src.model.pair_contact_geometry_3d import attach_geometry_to_pair_result, attach_geometry_to_pair_result_from_Q
from src.model.pair_contact_measure_pw_3d import compute_pair_contact_measure
from src.model.pair_gradient_3d import pair_orientation_gradient_analytic, pair_position_gradients
from src.model.step_context_3d import StepContext3D, build_step_context_3d, build_particle_geom_cache_3d
from src.monitoring.metrics_3d import StepMetrics3D
from src.monitoring.timers_3d import timed_field
from src.overlap_detection.candidate_pairs_3d import CandidatePairConfig3D, generate_candidate_pairs
from src.overlap_detection.directional_extent_3d import definitely_separated_by_directional_extent
from src.periodic.mic_3d import minimum_image_vector

def _validate_particles(particles: list[EllipsoidParticle3D]) -> None:
    if not isinstance(particles, list):
        raise TypeError('particles must be a list of EllipsoidParticle3D.')
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise TypeError(f'particles[{idx}] must be an EllipsoidParticle3D, got {type(particle).__name__}.')

def _validate_box(box: np.ndarray | None) -> np.ndarray | None:
    if box is None:
        return None
    box_array = np.asarray(box, dtype=float)
    if box_array.shape != (3,):
        raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
    if np.any(box_array <= 0.0):
        raise ValueError(f'Box lengths must be positive, got {box_array}.')
    return box_array

def iter_all_pairs(n: int) -> list[tuple[int, int]]:
    n_int = int(n)
    if n_int < 0:
        raise ValueError(f'Expected n >= 0, got {n}.')
    return [(i, j) for i in range(n_int) for j in range(i + 1, n_int)]

def evaluate_all_pairs(particles: list[EllipsoidParticle3D], box: np.ndarray | None=None, use_mic: bool=False, attach_geometry: bool=True, pair_config: CandidatePairConfig3D | None=None, metrics: StepMetrics3D | None=None, *, context: StepContext3D | None=None, geometry_attach_mode: Literal['all', 'overlap_only']='all', use_extent_filter: bool=False, extent_sep_tol: float=1e-12) -> list[PairContactResult3D]:
    _validate_particles(particles)
    box_array = _validate_box(box)
    if extent_sep_tol < 0.0:
        raise ValueError(f'extent_sep_tol must be >= 0, got {extent_sep_tol}.')
    if metrics is not None and (not isinstance(metrics, StepMetrics3D)):
        raise TypeError(f'metrics must be StepMetrics3D or None, got {type(metrics).__name__}.')
    if pair_config is None:
        pair_config = CandidatePairConfig3D()
    if not isinstance(pair_config, CandidatePairConfig3D):
        raise TypeError(f'pair_config must be CandidatePairConfig3D, got {type(pair_config).__name__}.')
    if context is None:
        context = build_step_context_3d(particles, box=box_array, use_mic=bool(use_mic), pair_config=pair_config, use_pw85_q=True, cache_Q_full=bool(attach_geometry), metrics=metrics)
    candidate_pairs = context.candidate_pairs
    q_sym = context.particle_cache.q_sym
    Q_full = context.particle_cache.Q_full
    pair_results: list[PairContactResult3D] = []
    if metrics is not None:
        metrics.num_overlap_pairs = 0
        metrics.num_overlapping_pairs = 0
        metrics.max_overlap = 0.0
        metrics.num_geometry_attached_pairs = 0
        metrics.num_extent_tests = 0
        metrics.num_pairs_rejected_by_extent = 0
        metrics.time_extent_filter = float(getattr(metrics, 'time_extent_filter', 0.0))
        metrics.time_pw_contact = float(getattr(metrics, 'time_pw_contact', 0.0))
    out = np.empty(2, dtype=float)
    max_overlap = 0.0
    num_overlap_pairs = 0
    extent_time = 0.0
    pw_contact_time = 0.0
    with timed_field(metrics, 'time_pw_eval'):
        for i, j in candidate_pairs:
            i_int = int(i)
            j_int = int(j)
            particle_i = particles[i_int]
            particle_j = particles[j_int]
            if use_mic:
                if box_array is None:
                    raise ValueError('box must be provided when use_mic=True.')
                r_ij = minimum_image_vector(particle_i.c, particle_j.c, box_array)
            else:
                r_ij = np.asarray(particle_j.c - particle_i.c, dtype=float)
            if use_extent_filter:
                if metrics is not None:
                    metrics.num_extent_tests += 1
                t0 = perf_counter() if metrics is not None else 0.0
                rejected = definitely_separated_by_directional_extent(a_i=float(particle_i.a), b_i=float(particle_i.b), p_i=np.asarray(particle_i.p, dtype=float), a_j=float(particle_j.a), b_j=float(particle_j.b), p_j=np.asarray(particle_j.p, dtype=float), r_ij=np.asarray(r_ij, dtype=float), sep_tol_abs=float(extent_sep_tol))
                if metrics is not None:
                    extent_time += perf_counter() - t0
                if rejected:
                    if metrics is not None:
                        metrics.num_pairs_rejected_by_extent += 1
                    continue
            t0 = perf_counter() if metrics is not None else 0.0
            F_ij, lambda_star = compute_pair_contact_measure(r_ij=r_ij, q_i=q_sym[i_int], q_j=q_sym[j_int], out=out, validate_shapes=False)
            if metrics is not None:
                pw_contact_time += perf_counter() - t0
            gap = 1.0 - float(F_ij)
            overlap = float(gap) if gap > 0.0 else 0.0
            psi_ij = 0.5 * overlap * overlap
            pr = PairContactResult3D(i=i_int, j=j_int, r_ij=np.asarray(r_ij, dtype=float), F_ij=float(F_ij), lambda_star=float(lambda_star), overlap=float(overlap), psi_ij=float(psi_ij), Q_star=None, s_ij=None, x0_ij=None)
            if metrics is not None:
                metrics.num_pw_calls += 1
            if overlap > 0.0:
                num_overlap_pairs += 1
                if overlap > max_overlap:
                    max_overlap = overlap
            if attach_geometry:
                if geometry_attach_mode == 'overlap_only':
                    do_attach = overlap > 0.0
                elif geometry_attach_mode == 'all':
                    do_attach = True
                else:
                    raise ValueError(f'Unknown geometry_attach_mode={geometry_attach_mode!r}')
                if do_attach:
                    with timed_field(metrics, 'time_geometry_attach'):
                        if Q_full is not None:
                            pr = attach_geometry_to_pair_result_from_Q(pair_result=pr, c_i=particle_i.c, Q_i=Q_full[i_int], Q_j=Q_full[j_int])
                        else:
                            pr = attach_geometry_to_pair_result(particle_i=particle_i, particle_j=particle_j, pair_result=pr)
                    if metrics is not None:
                        metrics.num_geometry_attached_pairs += 1
            pair_results.append(pr)
    if metrics is not None:
        metrics.num_overlap_pairs = int(num_overlap_pairs)
        metrics.num_overlapping_pairs = int(num_overlap_pairs)
        metrics.max_overlap = float(max_overlap) if len(pair_results) > 0 else 0.0
        metrics.time_extent_filter += float(extent_time)
        metrics.time_pw_contact += float(pw_contact_time)
    return pair_results

def compute_total_potential(pair_results: list[PairContactResult3D], particles: list[EllipsoidParticle3D] | None=None, include_orientation_regularization: bool=False, *, box: np.ndarray | None=None, include_geometry_distortion: bool=False, geometry_config: GeometryDistortionConfig3D | None=None, physical_axes_override: Any | None=None, geometry_candidate_cache: GeometryCandidateCache3D | None=None, orientation_target: str='3d-isotropic') -> float:
    if not isinstance(pair_results, list):
        raise TypeError('pair_results must be a list of PairContactResult3D.')
    for idx, pair_result in enumerate(pair_results):
        if not isinstance(pair_result, PairContactResult3D):
            raise TypeError(f'pair_results[{idx}] must be PairContactResult3D, got {type(pair_result).__name__}.')
    total_energy = 0.0
    for pair_result in pair_results:
        total_energy += float(pair_result.psi_ij)
    if include_orientation_regularization:
        if particles is None:
            raise ValueError('particles must be provided when include_orientation_regularization=True.')
        _validate_particles(particles)
        total_energy += compute_orientation_regularization_energy(particles, orientation_target=orientation_target)
    if include_geometry_distortion:
        if particles is None:
            raise ValueError('particles must be provided when include_geometry_distortion=True.')
        _validate_particles(particles)
        box_array = _validate_box(box)
        if box_array is None:
            raise ValueError('box must be provided when include_geometry_distortion=True.')
        if geometry_config is None:
            geometry_config = GeometryDistortionConfig3D()
        if not isinstance(geometry_config, GeometryDistortionConfig3D):
            raise TypeError(f'geometry_config must be GeometryDistortionConfig3D or None, got {type(geometry_config).__name__}.')
        geo_result = evaluate_geometry_distortion_energy_and_gradients(particles=particles, box=box_array, config=geometry_config, physical_axes_override=physical_axes_override, candidate_cache=geometry_candidate_cache, compute_grad=False, apply_lambda=True, metrics=None)
        total_energy += float(geo_result.total_energy)
    return float(total_energy)

def compute_total_gradients(particles: list[EllipsoidParticle3D], pair_results: list[PairContactResult3D] | None=None, box: np.ndarray | None=None, use_mic: bool=False, pair_config: CandidatePairConfig3D | None=None, metrics: StepMetrics3D | None=None, include_orientation_regularization: bool=False, include_geometry_distortion: bool=False, geometry_config: GeometryDistortionConfig3D | None=None, physical_axes_override: Any | None=None, geometry_candidate_cache: GeometryCandidateCache3D | None=None, orientation_target: str='3d-isotropic') -> tuple[list[np.ndarray], list[np.ndarray], list[PairContactResult3D]]:
    _validate_particles(particles)
    box_array = _validate_box(box)
    if metrics is not None and (not isinstance(metrics, StepMetrics3D)):
        raise TypeError(f'metrics must be StepMetrics3D or None, got {type(metrics).__name__}.')
    if pair_results is None:
        pair_results = evaluate_all_pairs(particles=particles, box=box_array, use_mic=bool(use_mic), attach_geometry=True, geometry_attach_mode='overlap_only', pair_config=pair_config, metrics=metrics)
    grad_cs = [np.zeros(3, dtype=float) for _ in particles]
    grad_ps = [np.zeros(3, dtype=float) for _ in particles]
    with timed_field(metrics, 'time_gradient_eval'):
        for idx, pair_result in enumerate(pair_results):
            if not isinstance(pair_result, PairContactResult3D):
                raise TypeError(f'pair_results[{idx}] must be PairContactResult3D, got {type(pair_result).__name__}.')
            i = int(pair_result.i)
            j = int(pair_result.j)
            particle_i = particles[i]
            particle_j = particles[j]
            grad_c_i, grad_c_j = pair_position_gradients(particle_i=particle_i, particle_j=particle_j, pair_result=pair_result)
            grad_p_i, grad_p_j = pair_orientation_gradient_analytic(particle_i=particle_i, particle_j=particle_j, pair_result=pair_result)
            grad_cs[i] += grad_c_i
            grad_cs[j] += grad_c_j
            grad_ps[i] += grad_p_i
            grad_ps[j] += grad_p_j
    if include_orientation_regularization:
        orientation_grads = compute_orientation_regularization_gradients(particles, orientation_target=orientation_target)
        for i in range(len(grad_ps)):
            grad_ps[i] += orientation_grads[i]
    if include_geometry_distortion:
        if box_array is None:
            raise ValueError('box must be provided when include_geometry_distortion=True.')
        if geometry_config is None:
            geometry_config = GeometryDistortionConfig3D()
        if not isinstance(geometry_config, GeometryDistortionConfig3D):
            raise TypeError(f'geometry_config must be GeometryDistortionConfig3D or None, got {type(geometry_config).__name__}.')
        geo_result = evaluate_geometry_distortion_energy_and_gradients(particles=particles, box=box_array, config=geometry_config, physical_axes_override=physical_axes_override, candidate_cache=geometry_candidate_cache, compute_grad=True, apply_lambda=True, metrics=metrics)
        for i in range(len(particles)):
            grad_cs[i] += geo_result.grad_cs[i]
            grad_ps[i] += geo_result.grad_ps[i]
    return (grad_cs, grad_ps, pair_results)

def max_overlap_info(pair_results: list[PairContactResult3D]) -> dict[str, Any]:
    if len(pair_results) == 0:
        return {'max_overlap': 0.0, 'pair_indices': None, 'F_ij': 0.0, 'lambda_star': 0.0, 'psi_ij': 0.0}
    max_pair = max(pair_results, key=lambda result: float(result.overlap))
    return {'max_overlap': float(max_pair.overlap), 'pair_indices': (int(max_pair.i), int(max_pair.j)), 'F_ij': float(max_pair.F_ij), 'lambda_star': float(max_pair.lambda_star), 'psi_ij': float(max_pair.psi_ij)}

@dataclass(frozen=True)
class FastEnergyEvalResult3D:
    energy: float
    overlap_energy: float
    orientation_energy: float
    geometry_energy: float
    face_energy: float
    edge_energy: float
    corner_energy: float
    max_overlap: float
    num_overlapping_pairs: int
    num_pw_calls: int
    early_aborted: bool

def evaluate_total_energy_fast(particles: list[EllipsoidParticle3D], *, box: np.ndarray | None, use_mic: bool, pair_config: CandidatePairConfig3D | None, include_orientation_regularization: bool=False, orientation_target: str='3d-isotropic', include_geometry_distortion: bool=False, geometry_config: GeometryDistortionConfig3D | None=None, physical_axes_override: Any | None=None, geometry_candidate_cache: GeometryCandidateCache3D | None=None, energy_abort_threshold: float | None=None, overlap_count_tol: float=0.0, candidate_pairs_override: list[tuple[int, int]] | None=None, use_extent_filter: bool=False, extent_sep_tol: float=1e-12, metrics: StepMetrics3D | None=None) -> FastEnergyEvalResult3D:
    _validate_particles(particles)
    box_array = _validate_box(box)
    if extent_sep_tol < 0.0:
        raise ValueError(f'extent_sep_tol must be >= 0, got {extent_sep_tol}.')
    if metrics is not None and (not isinstance(metrics, StepMetrics3D)):
        raise TypeError(f'metrics must be StepMetrics3D or None, got {type(metrics).__name__}.')
    if pair_config is None:
        pair_config = CandidatePairConfig3D()
    orientation_energy = float(compute_orientation_regularization_energy(particles, orientation_target=orientation_target)) if include_orientation_regularization else 0.0
    geometry_energy = 0.0
    face_energy = 0.0
    edge_energy = 0.0
    corner_energy = 0.0
    if include_geometry_distortion:
        if box_array is None:
            raise ValueError('box must be provided when include_geometry_distortion=True.')
        if geometry_config is None:
            geometry_config = GeometryDistortionConfig3D()
        if not isinstance(geometry_config, GeometryDistortionConfig3D):
            raise TypeError(f'geometry_config must be GeometryDistortionConfig3D or None, got {type(geometry_config).__name__}.')
        geo_result = evaluate_geometry_distortion_energy_and_gradients(particles=particles, box=box_array, config=geometry_config, physical_axes_override=physical_axes_override, candidate_cache=geometry_candidate_cache, compute_grad=False, apply_lambda=True, metrics=metrics)
        geometry_energy = float(geo_result.total_energy)
        face_energy = float(geo_result.face_energy)
        edge_energy = float(geo_result.edge_energy)
        corner_energy = float(geo_result.corner_energy)
    fixed_energy = orientation_energy + geometry_energy
    if energy_abort_threshold is not None and fixed_energy > float(energy_abort_threshold):
        return FastEnergyEvalResult3D(energy=float(fixed_energy), overlap_energy=0.0, orientation_energy=float(orientation_energy), geometry_energy=float(geometry_energy), face_energy=float(face_energy), edge_energy=float(edge_energy), corner_energy=float(corner_energy), max_overlap=0.0, num_overlapping_pairs=0, num_pw_calls=0, early_aborted=True)
    overlap_abort_threshold = None
    if energy_abort_threshold is not None:
        overlap_abort_threshold = float(energy_abort_threshold) - fixed_energy
    if candidate_pairs_override is not None:
        candidate_pairs = candidate_pairs_override
    else:
        candidate_pairs = generate_candidate_pairs(particles=particles, box=box_array, use_mic=bool(use_mic), config=pair_config, metrics=None)
    if len(candidate_pairs) > 0:
        particle_cache = build_particle_geom_cache_3d(particles, use_pw85_q=True, cache_Q_full=False)
        q_sym = particle_cache.q_sym
    else:
        q_sym = []
    out = np.empty(2, dtype=float)
    overlap_energy = 0.0
    max_overlap = 0.0
    num_overlapping_pairs = 0
    num_pw_calls = 0
    early_aborted = False
    count_tol = float(overlap_count_tol)
    extent_time = 0.0
    pw_contact_time = 0.0
    for i, j in candidate_pairs:
        i_int = int(i)
        j_int = int(j)
        pi = particles[i_int]
        pj = particles[j_int]
        if use_mic:
            if box_array is None:
                raise ValueError('box must be provided when use_mic=True.')
            r_ij = minimum_image_vector(pi.c, pj.c, box_array)
        else:
            r_ij = np.asarray(pj.c - pi.c, dtype=float)
        if use_extent_filter:
            t0 = perf_counter() if metrics is not None else 0.0
            rejected = definitely_separated_by_directional_extent(a_i=float(pi.a), b_i=float(pi.b), p_i=np.asarray(pi.p, dtype=float), a_j=float(pj.a), b_j=float(pj.b), p_j=np.asarray(pj.p, dtype=float), r_ij=np.asarray(r_ij, dtype=float), sep_tol_abs=float(extent_sep_tol))
            if metrics is not None:
                extent_time += perf_counter() - t0
            if rejected:
                continue
        t0 = perf_counter() if metrics is not None else 0.0
        F_ij, _lam = compute_pair_contact_measure(r_ij=r_ij, q_i=q_sym[i_int], q_j=q_sym[j_int], out=out, validate_shapes=False)
        if metrics is not None:
            pw_contact_time += perf_counter() - t0
        num_pw_calls += 1
        gap = 1.0 - float(F_ij)
        if gap > 0.0:
            overlap = float(gap)
            if overlap > max_overlap:
                max_overlap = overlap
            overlap_energy += 0.5 * overlap * overlap
            if overlap > count_tol:
                num_overlapping_pairs += 1
            if overlap_abort_threshold is not None and overlap_energy > overlap_abort_threshold:
                early_aborted = True
                break
    total_energy = overlap_energy + orientation_energy + geometry_energy
    if energy_abort_threshold is not None and total_energy > float(energy_abort_threshold):
        early_aborted = True
    if metrics is not None:
        metrics.time_fast_extent_filter += float(extent_time)
        metrics.time_fast_pw_contact += float(pw_contact_time)
    return FastEnergyEvalResult3D(energy=float(total_energy), overlap_energy=float(overlap_energy), orientation_energy=float(orientation_energy), geometry_energy=float(geometry_energy), face_energy=float(face_energy), edge_energy=float(edge_energy), corner_energy=float(corner_energy), max_overlap=float(max_overlap), num_overlapping_pairs=int(num_overlapping_pairs), num_pw_calls=int(num_pw_calls), early_aborted=bool(early_aborted))

def _compute_step_objective_and_gradients(particles: list[EllipsoidParticle3D], pair_results: list[PairContactResult3D], *, box: np.ndarray | None=None, use_mic: bool=False, pair_config: CandidatePairConfig3D | None=None, metrics: StepMetrics3D | None=None, include_orientation_regularization: bool=False, include_geometry_distortion: bool=False, geometry_config: GeometryDistortionConfig3D | None=None, physical_axes_override: Any | None=None, geometry_candidate_cache: GeometryCandidateCache3D | None=None, orientation_target: str='3d-isotropic') -> tuple[float, list[np.ndarray], list[np.ndarray], list[PairContactResult3D]]:
    shared_geometry_result: Any | None = None

    def _geometry_result_shareable(result: Any) -> bool:
        if result is None:
            return False
        required_scalars = ('total_energy', 'face_energy', 'edge_energy', 'corner_energy', 'worst_violation', 'max_face_violation', 'max_edge_violation', 'max_corner_violation')
        try:
            for name in required_scalars:
                if not np.isfinite(float(getattr(result, name))):
                    return False
            grad_cs = getattr(result, 'grad_cs')
            grad_ps = getattr(result, 'grad_ps')
            if not isinstance(grad_cs, list) or not isinstance(grad_ps, list):
                return False
            if len(grad_cs) != len(particles) or len(grad_ps) != len(particles):
                return False
            for grad in [*grad_cs, *grad_ps]:
                arr = np.asarray(grad, dtype=float)
                if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                    return False
            for name in ('num_active_face', 'num_active_edge', 'num_active_corner'):
                int(getattr(result, name))
            getattr(result, 'worst_particle_index')
            getattr(result, 'worst_feature_name')
            getattr(result, 'worst_feature_kind')
            float(getattr(result, 'min_abs_signed_clearance'))
            candidate_stats = getattr(result, 'candidate_stats')
            if not isinstance(candidate_stats, dict):
                return False
            for key, value in candidate_stats.items():
                if not isinstance(key, str):
                    return False
                int(value)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        return True

    def _publish_geometry_metrics(result: Any) -> None:
        if metrics is None:
            return
        setattr(metrics, 'geometry_energy', float(result.total_energy))
        setattr(metrics, 'face_energy', float(result.face_energy))
        setattr(metrics, 'edge_energy', float(result.edge_energy))
        setattr(metrics, 'corner_energy', float(result.corner_energy))
        setattr(metrics, 'num_active_face_constraints', int(result.num_active_face))
        setattr(metrics, 'num_active_edge_constraints', int(result.num_active_edge))
        setattr(metrics, 'num_active_corner_constraints', int(result.num_active_corner))
        for key, value in result.candidate_stats.items():
            current = getattr(metrics, key, 0)
            setattr(metrics, key, int(current) + int(value))
        setattr(metrics, 'worst_geometry_violation', float(result.worst_violation))
        setattr(metrics, 'max_geometry_violation', float(result.worst_violation))
        setattr(metrics, 'max_face_violation', float(result.max_face_violation))
        setattr(metrics, 'max_edge_violation', float(result.max_edge_violation))
        setattr(metrics, 'max_corner_violation', float(result.max_corner_violation))
        setattr(metrics, 'max_face_geometry_violation', float(result.max_face_violation))
        setattr(metrics, 'max_edge_geometry_violation', float(result.max_edge_violation))
        setattr(metrics, 'max_corner_geometry_violation', float(result.max_corner_violation))
        setattr(metrics, 'worst_geometry_particle_id', result.worst_particle_index)
        setattr(metrics, 'worst_geometry_feature', result.worst_feature_name)
        setattr(metrics, 'min_abs_signed_clearance', float(result.min_abs_signed_clearance))

    def _energy_geometry(*, local_particles: list[EllipsoidParticle3D], local_box: np.ndarray, local_config: GeometryDistortionConfig3D, local_physical_axes_override: Any | None, local_candidate_cache: GeometryCandidateCache3D | None) -> Any:
        nonlocal shared_geometry_result
        shared_geometry_result = None
        try:
            candidate = evaluate_geometry_distortion_energy_and_gradients(particles=local_particles, box=local_box, config=local_config, physical_axes_override=local_physical_axes_override, candidate_cache=local_candidate_cache, compute_grad=True, apply_lambda=True, metrics=None)
        except Exception:
            candidate = None
        if _geometry_result_shareable(candidate):
            shared_geometry_result = candidate
            return candidate
        return evaluate_geometry_distortion_energy_and_gradients(particles=local_particles, box=local_box, config=local_config, physical_axes_override=local_physical_axes_override, candidate_cache=local_candidate_cache, compute_grad=False, apply_lambda=True, metrics=None)

    def _gradient_geometry(*, local_particles: list[EllipsoidParticle3D], local_box: np.ndarray, local_config: GeometryDistortionConfig3D, local_physical_axes_override: Any | None, local_candidate_cache: GeometryCandidateCache3D | None) -> Any:
        nonlocal shared_geometry_result
        result, shared_geometry_result = (shared_geometry_result, None)
        if result is not None:
            _publish_geometry_metrics(result)
            return result
        return evaluate_geometry_distortion_energy_and_gradients(particles=local_particles, box=local_box, config=local_config, physical_axes_override=local_physical_axes_override, candidate_cache=local_candidate_cache, compute_grad=True, apply_lambda=True, metrics=metrics)

    def _step_total_potential() -> float:
        if not isinstance(pair_results, list):
            raise TypeError('pair_results must be a list of PairContactResult3D.')
        for idx, pair_result in enumerate(pair_results):
            if not isinstance(pair_result, PairContactResult3D):
                raise TypeError(f'pair_results[{idx}] must be PairContactResult3D, got {type(pair_result).__name__}.')
        total_energy = 0.0
        for pair_result in pair_results:
            total_energy += float(pair_result.psi_ij)
        if include_orientation_regularization:
            _validate_particles(particles)
            total_energy += compute_orientation_regularization_energy(particles, orientation_target=orientation_target)
        if include_geometry_distortion:
            _validate_particles(particles)
            box_array = _validate_box(box)
            if box_array is None:
                raise ValueError('box must be provided when include_geometry_distortion=True.')
            local_geometry_config = geometry_config
            if local_geometry_config is None:
                local_geometry_config = GeometryDistortionConfig3D()
            if not isinstance(local_geometry_config, GeometryDistortionConfig3D):
                raise TypeError(f'geometry_config must be GeometryDistortionConfig3D or None, got {type(local_geometry_config).__name__}.')
            geo_result = _energy_geometry(local_particles=particles, local_box=box_array, local_config=local_geometry_config, local_physical_axes_override=physical_axes_override, local_candidate_cache=geometry_candidate_cache)
            total_energy += float(geo_result.total_energy)
        return float(total_energy)

    def _step_total_gradients() -> tuple[list[np.ndarray], list[np.ndarray], list[PairContactResult3D]]:
        _validate_particles(particles)
        box_array = _validate_box(box)
        if metrics is not None and (not isinstance(metrics, StepMetrics3D)):
            raise TypeError(f'metrics must be StepMetrics3D or None, got {type(metrics).__name__}.')
        local_pair_results = pair_results
        grad_cs = [np.zeros(3, dtype=float) for _ in particles]
        grad_ps = [np.zeros(3, dtype=float) for _ in particles]
        with timed_field(metrics, 'time_gradient_eval'):
            for idx, pair_result in enumerate(local_pair_results):
                if not isinstance(pair_result, PairContactResult3D):
                    raise TypeError(f'pair_results[{idx}] must be PairContactResult3D, got {type(pair_result).__name__}.')
                i = int(pair_result.i)
                j = int(pair_result.j)
                particle_i = particles[i]
                particle_j = particles[j]
                grad_c_i, grad_c_j = pair_position_gradients(particle_i=particle_i, particle_j=particle_j, pair_result=pair_result)
                grad_p_i, grad_p_j = pair_orientation_gradient_analytic(particle_i=particle_i, particle_j=particle_j, pair_result=pair_result)
                grad_cs[i] += grad_c_i
                grad_cs[j] += grad_c_j
                grad_ps[i] += grad_p_i
                grad_ps[j] += grad_p_j
        if include_orientation_regularization:
            orientation_grads = compute_orientation_regularization_gradients(particles, orientation_target=orientation_target)
            for i in range(len(grad_ps)):
                grad_ps[i] += orientation_grads[i]
        if include_geometry_distortion:
            if box_array is None:
                raise ValueError('box must be provided when include_geometry_distortion=True.')
            local_geometry_config = geometry_config
            if local_geometry_config is None:
                local_geometry_config = GeometryDistortionConfig3D()
            if not isinstance(local_geometry_config, GeometryDistortionConfig3D):
                raise TypeError(f'geometry_config must be GeometryDistortionConfig3D or None, got {type(local_geometry_config).__name__}.')
            geo_result = _gradient_geometry(local_particles=particles, local_box=box_array, local_config=local_geometry_config, local_physical_axes_override=physical_axes_override, local_candidate_cache=geometry_candidate_cache)
            for i in range(len(particles)):
                grad_cs[i] += geo_result.grad_cs[i]
                grad_ps[i] += geo_result.grad_ps[i]
        return (grad_cs, grad_ps, local_pair_results)
    try:
        total_energy = _step_total_potential()
        grad_cs, grad_ps, out_pair_results = _step_total_gradients()
        return (total_energy, grad_cs, grad_ps, out_pair_results)
    finally:
        shared_geometry_result = None
