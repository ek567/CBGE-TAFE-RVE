from __future__ import annotations
from dataclasses import replace
from typing import Any
import time
import numpy as np
from src.constraints.min_distance_3d import inflate_particles_min_distance_3d, is_min_distance_enabled, restore_physical_particles_after_optimization_3d
from src.data.particle_3d import EllipsoidParticle3D
from src.model.geometry_distortion_energy_3d import evaluate_geometry_distortion_energy_and_gradients
from src.model.overlap_status_3d import system_overlap_status
from src.model.orientation_targets_3d import resolve_orientation_tolerance
from src.optimizer.descent_3d import DescentConfig, DescentResult, effective_inflated_search_overlap_tol, run_descent
from src.postprocess.origin_shift_3d import OriginShiftConfig3D, origin_shift_postprocess_3d
from src.resource_budget import RealizationIterationBudget, apply_resource_result
from src.validation.orientation_isotropy_check_3d import check_orientation_isotropy

def _geometry_distortion_report_from_config(config: DescentConfig) -> dict[str, Any]:
    geo_config = getattr(config, 'geometry_config', None)
    enabled = bool(getattr(config, 'include_geometry_distortion', False))
    if geo_config is None:
        return {'enabled': enabled}
    return {'enabled': enabled and bool(getattr(geo_config, 'enabled', True)), 'include_face': bool(getattr(geo_config, 'include_face', False)), 'include_edge': bool(getattr(geo_config, 'include_edge', False)), 'include_corner': bool(getattr(geo_config, 'include_corner', False)), 'd_face': float(getattr(geo_config, 'd_face', 0.0)), 'd_edge': float(getattr(geo_config, 'd_edge', 0.0)), 'd_corner': float(getattr(geo_config, 'd_corner', 0.0)), 'w_face': float(getattr(geo_config, 'w_face', 0.0)), 'w_edge': float(getattr(geo_config, 'w_edge', 0.0)), 'w_corner': float(getattr(geo_config, 'w_corner', 0.0)), 'lambda_geo': float(getattr(geo_config, 'lambda_geo', 0.0)), 'normalize_by': str(getattr(geo_config, 'normalize_by', 'num_particles')), 'use_physical_axes': bool(getattr(geo_config, 'use_physical_axes', True)), 'active_violation_abs_tol': float(getattr(geo_config, 'active_violation_abs_tol', 0.0)), 'active_violation_rel_tol': float(getattr(geo_config, 'active_violation_rel_tol', 0.0)), 'candidate_filter': bool(getattr(geo_config, 'candidate_filter', False)), 'candidate_margin_abs': float(getattr(geo_config, 'candidate_margin_abs', 0.0)), 'candidate_margin_rel': float(getattr(geo_config, 'candidate_margin_rel', 0.0)), 'candidate_cache': bool(getattr(config, 'use_geometry_candidate_cache', False)), 'geometry_stop': bool(getattr(config, 'geometry_stop', True)), 'geometry_energy_tol': float(getattr(config, 'geometry_energy_tol', 0.0)), 'geometry_violation_tol': float(getattr(config, 'geometry_violation_tol', 0.0)), 'geometry_face_violation_tol': float(getattr(config, 'geometry_face_violation_tol', getattr(config, 'geometry_violation_tol', 0.0))), 'geometry_edge_violation_tol': float(getattr(config, 'geometry_edge_violation_tol', getattr(config, 'geometry_violation_tol', 0.0))), 'geometry_corner_violation_tol': float(getattr(config, 'geometry_corner_violation_tol', getattr(config, 'geometry_violation_tol', 0.0)))}

def run_descent_with_constraints_3d(particles_physical: list[EllipsoidParticle3D], config: DescentConfig, *, box: np.ndarray | None=None, collect_metrics: bool=False, delta_min: float=0.0, origin_shift_config: OriginShiftConfig3D | None=None, realization_budget: RealizationIterationBudget | None=None) -> tuple[DescentResult, dict[str, Any]]:
    delta_min = float(delta_min)
    if delta_min < 0.0:
        raise ValueError(f'delta_min must be >=0, got {delta_min}.')
    if realization_budget is None:
        realization_budget = RealizationIterationBudget()
    if is_min_distance_enabled(delta_min):
        particles_opt = inflate_particles_min_distance_3d(particles_physical, delta_min=delta_min)
        min_dist_info = {'enabled': True, 'delta_min': float(delta_min)}
    else:
        particles_opt = inflate_particles_min_distance_3d(particles_physical, delta_min=0.0)
        min_dist_info = {'enabled': False, 'delta_min': 0.0}
    physical_axes_override = [(float(p.a), float(p.b)) for p in particles_physical]
    result_opt = run_descent(particles=particles_opt, config=config, box=box, collect_metrics=collect_metrics, physical_axes_override=physical_axes_override, realization_budget=realization_budget)
    inflated_status = getattr(result_opt, 'inflated_final_status', None)
    if inflated_status is None:
        inflated_status = result_opt.final_status
    inflated_status = None if inflated_status is None else dict(inflated_status)
    search_overlap_tol = effective_inflated_search_overlap_tol(config)
    particles_physical_final = restore_physical_particles_after_optimization_3d(particles_physical=particles_physical, particles_opt_final=result_opt.particles)
    origin_info: dict[str, Any] = {'enabled': False}
    if origin_shift_config is not None and origin_shift_config.enabled and (realization_budget.resource_status != 'RESOURCE_BUDGET_EXHAUSTED'):
        if box is None:
            raise ValueError('origin_shift is enabled but box is None (non-periodic).')
        shifted_particles, shift_vec, shift_report = origin_shift_postprocess_3d(particles=particles_physical_final, box=np.asarray(box, dtype=float), config=origin_shift_config)
        particles_physical_final = shifted_particles
        origin_info = {'enabled': True, 'shift_vector': np.asarray(shift_vec, dtype=float).tolist(), 'report': shift_report}
    elif origin_shift_config is not None and origin_shift_config.enabled:
        origin_info['skipped_reason'] = 'RESOURCE_BUDGET_EXHAUSTED'
    physical_restore_audit_start = time.perf_counter()
    physical_status = system_overlap_status(particles=particles_physical_final, box=None if box is None else np.asarray(box, dtype=float), use_mic=bool(getattr(config, 'use_mic', False)), overlap_tol=float(getattr(config, 'overlap_tol', 0.0)))
    geometry_ok = True
    if bool(getattr(config, 'include_geometry_distortion', False)) and bool(getattr(config, 'geometry_stop', True)):
        if box is None:
            raise ValueError('box must be provided when geometry distortion is enabled.')
        geo_result = evaluate_geometry_distortion_energy_and_gradients(particles=particles_physical_final, box=np.asarray(box, dtype=float), config=config.geometry_config, compute_grad=False, apply_lambda=True, metrics=None)
        physical_status = dict(physical_status)
        physical_status.update({'geometry_energy': float(geo_result.total_energy), 'worst_geometry_violation': float(geo_result.worst_violation), 'max_geometry_violation': float(geo_result.worst_violation), 'max_face_violation': float(geo_result.max_face_violation), 'max_edge_violation': float(geo_result.max_edge_violation), 'max_corner_violation': float(geo_result.max_corner_violation)})
        geometry_ok = bool(float(geo_result.max_face_violation) <= float(config.geometry_face_violation_tol) and float(geo_result.max_edge_violation) <= float(config.geometry_edge_violation_tol) and (float(geo_result.max_corner_violation) <= float(config.geometry_corner_violation_tol)))
    orientation_ok = True
    orientation_result = None
    if bool(getattr(config, 'include_orientation_regularization', False)) and bool(getattr(config, 'orientation_stop', True)):
        orientation_result = check_orientation_isotropy(particles=particles_physical_final, tolerance=resolve_orientation_tolerance(getattr(config, 'orientation_target', '3d-isotropic'), getattr(config, 'orientation_tol', None)), orientation_target=getattr(config, 'orientation_target', '3d-isotropic'))
        orientation_ok = bool(orientation_result['is_target_aligned'])
    physical_overlap_ok = bool(float(physical_status['max_overlap']) <= float(getattr(config, 'overlap_tol', 0.0)))
    physical_success = bool(physical_overlap_ok and geometry_ok and orientation_ok)
    physical_restore_audit_elapsed_seconds = float(time.perf_counter() - physical_restore_audit_start)
    if physical_success:
        stop_parts = ['physical_overlap']
        if bool(getattr(config, 'include_geometry_distortion', False)) and bool(getattr(config, 'geometry_stop', True)):
            stop_parts.append('geometry')
        if bool(getattr(config, 'include_orientation_regularization', False)) and bool(getattr(config, 'orientation_stop', True)):
            stop_parts.append('orientation')
        stop_reason = '_and_'.join(stop_parts) + '_target_reached_after_restore'
    else:
        missing = []
        if not physical_overlap_ok:
            missing.append('physical_overlap')
        if not geometry_ok:
            missing.append('geometry')
        if not orientation_ok:
            missing.append('orientation')
        stop_reason = 'physical_restore_check_failed:' + ','.join(missing)
    if result_opt.metrics is not None:
        result_opt.metrics.final_max_overlap = float(physical_status['max_overlap'])
        result_opt.metrics.converged = bool(physical_success)
    result_final = replace(result_opt, particles=particles_physical_final, final_status=physical_status, inflated_final_status=inflated_status, final_pair_results=list(physical_status['pair_results']), converged=bool(physical_success), success=bool(physical_success), independent_audit_elapsed_seconds=result_opt.independent_audit_elapsed_seconds + physical_restore_audit_elapsed_seconds, physical_restore_audit_elapsed_seconds=result_opt.physical_restore_audit_elapsed_seconds + physical_restore_audit_elapsed_seconds, stop_reason=stop_reason)
    result_final = apply_resource_result(result_final, realization_budget)
    if result_final.metrics is not None:
        result_final.metrics.converged = bool(result_final.success)
    constraints_report = {'pw_search_acceptance': {'inflated_search_overlap_tol': search_overlap_tol, 'physical_acceptance_overlap_tol': float(config.overlap_tol), 'overlap_count_tol': float(config.overlap_tol), 'inflated_final_max_overlap': None if inflated_status is None else float(inflated_status['max_overlap']), 'inflated_search_passed': None if inflated_status is None else bool(float(inflated_status['max_overlap']) <= search_overlap_tol), 'physical_final_max_overlap': float(physical_status['max_overlap']), 'physical_overlap_passed': physical_overlap_ok, 'orientation_required': bool(config.include_orientation_regularization and config.orientation_stop), 'orientation_tol': float(config.orientation_tol), 'orientation_passed': orientation_ok, 'orientation_max_abs_deviation': None if orientation_result is None else float(orientation_result['max_abs_deviation']), 'native_search_success': bool(result_opt.success), 'native_search_stop_reason': str(result_opt.stop_reason), 'min_distance_acceptance_performed': False}, 'independent_audit_timing': {'elapsed_seconds': result_final.independent_audit_elapsed_seconds, 'native_final_elapsed_seconds': result_final.native_final_audit_elapsed_seconds, 'physical_restore_elapsed_seconds': result_final.physical_restore_audit_elapsed_seconds, 'scope': result_final.independent_audit_timing_scope}, 'min_distance': min_dist_info, 'origin_shift': origin_info, 'geometry_distortion': _geometry_distortion_report_from_config(config), 'resource_budget': {**realization_budget.snapshot(), 'generation_success': bool(result_final.generation_success), 'terminal_state_sha256': result_final.terminal_state_sha256, 'physical_restore_stop_reason': stop_reason}, 'terminal_descriptive_audit': {'state_sha256': result_final.terminal_state_sha256, 'resource_failure_cannot_be_overridden': True, 'native_physical_max_overlap': float(physical_status['max_overlap']), 'stage_geometry_violations': {key: physical_status.get(key) for key in ('max_face_violation', 'max_edge_violation', 'max_corner_violation')}, 'orientation_max_abs_deviation': None if orientation_result is None else float(orientation_result['max_abs_deviation']), 'orientation_target': getattr(config, 'orientation_target', '3d-isotropic'), 'orientation_is_target_aligned': None if orientation_result is None else bool(orientation_result['is_target_aligned']), 'orientation_frobenius_norm': None if orientation_result is None else float(orientation_result['frobenius_norm'])}}
    return (result_final, constraints_report)
