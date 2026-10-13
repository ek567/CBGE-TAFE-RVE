from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import time
from pathlib import Path
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.resource_budget import RealizationIterationBudget, RESOURCE_BUDGET_EXHAUSTED, TOTAL_REALIZATION_ITERATION_BUDGET, WITHIN_RESOURCE_BUDGET, apply_resource_result
from src.model.overlap_status_3d import system_overlap_status
from src.model.geometry_distortion_energy_3d import GeometryDistortionConfig3D, evaluate_geometry_distortion_energy_and_gradients
from src.model.total_potential_3d import evaluate_all_pairs
from src.monitoring.metrics_3d import RunMetrics3D, StepMetrics3D
from src.monitoring.timers_3d import start_timer, stop_timer, timed_field
from src.optimizer.line_search_3d import _compute_step_objective_and_gradients, compute_total_potential, evaluate_total_energy_fast, backtracking_line_search_with_stats
from src.optimizer.step_policy_3d import LineSearchConfig, LineSearchState
from src.overlap_detection.candidate_pairs_3d import CandidatePairConfig3D
from src.periodic.mic_3d import wrap_position
from src.utils.math_utils import normalize
from src.validation.orientation_isotropy_check_3d import check_orientation_isotropy
from src.model.orientation_targets_3d import resolve_orientation_tolerance, validate_orientation_target

@dataclass
class DescentConfig:
    tau_c: float = 0.01
    tau_p: float = 0.01
    max_steps: int = 100
    grad_tol: float = 1e-08
    energy_tol: float = 1e-12
    physical_stop: bool = True
    overlap_tol: float = 1e-09
    check_system_status_each_step: bool = True
    progress_window: int = 10
    progress_rel_tol: float = 1e-06
    stop_on_stagnation: bool = True
    use_mic: bool = False
    attach_geometry: bool = True
    verbose: bool = True
    wrap_positions: bool = False
    include_orientation_regularization: bool = False
    orientation_regularization_mode: str = 'baseline'
    orientation_stop: bool = True
    orientation_tol: float | None = None
    orientation_tol_selection: str = field(init=False, default='model_default')
    c01_ls_rescue_enabled: bool = True
    orientation_geometry_guard_enabled: bool = False
    include_geometry_distortion: bool = False
    geometry_regularization_mode: str = 'baseline'
    geometry_stop: bool = True
    geometry_energy_tol: float = 1e-12
    geometry_violation_tol: float = 0.0001
    geometry_face_violation_tol: float = 0.0001
    geometry_edge_violation_tol: float = 0.0001
    geometry_corner_violation_tol: float = 0.0001
    use_geometry_candidate_cache: bool = False
    geometry_config: GeometryDistortionConfig3D = field(default_factory=GeometryDistortionConfig3D)
    line_search: LineSearchConfig = field(default_factory=LineSearchConfig)
    pair_config: CandidatePairConfig3D = field(default_factory=CandidatePairConfig3D)
    heartbeat_interval: int = 0
    heartbeat_path: str | None = None
    heartbeat_label: str = ''
    heartbeat_print: bool = True
    heartbeat_json: bool = True
    use_extent_filter: bool = False
    extent_sep_tol: float = 1e-12
    orientation_target: str = '3d-isotropic'
    inflated_search_overlap_tol: float | None = None
    stage0_inflated_search_overlap_tol: float | None = None
    stage123_inflated_search_overlap_tol: float | None = None

    def __post_init__(self) -> None:
        self.tau_c = float(self.tau_c)
        self.tau_p = float(self.tau_p)
        self.max_steps = int(self.max_steps)
        self.grad_tol = float(self.grad_tol)
        self.energy_tol = float(self.energy_tol)
        self.physical_stop = bool(self.physical_stop)
        self.overlap_tol = float(self.overlap_tol)
        for name in ('inflated_search_overlap_tol', 'stage0_inflated_search_overlap_tol', 'stage123_inflated_search_overlap_tol'):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f'Expected finite {name} >= 0, got {value}.')
                setattr(self, name, value)
        self.check_system_status_each_step = bool(self.check_system_status_each_step)
        self.progress_window = int(self.progress_window)
        self.progress_rel_tol = float(self.progress_rel_tol)
        self.stop_on_stagnation = bool(self.stop_on_stagnation)
        self.use_mic = bool(self.use_mic)
        self.attach_geometry = bool(self.attach_geometry)
        self.verbose = bool(self.verbose)
        self.wrap_positions = bool(self.wrap_positions)
        self.include_orientation_regularization = bool(self.include_orientation_regularization)
        self.orientation_regularization_mode = str(self.orientation_regularization_mode)
        self.orientation_stop = bool(self.orientation_stop)
        self.orientation_target = validate_orientation_target(self.orientation_target)
        self.orientation_tol_selection = 'model_default' if self.orientation_tol is None else 'explicit_override'
        self.orientation_tol = resolve_orientation_tolerance(self.orientation_target, self.orientation_tol)
        self.c01_ls_rescue_enabled = bool(self.c01_ls_rescue_enabled)
        self.orientation_geometry_guard_enabled = bool(self.orientation_geometry_guard_enabled)
        self.include_geometry_distortion = bool(self.include_geometry_distortion)
        self.geometry_regularization_mode = str(self.geometry_regularization_mode)
        self.geometry_stop = bool(self.geometry_stop)
        self.geometry_energy_tol = float(self.geometry_energy_tol)
        self.geometry_violation_tol = float(self.geometry_violation_tol)
        self.geometry_face_violation_tol = float(self.geometry_face_violation_tol)
        self.geometry_edge_violation_tol = float(self.geometry_edge_violation_tol)
        self.geometry_corner_violation_tol = float(self.geometry_corner_violation_tol)
        self.use_geometry_candidate_cache = bool(self.use_geometry_candidate_cache)
        self.use_extent_filter = bool(self.use_extent_filter)
        self.extent_sep_tol = float(self.extent_sep_tol)
        self.heartbeat_interval = int(self.heartbeat_interval)
        self.heartbeat_path = None if self.heartbeat_path is None else str(self.heartbeat_path)
        self.heartbeat_label = str(self.heartbeat_label)
        self.heartbeat_print = bool(self.heartbeat_print)
        self.heartbeat_json = bool(self.heartbeat_json)
        if not isinstance(self.line_search, LineSearchConfig):
            raise TypeError(f'line_search must be LineSearchConfig, got {type(self.line_search).__name__}.')
        if not isinstance(self.pair_config, CandidatePairConfig3D):
            raise TypeError(f'pair_config must be CandidatePairConfig3D, got {type(self.pair_config).__name__}.')
        if not isinstance(self.geometry_config, GeometryDistortionConfig3D):
            raise TypeError(f'geometry_config must be GeometryDistortionConfig3D, got {type(self.geometry_config).__name__}.')
        if self.tau_c < 0.0:
            raise ValueError(f'Expected tau_c >= 0, got {self.tau_c}.')
        if self.tau_p < 0.0:
            raise ValueError(f'Expected tau_p >= 0, got {self.tau_p}.')
        if self.max_steps < 0:
            raise ValueError(f'Expected max_steps >= 0, got {self.max_steps}.')
        if self.grad_tol < 0.0:
            raise ValueError(f'Expected grad_tol >= 0, got {self.grad_tol}.')
        if self.energy_tol < 0.0:
            raise ValueError(f'Expected energy_tol >= 0, got {self.energy_tol}.')
        if self.overlap_tol < 0.0:
            raise ValueError(f'Expected overlap_tol >= 0, got {self.overlap_tol}.')
        if self.progress_window <= 0:
            raise ValueError(f'Expected progress_window > 0, got {self.progress_window}.')
        if self.progress_rel_tol < 0.0:
            raise ValueError(f'Expected progress_rel_tol >= 0, got {self.progress_rel_tol}.')
        if self.orientation_regularization_mode not in ('baseline', 'acceptance_band'):
            raise ValueError(f"orientation_regularization_mode must be 'baseline' or 'acceptance_band', got {self.orientation_regularization_mode!r}.")
        if self.geometry_regularization_mode not in ('baseline', 'acceptance_band'):
            raise ValueError(f"geometry_regularization_mode must be 'baseline' or 'acceptance_band', got {self.geometry_regularization_mode!r}.")
        if self.geometry_energy_tol < 0.0:
            raise ValueError(f'Expected geometry_energy_tol >= 0, got {self.geometry_energy_tol}.')
        if self.geometry_violation_tol < 0.0:
            raise ValueError(f'Expected geometry_violation_tol >= 0, got {self.geometry_violation_tol}.')
        for _name in ('geometry_face_violation_tol', 'geometry_edge_violation_tol', 'geometry_corner_violation_tol'):
            _value = float(getattr(self, _name))
            if _value < 0.0:
                raise ValueError(f'Expected {_name} >= 0, got {_value}.')
        if self.extent_sep_tol < 0.0:
            raise ValueError(f'Expected extent_sep_tol >= 0, got {self.extent_sep_tol}.')
        if self.heartbeat_interval < 0:
            raise ValueError(f'Expected heartbeat_interval >= 0, got {self.heartbeat_interval}.')

@dataclass
class DescentResult:
    particles: list[EllipsoidParticle3D]
    history_energy: list[float]
    history_grad_norm: list[float]
    history_max_overlap: list[float]
    history_num_overlapping_pairs: list[int]
    converged: bool
    success: bool
    stop_reason: str
    steps_taken: int
    final_pair_results: list[Any]
    final_status: dict[str, Any] | None = None
    metrics: RunMetrics3D | None = None
    cumulative_completed_iterations: int = 0
    total_realization_iteration_budget: int = TOTAL_REALIZATION_ITERATION_BUDGET
    resource_status: str = WITHIN_RESOURCE_BUDGET
    generation_success: bool = False
    last_committed_state_sha256: str | None = None
    terminal_state_sha256: str | None = None
    native_stop_context: list[dict[str, Any]] = field(default_factory=list)
    optimization_elapsed_seconds: float = 0.0
    optimization_timing_scope: str = 'native_iteration_loop_only; excludes_initialization_and_independent_final_audits'
    independent_audit_elapsed_seconds: float = 0.0
    native_final_audit_elapsed_seconds: float = 0.0
    physical_restore_audit_elapsed_seconds: float = 0.0
    independent_audit_timing_scope: str = 'native_final_status_and_metrics_energy_plus_physical_restore_acceptance; excludes_initialization_origin_shift_and_export'
    inflated_final_status: dict[str, Any] | None = None

def effective_inflated_search_overlap_tol(config: DescentConfig) -> float:
    value = getattr(config, 'inflated_search_overlap_tol', None)
    return float(config.overlap_tol if value is None else value)

def _validate_particles(particles) -> None:
    if not isinstance(particles, list):
        raise TypeError('particles must be a list of EllipsoidParticle3D.')
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise TypeError(f'particles[{idx}] must be EllipsoidParticle3D, got {type(particle).__name__}.')

def _validate_box(box) -> np.ndarray | None:
    if box is None:
        return None
    box_array = np.asarray(box, dtype=float)
    if box_array.shape != (3,):
        raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
    if np.any(box_array <= 0.0):
        raise ValueError(f'Box lengths must be positive, got {box_array}.')
    return box_array

def _clone_particles(particles) -> list[EllipsoidParticle3D]:
    _validate_particles(particles)
    return [particle.clone() for particle in particles]

def _relative_change_sequence(values, floor=1e-15) -> list[float]:
    values_array = [float(v) for v in values]
    if len(values_array) < 2:
        return []
    rel_changes = []
    floor_value = float(floor)
    for k in range(1, len(values_array)):
        denom = max(abs(values_array[k - 1]), floor_value)
        rel_changes.append(abs(values_array[k] - values_array[k - 1]) / denom)
    return rel_changes

def _detect_progress_stagnation(history_max_overlap, history_energy, progress_window, progress_rel_tol) -> bool:
    window = int(progress_window)
    tol = float(progress_rel_tol)
    if len(history_max_overlap) < window + 1:
        return False
    if len(history_energy) < window + 1:
        return False
    overlap_changes = _relative_change_sequence(history_max_overlap[-(window + 1):])
    energy_changes = _relative_change_sequence(history_energy[-(window + 1):])
    if len(overlap_changes) != window or len(energy_changes) != window:
        return False
    return bool(all((change <= tol for change in overlap_changes)) and all((change <= tol for change in energy_changes)))

def _format_scalar(value) -> str:
    return f'{float(value):.12e}'

def _heartbeat_enabled(config: DescentConfig) -> bool:
    return bool(int(config.heartbeat_interval) > 0)

def _emit_heartbeat(config: DescentConfig, payload: dict[str, Any]) -> None:
    if not _heartbeat_enabled(config):
        return
    label = str(payload.get('label', '') or config.heartbeat_label or 'descent')
    step = int(payload.get('step', 0))
    max_steps = int(payload.get('max_steps', 0))
    max_overlap = float(payload.get('max_overlap', 0.0))
    overlap_tol = float(payload.get('overlap_tol', 0.0))
    alpha = payload.get('alpha', None)
    trials = int(payload.get('line_search_trials', 0))
    elapsed = float(payload.get('elapsed_sec', 0.0))
    if bool(config.heartbeat_print):
        alpha_text = 'None' if alpha is None else f'{float(alpha):.3e}'
        print(f'[heartbeat] {label} step={step}/{max_steps} max_overlap={max_overlap:.3e} tol={overlap_tol:.3e} alpha={alpha_text} trials={trials} elapsed={elapsed:.1f}s')
    if bool(config.heartbeat_json) and config.heartbeat_path:
        path = Path(str(config.heartbeat_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

def _orientation_stop_satisfied(particles: list[EllipsoidParticle3D], orientation_tol: float, orientation_target: str='3d-isotropic') -> tuple[bool, dict[str, Any]]:
    result = check_orientation_isotropy(particles=particles, tolerance=float(orientation_tol), orientation_target=orientation_target)
    return (bool(result['is_target_aligned']), result)

def _geometry_enabled(config: DescentConfig) -> bool:
    return bool(config.include_geometry_distortion and config.geometry_config.enabled)

def _compute_geometry_stop_values(particles: list[EllipsoidParticle3D], config: DescentConfig, box: np.ndarray | None, physical_axes_override=None) -> dict[str, float]:
    if not _geometry_enabled(config):
        return {'geometry_energy': 0.0, 'worst_geometry_violation': 0.0, 'max_geometry_violation': 0.0, 'max_face_violation': 0.0, 'max_edge_violation': 0.0, 'max_corner_violation': 0.0}
    box_array = _validate_box(box)
    if box_array is None:
        raise ValueError('box must be provided when include_geometry_distortion=True.')
    result = evaluate_geometry_distortion_energy_and_gradients(particles=particles, box=box_array, config=config.geometry_config, physical_axes_override=physical_axes_override, compute_grad=False, apply_lambda=True, metrics=None)
    return {'geometry_energy': float(result.total_energy), 'worst_geometry_violation': float(result.worst_violation), 'max_geometry_violation': float(result.worst_violation), 'max_face_violation': float(result.max_face_violation), 'max_edge_violation': float(result.max_edge_violation), 'max_corner_violation': float(result.max_corner_violation)}

def _compute_geometry_energy_for_stop(particles: list[EllipsoidParticle3D], config: DescentConfig, box: np.ndarray | None, physical_axes_override=None) -> float:
    return _compute_geometry_stop_values(particles=particles, config=config, box=box, physical_axes_override=physical_axes_override)['geometry_energy']

def _geometry_stop_satisfied(geometry_energy: float, config: DescentConfig, geometry_worst_violation: float=0.0, max_face_violation: float=0.0, max_edge_violation: float=0.0, max_corner_violation: float=0.0) -> bool:
    if not _geometry_enabled(config) or not bool(config.geometry_stop):
        return True
    return bool(float(max_face_violation) <= float(config.geometry_face_violation_tol) and float(max_edge_violation) <= float(config.geometry_edge_violation_tol) and (float(max_corner_violation) <= float(config.geometry_corner_violation_tol)))

def _geometry_source_active_for_status(config: DescentConfig, status: dict[str, Any]) -> bool:
    if not _geometry_enabled(config):
        return False
    if str(config.geometry_regularization_mode) == 'baseline':
        return True
    if str(config.geometry_regularization_mode) != 'acceptance_band':
        raise ValueError(f"geometry_regularization_mode must be 'baseline' or 'acceptance_band', got {config.geometry_regularization_mode!r}.")
    feasible = _geometry_stop_satisfied(float(status.get('geometry_energy', 0.0)), config, float(status.get('worst_geometry_violation', 0.0)), max_face_violation=float(status.get('max_face_violation', 0.0)), max_edge_violation=float(status.get('max_edge_violation', 0.0)), max_corner_violation=float(status.get('max_corner_violation', 0.0)))
    return not bool(feasible)

def _physical_target_stop_reason(config: DescentConfig, *, include_orientation: bool=False, initial: bool=False) -> str:
    parts = ['physical_overlap']
    if _geometry_enabled(config) and bool(config.geometry_stop):
        parts.append('geometry')
    if include_orientation:
        parts.append('orientation')
    suffix = 'target_reached_initial' if initial else 'target_reached'
    prefix = '_and_'.join(parts)
    return f'{prefix}_{suffix}'

def compute_total_gradient_norm(grad_cs: list[np.ndarray], grad_ps: list[np.ndarray]) -> float:
    if not isinstance(grad_cs, list):
        raise TypeError('grad_cs must be a list.')
    if not isinstance(grad_ps, list):
        raise TypeError('grad_ps must be a list.')
    if len(grad_cs) != len(grad_ps):
        raise ValueError('grad_cs and grad_ps must have the same length.')
    total_sq = 0.0
    for i, grad_c in enumerate(grad_cs):
        grad_c_array = np.asarray(grad_c, dtype=float)
        if grad_c_array.shape != (3,):
            raise ValueError(f'grad_cs[{i}] must have shape (3,), got {grad_c_array.shape}.')
        total_sq += float(np.dot(grad_c_array, grad_c_array))
    for i, grad_p in enumerate(grad_ps):
        grad_p_array = np.asarray(grad_p, dtype=float)
        if grad_p_array.shape != (3,):
            raise ValueError(f'grad_ps[{i}] must have shape (3,), got {grad_p_array.shape}.')
        total_sq += float(np.dot(grad_p_array, grad_p_array))
    return float(np.sqrt(total_sq))

def take_one_descent_step(particles, config, box=None, line_search_state=None, metrics=None, *, physical_axes_override=None, return_accept_stats: bool=False, geometry_source_active: bool | None=None, orientation_regularization_mode_override: str | None=None):
    _validate_particles(particles)
    if not isinstance(config, DescentConfig):
        raise TypeError(f'config must be DescentConfig, got {type(config).__name__}.')
    box_array = _validate_box(box)
    if metrics is not None and (not isinstance(metrics, StepMetrics3D)):
        raise TypeError(f'metrics must be StepMetrics3D or None, got {type(metrics).__name__}.')
    if line_search_state is not None and (not isinstance(line_search_state, LineSearchState)):
        raise TypeError(f'line_search_state must be LineSearchState or None, got {type(line_search_state).__name__}.')
    working_particles = _clone_particles(particles)
    t_step = start_timer()
    geometry_source_enabled = _geometry_enabled(config) if geometry_source_active is None else bool(_geometry_enabled(config) and bool(geometry_source_active))
    effective_orientation_mode = str(config.orientation_regularization_mode if orientation_regularization_mode_override is None else orientation_regularization_mode_override)
    if effective_orientation_mode not in ('baseline', 'acceptance_band'):
        raise ValueError(f"effective orientation regularization mode must be 'baseline' or 'acceptance_band', got {effective_orientation_mode!r}.")
    pair_results = evaluate_all_pairs(particles=working_particles, box=box_array, use_mic=config.use_mic, attach_geometry=config.attach_geometry, geometry_attach_mode='overlap_only', pair_config=config.pair_config, metrics=metrics, use_extent_filter=bool(config.use_extent_filter), extent_sep_tol=float(config.extent_sep_tol))
    total_energy, grad_cs, grad_ps, pair_results = _compute_step_objective_and_gradients(particles=working_particles, pair_results=pair_results, box=box_array, use_mic=config.use_mic, pair_config=config.pair_config, metrics=metrics, include_orientation_regularization=config.include_orientation_regularization, orientation_regularization_mode=effective_orientation_mode, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target, include_geometry_distortion=geometry_source_enabled, geometry_config=config.geometry_config, physical_axes_override=physical_axes_override)
    if metrics is not None:
        metrics.energy_before = float(total_energy)
    total_grad_norm = compute_total_gradient_norm(grad_cs, grad_ps)
    if metrics is not None:
        metrics.grad_norm = float(total_grad_norm)
    accepted_geometry_energy: float | None = None
    if config.line_search.enabled:
        new_particles, accepted_energy, accepted_alpha, _success, updated_line_search_state, accepted_max_overlap, accepted_num_overlapping_pairs, accepted_geometry_energy = backtracking_line_search_with_stats(particles=working_particles, grad_cs=grad_cs, grad_ps=grad_ps, energy_before=total_energy, box=box_array, use_mic=config.use_mic, attach_geometry=config.attach_geometry, tau_c=config.tau_c, tau_p=config.tau_p, line_search_config=config.line_search, line_search_state=line_search_state, wrap_positions=config.wrap_positions, include_orientation_regularization=config.include_orientation_regularization, orientation_regularization_mode=effective_orientation_mode, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target, c01_ls_rescue_enabled=config.c01_ls_rescue_enabled, include_geometry_distortion=geometry_source_enabled, geometry_config=config.geometry_config, physical_axes_override=physical_axes_override, pair_config=config.pair_config, overlap_count_tol=config.overlap_tol, metrics=metrics, use_extent_filter=bool(config.use_extent_filter), extent_sep_tol=float(config.extent_sep_tol), use_geometry_candidate_cache=bool(config.use_geometry_candidate_cache))
        if accepted_max_overlap is None or accepted_num_overlapping_pairs is None:
            if float(accepted_alpha) == 0.0:
                accepted_max_overlap = float(max((pr.overlap for pr in pair_results), default=0.0))
                accepted_num_overlapping_pairs = int(sum((1 for pr in pair_results if float(pr.overlap) > float(config.overlap_tol))))
            else:
                fast_res = evaluate_total_energy_fast(particles=new_particles, box=box_array, use_mic=config.use_mic, pair_config=config.pair_config, include_orientation_regularization=config.include_orientation_regularization, orientation_regularization_mode=effective_orientation_mode, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target, include_geometry_distortion=geometry_source_enabled, geometry_config=config.geometry_config, physical_axes_override=physical_axes_override, energy_abort_threshold=None, overlap_count_tol=config.overlap_tol, use_extent_filter=bool(config.use_extent_filter), extent_sep_tol=float(config.extent_sep_tol), metrics=metrics)
                accepted_energy = float(fast_res.energy)
                accepted_max_overlap = float(fast_res.max_overlap)
                accepted_num_overlapping_pairs = int(fast_res.num_overlapping_pairs)
                accepted_geometry_energy = float(fast_res.geometry_energy)
                if metrics is not None:
                    metrics.num_pw_calls += int(fast_res.num_pw_calls)
                    metrics.max_overlap = float(accepted_max_overlap)
                    metrics.num_overlapping_pairs = int(accepted_num_overlapping_pairs)
    else:
        updated_line_search_state = line_search_state
        with timed_field(metrics, 'time_line_search'):
            new_particles = []
            for particle, grad_c, grad_p in zip(working_particles, grad_cs, grad_ps):
                new_c = particle.c - config.tau_c * np.asarray(grad_c, dtype=float)
                new_p = normalize(particle.p - config.tau_p * np.asarray(grad_p, dtype=float))
                if config.wrap_positions:
                    if box_array is None:
                        raise ValueError('box must be provided when wrap_positions=True.')
                    new_c = wrap_position(new_c, box_array)
                new_particles.append(EllipsoidParticle3D(a=particle.a, b=particle.b, c=np.asarray(new_c, dtype=float), p=np.asarray(new_p, dtype=float), id=particle.id))
        fast_res = evaluate_total_energy_fast(particles=new_particles, box=box_array, use_mic=config.use_mic, pair_config=config.pair_config, include_orientation_regularization=config.include_orientation_regularization, orientation_regularization_mode=effective_orientation_mode, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target, include_geometry_distortion=geometry_source_enabled, geometry_config=config.geometry_config, physical_axes_override=physical_axes_override, energy_abort_threshold=None, overlap_count_tol=config.overlap_tol, use_extent_filter=bool(config.use_extent_filter), extent_sep_tol=float(config.extent_sep_tol), metrics=metrics)
        accepted_energy = float(fast_res.energy)
        accepted_max_overlap = float(fast_res.max_overlap)
        accepted_num_overlapping_pairs = int(fast_res.num_overlapping_pairs)
        accepted_geometry_energy = float(fast_res.geometry_energy)
        if metrics is not None:
            metrics.step_alpha = None
            metrics.line_search_success = False
            metrics.num_pw_calls += int(fast_res.num_pw_calls)
            metrics.max_overlap = float(accepted_max_overlap)
            metrics.num_overlapping_pairs = int(accepted_num_overlapping_pairs)
    if accepted_geometry_energy is None:
        fast_res = evaluate_total_energy_fast(particles=new_particles, box=box_array, use_mic=config.use_mic, pair_config=config.pair_config, include_orientation_regularization=config.include_orientation_regularization, orientation_regularization_mode=effective_orientation_mode, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target, include_geometry_distortion=geometry_source_enabled, geometry_config=config.geometry_config, physical_axes_override=physical_axes_override, energy_abort_threshold=None, overlap_count_tol=config.overlap_tol, use_extent_filter=bool(config.use_extent_filter), extent_sep_tol=float(config.extent_sep_tol), metrics=metrics)
        accepted_energy = float(fast_res.energy)
        accepted_max_overlap = float(fast_res.max_overlap)
        accepted_num_overlapping_pairs = int(fast_res.num_overlapping_pairs)
        accepted_geometry_energy = float(fast_res.geometry_energy)
        if metrics is not None:
            metrics.num_pw_calls += int(fast_res.num_pw_calls)
            metrics.max_overlap = float(accepted_max_overlap)
            metrics.num_overlapping_pairs = int(accepted_num_overlapping_pairs)
    if metrics is not None:
        metrics.energy_after = float(accepted_energy)
        metrics.time_total_step += stop_timer(t_step)
    base_return = (new_particles, float(accepted_energy), float(total_grad_norm), pair_results, metrics, updated_line_search_state)
    if not return_accept_stats:
        return base_return
    return base_return + (float(accepted_max_overlap), int(accepted_num_overlapping_pairs), float(accepted_geometry_energy))

def run_descent(particles, config, box=None, collect_metrics=False, *, physical_axes_override=None, realization_budget: RealizationIterationBudget | None=None) -> DescentResult:
    _validate_particles(particles)
    if not isinstance(config, DescentConfig):
        raise TypeError(f'config must be DescentConfig, got {type(config).__name__}.')
    box_array = _validate_box(box)
    search_overlap_tol = effective_inflated_search_overlap_tol(config)
    if realization_budget is None:
        realization_budget = RealizationIterationBudget()
    if not isinstance(realization_budget, RealizationIterationBudget):
        raise TypeError('realization_budget must be RealizationIterationBudget.')
    current_particles = _clone_particles(particles)
    realization_budget.bind_initial_state(current_particles)
    history_energy: list[float] = []
    history_grad_norm: list[float] = []
    history_max_overlap: list[float] = []
    history_num_overlapping_pairs: list[int] = []
    final_pair_results: list[Any] = []
    final_status: dict[str, Any] | None = None
    converged = False
    success = False
    stop_reason = 'max_steps'
    steps_taken = 0
    run_metrics = RunMetrics3D() if collect_metrics else None
    t_run = start_timer()
    heartbeat_start = time.time()
    line_search_state = LineSearchState() if config.line_search.enabled else None
    initial_status = system_overlap_status(particles=current_particles, box=box_array, use_mic=config.use_mic, overlap_tol=config.overlap_tol)
    final_status = initial_status
    final_pair_results = list(initial_status['pair_results'])
    initial_geometry_values = _compute_geometry_stop_values(current_particles, config, box_array, physical_axes_override=physical_axes_override)
    status_current = {'max_overlap': float(initial_status['max_overlap']), 'num_overlapping_pairs': int(initial_status['num_overlapping_pairs']), **initial_geometry_values}
    initial_orientation_result: dict[str, Any] | None = None
    if config.include_orientation_regularization and config.orientation_stop:
        _, initial_orientation_result = _orientation_stop_satisfied(particles=current_particles, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target)
    if config.physical_stop and float(status_current['max_overlap']) <= search_overlap_tol and _geometry_stop_satisfied(float(status_current.get('geometry_energy', 0.0)), config, float(status_current.get('worst_geometry_violation', 0.0)), max_face_violation=float(status_current.get('max_face_violation', 0.0)), max_edge_violation=float(status_current.get('max_edge_violation', 0.0)), max_corner_violation=float(status_current.get('max_corner_violation', 0.0))):
        if config.include_orientation_regularization and config.orientation_stop:
            if initial_orientation_result is None:
                raise RuntimeError('initial_orientation_result must not be None here.')
            initial_orientation_max_dev = float(initial_orientation_result['max_abs_deviation'])
            if initial_orientation_max_dev <= config.orientation_tol:
                converged = True
                success = True
                stop_reason = _physical_target_stop_reason(config, include_orientation=True, initial=True)
        else:
            converged = True
            success = True
            stop_reason = _physical_target_stop_reason(config, initial=True)
        if converged:
            if run_metrics is not None:
                run_metrics.total_wall_time = stop_timer(t_run)
                run_metrics.num_steps = 0
                run_metrics.final_energy = compute_total_potential(initial_status['pair_results'], particles=current_particles, include_orientation_regularization=config.include_orientation_regularization, orientation_regularization_mode=config.orientation_regularization_mode, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target, box=box_array, include_geometry_distortion=_geometry_enabled(config), geometry_config=config.geometry_config, physical_axes_override=physical_axes_override)
                run_metrics.final_max_overlap = float(initial_status['max_overlap'])
                run_metrics.converged = bool(success)
            final_status = dict(final_status)
            final_status.update(initial_geometry_values)
            return apply_resource_result(DescentResult(particles=current_particles, history_energy=history_energy, history_grad_norm=history_grad_norm, history_max_overlap=history_max_overlap, history_num_overlapping_pairs=history_num_overlapping_pairs, converged=converged, success=success, stop_reason=stop_reason, steps_taken=steps_taken, final_pair_results=final_pair_results, final_status=final_status, inflated_final_status=dict(final_status), metrics=run_metrics), realization_budget)
    if config.max_steps == 0:
        converged = False
        success = False
        stop_reason = 'max_steps_zero'
        if run_metrics is not None:
            run_metrics.total_wall_time = stop_timer(t_run)
            run_metrics.num_steps = 0
            run_metrics.final_energy = compute_total_potential(initial_status['pair_results'], particles=current_particles, include_orientation_regularization=config.include_orientation_regularization, orientation_regularization_mode=config.orientation_regularization_mode, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target, box=box_array, include_geometry_distortion=_geometry_enabled(config), geometry_config=config.geometry_config, physical_axes_override=physical_axes_override)
            run_metrics.final_max_overlap = float(initial_status['max_overlap'])
            run_metrics.converged = bool(success)
        final_status = dict(final_status)
        final_status.update(initial_geometry_values)
        return apply_resource_result(DescentResult(particles=current_particles, history_energy=history_energy, history_grad_norm=history_grad_norm, history_max_overlap=history_max_overlap, history_num_overlapping_pairs=history_num_overlapping_pairs, converged=converged, success=success, stop_reason=stop_reason, steps_taken=steps_taken, final_pair_results=final_pair_results, final_status=final_status, inflated_final_status=dict(final_status), metrics=run_metrics), realization_budget)
    previous_energy: float | None = None
    iteration_start = time.perf_counter()
    for step in range(config.max_steps):
        status_before = status_current
        orientation_result_before: dict[str, Any] | None = None
        if config.include_orientation_regularization and config.orientation_stop:
            _, orientation_result_before = _orientation_stop_satisfied(particles=current_particles, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target)
        if config.physical_stop and float(status_before['max_overlap']) <= search_overlap_tol and _geometry_stop_satisfied(float(status_before.get('geometry_energy', 0.0)), config, float(status_before.get('worst_geometry_violation', 0.0)), max_face_violation=float(status_before.get('max_face_violation', 0.0)), max_edge_violation=float(status_before.get('max_edge_violation', 0.0)), max_corner_violation=float(status_before.get('max_corner_violation', 0.0))):
            if config.include_orientation_regularization and config.orientation_stop:
                if orientation_result_before is None:
                    raise RuntimeError('orientation_result_before must not be None here.')
                orientation_max_dev_before = float(orientation_result_before['max_abs_deviation'])
                if orientation_max_dev_before <= config.orientation_tol:
                    converged = True
                    success = True
                    stop_reason = _physical_target_stop_reason(config, include_orientation=True)
                    break
            else:
                converged = True
                success = True
                stop_reason = _physical_target_stop_reason(config)
                break
        if not realization_budget.admit_iteration():
            stop_reason = RESOURCE_BUDGET_EXHAUSTED
            converged = False
            success = False
            break
        step_metrics = StepMetrics3D(step_index=step) if collect_metrics or _heartbeat_enabled(config) else None
        geometry_source_active = _geometry_source_active_for_status(config, status_before)
        geometry_band_feasible_before = bool(_geometry_enabled(config) and _geometry_stop_satisfied(float(status_before.get('geometry_energy', 0.0)), config, float(status_before.get('worst_geometry_violation', 0.0)), max_face_violation=float(status_before.get('max_face_violation', 0.0)), max_edge_violation=float(status_before.get('max_edge_violation', 0.0)), max_corner_violation=float(status_before.get('max_corner_violation', 0.0))))
        orientation_mode_for_step = str(config.orientation_regularization_mode)
        orientation_geometry_guard_applied = bool(config.orientation_geometry_guard_enabled and config.include_orientation_regularization and _geometry_enabled(config) and (str(config.orientation_regularization_mode) == 'acceptance_band'))
        if orientation_geometry_guard_applied:
            orientation_mode_for_step = 'acceptance_band' if geometry_band_feasible_before else 'baseline'
        if step_metrics is not None:
            step_metrics.max_overlap = float(status_before['max_overlap'])
            step_metrics.num_overlapping_pairs = int(status_before['num_overlapping_pairs'])
            setattr(step_metrics, 'geo_abg_mode', str(config.geometry_regularization_mode))
            setattr(step_metrics, 'geo_abg_source_active', bool(geometry_source_active))
            setattr(step_metrics, 'geo_abg_band_feasible_before', bool(geometry_band_feasible_before))
            setattr(step_metrics, 'ori_gcg_guard_enabled', bool(orientation_geometry_guard_applied))
            setattr(step_metrics, 'ori_gcg_geometry_feasible_before', bool(geometry_band_feasible_before))
            setattr(step_metrics, 'ori_gcg_effective_mode', str(orientation_mode_for_step))
        new_particles, total_energy, total_grad_norm, _pair_results, step_metrics, line_search_state, max_overlap_after, num_overlapping_pairs_after, geometry_energy_after = take_one_descent_step(particles=current_particles, config=config, box=box_array, line_search_state=line_search_state, metrics=step_metrics, physical_axes_override=physical_axes_override, return_accept_stats=True, geometry_source_active=geometry_source_active, orientation_regularization_mode_override=orientation_mode_for_step)
        current_particles = new_particles
        steps_taken = step + 1
        realization_budget.commit_iteration(current_particles)
        geometry_values_after = {'geometry_energy': float(geometry_energy_after), 'worst_geometry_violation': 0.0, 'max_geometry_violation': 0.0, 'max_face_violation': 0.0, 'max_edge_violation': 0.0, 'max_corner_violation': 0.0}
        if _geometry_enabled(config):
            geometry_values_after = _compute_geometry_stop_values(current_particles, config, box_array, physical_axes_override=physical_axes_override)
            geometry_energy_after = float(geometry_values_after['geometry_energy'])
            if step_metrics is not None:
                step_metrics.geometry_energy = float(geometry_values_after['geometry_energy'])
                step_metrics.worst_geometry_violation = float(geometry_values_after['worst_geometry_violation'])
                step_metrics.max_geometry_violation = float(geometry_values_after['max_geometry_violation'])
                step_metrics.max_face_violation = float(geometry_values_after['max_face_violation'])
                step_metrics.max_edge_violation = float(geometry_values_after['max_edge_violation'])
                step_metrics.max_corner_violation = float(geometry_values_after['max_corner_violation'])
                step_metrics.max_face_geometry_violation = float(geometry_values_after['max_face_violation'])
                step_metrics.max_edge_geometry_violation = float(geometry_values_after['max_edge_violation'])
                step_metrics.max_corner_geometry_violation = float(geometry_values_after['max_corner_violation'])
        history_energy.append(float(total_energy))
        history_grad_norm.append(float(total_grad_norm))
        history_max_overlap.append(float(max_overlap_after))
        history_num_overlapping_pairs.append(int(num_overlapping_pairs_after))
        status_current = {'max_overlap': float(max_overlap_after), 'num_overlapping_pairs': int(num_overlapping_pairs_after), **geometry_values_after}
        orientation_result_after: dict[str, Any] | None = None
        orientation_max_dev_after: float | None = None
        if config.include_orientation_regularization and config.orientation_stop:
            _, orientation_result_after = _orientation_stop_satisfied(particles=current_particles, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target)
            orientation_max_dev_after = float(orientation_result_after['max_abs_deviation'])
        if step_metrics is not None:
            step_metrics.energy_after = float(total_energy)
            step_metrics.max_overlap = float(max_overlap_after)
            step_metrics.num_overlapping_pairs = int(num_overlapping_pairs_after)
        if run_metrics is not None and step_metrics is not None:
            run_metrics.append_step(step_metrics)
        if _heartbeat_enabled(config):
            interval = int(config.heartbeat_interval)
            if steps_taken == 1 or (interval > 0 and steps_taken % interval == 0):
                payload = {'label': str(config.heartbeat_label or 'descent'), 'step': int(steps_taken), 'max_steps': int(config.max_steps), 'elapsed_sec': float(time.time() - heartbeat_start), 'energy': float(total_energy), 'grad_norm': float(total_grad_norm), 'max_overlap': float(max_overlap_after), 'overlap_tol': search_overlap_tol, 'overlap_tol_role': 'inflated_search_stop', 'inflated_search_overlap_tol': search_overlap_tol, 'physical_acceptance_overlap_tol': float(config.overlap_tol), 'overlap_count_tol': float(config.overlap_tol), 'num_overlapping_pairs': int(num_overlapping_pairs_after), 'alpha': None if step_metrics is None or step_metrics.step_alpha is None else float(step_metrics.step_alpha), 'line_search_trials': 0 if step_metrics is None else int(step_metrics.line_search_trials), 'line_search_success': None if step_metrics is None else bool(step_metrics.line_search_success), 'geometry_enabled': bool(_geometry_enabled(config)), 'geometry_energy': float(geometry_values_after.get('geometry_energy', 0.0)), 'max_geometry_violation': float(geometry_values_after.get('max_geometry_violation', 0.0)), 'max_face_violation': float(geometry_values_after.get('max_face_violation', 0.0)), 'max_edge_violation': float(geometry_values_after.get('max_edge_violation', 0.0)), 'max_corner_violation': float(geometry_values_after.get('max_corner_violation', 0.0)), 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
                _emit_heartbeat(config, payload)
        if config.verbose:
            msg = f'[step {step + 1:04d}] energy={_format_scalar(total_energy)}, grad_norm={_format_scalar(total_grad_norm)}, max_overlap={_format_scalar(max_overlap_after)}, num_overlapping_pairs={num_overlapping_pairs_after}'
            if config.include_orientation_regularization and config.orientation_stop and (orientation_max_dev_after is not None):
                msg += f', orientation_max_dev={_format_scalar(orientation_max_dev_after)}, orientation_tol={_format_scalar(config.orientation_tol)}'
            print(msg)
            if step_metrics is not None:
                print(f'  pairs: all={step_metrics.num_all_pairs}, candidate={step_metrics.num_candidate_pairs}, pw_calls={step_metrics.num_pw_calls}, overlap_pairs={step_metrics.num_overlap_pairs}, geom_attached={step_metrics.num_geometry_attached_pairs}')
                if config.use_extent_filter:
                    print(f'  extent_filter: tests={step_metrics.num_extent_tests}, rejected={step_metrics.num_pairs_rejected_by_extent}')
                print(f"  times: total={_format_scalar(step_metrics.time_total_step)}, pair_gen={_format_scalar(step_metrics.time_pair_generation)}, broadphase={_format_scalar(step_metrics.time_broadphase)}, neighbor={_format_scalar(step_metrics.time_neighbor_build)}, pw_loop={_format_scalar(step_metrics.time_pw_eval)}, extent={_format_scalar(getattr(step_metrics, 'time_extent_filter', 0.0))}, pw_contact={_format_scalar(getattr(step_metrics, 'time_pw_contact', 0.0))}, geom={_format_scalar(step_metrics.time_geometry_attach)}, grad={_format_scalar(step_metrics.time_gradient_eval)}, ls={_format_scalar(step_metrics.time_line_search)}, fast_extent={_format_scalar(getattr(step_metrics, 'time_fast_extent_filter', 0.0))}, fast_pw={_format_scalar(getattr(step_metrics, 'time_fast_pw_contact', 0.0))}")
                if config.line_search.enabled:
                    print(f'  line_search: alpha={step_metrics.step_alpha}, trials={step_metrics.line_search_trials}, energy_evals={step_metrics.line_search_energy_evals}, early_aborts={step_metrics.line_search_early_aborts}, success={step_metrics.line_search_success}')
        overlap_free_after = bool(float(max_overlap_after) <= search_overlap_tol)
        geometry_free_after = _geometry_stop_satisfied(float(geometry_values_after.get('geometry_energy', 0.0)), config, float(geometry_values_after.get('worst_geometry_violation', 0.0)), max_face_violation=float(geometry_values_after.get('max_face_violation', 0.0)), max_edge_violation=float(geometry_values_after.get('max_edge_violation', 0.0)), max_corner_violation=float(geometry_values_after.get('max_corner_violation', 0.0)))
        if config.include_orientation_regularization and config.orientation_stop:
            if orientation_result_after is None or orientation_max_dev_after is None:
                raise RuntimeError('orientation_result_after and orientation_max_dev_after must be available when include_orientation_regularization=True and orientation_stop=True.')
            if overlap_free_after and geometry_free_after and (orientation_max_dev_after <= config.orientation_tol):
                converged = True
                success = True
                stop_reason = _physical_target_stop_reason(config, include_orientation=True)
                break
        elif config.physical_stop and overlap_free_after and geometry_free_after:
            converged = True
            success = True
            stop_reason = _physical_target_stop_reason(config)
            break
        if config.include_orientation_regularization and config.orientation_stop:
            if overlap_free_after and geometry_free_after and (total_grad_norm <= config.grad_tol):
                converged = True
                success = False
                stop_reason = 'orientation_target_not_reached_grad_small'
                if config.verbose:
                    print('Stopping: overlap-free reached, but orientation target not reached and gradient is small.')
                break
            if previous_energy is not None:
                energy_drop = abs(float(previous_energy) - float(total_energy))
                if overlap_free_after and geometry_free_after and (energy_drop <= config.energy_tol):
                    converged = True
                    success = False
                    stop_reason = 'orientation_target_not_reached_energy_small'
                    if config.verbose:
                        print('Stopping: overlap-free reached, but orientation target not reached and energy change is small.')
                    break
        previous_energy = float(total_energy)
        if config.stop_on_stagnation:
            stalled = _detect_progress_stagnation(history_max_overlap=history_max_overlap, history_energy=history_energy, progress_window=config.progress_window, progress_rel_tol=config.progress_rel_tol)
            if stalled:
                converged = True
                success = False
                stop_reason = 'progress_stalled'
                if config.verbose:
                    print('Stopping due to progress stagnation.')
                break
    optimization_elapsed_seconds = float(time.perf_counter() - iteration_start)
    if stop_reason == 'max_steps':
        converged = False
        success = False
    independent_audit_start = time.perf_counter()
    final_status = system_overlap_status(particles=current_particles, box=box_array, use_mic=config.use_mic, overlap_tol=config.overlap_tol)
    final_geometry_values = _compute_geometry_stop_values(current_particles, config, box_array, physical_axes_override=physical_axes_override)
    final_status = dict(final_status)
    final_status.update(final_geometry_values)
    final_pair_results = list(final_status['pair_results'])
    if run_metrics is not None:
        run_metrics.total_wall_time = stop_timer(t_run)
        run_metrics.num_steps = int(steps_taken)
        run_metrics.final_energy = compute_total_potential(final_status['pair_results'], particles=current_particles, include_orientation_regularization=config.include_orientation_regularization, orientation_regularization_mode=config.orientation_regularization_mode, orientation_tol=config.orientation_tol, orientation_target=config.orientation_target, box=box_array, include_geometry_distortion=_geometry_enabled(config), geometry_config=config.geometry_config, physical_axes_override=physical_axes_override)
        run_metrics.final_max_overlap = float(final_status['max_overlap'])
        run_metrics.converged = bool(success)
    independent_audit_elapsed_seconds = float(time.perf_counter() - independent_audit_start)
    if _heartbeat_enabled(config):
        _emit_heartbeat(config, {'label': str(config.heartbeat_label or 'descent'), 'step': int(steps_taken), 'max_steps': int(config.max_steps), 'elapsed_sec': float(time.time() - heartbeat_start), 'max_overlap': float(final_status['max_overlap']), 'overlap_tol': search_overlap_tol, 'overlap_tol_role': 'inflated_search_stop', 'inflated_search_overlap_tol': search_overlap_tol, 'physical_acceptance_overlap_tol': float(config.overlap_tol), 'overlap_count_tol': float(config.overlap_tol), 'num_overlapping_pairs': int(final_status['num_overlapping_pairs']), 'geometry_enabled': bool(_geometry_enabled(config)), 'geometry_energy': float(final_status.get('geometry_energy', 0.0)), 'max_geometry_violation': float(final_status.get('max_geometry_violation', 0.0)), 'max_face_violation': float(final_status.get('max_face_violation', 0.0)), 'max_edge_violation': float(final_status.get('max_edge_violation', 0.0)), 'max_corner_violation': float(final_status.get('max_corner_violation', 0.0)), 'success': bool(success), 'converged': bool(converged), 'stop_reason': str(stop_reason), 'final': True, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')})
    return apply_resource_result(DescentResult(particles=current_particles, history_energy=history_energy, history_grad_norm=history_grad_norm, history_max_overlap=history_max_overlap, history_num_overlapping_pairs=history_num_overlapping_pairs, converged=converged, success=success, stop_reason=stop_reason, steps_taken=steps_taken, final_pair_results=final_pair_results, final_status=final_status, inflated_final_status=dict(final_status), metrics=run_metrics, optimization_elapsed_seconds=optimization_elapsed_seconds, independent_audit_elapsed_seconds=independent_audit_elapsed_seconds, native_final_audit_elapsed_seconds=independent_audit_elapsed_seconds), realization_budget)
