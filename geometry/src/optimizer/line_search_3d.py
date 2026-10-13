from __future__ import annotations
from dataclasses import is_dataclass, replace
from functools import partial
from types import FunctionType
from typing import Any
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.model.geometry_distortion_energy_3d import GeometryDistortionConfig3D, build_geometry_candidate_cache_3d
from src.model.total_potential_3d import _compute_step_objective_and_gradients as _baseline_step_objective, compute_total_potential as _baseline_total_potential, evaluate_all_pairs, evaluate_total_energy_fast as _baseline_fast_energy
from src.model.orientation_regularization_3d import _validate_particles as _validate_orientation_particles, compute_orientation_deviation_tensor_3d, compute_orientation_regularization_energy as _baseline_orientation_energy, compute_orientation_regularization_gradients as _baseline_orientation_gradients
from src.model.orientation_targets_3d import resolve_orientation_tolerance
from src.monitoring.metrics_3d import StepMetrics3D
from src.monitoring.timers_3d import timed_field
from src.optimizer.retraction_3d import apply_trial_step
from src.optimizer.step_policy_3d import LineSearchConfig, LineSearchState, cap_trial_step_sizes, choose_initial_alpha, decide_fallback_alpha
from src.overlap_detection.candidate_pairs_3d import CandidatePairConfig3D, generate_candidate_pairs

def orientation_acceptance_band_bounds(orientation_tol: float) -> tuple[float, float]:
    tol = float(orientation_tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f'orientation_tol must be finite and >= 0, got {orientation_tol}.')
    if tol == 0.0:
        return (0.0, 0.0)
    q0_exact = float(np.sqrt(1.5) * tol)
    q0 = float(np.nextafter(q0_exact, 0.0))
    q1 = float(np.sqrt(2.0) * tol)
    if not 0.0 <= q0 < q1:
        raise RuntimeError(f'Invalid acceptance-band bounds q0={q0}, q1={q1}.')
    return (q0, q1)

def _orientation_acceptance_band_scale_from_q(q: float, orientation_tol: float) -> float:
    q_value = float(q)
    if not np.isfinite(q_value) or q_value < 0.0:
        raise ValueError(f'q must be finite and >= 0, got {q}.')
    q0, q1 = orientation_acceptance_band_bounds(orientation_tol)
    if q1 == 0.0:
        return 1.0 if q_value > 0.0 else 0.0
    if q_value <= q0:
        return 0.0
    if q_value >= q1:
        return 1.0
    t = (q_value - q0) / (q1 - q0)
    return float(3.0 * t * t - 2.0 * t * t * t)

def _orientation_acceptance_band_energy_from_q(q: float, orientation_tol: float) -> float:
    q_value = float(q)
    if not np.isfinite(q_value) or q_value < 0.0:
        raise ValueError(f'q must be finite and >= 0, got {q}.')
    q0, q1 = orientation_acceptance_band_bounds(orientation_tol)
    if q1 == 0.0:
        return 0.5 * q_value * q_value
    if q_value <= q0:
        return 0.0
    width = q1 - q0
    if q_value < q1:
        x = (q_value - q0) / width
        value = width * q0 * (x ** 3 - 0.5 * x ** 4) + width * width * (0.75 * x ** 4 - 0.4 * x ** 5)
        return float(value)
    transition_at_q1 = 0.5 * width * q0 + 7.0 / 20.0 * width * width
    return float(transition_at_q1 + 0.5 * (q_value * q_value - q1 * q1))

def _orientation_mode(mode: str) -> str:
    value = str(mode)
    if value not in ('baseline', 'acceptance_band'):
        raise ValueError(f"orientation_regularization_mode must be 'baseline' or 'acceptance_band', got {value!r}.")
    return value

def _ori_energy(particles, *, mode, orientation_tol, orientation_target='3d-isotropic'):
    mode_value = _orientation_mode(mode)
    if mode_value == 'baseline':
        return _baseline_orientation_energy(particles, orientation_target=orientation_target)
    deviation = compute_orientation_deviation_tensor_3d(particles, orientation_target=orientation_target)
    q = float(np.linalg.norm(deviation, ord='fro'))
    return _orientation_acceptance_band_energy_from_q(q, orientation_tol)

def _ori_gradients(particles, *, mode, orientation_tol, orientation_target='3d-isotropic'):
    _validate_orientation_particles(particles)
    mode_value = _orientation_mode(mode)
    if mode_value == 'baseline':
        return _baseline_orientation_gradients(particles, orientation_target=orientation_target)
    deviation = compute_orientation_deviation_tensor_3d(particles, orientation_target=orientation_target)
    q = float(np.linalg.norm(deviation, ord='fro'))
    scale = _orientation_acceptance_band_scale_from_q(q, orientation_tol)
    gradients = _baseline_orientation_gradients(particles, orientation_target=orientation_target)
    if scale != 1.0:
        gradients = [float(scale) * grad for grad in gradients]
    return gradients

def _bind_orientation_contribution(implementation, mode, orientation_tol):
    private_globals = implementation.__globals__.copy()
    private_globals['compute_orientation_regularization_energy'] = partial(_ori_energy, mode=mode, orientation_tol=orientation_tol)
    private_globals['compute_orientation_regularization_gradients'] = partial(_ori_gradients, mode=mode, orientation_tol=orientation_tol)
    bound = FunctionType(implementation.__code__, private_globals, implementation.__name__, implementation.__defaults__, implementation.__closure__)
    bound.__kwdefaults__ = None if implementation.__kwdefaults__ is None else implementation.__kwdefaults__.copy()
    return bound

def compute_total_potential(*args, orientation_regularization_mode='baseline', orientation_tol=None, orientation_target='3d-isotropic', **kwargs):
    orientation_tol = resolve_orientation_tolerance(orientation_target, orientation_tol)
    return _bind_orientation_contribution(_baseline_total_potential, orientation_regularization_mode, orientation_tol)(*args, orientation_target=orientation_target, **kwargs)

def evaluate_total_energy_fast(*args, orientation_regularization_mode='baseline', orientation_tol=None, orientation_target='3d-isotropic', **kwargs):
    orientation_tol = resolve_orientation_tolerance(orientation_target, orientation_tol)
    return _bind_orientation_contribution(_baseline_fast_energy, orientation_regularization_mode, orientation_tol)(*args, orientation_target=orientation_target, **kwargs)

def _compute_step_objective_and_gradients(*args, orientation_regularization_mode='baseline', orientation_tol=None, orientation_target='3d-isotropic', **kwargs):
    orientation_tol = resolve_orientation_tolerance(orientation_target, orientation_tol)
    return _bind_orientation_contribution(_baseline_step_objective, orientation_regularization_mode, orientation_tol)(*args, orientation_target=orientation_target, **kwargs)

def build_descent_directions(grad_cs: list[np.ndarray], grad_ps: list[np.ndarray], tau_c: float, tau_p: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not isinstance(grad_cs, list):
        raise TypeError('grad_cs must be a list of numpy arrays.')
    if not isinstance(grad_ps, list):
        raise TypeError('grad_ps must be a list of numpy arrays.')
    if len(grad_cs) != len(grad_ps):
        raise ValueError('grad_cs and grad_ps must have the same length.')
    tau_c_value = float(tau_c)
    tau_p_value = float(tau_p)
    direction_cs: list[np.ndarray] = []
    direction_ps: list[np.ndarray] = []
    for i, (grad_c, grad_p) in enumerate(zip(grad_cs, grad_ps)):
        grad_c_array = np.asarray(grad_c, dtype=float)
        grad_p_array = np.asarray(grad_p, dtype=float)
        if grad_c_array.shape != (3,):
            raise ValueError(f'grad_cs[{i}] must have shape (3,), got {grad_c_array.shape}.')
        if grad_p_array.shape != (3,):
            raise ValueError(f'grad_ps[{i}] must have shape (3,), got {grad_p_array.shape}.')
        direction_cs.append(-tau_c_value * grad_c_array)
        direction_ps.append(-tau_p_value * grad_p_array)
    return (direction_cs, direction_ps)

def directional_derivative(grad_cs: list[np.ndarray], grad_ps: list[np.ndarray], direction_cs: list[np.ndarray], direction_ps: list[np.ndarray]) -> float:
    if not (isinstance(grad_cs, list) and isinstance(grad_ps, list)):
        raise TypeError('grad_cs and grad_ps must be lists.')
    if not (isinstance(direction_cs, list) and isinstance(direction_ps, list)):
        raise TypeError('direction_cs and direction_ps must be lists.')
    n = len(grad_cs)
    if len(grad_ps) != n or len(direction_cs) != n or len(direction_ps) != n:
        raise ValueError('All gradient and direction lists must have the same length.')
    value = 0.0
    for i in range(n):
        grad_c = np.asarray(grad_cs[i], dtype=float)
        grad_p = np.asarray(grad_ps[i], dtype=float)
        dir_c = np.asarray(direction_cs[i], dtype=float)
        dir_p = np.asarray(direction_ps[i], dtype=float)
        if grad_c.shape != (3,):
            raise ValueError(f'grad_cs[{i}] must have shape (3,), got {grad_c.shape}.')
        if grad_p.shape != (3,):
            raise ValueError(f'grad_ps[{i}] must have shape (3,), got {grad_p.shape}.')
        if dir_c.shape != (3,):
            raise ValueError(f'direction_cs[{i}] must have shape (3,), got {dir_c.shape}.')
        if dir_p.shape != (3,):
            raise ValueError(f'direction_ps[{i}] must have shape (3,), got {dir_p.shape}.')
        value += float(np.dot(grad_c, dir_c) + np.dot(grad_p, dir_p))
    return float(value)

def evaluate_trial_energy(particles: list[EllipsoidParticle3D], box: np.ndarray | None, use_mic: bool, attach_geometry: bool, include_orientation_regularization: bool=False, orientation_regularization_mode: str='baseline', orientation_tol: float | None=None, include_geometry_distortion: bool=False, geometry_config: GeometryDistortionConfig3D | None=None, physical_axes_override: object | None=None, metrics: StepMetrics3D | None=None, orientation_target: str='3d-isotropic') -> float:
    orientation_tol = resolve_orientation_tolerance(orientation_target, orientation_tol)
    pair_results = evaluate_all_pairs(particles=particles, box=box, use_mic=bool(use_mic), attach_geometry=False, metrics=None)
    return compute_total_potential(pair_results, particles=particles, include_orientation_regularization=bool(include_orientation_regularization), orientation_regularization_mode=orientation_regularization_mode, orientation_tol=orientation_tol, orientation_target=orientation_target, box=box, include_geometry_distortion=bool(include_geometry_distortion), geometry_config=geometry_config, physical_axes_override=physical_axes_override)

def _max_direction_c_norm(direction_cs: list[np.ndarray]) -> float:
    m = 0.0
    for d in direction_cs:
        v = float(np.linalg.norm(np.asarray(d, dtype=float)))
        if v > m:
            m = v
    return float(m)

def _ls01_directional_derivative_blocks(grad_cs: list[np.ndarray], grad_ps: list[np.ndarray], direction_cs: list[np.ndarray], direction_ps: list[np.ndarray]) -> tuple[float, float]:
    n = len(grad_cs)
    if len(grad_ps) != n or len(direction_cs) != n or len(direction_ps) != n:
        raise ValueError('All gradient and direction lists must have the same length.')
    value_c = 0.0
    value_p = 0.0
    for i in range(n):
        grad_c = np.asarray(grad_cs[i], dtype=float)
        grad_p = np.asarray(grad_ps[i], dtype=float)
        dir_c = np.asarray(direction_cs[i], dtype=float)
        dir_p = np.asarray(direction_ps[i], dtype=float)
        if grad_c.shape != (3,):
            raise ValueError(f'grad_cs[{i}] must have shape (3,), got {grad_c.shape}.')
        if grad_p.shape != (3,):
            raise ValueError(f'grad_ps[{i}] must have shape (3,), got {grad_p.shape}.')
        if dir_c.shape != (3,):
            raise ValueError(f'direction_cs[{i}] must have shape (3,), got {dir_c.shape}.')
        if dir_p.shape != (3,):
            raise ValueError(f'direction_ps[{i}] must have shape (3,), got {dir_p.shape}.')
        value_c += float(np.dot(grad_c, dir_c))
        value_p += float(np.dot(grad_p, dir_p))
    return (float(value_c), float(value_p))

def _ls01_bump_metric(metrics: StepMetrics3D | None, name: str, amount: int=1) -> None:
    if metrics is None:
        return
    setattr(metrics, name, int(getattr(metrics, name, 0)) + int(amount))

def _ls01_set_metric(metrics: StepMetrics3D | None, name: str, value: float) -> None:
    if metrics is None:
        return
    setattr(metrics, name, float(value))

def _build_pair_config_for_line_search(base: CandidatePairConfig3D, *, sphere_padding_ls: float, cell_size_min: float | None) -> CandidatePairConfig3D:
    if is_dataclass(base):
        kwargs: dict[str, Any] = {'sphere_padding': float(sphere_padding_ls), 'use_bounding_sphere': True}
        method = getattr(base, 'method', None)
        if method == 'uniform_grid' and cell_size_min is not None:
            base_cell = getattr(base, 'cell_size', None)
            base_cell_val = float(base_cell) if base_cell is not None else 0.0
            kwargs['cell_size'] = float(max(base_cell_val, float(cell_size_min)))
            if hasattr(base, 'auto_cell_size'):
                kwargs['auto_cell_size'] = False
        return replace(base, **kwargs)
    cfg = CandidatePairConfig3D()
    for name in ('method', 'use_bounding_sphere', 'sphere_padding', 'cell_size', 'auto_cell_size'):
        if hasattr(base, name):
            setattr(cfg, name, getattr(base, name))
    cfg.sphere_padding = float(sphere_padding_ls)
    cfg.use_bounding_sphere = True
    if getattr(cfg, 'method', None) == 'uniform_grid' and cell_size_min is not None:
        base_cell = getattr(cfg, 'cell_size', None)
        base_cell_val = float(base_cell) if base_cell is not None else 0.0
        cfg.cell_size = float(max(base_cell_val, float(cell_size_min)))
        if hasattr(cfg, 'auto_cell_size'):
            cfg.auto_cell_size = False
    return cfg

def build_trial_safe_candidate_pairs(particles: list[EllipsoidParticle3D], *, box: np.ndarray | None, use_mic: bool, pair_config: CandidatePairConfig3D, delta_max: float) -> list[tuple[int, int]]:
    if not isinstance(particles, list):
        raise TypeError('particles must be a list.')
    if not isinstance(pair_config, CandidatePairConfig3D):
        raise TypeError('pair_config must be CandidatePairConfig3D.')
    delta = float(delta_max)
    if delta < 0.0:
        raise ValueError(f'delta_max must be >= 0, got {delta_max}.')
    base_padding = float(getattr(pair_config, 'sphere_padding', 0.0))
    sphere_padding_ls = base_padding + 2.0 * delta
    a_max = max((float(p.a) for p in particles)) if len(particles) > 0 else 0.0
    cutoff_max_ls = 2.0 * a_max + float(sphere_padding_ls)
    method = getattr(pair_config, 'method', 'all_pairs')
    cell_size_min = float(cutoff_max_ls) if method == 'uniform_grid' else None
    cfg_ls = _build_pair_config_for_line_search(pair_config, sphere_padding_ls=float(sphere_padding_ls), cell_size_min=cell_size_min)
    return generate_candidate_pairs(particles=particles, box=box, use_mic=bool(use_mic), config=cfg_ls, metrics=None)

def backtracking_line_search_with_stats(particles: list[EllipsoidParticle3D], grad_cs: list[np.ndarray], grad_ps: list[np.ndarray], energy_before: float, box: np.ndarray | None, use_mic: bool, attach_geometry: bool, tau_c: float, tau_p: float, line_search_config: LineSearchConfig, line_search_state: LineSearchState | None=None, wrap_positions: bool=False, include_orientation_regularization: bool=False, orientation_regularization_mode: str='baseline', orientation_tol: float | None=None, c01_ls_rescue_enabled: bool=True, include_geometry_distortion: bool=False, geometry_config: GeometryDistortionConfig3D | None=None, physical_axes_override: object | None=None, pair_config: CandidatePairConfig3D | None=None, overlap_count_tol: float=0.0, metrics: StepMetrics3D | None=None, use_extent_filter: bool=False, extent_sep_tol: float=1e-12, use_geometry_candidate_cache: bool=False, orientation_target: str='3d-isotropic') -> tuple[list[EllipsoidParticle3D], float, float, bool, LineSearchState, float | None, int | None, float | None]:
    orientation_tol = resolve_orientation_tolerance(orientation_target, orientation_tol)
    if not isinstance(particles, list):
        raise TypeError('particles must be a list of EllipsoidParticle3D.')
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise TypeError(f'particles[{idx}] must be EllipsoidParticle3D, got {type(particle).__name__}.')
    if not isinstance(line_search_config, LineSearchConfig):
        raise TypeError(f'line_search_config must be LineSearchConfig, got {type(line_search_config).__name__}.')
    if line_search_state is None:
        line_search_state = LineSearchState()
    if not isinstance(line_search_state, LineSearchState):
        raise TypeError(f'line_search_state must be LineSearchState, got {type(line_search_state).__name__}.')
    if pair_config is None:
        pair_config = CandidatePairConfig3D()
    if not isinstance(pair_config, CandidatePairConfig3D):
        raise TypeError(f'pair_config must be CandidatePairConfig3D or None, got {type(pair_config).__name__}.')
    direction_cs, direction_ps = build_descent_directions(grad_cs=grad_cs, grad_ps=grad_ps, tau_c=float(tau_c), tau_p=float(tau_p))
    dir_deriv = directional_derivative(grad_cs=grad_cs, grad_ps=grad_ps, direction_cs=direction_cs, direction_ps=direction_ps)
    ls02_enabled = bool(c01_ls_rescue_enabled) and bool(include_orientation_regularization) and (str(orientation_regularization_mode) == 'acceptance_band')
    ls02_phase_in = str(getattr(line_search_state, '_c01_ls02_phase', 'NONE'))
    if ls02_phase_in not in {'NONE', 'AFTER_DEEP_RESCUE', 'HOLD_NEXT_GROWTH'}:
        ls02_phase_in = 'NONE'
    alpha_nominal = choose_initial_alpha(line_search_config, line_search_state)
    alpha_nominal = cap_trial_step_sizes(direction_cs=direction_cs, direction_ps=direction_ps, alpha=alpha_nominal, position_step_cap=line_search_config.position_step_cap, orientation_step_cap=line_search_config.orientation_step_cap)
    alpha = float(alpha_nominal)
    if ls02_enabled and ls02_phase_in == 'HOLD_NEXT_GROWTH' and line_search_state.previous_success and (line_search_state.previous_alpha is not None):
        held_alpha = min(float(alpha), float(line_search_state.previous_alpha))
        if held_alpha < float(alpha):
            _ls01_bump_metric(metrics, 'ls02_growth_holds')
        alpha = float(held_alpha)
    if ls02_enabled:
        _ls01_set_metric(metrics, 'ls02_nominal_initial_alpha', float(alpha_nominal))
        _ls01_set_metric(metrics, 'ls02_effective_initial_alpha', float(alpha))
    accepted_particles = [particle.clone() for particle in particles]
    accepted_energy = float(energy_before)
    accepted_alpha = 0.0
    accepted_state_alpha_override: float | None = None
    accepted_via_rescue = False
    accepted_via_shared_first_trial = False
    accepted_rescue_alpha_center: float | None = None
    accepted_max_overlap: float | None = None
    accepted_num_overlapping_pairs: int | None = None
    accepted_geometry_energy: float | None = None
    success = False
    updated_state = LineSearchState(previous_alpha=line_search_state.previous_alpha, previous_success=line_search_state.previous_success)
    use_fast = bool(line_search_config.fast_energy_eval)
    use_early_abort = bool(line_search_config.early_abort)
    trial_pairs_override: list[tuple[int, int]] | None = None
    geometry_candidate_cache = None
    direction_c_norm_for_cache = _max_direction_c_norm(direction_cs)
    if use_fast:
        fallback_alpha_for_superset = decide_fallback_alpha(line_search_config)
        if fallback_alpha_for_superset is not None:
            fallback_alpha_for_superset = cap_trial_step_sizes(direction_cs=direction_cs, direction_ps=direction_ps, alpha=float(fallback_alpha_for_superset), position_step_cap=line_search_config.position_step_cap, orientation_step_cap=line_search_config.orientation_step_cap)
            alpha_upper = max(float(alpha), float(fallback_alpha_for_superset))
        else:
            alpha_upper = float(alpha)
        if line_search_config.position_step_cap is not None:
            delta_max = float(line_search_config.position_step_cap)
        else:
            delta_max = float(alpha_upper) * direction_c_norm_for_cache
        trial_pairs_override = build_trial_safe_candidate_pairs(particles=particles, box=box, use_mic=bool(use_mic), pair_config=pair_config, delta_max=float(delta_max))
        if bool(use_geometry_candidate_cache) and bool(include_geometry_distortion) and (geometry_config is not None) and (box is not None):
            geometry_candidate_cache = build_geometry_candidate_cache_3d(particles=particles, box=np.asarray(box, dtype=float), config=geometry_config, physical_axes_override=physical_axes_override, guard_margin=float(delta_max))
            if metrics is not None:
                metrics.geo_cache_enabled = True
                metrics.geo_cache_rebuilds += 1
                metrics.geo_cache_face_candidates = int(len(geometry_candidate_cache.face_candidates))
                metrics.geo_cache_edge_candidates = int(len(geometry_candidate_cache.edge_candidates))
                metrics.geo_cache_corner_candidates = int(len(geometry_candidate_cache.corner_candidates))
    ls01_enabled = bool(ls02_enabled)
    regular_trials_used = 0

    def _try_ls01_center_preserving_rescue(alpha_center: float) -> tuple[bool, list[EllipsoidParticle3D] | None, float | None, float | None, int | None, float | None]:
        nonlocal regular_trials_used
        if not ls01_enabled:
            return (False, None, None, None, None, None)
        if regular_trials_used >= int(line_search_config.max_trials):
            return (False, None, None, None, None, None)
        alpha_c = float(alpha_center)
        alpha_p = float(alpha_c) * float(line_search_config.rho)
        if alpha_p < float(line_search_config.alpha_min):
            return (False, None, None, None, None, None)
        dir_deriv_c, dir_deriv_p = _ls01_directional_derivative_blocks(grad_cs=grad_cs, grad_ps=grad_ps, direction_cs=direction_cs, direction_ps=direction_ps)
        rhs_rescue = float(energy_before) + float(line_search_config.c1) * (float(alpha_c) * float(dir_deriv_c) + float(alpha_p) * float(dir_deriv_p))
        scaled_direction_ps = [float(line_search_config.rho) * np.asarray(d, dtype=float) for d in direction_ps]
        rescue_particles = apply_trial_step(particles=particles, direction_cs=direction_cs, direction_ps=scaled_direction_ps, alpha=alpha_c, box=box, wrap_positions=bool(wrap_positions))
        regular_trials_used += 1
        if metrics is not None:
            metrics.line_search_trials += 1
            metrics.line_search_energy_evals += 1
        _ls01_bump_metric(metrics, 'ls01_rescue_trials')
        _ls01_set_metric(metrics, 'ls01_last_rescue_alpha_c', alpha_c)
        _ls01_set_metric(metrics, 'ls01_last_rescue_alpha_p', alpha_p)
        _ls01_set_metric(metrics, 'ls01_last_rescue_rhs', rhs_rescue)
        if use_fast:
            geometry_cache_for_rescue = geometry_candidate_cache
            if geometry_candidate_cache is not None:
                rescue_delta = float(alpha_c) * float(direction_c_norm_for_cache)
                if rescue_delta > float(geometry_candidate_cache.guard_margin) * (1.0 + 1e-12):
                    geometry_cache_for_rescue = None
                    if metrics is not None:
                        metrics.geo_fast_eval_fallback_full_count += 1
            fast_res = evaluate_total_energy_fast(particles=rescue_particles, box=box, use_mic=bool(use_mic), pair_config=pair_config, include_orientation_regularization=bool(include_orientation_regularization), orientation_regularization_mode=orientation_regularization_mode, orientation_tol=orientation_tol, orientation_target=orientation_target, include_geometry_distortion=bool(include_geometry_distortion), geometry_config=geometry_config, physical_axes_override=physical_axes_override, geometry_candidate_cache=geometry_cache_for_rescue, energy_abort_threshold=rhs_rescue if use_early_abort else None, overlap_count_tol=float(overlap_count_tol), candidate_pairs_override=trial_pairs_override, use_extent_filter=bool(use_extent_filter), extent_sep_tol=float(extent_sep_tol), metrics=metrics)
            if metrics is not None:
                metrics.num_pw_calls += int(fast_res.num_pw_calls)
                metrics.geo_fast_eval_used += 1
            if fast_res.early_aborted and use_early_abort:
                if metrics is not None:
                    metrics.line_search_early_aborts += 1
                _ls01_bump_metric(metrics, 'ls01_rescue_early_aborts')
                _ls01_bump_metric(metrics, 'ls01_rescue_rejects')
                return (False, None, None, None, None, None)
            rescue_energy = float(fast_res.energy)
            rescue_max_overlap = float(fast_res.max_overlap)
            rescue_num_overlapping_pairs = int(fast_res.num_overlapping_pairs)
            rescue_geometry_energy = float(fast_res.geometry_energy)
        else:
            rescue_energy = evaluate_trial_energy(particles=rescue_particles, box=box, use_mic=bool(use_mic), attach_geometry=False, include_orientation_regularization=bool(include_orientation_regularization), orientation_regularization_mode=orientation_regularization_mode, orientation_tol=orientation_tol, orientation_target=orientation_target, include_geometry_distortion=bool(include_geometry_distortion), geometry_config=geometry_config, physical_axes_override=physical_axes_override, metrics=None)
            rescue_max_overlap = None
            rescue_num_overlapping_pairs = None
            rescue_geometry_energy = None
        _ls01_set_metric(metrics, 'ls01_last_rescue_energy', rescue_energy)
        if rescue_energy <= rhs_rescue:
            _ls01_bump_metric(metrics, 'ls01_rescue_accepts')
            _ls01_set_metric(metrics, 'ls01_accepted_alpha_c', alpha_c)
            _ls01_set_metric(metrics, 'ls01_accepted_alpha_p', alpha_p)
            return (True, rescue_particles, float(rescue_energy), rescue_max_overlap, rescue_num_overlapping_pairs, rescue_geometry_energy)
        _ls01_bump_metric(metrics, 'ls01_rescue_rejects')
        return (False, None, None, None, None, None)
    with timed_field(metrics, 'time_line_search'):
        for _ in range(line_search_config.max_trials):
            if regular_trials_used >= int(line_search_config.max_trials):
                break
            regular_trials_used += 1
            if metrics is not None:
                metrics.line_search_trials += 1
            if alpha < line_search_config.alpha_min:
                break
            trial_particles = apply_trial_step(particles=particles, direction_cs=direction_cs, direction_ps=direction_ps, alpha=alpha, box=box, wrap_positions=bool(wrap_positions))
            if metrics is not None:
                metrics.line_search_energy_evals += 1
            rhs = float(energy_before) + float(line_search_config.c1) * float(alpha) * float(dir_deriv)
            if use_fast:
                geometry_cache_for_trial = geometry_candidate_cache
                if geometry_candidate_cache is not None:
                    trial_delta = float(alpha) * float(direction_c_norm_for_cache)
                    if trial_delta > float(geometry_candidate_cache.guard_margin) * (1.0 + 1e-12):
                        geometry_cache_for_trial = None
                        if metrics is not None:
                            metrics.geo_fast_eval_fallback_full_count += 1
                fast_res = evaluate_total_energy_fast(particles=trial_particles, box=box, use_mic=bool(use_mic), pair_config=pair_config, include_orientation_regularization=bool(include_orientation_regularization), orientation_regularization_mode=orientation_regularization_mode, orientation_tol=orientation_tol, orientation_target=orientation_target, include_geometry_distortion=bool(include_geometry_distortion), geometry_config=geometry_config, physical_axes_override=physical_axes_override, geometry_candidate_cache=geometry_cache_for_trial, energy_abort_threshold=rhs if use_early_abort else None, overlap_count_tol=float(overlap_count_tol), candidate_pairs_override=trial_pairs_override, use_extent_filter=bool(use_extent_filter), extent_sep_tol=float(extent_sep_tol), metrics=metrics)
                if metrics is not None:
                    metrics.num_pw_calls += int(fast_res.num_pw_calls)
                    metrics.geo_fast_eval_used += 1
                if fast_res.early_aborted and use_early_abort:
                    if metrics is not None:
                        metrics.line_search_early_aborts += 1
                    rescue_success, rescue_particles, rescue_energy, rescue_max_overlap, rescue_num_overlapping_pairs, rescue_geometry_energy = _try_ls01_center_preserving_rescue(alpha)
                    if rescue_success:
                        accepted_particles = rescue_particles
                        accepted_energy = float(rescue_energy)
                        accepted_alpha = float(alpha)
                        accepted_state_alpha_override = float(alpha) * float(line_search_config.rho)
                        accepted_via_rescue = True
                        accepted_rescue_alpha_center = float(alpha)
                        _ls01_set_metric(metrics, 'ls01_warm_start_alpha', accepted_state_alpha_override)
                        accepted_max_overlap = rescue_max_overlap
                        accepted_num_overlapping_pairs = rescue_num_overlapping_pairs
                        accepted_geometry_energy = rescue_geometry_energy
                        success = True
                        break
                    alpha *= float(line_search_config.rho)
                    continue
                trial_energy = float(fast_res.energy)
                trial_max_overlap = float(fast_res.max_overlap)
                trial_num_overlapping_pairs = int(fast_res.num_overlapping_pairs)
                trial_geometry_energy = float(fast_res.geometry_energy)
            else:
                trial_energy = evaluate_trial_energy(particles=trial_particles, box=box, use_mic=bool(use_mic), attach_geometry=False, include_orientation_regularization=bool(include_orientation_regularization), orientation_regularization_mode=orientation_regularization_mode, orientation_tol=orientation_tol, orientation_target=orientation_target, include_geometry_distortion=bool(include_geometry_distortion), geometry_config=geometry_config, physical_axes_override=physical_axes_override, metrics=None)
                trial_max_overlap = None
                trial_num_overlapping_pairs = None
                trial_geometry_energy = None
            if trial_energy <= rhs:
                accepted_particles = trial_particles
                accepted_energy = float(trial_energy)
                accepted_alpha = float(alpha)
                accepted_via_shared_first_trial = bool(regular_trials_used == 1)
                accepted_max_overlap = trial_max_overlap
                accepted_num_overlapping_pairs = trial_num_overlapping_pairs
                accepted_geometry_energy = trial_geometry_energy
                success = True
                break
            rescue_success, rescue_particles, rescue_energy, rescue_max_overlap, rescue_num_overlapping_pairs, rescue_geometry_energy = _try_ls01_center_preserving_rescue(alpha)
            if rescue_success:
                accepted_particles = rescue_particles
                accepted_energy = float(rescue_energy)
                accepted_alpha = float(alpha)
                accepted_state_alpha_override = float(alpha) * float(line_search_config.rho)
                accepted_via_rescue = True
                accepted_rescue_alpha_center = float(alpha)
                _ls01_set_metric(metrics, 'ls01_warm_start_alpha', accepted_state_alpha_override)
                accepted_max_overlap = rescue_max_overlap
                accepted_num_overlapping_pairs = rescue_num_overlapping_pairs
                accepted_geometry_energy = rescue_geometry_energy
                success = True
                break
            alpha *= float(line_search_config.rho)
        if not success:
            fallback_alpha = decide_fallback_alpha(line_search_config)
            if fallback_alpha is not None:
                fallback_alpha = cap_trial_step_sizes(direction_cs=direction_cs, direction_ps=direction_ps, alpha=float(fallback_alpha), position_step_cap=line_search_config.position_step_cap, orientation_step_cap=line_search_config.orientation_step_cap)
                if fallback_alpha >= line_search_config.alpha_min:
                    trial_particles = apply_trial_step(particles=particles, direction_cs=direction_cs, direction_ps=direction_ps, alpha=fallback_alpha, box=box, wrap_positions=bool(wrap_positions))
                    if metrics is not None:
                        metrics.line_search_trials += 1
                        metrics.line_search_energy_evals += 1
                    if use_fast:
                        geometry_cache_for_trial = geometry_candidate_cache
                        if geometry_candidate_cache is not None:
                            trial_delta = float(fallback_alpha) * float(direction_c_norm_for_cache)
                            if trial_delta > float(geometry_candidate_cache.guard_margin) * (1.0 + 1e-12):
                                geometry_cache_for_trial = None
                                if metrics is not None:
                                    metrics.geo_fast_eval_fallback_full_count += 1
                        fast_res = evaluate_total_energy_fast(particles=trial_particles, box=box, use_mic=bool(use_mic), pair_config=pair_config, include_orientation_regularization=bool(include_orientation_regularization), orientation_regularization_mode=orientation_regularization_mode, orientation_tol=orientation_tol, orientation_target=orientation_target, include_geometry_distortion=bool(include_geometry_distortion), geometry_config=geometry_config, physical_axes_override=physical_axes_override, geometry_candidate_cache=geometry_cache_for_trial, energy_abort_threshold=None, overlap_count_tol=float(overlap_count_tol), candidate_pairs_override=trial_pairs_override, use_extent_filter=bool(use_extent_filter), extent_sep_tol=float(extent_sep_tol), metrics=metrics)
                        if metrics is not None:
                            metrics.num_pw_calls += int(fast_res.num_pw_calls)
                            metrics.geo_fast_eval_used += 1
                        fallback_energy = float(fast_res.energy)
                        fallback_max_overlap = float(fast_res.max_overlap)
                        fallback_num_overlapping_pairs = int(fast_res.num_overlapping_pairs)
                        fallback_geometry_energy = float(fast_res.geometry_energy)
                    else:
                        fallback_energy = evaluate_trial_energy(particles=trial_particles, box=box, use_mic=bool(use_mic), attach_geometry=False, include_orientation_regularization=bool(include_orientation_regularization), orientation_regularization_mode=orientation_regularization_mode, orientation_tol=orientation_tol, orientation_target=orientation_target, include_geometry_distortion=bool(include_geometry_distortion), geometry_config=geometry_config, physical_axes_override=physical_axes_override, metrics=None)
                        fallback_max_overlap = None
                        fallback_num_overlapping_pairs = None
                        fallback_geometry_energy = None
                    if fallback_energy <= float(energy_before):
                        accepted_particles = trial_particles
                        accepted_energy = float(fallback_energy)
                        accepted_alpha = float(fallback_alpha)
                        accepted_max_overlap = fallback_max_overlap
                        accepted_num_overlapping_pairs = fallback_num_overlapping_pairs
                        accepted_geometry_energy = fallback_geometry_energy
                        success = True
    if metrics is not None:
        metrics.step_alpha = float(accepted_alpha) if success else 0.0
        metrics.line_search_success = bool(success)
        if success and accepted_max_overlap is not None:
            metrics.max_overlap = float(accepted_max_overlap)
        if success and accepted_num_overlapping_pairs is not None:
            metrics.num_overlapping_pairs = int(accepted_num_overlapping_pairs)
    if success and accepted_alpha > 0.0:
        updated_state.previous_alpha = float(accepted_alpha if accepted_state_alpha_override is None else accepted_state_alpha_override)
    else:
        updated_state.previous_alpha = None
    updated_state.previous_success = bool(success)
    ls02_phase_out = 'NONE'
    if success and ls02_enabled:
        if accepted_via_rescue and accepted_rescue_alpha_center is not None and (float(accepted_rescue_alpha_center) < float(alpha_nominal)):
            ls02_phase_out = 'AFTER_DEEP_RESCUE'
            _ls01_bump_metric(metrics, 'ls02_deep_rescue_accepts')
        elif ls02_phase_in == 'AFTER_DEEP_RESCUE' and accepted_via_shared_first_trial:
            ls02_phase_out = 'HOLD_NEXT_GROWTH'
            _ls01_bump_metric(metrics, 'ls02_growth_hold_armed')
    setattr(updated_state, '_c01_ls02_phase', ls02_phase_out)
    return (accepted_particles, accepted_energy, accepted_alpha, success, updated_state, accepted_max_overlap, accepted_num_overlapping_pairs, accepted_geometry_energy)

def backtracking_line_search(particles: list[EllipsoidParticle3D], grad_cs: list[np.ndarray], grad_ps: list[np.ndarray], energy_before: float, box: np.ndarray | None, use_mic: bool, attach_geometry: bool, tau_c: float, tau_p: float, line_search_config: LineSearchConfig, line_search_state: LineSearchState | None=None, wrap_positions: bool=False, include_orientation_regularization: bool=False, orientation_regularization_mode: str='baseline', orientation_tol: float | None=None, c01_ls_rescue_enabled: bool=True, metrics: StepMetrics3D | None=None, orientation_target: str='3d-isotropic') -> tuple[list[EllipsoidParticle3D], float, float, bool, LineSearchState]:
    orientation_tol = resolve_orientation_tolerance(orientation_target, orientation_tol)
    accepted_particles, accepted_energy, accepted_alpha, success, updated_state, _, _, _ = backtracking_line_search_with_stats(particles=particles, grad_cs=grad_cs, grad_ps=grad_ps, energy_before=energy_before, box=box, use_mic=use_mic, attach_geometry=attach_geometry, tau_c=tau_c, tau_p=tau_p, line_search_config=line_search_config, line_search_state=line_search_state, wrap_positions=wrap_positions, include_orientation_regularization=include_orientation_regularization, orientation_regularization_mode=orientation_regularization_mode, orientation_tol=orientation_tol, orientation_target=orientation_target, c01_ls_rescue_enabled=c01_ls_rescue_enabled, pair_config=None, overlap_count_tol=0.0, metrics=metrics)
    return (accepted_particles, accepted_energy, accepted_alpha, success, updated_state)
