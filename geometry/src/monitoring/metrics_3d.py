from __future__ import annotations
from dataclasses import asdict, dataclass, field

@dataclass
class StepMetrics3D:
    step_index: int = -1
    num_particles: int = 0
    num_all_pairs: int = 0
    num_candidate_pairs: int = 0
    num_pw_calls: int = 0
    num_overlap_pairs: int = 0
    energy_before: float = 0.0
    energy_after: float = 0.0
    grad_norm: float = 0.0
    max_overlap: float = 0.0
    num_overlapping_pairs: int = 0
    num_geometry_attached_pairs: int = 0
    line_search_early_aborts: int = 0
    geometry_energy: float = 0.0
    face_energy: float = 0.0
    edge_energy: float = 0.0
    corner_energy: float = 0.0
    num_active_face_constraints: int = 0
    num_active_edge_constraints: int = 0
    num_active_corner_constraints: int = 0
    num_face_tests_total: int = 0
    num_face_candidates: int = 0
    num_face_skipped_by_filter: int = 0
    num_edge_tests_total: int = 0
    num_edge_candidates: int = 0
    num_edge_skipped_by_filter: int = 0
    num_corner_tests_total: int = 0
    num_corner_candidates: int = 0
    num_corner_skipped_by_filter: int = 0
    num_corner_exact_distance_calls: int = 0
    geo_cache_enabled: bool = False
    geo_cache_rebuilds: int = 0
    geo_cache_face_candidates: int = 0
    geo_cache_edge_candidates: int = 0
    geo_cache_corner_candidates: int = 0
    geo_fast_eval_used: int = 0
    geo_fast_eval_fallback_full_count: int = 0
    worst_geometry_violation: float = 0.0
    max_geometry_violation: float = 0.0
    max_face_violation: float = 0.0
    max_edge_violation: float = 0.0
    max_corner_violation: float = 0.0
    max_face_geometry_violation: float = 0.0
    max_edge_geometry_violation: float = 0.0
    max_corner_geometry_violation: float = 0.0
    worst_geometry_particle_id: int | None = None
    worst_geometry_feature: str | None = None
    min_abs_signed_clearance: float = 0.0
    num_extent_tests: int = 0
    num_pairs_rejected_by_extent: int = 0
    step_alpha: float | None = None
    line_search_trials: int = 0
    line_search_energy_evals: int = 0
    line_search_success: bool = False
    time_total_step: float = 0.0
    time_pair_generation: float = 0.0
    time_broadphase: float = 0.0
    time_neighbor_build: float = 0.0
    time_pw_eval: float = 0.0
    time_extent_filter: float = 0.0
    time_pw_contact: float = 0.0
    time_geometry_attach: float = 0.0
    time_gradient_eval: float = 0.0
    time_line_search: float = 0.0
    time_fast_extent_filter: float = 0.0
    time_fast_pw_contact: float = 0.0
    time_system_status: float = 0.0
    time_strict_audit: float = 0.0

    def __post_init__(self) -> None:
        self.step_index = int(self.step_index)
        self.num_particles = int(self.num_particles)
        self.num_all_pairs = int(self.num_all_pairs)
        self.num_candidate_pairs = int(self.num_candidate_pairs)
        self.num_pw_calls = int(self.num_pw_calls)
        self.num_overlap_pairs = int(self.num_overlap_pairs)
        self.energy_before = float(self.energy_before)
        self.energy_after = float(self.energy_after)
        self.grad_norm = float(self.grad_norm)
        self.max_overlap = float(self.max_overlap)
        self.num_overlapping_pairs = int(self.num_overlapping_pairs)
        self.num_geometry_attached_pairs = int(self.num_geometry_attached_pairs)
        self.line_search_early_aborts = int(self.line_search_early_aborts)
        self.geometry_energy = float(self.geometry_energy)
        self.face_energy = float(self.face_energy)
        self.edge_energy = float(self.edge_energy)
        self.corner_energy = float(self.corner_energy)
        self.num_active_face_constraints = int(self.num_active_face_constraints)
        self.num_active_edge_constraints = int(self.num_active_edge_constraints)
        self.num_active_corner_constraints = int(self.num_active_corner_constraints)
        self.num_face_tests_total = int(self.num_face_tests_total)
        self.num_face_candidates = int(self.num_face_candidates)
        self.num_face_skipped_by_filter = int(self.num_face_skipped_by_filter)
        self.num_edge_tests_total = int(self.num_edge_tests_total)
        self.num_edge_candidates = int(self.num_edge_candidates)
        self.num_edge_skipped_by_filter = int(self.num_edge_skipped_by_filter)
        self.num_corner_tests_total = int(self.num_corner_tests_total)
        self.num_corner_candidates = int(self.num_corner_candidates)
        self.num_corner_skipped_by_filter = int(self.num_corner_skipped_by_filter)
        self.num_corner_exact_distance_calls = int(self.num_corner_exact_distance_calls)
        self.geo_cache_enabled = bool(self.geo_cache_enabled)
        self.geo_cache_rebuilds = int(self.geo_cache_rebuilds)
        self.geo_cache_face_candidates = int(self.geo_cache_face_candidates)
        self.geo_cache_edge_candidates = int(self.geo_cache_edge_candidates)
        self.geo_cache_corner_candidates = int(self.geo_cache_corner_candidates)
        self.geo_fast_eval_used = int(self.geo_fast_eval_used)
        self.geo_fast_eval_fallback_full_count = int(self.geo_fast_eval_fallback_full_count)
        self.worst_geometry_violation = float(self.worst_geometry_violation)
        self.max_geometry_violation = float(self.max_geometry_violation)
        self.max_face_violation = float(self.max_face_violation)
        self.max_edge_violation = float(self.max_edge_violation)
        self.max_corner_violation = float(self.max_corner_violation)
        self.max_face_geometry_violation = float(self.max_face_geometry_violation)
        self.max_edge_geometry_violation = float(self.max_edge_geometry_violation)
        self.max_corner_geometry_violation = float(self.max_corner_geometry_violation)
        self.worst_geometry_particle_id = None if self.worst_geometry_particle_id is None else int(self.worst_geometry_particle_id)
        self.worst_geometry_feature = None if self.worst_geometry_feature is None else str(self.worst_geometry_feature)
        self.min_abs_signed_clearance = float(self.min_abs_signed_clearance)
        self.num_extent_tests = int(self.num_extent_tests)
        self.num_pairs_rejected_by_extent = int(self.num_pairs_rejected_by_extent)
        self.step_alpha = None if self.step_alpha is None else float(self.step_alpha)
        self.line_search_trials = int(self.line_search_trials)
        self.line_search_energy_evals = int(self.line_search_energy_evals)
        self.line_search_success = bool(self.line_search_success)
        self.time_total_step = float(self.time_total_step)
        self.time_pair_generation = float(self.time_pair_generation)
        self.time_broadphase = float(self.time_broadphase)
        self.time_neighbor_build = float(self.time_neighbor_build)
        self.time_pw_eval = float(self.time_pw_eval)
        self.time_extent_filter = float(self.time_extent_filter)
        self.time_pw_contact = float(self.time_pw_contact)
        self.time_geometry_attach = float(self.time_geometry_attach)
        self.time_gradient_eval = float(self.time_gradient_eval)
        self.time_line_search = float(self.time_line_search)
        self.time_fast_extent_filter = float(self.time_fast_extent_filter)
        self.time_fast_pw_contact = float(self.time_fast_pw_contact)
        self.time_system_status = float(self.time_system_status)
        self.time_strict_audit = float(self.time_strict_audit)
        int_fields = [('num_particles', self.num_particles), ('num_all_pairs', self.num_all_pairs), ('num_candidate_pairs', self.num_candidate_pairs), ('num_pw_calls', self.num_pw_calls), ('num_overlap_pairs', self.num_overlap_pairs), ('num_overlapping_pairs', self.num_overlapping_pairs), ('num_geometry_attached_pairs', self.num_geometry_attached_pairs), ('line_search_trials', self.line_search_trials), ('line_search_energy_evals', self.line_search_energy_evals), ('line_search_early_aborts', self.line_search_early_aborts), ('num_active_face_constraints', self.num_active_face_constraints), ('num_active_edge_constraints', self.num_active_edge_constraints), ('num_active_corner_constraints', self.num_active_corner_constraints), ('num_face_tests_total', self.num_face_tests_total), ('num_face_candidates', self.num_face_candidates), ('num_face_skipped_by_filter', self.num_face_skipped_by_filter), ('num_edge_tests_total', self.num_edge_tests_total), ('num_edge_candidates', self.num_edge_candidates), ('num_edge_skipped_by_filter', self.num_edge_skipped_by_filter), ('num_corner_tests_total', self.num_corner_tests_total), ('num_corner_candidates', self.num_corner_candidates), ('num_corner_skipped_by_filter', self.num_corner_skipped_by_filter), ('num_corner_exact_distance_calls', self.num_corner_exact_distance_calls), ('geo_cache_rebuilds', self.geo_cache_rebuilds), ('geo_cache_face_candidates', self.geo_cache_face_candidates), ('geo_cache_edge_candidates', self.geo_cache_edge_candidates), ('geo_cache_corner_candidates', self.geo_cache_corner_candidates), ('geo_fast_eval_used', self.geo_fast_eval_used), ('geo_fast_eval_fallback_full_count', self.geo_fast_eval_fallback_full_count), ('num_extent_tests', self.num_extent_tests), ('num_pairs_rejected_by_extent', self.num_pairs_rejected_by_extent)]
        for name, value in int_fields:
            if value < 0:
                raise ValueError(f'{name} must be >= 0, got {value}.')
        float_fields = [('time_total_step', self.time_total_step), ('time_pair_generation', self.time_pair_generation), ('time_broadphase', self.time_broadphase), ('time_neighbor_build', self.time_neighbor_build), ('time_pw_eval', self.time_pw_eval), ('time_extent_filter', self.time_extent_filter), ('time_pw_contact', self.time_pw_contact), ('time_geometry_attach', self.time_geometry_attach), ('time_gradient_eval', self.time_gradient_eval), ('time_line_search', self.time_line_search), ('time_fast_extent_filter', self.time_fast_extent_filter), ('time_fast_pw_contact', self.time_fast_pw_contact), ('time_system_status', self.time_system_status), ('time_strict_audit', self.time_strict_audit), ('geometry_energy', self.geometry_energy), ('face_energy', self.face_energy), ('edge_energy', self.edge_energy), ('corner_energy', self.corner_energy), ('worst_geometry_violation', self.worst_geometry_violation), ('max_geometry_violation', self.max_geometry_violation), ('max_face_violation', self.max_face_violation), ('max_edge_violation', self.max_edge_violation), ('max_corner_violation', self.max_corner_violation), ('max_face_geometry_violation', self.max_face_geometry_violation), ('max_edge_geometry_violation', self.max_edge_geometry_violation), ('max_corner_geometry_violation', self.max_corner_geometry_violation), ('min_abs_signed_clearance', self.min_abs_signed_clearance)]
        for name, value in float_fields:
            if value < 0.0:
                raise ValueError(f'{name} must be >= 0, got {value}.')

    def candidate_ratio(self) -> float:
        if self.num_all_pairs <= 0:
            return 0.0
        return float(self.num_candidate_pairs) / float(self.num_all_pairs)

    def pw_call_ratio(self) -> float:
        if self.num_all_pairs <= 0:
            return 0.0
        return float(self.num_pw_calls) / float(self.num_all_pairs)

    def energy_drop(self) -> float:
        return float(self.energy_before - self.energy_after)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data['candidate_ratio'] = self.candidate_ratio()
        data['pw_call_ratio'] = self.pw_call_ratio()
        data['energy_drop'] = self.energy_drop()
        return data

@dataclass
class RunMetrics3D:
    steps: list[StepMetrics3D] = field(default_factory=list)
    total_wall_time: float = 0.0
    total_pw_calls: int = 0
    total_candidate_pairs: int = 0
    total_line_search_trials: int = 0
    total_line_search_energy_evals: int = 0
    total_neighbor_rebuilds: int = 0
    num_steps: int = 0
    final_energy: float = 0.0
    final_max_overlap: float = 0.0
    converged: bool = False

    def __post_init__(self) -> None:
        self.steps = list(self.steps)
        for idx, step in enumerate(self.steps):
            if not isinstance(step, StepMetrics3D):
                raise ValueError(f'steps[{idx}] must be StepMetrics3D, got {type(step)!r}.')
        self.total_wall_time = float(self.total_wall_time)
        self.total_pw_calls = int(self.total_pw_calls)
        self.total_candidate_pairs = int(self.total_candidate_pairs)
        self.total_line_search_trials = int(self.total_line_search_trials)
        self.total_line_search_energy_evals = int(self.total_line_search_energy_evals)
        self.total_neighbor_rebuilds = int(self.total_neighbor_rebuilds)
        self.num_steps = int(self.num_steps)
        self.final_energy = float(self.final_energy)
        self.final_max_overlap = float(self.final_max_overlap)
        self.converged = bool(self.converged)
        if self.total_wall_time < 0.0:
            raise ValueError(f'total_wall_time must be >= 0, got {self.total_wall_time}.')
        if self.total_pw_calls < 0:
            raise ValueError(f'total_pw_calls must be >= 0, got {self.total_pw_calls}.')
        if self.total_candidate_pairs < 0:
            raise ValueError(f'total_candidate_pairs must be >= 0, got {self.total_candidate_pairs}.')
        if self.total_line_search_trials < 0:
            raise ValueError(f'total_line_search_trials must be >= 0, got {self.total_line_search_trials}.')
        if self.total_line_search_energy_evals < 0:
            raise ValueError(f'total_line_search_energy_evals must be >= 0, got {self.total_line_search_energy_evals}.')
        if self.total_neighbor_rebuilds < 0:
            raise ValueError(f'total_neighbor_rebuilds must be >= 0, got {self.total_neighbor_rebuilds}.')
        if self.num_steps < 0:
            raise ValueError(f'num_steps must be >= 0, got {self.num_steps}.')

    def append_step(self, step_metrics: StepMetrics3D) -> None:
        if not isinstance(step_metrics, StepMetrics3D):
            raise ValueError(f'step_metrics must be StepMetrics3D, got {type(step_metrics)!r}.')
        self.steps.append(step_metrics)
        self.num_steps = int(len(self.steps))
        self.total_pw_calls += int(step_metrics.num_pw_calls)
        self.total_candidate_pairs += int(step_metrics.num_candidate_pairs)
        self.total_line_search_trials += int(step_metrics.line_search_trials)
        self.total_line_search_energy_evals += int(step_metrics.line_search_energy_evals)
        self.final_energy = float(step_metrics.energy_after)
        self.final_max_overlap = float(step_metrics.max_overlap)

    def to_dict(self) -> dict[str, object]:
        return {'total_wall_time': float(self.total_wall_time), 'total_pw_calls': int(self.total_pw_calls), 'total_candidate_pairs': int(self.total_candidate_pairs), 'total_line_search_trials': int(self.total_line_search_trials), 'total_line_search_energy_evals': int(self.total_line_search_energy_evals), 'total_neighbor_rebuilds': int(self.total_neighbor_rebuilds), 'num_steps': int(self.num_steps), 'final_energy': float(self.final_energy), 'final_max_overlap': float(self.final_max_overlap), 'converged': bool(self.converged), 'steps': [step.to_dict() for step in self.steps]}

    def summary_dict(self) -> dict[str, object]:
        return {'total_wall_time': float(self.total_wall_time), 'total_pw_calls': int(self.total_pw_calls), 'total_candidate_pairs': int(self.total_candidate_pairs), 'total_line_search_trials': int(self.total_line_search_trials), 'total_line_search_energy_evals': int(self.total_line_search_energy_evals), 'total_neighbor_rebuilds': int(self.total_neighbor_rebuilds), 'num_steps': int(self.num_steps), 'final_energy': float(self.final_energy), 'final_max_overlap': float(self.final_max_overlap), 'converged': bool(self.converged)}
