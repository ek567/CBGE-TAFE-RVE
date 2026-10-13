from __future__ import annotations
from typing import Any
import numpy as np
from src.data.pair_result_3d import PairContactResult3D
from src.data.particle_3d import EllipsoidParticle3D
from src.model.total_potential_3d import compute_total_gradients, compute_total_potential, evaluate_all_pairs
from src.overlap_detection.candidate_pairs_3d import CandidatePairConfig3D

def _validate_particles(particles: list[EllipsoidParticle3D]) -> None:
    if not isinstance(particles, list):
        raise ValueError('particles must be a list.')
    for idx, particle in enumerate(particles):
        if not isinstance(particle, EllipsoidParticle3D):
            raise ValueError(f'particles[{idx}] must be EllipsoidParticle3D, got {type(particle)!r}.')

def _validate_box(box: np.ndarray | None) -> np.ndarray | None:
    if box is None:
        return None
    box_array = np.asarray(box, dtype=float)
    if box_array.shape != (3,):
        raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
    if np.any(box_array <= 0.0):
        raise ValueError(f'Box lengths must be positive, got {box_array}.')
    return box_array

def _validate_pair_result(result: PairContactResult3D, name: str) -> None:
    if not isinstance(result, PairContactResult3D):
        raise ValueError(f'{name} must be PairContactResult3D, got {type(result)!r}.')
    if result.r_ij.shape != (3,):
        raise ValueError(f'Expected {name}.r_ij.shape == (3,), got {result.r_ij.shape}.')
    if result.Q_star is not None and np.asarray(result.Q_star).shape != (3, 3):
        raise ValueError(f'Expected {name}.Q_star.shape == (3, 3), got {np.asarray(result.Q_star).shape}.')
    if result.s_ij is not None and np.asarray(result.s_ij).shape != (3,):
        raise ValueError(f'Expected {name}.s_ij.shape == (3,), got {np.asarray(result.s_ij).shape}.')
    if result.x0_ij is not None and np.asarray(result.x0_ij).shape != (3,):
        raise ValueError(f'Expected {name}.x0_ij.shape == (3,), got {np.asarray(result.x0_ij).shape}.')

def _validate_pair_results(pair_results: list[PairContactResult3D], name: str) -> None:
    if not isinstance(pair_results, list):
        raise ValueError(f'{name} must be a list.')
    for idx, result in enumerate(pair_results):
        _validate_pair_result(result, f'{name}[{idx}]')

def _pair_key(result: PairContactResult3D) -> tuple[int, int]:
    i = int(result.i)
    j = int(result.j)
    if i == j:
        raise ValueError(f'Pair key must satisfy i != j, got ({i}, {j}).')
    if i > j:
        i, j = (j, i)
    return (i, j)

def pair_results_to_map(pair_results: list[PairContactResult3D]) -> dict[tuple[int, int], PairContactResult3D]:
    _validate_pair_results(pair_results, 'pair_results')
    result_map: dict[tuple[int, int], PairContactResult3D] = {}
    for idx, result in enumerate(pair_results):
        key = _pair_key(result)
        if key in result_map:
            raise ValueError(f'Duplicate pair key {key} found in pair_results at index {idx}.')
        result_map[key] = result
    return result_map

def _max_abs_array_diff(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None and b is None:
        return 0.0
    if (a is None) != (b is None):
        return float(np.inf)
    a_array = np.asarray(a, dtype=float)
    b_array = np.asarray(b, dtype=float)
    if a_array.shape != b_array.shape:
        return float(np.inf)
    return float(np.max(np.abs(a_array - b_array)))

def compare_pair_result_maps(map_a: dict[tuple[int, int], PairContactResult3D], map_b: dict[tuple[int, int], PairContactResult3D], scalar_tol: float=1e-10, array_tol: float=1e-10, compare_geometry: bool=True) -> dict[str, Any]:
    if not isinstance(map_a, dict):
        raise ValueError('map_a must be a dict.')
    if not isinstance(map_b, dict):
        raise ValueError('map_b must be a dict.')
    scalar_tol_value = float(scalar_tol)
    array_tol_value = float(array_tol)
    if scalar_tol_value < 0.0:
        raise ValueError(f'scalar_tol must be >= 0, got {scalar_tol_value}.')
    if array_tol_value < 0.0:
        raise ValueError(f'array_tol must be >= 0, got {array_tol_value}.')
    keys_a = set(map_a.keys())
    keys_b = set(map_b.keys())
    missing_in_b = sorted(keys_a - keys_b)
    extra_in_b = sorted(keys_b - keys_a)
    common_keys = sorted(keys_a & keys_b)
    max_abs_diff_F = 0.0
    max_abs_diff_lambda = 0.0
    max_abs_diff_overlap = 0.0
    max_abs_diff_psi = 0.0
    max_abs_diff_r = 0.0
    max_abs_diff_Q_star = 0.0
    max_abs_diff_s_ij = 0.0
    max_abs_diff_x0_ij = 0.0
    mismatch_records: list[dict[str, Any]] = []
    for key in common_keys:
        result_a = map_a[key]
        result_b = map_b[key]
        _validate_pair_result(result_a, f'map_a[{key}]')
        _validate_pair_result(result_b, f'map_b[{key}]')
        diff_F = abs(float(result_a.F_ij) - float(result_b.F_ij))
        diff_lambda = abs(float(result_a.lambda_star) - float(result_b.lambda_star))
        diff_overlap = abs(float(result_a.overlap) - float(result_b.overlap))
        diff_psi = abs(float(result_a.psi_ij) - float(result_b.psi_ij))
        diff_r = float(np.max(np.abs(np.asarray(result_a.r_ij) - np.asarray(result_b.r_ij))))
        max_abs_diff_F = max(max_abs_diff_F, diff_F)
        max_abs_diff_lambda = max(max_abs_diff_lambda, diff_lambda)
        max_abs_diff_overlap = max(max_abs_diff_overlap, diff_overlap)
        max_abs_diff_psi = max(max_abs_diff_psi, diff_psi)
        max_abs_diff_r = max(max_abs_diff_r, diff_r)
        geometry_record: dict[str, float] = {}
        geometry_failed = False
        if compare_geometry:
            diff_Q = _max_abs_array_diff(result_a.Q_star, result_b.Q_star)
            diff_s = _max_abs_array_diff(result_a.s_ij, result_b.s_ij)
            diff_x0 = _max_abs_array_diff(result_a.x0_ij, result_b.x0_ij)
            max_abs_diff_Q_star = max(max_abs_diff_Q_star, diff_Q)
            max_abs_diff_s_ij = max(max_abs_diff_s_ij, diff_s)
            max_abs_diff_x0_ij = max(max_abs_diff_x0_ij, diff_x0)
            geometry_record = {'Q_star': float(diff_Q), 's_ij': float(diff_s), 'x0_ij': float(diff_x0)}
            if diff_Q > array_tol_value or diff_s > array_tol_value or diff_x0 > array_tol_value:
                geometry_failed = True
        scalar_failed = diff_F > scalar_tol_value or diff_lambda > scalar_tol_value or diff_overlap > scalar_tol_value or (diff_psi > scalar_tol_value) or (diff_r > array_tol_value)
        if scalar_failed or geometry_failed:
            record: dict[str, Any] = {'pair': key, 'diff_F_ij': float(diff_F), 'diff_lambda_star': float(diff_lambda), 'diff_overlap': float(diff_overlap), 'diff_psi_ij': float(diff_psi), 'diff_r_ij': float(diff_r)}
            if compare_geometry:
                record['geometry_diffs'] = geometry_record
            mismatch_records.append(record)
    all_ok = len(missing_in_b) == 0 and len(extra_in_b) == 0 and (max_abs_diff_F <= scalar_tol_value) and (max_abs_diff_lambda <= scalar_tol_value) and (max_abs_diff_overlap <= scalar_tol_value) and (max_abs_diff_psi <= scalar_tol_value) and (max_abs_diff_r <= array_tol_value)
    if compare_geometry:
        all_ok = all_ok and (max_abs_diff_Q_star <= array_tol_value and max_abs_diff_s_ij <= array_tol_value and (max_abs_diff_x0_ij <= array_tol_value))
    return {'ok': bool(all_ok), 'num_keys_a': int(len(keys_a)), 'num_keys_b': int(len(keys_b)), 'num_common_keys': int(len(common_keys)), 'missing_in_b': missing_in_b, 'extra_in_b': extra_in_b, 'max_abs_diff_F_ij': float(max_abs_diff_F), 'max_abs_diff_lambda_star': float(max_abs_diff_lambda), 'max_abs_diff_overlap': float(max_abs_diff_overlap), 'max_abs_diff_psi_ij': float(max_abs_diff_psi), 'max_abs_diff_r_ij': float(max_abs_diff_r), 'max_abs_diff_Q_star': float(max_abs_diff_Q_star), 'max_abs_diff_s_ij': float(max_abs_diff_s_ij), 'max_abs_diff_x0_ij': float(max_abs_diff_x0_ij), 'mismatch_records': mismatch_records}

def _grad_list_max_abs_diff(grad_list_a: list[np.ndarray], grad_list_b: list[np.ndarray]) -> float:
    if not isinstance(grad_list_a, list):
        raise ValueError('grad_list_a must be a list.')
    if not isinstance(grad_list_b, list):
        raise ValueError('grad_list_b must be a list.')
    if len(grad_list_a) != len(grad_list_b):
        raise ValueError(f'Gradient list lengths differ: {len(grad_list_a)} != {len(grad_list_b)}.')
    max_diff = 0.0
    for idx, (ga, gb) in enumerate(zip(grad_list_a, grad_list_b)):
        ga_array = np.asarray(ga, dtype=float)
        gb_array = np.asarray(gb, dtype=float)
        if ga_array.shape != (3,):
            raise ValueError(f'Expected grad_list_a[{idx}].shape == (3,), got {ga_array.shape}.')
        if gb_array.shape != (3,):
            raise ValueError(f'Expected grad_list_b[{idx}].shape == (3,), got {gb_array.shape}.')
        max_diff = max(max_diff, float(np.max(np.abs(ga_array - gb_array))))
    return float(max_diff)

def validate_candidate_pair_consistency(particles: list[EllipsoidParticle3D], box: np.ndarray | None=None, use_mic: bool=False, all_pairs_config: CandidatePairConfig3D | None=None, grid_config: CandidatePairConfig3D | None=None, attach_geometry: bool=True, scalar_tol: float=1e-10, array_tol: float=1e-10, gradient_tol: float=1e-10, compare_geometry: bool=True) -> dict[str, Any]:
    _validate_particles(particles)
    box_array = _validate_box(box)
    scalar_tol_value = float(scalar_tol)
    array_tol_value = float(array_tol)
    gradient_tol_value = float(gradient_tol)
    if scalar_tol_value < 0.0:
        raise ValueError(f'Expected scalar_tol >= 0, got {scalar_tol_value}.')
    if array_tol_value < 0.0:
        raise ValueError(f'Expected array_tol >= 0, got {array_tol_value}.')
    if gradient_tol_value < 0.0:
        raise ValueError(f'Expected gradient_tol >= 0, got {gradient_tol_value}.')
    if all_pairs_config is None:
        all_pairs_config = CandidatePairConfig3D(method='all_pairs', use_bounding_sphere=True, sphere_padding=0.0, cell_size=None, auto_cell_size=True)
    if grid_config is None:
        grid_config = CandidatePairConfig3D(method='uniform_grid', use_bounding_sphere=True, sphere_padding=0.0, cell_size=None, auto_cell_size=True)
    if not isinstance(all_pairs_config, CandidatePairConfig3D):
        raise ValueError(f'all_pairs_config must be CandidatePairConfig3D or None, got {type(all_pairs_config)!r}.')
    if not isinstance(grid_config, CandidatePairConfig3D):
        raise ValueError(f'grid_config must be CandidatePairConfig3D or None, got {type(grid_config)!r}.')
    pair_results_all = evaluate_all_pairs(particles=particles, box=box_array, use_mic=use_mic, attach_geometry=attach_geometry, pair_config=all_pairs_config, metrics=None)
    pair_results_grid = evaluate_all_pairs(particles=particles, box=box_array, use_mic=use_mic, attach_geometry=attach_geometry, pair_config=grid_config, metrics=None)
    pair_map_all = pair_results_to_map(pair_results_all)
    pair_map_grid = pair_results_to_map(pair_results_grid)
    pair_comparison = compare_pair_result_maps(map_a=pair_map_all, map_b=pair_map_grid, scalar_tol=scalar_tol_value, array_tol=array_tol_value, compare_geometry=compare_geometry)
    total_energy_all = float(compute_total_potential(pair_results_all))
    total_energy_grid = float(compute_total_potential(pair_results_grid))
    total_energy_abs_diff = abs(total_energy_all - total_energy_grid)
    grad_cs_all, grad_ps_all, _ = compute_total_gradients(particles=particles, pair_results=pair_results_all, box=box_array, use_mic=use_mic, pair_config=all_pairs_config, metrics=None)
    grad_cs_grid, grad_ps_grid, _ = compute_total_gradients(particles=particles, pair_results=pair_results_grid, box=box_array, use_mic=use_mic, pair_config=grid_config, metrics=None)
    grad_c_max_abs_diff = _grad_list_max_abs_diff(grad_cs_all, grad_cs_grid)
    grad_p_max_abs_diff = _grad_list_max_abs_diff(grad_ps_all, grad_ps_grid)
    energy_ok = total_energy_abs_diff <= scalar_tol_value
    gradient_ok = grad_c_max_abs_diff <= gradient_tol_value and grad_p_max_abs_diff <= gradient_tol_value
    ok = bool(pair_comparison['ok'] and energy_ok and gradient_ok)
    return {'ok': ok, 'use_mic': bool(use_mic), 'attach_geometry': bool(attach_geometry), 'all_pairs_config': {'method': all_pairs_config.method, 'use_bounding_sphere': all_pairs_config.use_bounding_sphere, 'sphere_padding': float(all_pairs_config.sphere_padding), 'cell_size': all_pairs_config.cell_size, 'auto_cell_size': bool(all_pairs_config.auto_cell_size)}, 'grid_config': {'method': grid_config.method, 'use_bounding_sphere': grid_config.use_bounding_sphere, 'sphere_padding': float(grid_config.sphere_padding), 'cell_size': grid_config.cell_size, 'auto_cell_size': bool(grid_config.auto_cell_size)}, 'pair_comparison': pair_comparison, 'total_energy_all_pairs': float(total_energy_all), 'total_energy_uniform_grid': float(total_energy_grid), 'total_energy_abs_diff': float(total_energy_abs_diff), 'grad_c_max_abs_diff': float(grad_c_max_abs_diff), 'grad_p_max_abs_diff': float(grad_p_max_abs_diff), 'energy_ok': bool(energy_ok), 'gradient_ok': bool(gradient_ok)}
