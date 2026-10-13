from __future__ import annotations
import sys
import argparse
import copy
from dataclasses import replace, asdict
from typing import Sequence, Any
import numpy as np
import json
import hashlib
import uuid
import os
import time
from pathlib import Path
from datetime import datetime
try:
    from src.validation.spatial_statistics_3d import SpatialStatsMVPConfig3D, compute_spatial_statistics_mvp_3d
    from src.validation.spatial_statistics_plots_3d import plot_nnd_pdfs, plot_nnoa_cdfs, plot_ripley_and_rdf, plot_orientation_isotropy_boxplot
    _SPATIAL_STATS_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    SpatialStatsMVPConfig3D = None
    compute_spatial_statistics_mvp_3d = None
    plot_nnd_pdfs = None
    plot_nnoa_cdfs = None
    plot_ripley_and_rdf = None
    plot_orientation_isotropy_boxplot = None
    _SPATIAL_STATS_IMPORT_ERROR = exc
from src.export.export_json_3d import export_particles_json_3d
from src.data.particle_3d import EllipsoidParticle3D
from src.model.orientation_regularization_3d import orientation_regularization_summary
from src.model.orientation_targets_3d import ORIENTATION_TOLERANCE_DEFAULTS, orientation_target_tensor_3d, resolve_orientation_tolerance, validate_orientation_target
from src.model.overlap_status_3d import system_overlap_status
from src.model.total_potential_3d import compute_total_potential
from src.monitoring.exporters_3d import save_run_metrics_json, save_run_summary_json, save_step_metrics_csv
from src.monitoring.metrics_3d import StepMetrics3D
from src.monitoring.timers_3d import timed_field
from src.optimizer.descent_3d import DescentConfig, effective_inflated_search_overlap_tol
from src.optimizer.step_policy_3d import LineSearchConfig
from src.resource_budget import RealizationIterationBudget, apply_resource_result
from src.overlap_detection.candidate_pairs_3d import CandidatePairConfig3D
from src.periodic.mic_3d import wrap_position
from src.validation.expanded_periodic_overlap_audit_3d import audit_periodic_system_overlap
from src.validation.orientation_isotropy_check_3d import check_orientation_isotropy, format_orientation_isotropy_report
from src.generation.size_generator_3d import TruncNormalSpec, SizeGenerationConfig3D, generate_ellipsoid_sizes_from_truncnorm
from src.generation.size_generation_retry_3d import generate_sizes_with_seed_retries
from src.workflow.run_descent_with_constraints_3d import run_descent_with_constraints_3d
from src.postprocess.origin_shift_3d import OriginShiftConfig3D
from src.model.geometry_distortion_energy_3d import GeometryDistortionConfig3D
from src.validation.geometry_distortion_check_3d import audit_geometry_distortion_3d
from src.adaptation_identity import ADAPTATION_REVISION_ID, DEFAULT_PROFILE_ID, PARENT_BASELINE_ID, read_current_adaptation_identity
SPATIAL_STATS_SCHEMA_VERSION = 'spatial_stats_v2_theta_phi_scaledK'
GEO_EDGE_TARGET_FACE_FACTOR = 1.0
GEO_EDGE_TARGET_EDGE_FACTOR = 1.0
GEO_EDGE_TARGET_CORNER_FACTOR = 1.0

def _require_spatial_statistics_available() -> None:
    if _SPATIAL_STATS_IMPORT_ERROR is not None:
        raise RuntimeError(f'--spatial-stats requires optional spatial statistics dependencies (for example scipy/matplotlib). Import failed: {_SPATIAL_STATS_IMPORT_ERROR}')

def _ensure_pypw85_available() -> bool:
    try:
        import pypw85
    except ImportError:
        return False
    return True

def _format_scalar(value: float) -> str:
    return f'{float(value):.12e}'

def _parse_box(box_values: Sequence[float] | None) -> np.ndarray | None:
    if box_values is None:
        return None
    if len(box_values) != 3:
        raise ValueError(f'Expected exactly 3 box values, got {len(box_values)}.')
    box = np.asarray(box_values, dtype=float)
    if box.shape != (3,):
        raise ValueError(f'Expected box.shape == (3,), got {box.shape}.')
    if np.any(box <= 0.0):
        raise ValueError(f'Box lengths must be positive, got {box}.')
    return box

def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    norm = float(np.linalg.norm(v))
    while norm == 0.0:
        v = rng.normal(size=3)
        norm = float(np.linalg.norm(v))
    return v / norm

def _apply_orientation_initialization(particles: list[EllipsoidParticle3D], rng: np.random.Generator, *, orientation_target: str='3d-isotropic', orientation_regularization: bool=False) -> list[EllipsoidParticle3D]:
    target = validate_orientation_target(orientation_target)
    if not orientation_regularization or target == '3d-isotropic':
        return particles
    count = len(particles)
    if target == '1d-x':
        for particle in particles:
            particle.p = np.array([1.0, 0.0, 0.0], dtype=float)
        return particles
    if count < 2:
        raise ValueError('2d-xz orientation initialization requires at least two particles when ORI is enabled.')
    phase = float(rng.uniform(0.0, np.pi))
    angles = phase + np.pi / count * np.arange(count, dtype=float)
    directions = np.column_stack((np.cos(angles), np.zeros(count), np.sin(angles)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    assignment = rng.permutation(count)
    for particle, direction in zip(particles, directions[assignment]):
        particle.p = direction.copy()
    return particles

def _orientation_initialization_policy(args) -> dict[str, Any]:
    target = validate_orientation_target(str(args.orientation_target))
    enabled = bool(args.orientation_regularization)
    fixed_demo = bool(getattr(args, 'fixed_demo', False))
    if fixed_demo:
        method, reason, sampling = ('fixed_demo_preserved', 'fixed_demo', 'native_fixture')
    elif not enabled:
        method, reason, sampling = ('legacy_sphere_random_ori_off', 'ori_off', 'legacy_sphere_random')
    elif target == '1d-x':
        method, reason, sampling = ('aligned_x', 'target_aware', 'single_apolar_axis')
    elif target == '2d-xz':
        method, reason, sampling = ('planar_xz_even_angles', 'target_aware', 'even_apolar_angles_with_random_phase_and_permutation')
    else:
        method, reason, sampling = ('legacy_sphere_random', 'legacy_3d', 'legacy_sphere_random')
    return {'schema': 'pw-rve-orientation-initialization-policy-v1', 'orientation_target': target, 'orientation_optimization_enabled': enabled, 'method': method, 'target_adjustment_applied': reason == 'target_aware', 'selection_reason': reason, 'scope': 'random_particle_generation_only', 'minimum_particle_count': 2 if method == 'planar_xz_even_angles' else 1, 'angular_sampling': sampling, 'center_rng_policy': 'preserve_all_legacy_center_and_sphere_draws_before_target_adjustment', 'directions_frozen_during_optimization': False}

def _initial_orientation_metadata(args, particles) -> dict[str, Any]:
    if not particles:
        raise ValueError('Cannot summarize an empty orientation initialization.')
    directions = np.asarray([particle.p for particle in particles], dtype=float)
    tensor = directions.T @ directions / len(particles)
    target = orientation_target_tensor_3d(str(args.orientation_target))
    residual = tensor - target
    return {'policy': _orientation_initialization_policy(args), 'particle_count': len(particles), 'weighting': 'number_average', 'initial_second_order_tensor': tensor.tolist(), 'target_second_order_tensor': target.tolist(), 'spectral_residual': float(np.linalg.norm(residual, ord=2)), 'frobenius_residual': float(np.linalg.norm(residual, ord='fro')), 'max_abs_entry_residual': float(np.max(np.abs(residual))), 'max_unit_norm_error': float(np.max(np.abs(np.linalg.norm(directions, axis=1) - 1.0))), 'initialization_only': True, 'initialization_executed': True}

def build_demo_particles_nonperiodic() -> list[EllipsoidParticle3D]:
    return [EllipsoidParticle3D(a=3.0, b=1.0, c=np.array([0.0, 0.0, 0.0], dtype=float), p=np.array([1.0, 0.0, 0.0], dtype=float), id=0), EllipsoidParticle3D(a=3.0, b=1.0, c=np.array([2.0, 0.0, 0.0], dtype=float), p=np.array([0.0, 1.0, 0.0], dtype=float), id=1)]

def build_demo_particles_periodic(box: np.ndarray) -> list[EllipsoidParticle3D]:
    box_array = np.asarray(box, dtype=float)
    if box_array.shape != (3,):
        raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
    return [EllipsoidParticle3D(a=2.8, b=1.0, c=np.array([1.0, 2.5, 2.5], dtype=float), p=np.array([1.0, 0.0, 0.0], dtype=float), id=0), EllipsoidParticle3D(a=2.8, b=1.0, c=np.array([box_array[0] - 1.2, 2.5, 2.5], dtype=float), p=np.array([0.0, 1.0, 0.0], dtype=float), id=1), EllipsoidParticle3D(a=2.2, b=0.9, c=np.array([4.5, 4.0, 2.5], dtype=float), p=np.array([0.0, 0.0, 1.0], dtype=float), id=2)]

def build_random_demo_particles(n_particles: int, a: float, b: float, box: np.ndarray | None, seed: int, periodic: bool, *, orientation_target: str='3d-isotropic', orientation_regularization: bool=False) -> list[EllipsoidParticle3D]:
    if n_particles <= 0:
        raise ValueError(f'Expected n_particles > 0, got {n_particles}.')
    if not a >= b > 0.0:
        raise ValueError(f'Expected a >= b > 0, got a={a}, b={b}.')
    rng = np.random.default_rng(seed)
    particles: list[EllipsoidParticle3D] = []
    if periodic:
        if box is None:
            raise ValueError('box must be provided for periodic random demo.')
        box_array = np.asarray(box, dtype=float)
        if box_array.shape != (3,):
            raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
        for idx in range(n_particles):
            c = rng.uniform(low=0.0, high=1.0, size=3) * box_array
            p = _random_unit_vector(rng)
            particles.append(EllipsoidParticle3D(a=a, b=b, c=c, p=p, id=idx))
    else:
        scale = max(2.0 * a, 2.0)
        for idx in range(n_particles):
            c = rng.uniform(low=-1.0, high=1.0, size=3) * scale
            p = _random_unit_vector(rng)
            particles.append(EllipsoidParticle3D(a=a, b=b, c=c, p=p, id=idx))
    return _apply_orientation_initialization(particles, rng, orientation_target=orientation_target, orientation_regularization=orientation_regularization)

def build_random_particles_from_size_arrays(a_arr: np.ndarray, b_arr: np.ndarray, box: np.ndarray, seed: int, *, orientation_target: str='3d-isotropic', orientation_regularization: bool=False) -> list[EllipsoidParticle3D]:
    a_arr = np.asarray(a_arr, dtype=float).reshape(-1)
    b_arr = np.asarray(b_arr, dtype=float).reshape(-1)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f'a_arr shape {a_arr.shape} != b_arr shape {b_arr.shape}')
    if a_arr.size == 0:
        raise ValueError('Empty size arrays.')
    if np.any(a_arr < b_arr) or np.any(b_arr <= 0.0):
        raise ValueError('Invalid sizes: require a_i >= b_i > 0 for all i.')
    box_array = np.asarray(box, dtype=float).reshape(3)
    if np.any(box_array <= 0.0):
        raise ValueError('Invalid box for periodic particles.')
    rng = np.random.default_rng(seed)
    particles: list[EllipsoidParticle3D] = []
    for idx in range(a_arr.size):
        c = rng.uniform(low=0.0, high=1.0, size=3) * box_array
        p = _random_unit_vector(rng)
        particles.append(EllipsoidParticle3D(a=float(a_arr[idx]), b=float(b_arr[idx]), c=c, p=p, id=idx))
    return _apply_orientation_initialization(particles, rng, orientation_target=orientation_target, orientation_regularization=orientation_regularization)

def _print_system_summary(particles: list[EllipsoidParticle3D], box: np.ndarray | None, use_mic: bool, overlap_tol: float, label: str, metrics: StepMetrics3D | None=None, include_orientation_regularization: bool=False, orientation_target: str='3d-isotropic') -> tuple[float, dict]:
    with timed_field(metrics, 'time_system_status'):
        status = system_overlap_status(particles=particles, box=box, use_mic=use_mic, overlap_tol=overlap_tol)
    total_energy = compute_total_potential(status['pair_results'], particles=particles, include_orientation_regularization=bool(include_orientation_regularization), orientation_target=orientation_target)
    max_info = status.get('max_info', None)
    print(f'{label} total potential: {_format_scalar(total_energy)}')
    if max_info is not None:
        print(f"{label} max-overlap pair: {max_info['pair_indices']}, overlap={_format_scalar(max_info['max_overlap'])}, F_ij={_format_scalar(max_info['F_ij'])}, lambda_star={_format_scalar(max_info['lambda_star'])}, psi_ij={_format_scalar(max_info['psi_ij'])}")
    print(f"{label} overlap status: has_overlap={status['has_overlap']}, num_overlapping_pairs={status['num_overlapping_pairs']}, overlap_tol={_format_scalar(overlap_tol)}")
    if include_orientation_regularization:
        ori_summary = orientation_regularization_summary(particles, orientation_target=orientation_target)
        print(f"{label} orientation regularization energy: {_format_scalar(ori_summary['energy'])}")
        print(f"{label} orientation tensor eigenvalues: {np.array2string(ori_summary['eigenvalues'], precision=8, suppress_small=False)}")
    if metrics is not None:
        metrics.energy_after = float(total_energy)
        metrics.max_overlap = float(status['max_overlap'])
        metrics.num_overlapping_pairs = int(status['num_overlapping_pairs'])
    return (float(total_energy), status)

class _AspectRatioKAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, '_aspect_ratio_k_explicit', True)

class _NativeArgumentParser(argparse.ArgumentParser):

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        try:
            resolve_orientation_tolerance(parsed.orientation_target, parsed.orientation_tol)
        except ValueError as exc:
            self.error(str(exc))
        if parsed.size_gen_seed_retries < 0:
            self.error('--size-gen-seed-retries must be >= 0 (extra retries)')
        if parsed.heartbeat_steps < 0:
            self.error('--heartbeat-steps must be >= 0')
        if parsed.aspect_ratio_mode == 'fixed':
            if parsed.aspect_ratio_min is not None or parsed.aspect_ratio_max is not None:
                self.error('--aspect-ratio-min/--aspect-ratio-max require --aspect-ratio-mode uniform')
        else:
            if parsed._aspect_ratio_k_explicit:
                self.error('--aspect-ratio-k conflicts with --aspect-ratio-mode uniform')
            lo, hi = (parsed.aspect_ratio_min, parsed.aspect_ratio_max)
            if lo is None or hi is None or (not (np.isfinite(lo) and np.isfinite(hi) and (1 <= lo < hi))):
                self.error('uniform requires finite --aspect-ratio-min/--aspect-ratio-max with 1 <= min < max')
            if not _size_dist_requested(parsed) or parsed.fixed_demo or parsed.nonperiodic:
                self.error('uniform requires the periodic size-distribution path, not a demo or --nonperiodic')
            try:
                _require_all_size_dist_args(parsed)
            except ValueError as exc:
                self.error(str(exc))
        return parsed

def _add_default_on_switch(parser, enabled_flag, disabled_flag, *, dest, help=None):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(enabled_flag, dest=dest, action='store_true', help=help)
    group.add_argument(disabled_flag, dest=dest, action='store_false', help=f'Disable {enabled_flag}; the production CLI default is enabled.')
    parser.set_defaults(**{dest: True})

def build_arg_parser() -> argparse.ArgumentParser:
    parser = _NativeArgumentParser(description='3D uniaxial ellipsoid PW85 overlap-resolution default workflow demo.')
    parser.add_argument('--nonperiodic', action='store_true')
    parser.add_argument('--fixed-demo', action='store_true')
    parser.add_argument('--box', nargs=3, type=float, default=(60.0, 60.0, 60.0))
    parser.add_argument('--realizations', type=int, default=1, help='Number of geometries to generate.')
    parser.add_argument('--base-seed', type=int, default=None, help='Base seed for batch mode. Default: use --seed.')
    parser.add_argument('--batch-outdir', type=str, default=None, help='Output directory for batch mode (real_###/ + plots/). Default: outputs/spatial_stats_batch_<timestamp>/')
    parser.add_argument('--batch-export-json-dir', type=str, default=None, help='If set, export final particles JSON for each realization into this directory as real_###.json')
    parser.add_argument('--batch-no-aggregate-plots', action='store_true', help='If set, do not generate aggregated plots in batch mode (still writes per-real stats.json).')
    parser.add_argument('--tau-c', type=float, default=1.5)
    parser.add_argument('--tau-p', type=float, default=1.5)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--grad-tol', type=float, default=1e-10)
    parser.add_argument('--energy-tol', type=float, default=1e-14)
    parser.add_argument('--overlap-tol', type=float, default=1e-08, help='Physical PW acceptance, strict periodic audit, and PW pair-count tolerance.')
    parser.add_argument('--stage0-search-overlap-tol', type=float, default=0.002, help='Inflated PW search stopping tolerance for staged overlap-only stage 0.')
    parser.add_argument('--stage123-search-overlap-tol', type=float, default=0.0001, help='Inflated PW search stopping tolerance for geometry stages 1, 2, and 3.')
    parser.add_argument('--progress-window', type=int, default=100)
    parser.add_argument('--progress-rel-tol', type=float, default=1e-07)
    parser.add_argument('--no-stop-on-stagnation', action='store_true')
    parser.add_argument('--no-wrap', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save-prefix', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--n-particles', type=int, default=55)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--a', type=float, default=2.0)
    parser.add_argument('--b', type=float, default=1.0)
    parser.add_argument('--vf-target', type=float, default=None)
    parser.add_argument('--vf-abs-tol', type=float, default=0.005)
    parser.add_argument('--deq-mu', type=float, default=None)
    parser.add_argument('--deq-sigma', type=float, default=None)
    parser.add_argument('--deq-min', type=float, default=None)
    parser.add_argument('--deq-max', type=float, default=None)
    parser.set_defaults(_aspect_ratio_k_explicit=False)
    parser.add_argument('--aspect-ratio-k', type=float, default=1.5, action=_AspectRatioKAction)
    parser.add_argument('--aspect-ratio-mode', choices=('fixed', 'uniform'), default='fixed', help='Select fixed or uniform aspect-ratio mode.')
    parser.add_argument('--aspect-ratio-min', type=float, default=None, help='Required uniform lower bound, >= 1; forbidden in fixed mode.')
    parser.add_argument('--aspect-ratio-max', type=float, default=None, help='Required uniform upper bound, greater than min; forbidden in fixed mode.')
    parser.add_argument('--size-seed', type=int, default=None)
    parser.add_argument('--size-moment-mc-samples', type=int, default=200000)
    parser.add_argument('--size-max-adjust-iters', type=int, default=20)
    parser.add_argument('--size-gen-seed-retries', type=int, default=50, help='Extra VF-admission seed retries; default 50 allows at most 51 attempts; 0 disables seed retries.')
    parser.add_argument('--heartbeat-steps', type=int, default=0, help='Progress output interval; 0 disables it.')
    parser.add_argument('--collect-metrics', action='store_true')
    parser.add_argument('--print-metrics', action='store_true')
    parser.add_argument('--metrics-json', type=str, default=None)
    parser.add_argument('--metrics-summary-json', type=str, default=None)
    parser.add_argument('--metrics-csv', type=str, default=None)
    _add_default_on_switch(parser, '--line-search', '--no-line-search', dest='line_search')
    parser.add_argument('--ls-alpha-init', type=float, default=1.0)
    parser.add_argument('--ls-rho', type=float, default=0.5)
    parser.add_argument('--ls-c1', type=float, default=0.0001)
    parser.add_argument('--ls-alpha-min', type=float, default=1e-10)
    parser.add_argument('--ls-max-trials', type=int, default=25)
    parser.add_argument('--ls-no-warm-start', action='store_true')
    parser.add_argument('--ls-alpha-growth', type=float, default=1.2)
    parser.add_argument('--ls-alpha-max', type=float, default=1.0)
    parser.add_argument('--ls-no-fallback', action='store_true')
    parser.add_argument('--ls-safe-alpha', type=float, default=0.001)
    parser.add_argument('--ls-orientation-cap', type=float, default=None)
    parser.add_argument('--ls-position-cap', type=float, default=None)
    parser.add_argument('--ls-no-fast-energy-eval', action='store_true')
    parser.add_argument('--ls-no-early-abort', action='store_true')
    parser.add_argument('--pair-method', type=str, default='uniform_grid', choices=('all_pairs', 'uniform_grid'))
    parser.add_argument('--no-bounding-sphere', action='store_true')
    parser.add_argument('--sphere-padding', type=float, default=0.0)
    parser.add_argument('--grid-cell-size', type=float, default=None)
    parser.add_argument('--grid-no-auto-cell-size', action='store_true')
    _add_default_on_switch(parser, '--extent-filter', '--no-extent-filter', dest='extent_filter')
    parser.add_argument('--extent-sep-tol', type=float, default=1e-12)
    parser.add_argument('--export-json', type=str, default=None)
    parser.add_argument('--export-json-indent', type=int, default=2)
    _add_default_on_switch(parser, '--orientation-regularization', '--no-orientation-regularization', dest='orientation_regularization')
    parser.add_argument('--orientation-target', choices=('1d-x', '2d-xz', '3d-isotropic'), default='3d-isotropic', help='Second-order target; optimization uses it only when ORI is enabled.')
    parser.add_argument('--orientation-tol', type=float, default=None, help='Explicit ORI spectral tolerance override; defaults by target: ' + ', '.join((f'{name}={value:g}' for name, value in ORIENTATION_TOLERANCE_DEFAULTS.items())) + '.')
    parser.add_argument('--delta-min', type=float, default=0.0, help='Minimum inter-particle distance constraint via (a,b) inflation: a+=0.5*delta_min, b+=0.5*delta_min during optimization.')
    _add_default_on_switch(parser, '--geometry-distortion', '--no-geometry-distortion', dest='geometry_distortion', help='Enable final face/edge/corner geometry-distortion constraints. By default, geometry is activated progressively after the overlap-only stage passes restored physical PW and required ORI acceptance.')
    parser.add_argument('--geo-immediate', action='store_true', help='Start geometry distortion from step 0. By default, --geometry-distortion uses staged activation: overlap-only first, then weak/intermediate/final geometry stages.')
    parser.add_argument('--edge-target-len', type=float, default=None, help='Abaqus boundary edge target seed spacing used to derive default geometry safety bands. If --geo-d-face/edge/corner are not explicitly given, defaults are: d_face=edge_target_len, d_edge=edge_target_len, d_corner=1.0*edge_target_len.')
    parser.add_argument('--geo-d-face', type=float, default=None, help='Face safety-band thickness d_P. Explicit value overrides --edge-target-len default.')
    parser.add_argument('--geo-d-edge', type=float, default=None, help='Edge safety-band thickness d_L. Explicit value overrides --edge-target-len default.')
    parser.add_argument('--geo-d-corner', type=float, default=None, help='Corner safety-band thickness d_C. Explicit value overrides --edge-target-len default.')
    parser.add_argument('--geo-w-face', type=float, default=2.0, help='Face geometry-distortion penalty weight w_P.')
    parser.add_argument('--geo-w-edge', type=float, default=2.0, help='Edge geometry-distortion penalty weight w_L.')
    parser.add_argument('--geo-w-corner', type=float, default=2.5, help='Corner geometry-distortion penalty weight w_C.')
    parser.add_argument('--geo-energy-tol', type=float, default=1e-10, help='Monitoring tolerance for normalized geometry-distortion energy. Geometry feasibility is judged by max face/edge/corner violations.')
    parser.add_argument('--geo-violation-tol', type=float, default=0.01, help='Default tolerance for max face/edge/corner geometry violation. Used when per-kind tolerances are not explicitly given.')
    parser.add_argument('--geo-face-violation-tol', type=float, default=None, help='Tolerance for max face violation max(d_face-|s_face|)_+. Default: --geo-violation-tol.')
    parser.add_argument('--geo-edge-violation-tol', type=float, default=None, help='Tolerance for max edge violation max(d_edge-|s_edge|)_+. Default: --geo-violation-tol.')
    parser.add_argument('--geo-corner-violation-tol', type=float, default=None, help='Tolerance for max corner violation max(d_corner-|s_corner|)_+. Default: --geo-violation-tol.')
    parser.add_argument('--geo-active-violation-abs-tol', type=float, default=1e-12, help='Strict audit/active-record absolute tolerance for geometry violation.')
    parser.add_argument('--geo-active-violation-rel-tol', type=float, default=1e-10, help='Strict audit/active-record relative tolerance for geometry violation.')
    parser.add_argument('--geo-no-candidate-filter', action='store_true', help='Disable conservative face/edge/corner geometry candidate filtering for debug/full-eval comparisons.')
    _add_default_on_switch(parser, '--geo-candidate-cache', '--geo-no-candidate-cache', dest='geo_candidate_cache', help='Use a conservative near-feature geometry candidate cache during line-search fast energy evaluation.')
    parser.add_argument('--geo-audit', action='store_true', help='Print initial/final geometry-distortion audit when geometry distortion is enabled.')
    parser.add_argument('--origin-shift', action='store_true', help='Enable periodic origin shift postprocess to mitigate boundary thin-slice artifacts (periodic only).')
    parser.add_argument('--origin-shift-epsilon-b', type=float, default=None, help='Boundary danger-band thickness epsilon_b. If not provided, auto-estimate by beta*median(b).')
    parser.add_argument('--origin-shift-beta', type=float, default=0.1, help='Auto-estimation factor for epsilon_b = beta*median(b). Default 0.10.')
    parser.add_argument('--origin-shift-M', type=int, default=128, help='Number of coarse samples per axis for origin shift search. Default 64.')
    parser.add_argument('--origin-shift-no-refine', action='store_true', help='Disable local refinement in origin shift search (not recommended).')
    parser.add_argument('--origin-shift-refine-M', type=int, default=32, help='Number of refinement samples per axis near the coarse best. Default 32.')
    parser.add_argument('--spatial-stats', action='store_true')
    parser.add_argument('--spatial-stats-outdir', type=str, default=None)
    parser.add_argument('--spatial-stats-no-plots', action='store_true')
    parser.add_argument('--stats-r-max-factor', type=float, default=0.5)
    parser.add_argument('--stats-n-r-bins', type=int, default=120)
    parser.add_argument('--stats-n-k-points', type=int, default=60)
    parser.add_argument('--stats-orientation-tol', type=float, default=0.03)
    parser.add_argument('--version', action='version', version=f'{ADAPTATION_REVISION_ID}; parent={PARENT_BASELINE_ID}; defaults={DEFAULT_PROFILE_ID}')
    parser.add_argument('--baseline-identity', action='store_true', help='Verify and print source identity.')
    parser.add_argument('--show-effective-config', action='store_true', help='Print the native CLI/stage configurations without generating or optimizing particles; enabled GEO requires its explicit safety-band scale.')
    return parser

def _size_dist_requested(args: argparse.Namespace) -> bool:
    if args.vf_target is not None:
        return True
    if args.deq_mu is not None or args.deq_sigma is not None or args.deq_min is not None or (args.deq_max is not None):
        return True
    return False

def _require_all_size_dist_args(args: argparse.Namespace) -> None:
    missing = []
    if args.vf_target is None:
        missing.append('--vf-target')
    if args.deq_mu is None:
        missing.append('--deq-mu')
    if args.deq_sigma is None:
        missing.append('--deq-sigma')
    if args.deq_min is None:
        missing.append('--deq-min')
    if args.deq_max is None:
        missing.append('--deq-max')
    if missing:
        raise ValueError('Size-distribution mode requested but missing args: ' + ', '.join(missing))

def _resolve_geometry_distance(name: str, explicit_value: float | None, edge_target_len: float | None, factor: float) -> float:
    if explicit_value is not None:
        value = float(explicit_value)
        source = f'explicit --geo-d-{name}'
    else:
        if edge_target_len is None:
            raise ValueError(f'--geometry-distortion requires either --geo-d-{name} or --edge-target-len to derive d_{name}.')
        value = float(factor) * float(edge_target_len)
        source = f'{factor:g} * --edge-target-len'
    if value <= 0.0:
        raise ValueError(f'Geometry distance d_{name} must be > 0, got {value} from {source}.')
    return value

def _build_geometry_distortion_config(args: argparse.Namespace) -> GeometryDistortionConfig3D:
    edge_target_len = None if args.edge_target_len is None else float(args.edge_target_len)
    if edge_target_len is not None and edge_target_len <= 0.0:
        raise ValueError(f'--edge-target-len must be > 0 when provided, got {edge_target_len}.')
    d_face = _resolve_geometry_distance('face', args.geo_d_face, edge_target_len, GEO_EDGE_TARGET_FACE_FACTOR)
    d_edge = _resolve_geometry_distance('edge', args.geo_d_edge, edge_target_len, GEO_EDGE_TARGET_EDGE_FACTOR)
    d_corner = _resolve_geometry_distance('corner', args.geo_d_corner, edge_target_len, GEO_EDGE_TARGET_CORNER_FACTOR)
    return GeometryDistortionConfig3D(enabled=True, include_face=True, include_edge=True, include_corner=True, d_face=float(d_face), d_edge=float(d_edge), d_corner=float(d_corner), w_face=float(args.geo_w_face), w_edge=float(args.geo_w_edge), w_corner=float(args.geo_w_corner), lambda_geo=1.0, normalize_by='num_particles', use_physical_axes=True, active_violation_abs_tol=float(args.geo_active_violation_abs_tol), active_violation_rel_tol=float(args.geo_active_violation_rel_tol), edge_tangent_filter=True, candidate_filter=not bool(args.geo_no_candidate_filter), candidate_margin_abs=0.0, candidate_margin_rel=1.0)

def _print_geometry_config_summary(config: DescentConfig) -> None:
    if not bool(config.include_geometry_distortion):
        return
    geo = config.geometry_config
    print('Geometry distortion: enabled')
    print(f'  safety bands: d_face={geo.d_face:.12e}, d_edge={geo.d_edge:.12e}, d_corner={geo.d_corner:.12e}')
    print(f'  weights: w_face={geo.w_face:.6g}, w_edge={geo.w_edge:.6g}, w_corner={geo.w_corner:.6g}')
    print(f'  geometry feasibility tolerances: face<={config.geometry_face_violation_tol:.3e}, edge<={config.geometry_edge_violation_tol:.3e}, corner<={config.geometry_corner_violation_tol:.3e}')
    print(f'  geometry energy monitor tolerance: energy<={config.geometry_energy_tol:.3e}')
    print(f"  geometry candidate filter: {bool(getattr(geo, 'candidate_filter', False))}")
    print(f'  line-search geometry candidate cache: {bool(config.use_geometry_candidate_cache)}')

def _print_geometry_audit(label: str, particles: list[EllipsoidParticle3D], box: np.ndarray | None, config: DescentConfig) -> dict | None:
    if not bool(config.include_geometry_distortion):
        return None
    if box is None:
        print(f'[{label} geometry audit skipped] geometry distortion is periodic-box only.')
        return None
    audit = audit_geometry_distortion_3d(particles, box, config.geometry_config, feasible_violation_tol=float(config.geometry_violation_tol), feasible_face_violation_tol=float(config.geometry_face_violation_tol), feasible_edge_violation_tol=float(config.geometry_edge_violation_tol), feasible_corner_violation_tol=float(config.geometry_corner_violation_tol))
    print(f'\n=== {label} Geometry Distortion Audit ===')
    keys = ('geometry_energy', 'face_energy', 'edge_energy', 'corner_energy', 'max_face_violation', 'max_edge_violation', 'max_corner_violation', 'max_geometry_violation', 'worst_violation', 'min_abs_signed_clearance', 'num_face_forbidden_raw', 'num_edge_forbidden_raw', 'num_corner_forbidden_raw', 'feasible_violation_tol', 'feasible_face_violation_tol', 'feasible_edge_violation_tol', 'feasible_corner_violation_tol', 'num_face_forbidden_feasible', 'num_edge_forbidden_feasible', 'num_corner_forbidden_feasible')
    for key in keys:
        if key in audit:
            print(f'{key}: {audit[key]}')
    if audit.get('worst_feature_name') is not None:
        print(f"worst feature: particle={audit.get('worst_particle_index')}, kind={audit.get('worst_feature_kind')}, name={audit.get('worst_feature_name')}")
    return audit

def _clone_particles(particles: list[EllipsoidParticle3D]) -> list[EllipsoidParticle3D]:
    return [p.clone() for p in particles]

def _scale_geometry_config(geo: GeometryDistortionConfig3D, *, d_scale: float, w_scale: float) -> GeometryDistortionConfig3D:
    d_scale = float(d_scale)
    w_scale = float(w_scale)
    if d_scale <= 0.0:
        raise ValueError(f'Expected d_scale > 0 for an active geometry stage, got {d_scale}.')
    if w_scale <= 0.0:
        raise ValueError(f'Expected w_scale > 0 for an active geometry stage, got {w_scale}.')
    return replace(geo, enabled=True, d_face=float(geo.d_face) * d_scale, d_edge=float(geo.d_edge) * d_scale, d_corner=float(geo.d_corner) * d_scale, w_face=float(geo.w_face) * w_scale, w_edge=float(geo.w_edge) * w_scale, w_corner=float(geo.w_corner) * w_scale)

def _make_stage_config(base_config: DescentConfig, *, include_geometry: bool, geometry_config: GeometryDistortionConfig3D | None, preliminary_geometry_stop: bool=False, preliminary_violation_tol_scale: float | None=None) -> DescentConfig:
    cfg = copy.deepcopy(base_config)
    cfg.include_geometry_distortion = bool(include_geometry)
    cfg.geometry_stop = bool(include_geometry)
    if include_geometry:
        if geometry_config is None:
            raise ValueError('geometry_config must be provided when include_geometry=True.')
        cfg.geometry_config = copy.deepcopy(geometry_config)
        if preliminary_geometry_stop:
            cfg.geometry_energy_tol = max(float(base_config.geometry_energy_tol), 1e-08)
            scale = 0.001 if preliminary_violation_tol_scale is None else float(preliminary_violation_tol_scale)
            if scale < 0.0:
                raise ValueError(f'preliminary_violation_tol_scale must be >= 0, got {scale}.')
            target_geo = base_config.geometry_config
            cfg.geometry_face_violation_tol = max(float(base_config.geometry_face_violation_tol), scale * float(target_geo.d_face))
            cfg.geometry_edge_violation_tol = max(float(base_config.geometry_edge_violation_tol), scale * float(target_geo.d_edge))
            cfg.geometry_corner_violation_tol = max(float(base_config.geometry_corner_violation_tol), scale * float(target_geo.d_corner))
            cfg.geometry_violation_tol = max(float(base_config.geometry_violation_tol), cfg.geometry_face_violation_tol, cfg.geometry_edge_violation_tol, cfg.geometry_corner_violation_tol)
    else:
        cfg.geometry_stop = False
        cfg.geometry_config = GeometryDistortionConfig3D(enabled=False)
    return cfg

def _apply_stage_search_overlap_tol(stage_config: DescentConfig, base_config: DescentConfig, stage_index: int) -> DescentConfig:
    name = 'stage0_inflated_search_overlap_tol' if stage_index == 0 else 'stage123_inflated_search_overlap_tol'
    value = getattr(base_config, name, None)
    if value is not None:
        stage_config.inflated_search_overlap_tol = float(value)
    return stage_config

def _bind_ori_gcg_stage_policy_factory(original_make):
    policy = {'stages': ({'configured_orientation_mode': 'baseline', 'orientation_geometry_guard_enabled': False, 'c01_ls_rescue_enabled': True, 'geometry_mode': 'baseline'}, {'configured_orientation_mode': 'acceptance_band', 'orientation_geometry_guard_enabled': True, 'c01_ls_rescue_enabled': False, 'geometry_mode': 'baseline'}, {'configured_orientation_mode': 'acceptance_band', 'orientation_geometry_guard_enabled': True, 'c01_ls_rescue_enabled': False, 'geometry_mode': 'baseline'}, {'configured_orientation_mode': 'acceptance_band', 'orientation_geometry_guard_enabled': False, 'c01_ls_rescue_enabled': True, 'geometry_mode': 'acceptance_band'})}
    stage_count = 0

    def configure_stage(*args, **kwargs):
        nonlocal stage_count
        cfg = original_make(*args, **kwargs)
        idx = stage_count
        if idx >= 4:
            raise RuntimeError('additional stage/realization exceeds fixed scope')
        stage = policy['stages'][idx]
        if bool(kwargs.get('include_geometry')) != (idx != 0) or bool(kwargs.get('preliminary_geometry_stop')) != (idx in (1, 2)):
            raise RuntimeError('native stage structure differs from approved policy')
        fields = {'orientation_regularization_mode': stage['configured_orientation_mode'], 'orientation_geometry_guard_enabled': stage['orientation_geometry_guard_enabled'], 'c01_ls_rescue_enabled': stage['c01_ls_rescue_enabled'], 'geometry_regularization_mode': stage['geometry_mode']}
        for name, value in fields.items():
            if not hasattr(cfg, name):
                raise RuntimeError(f'candidate configuration missing {name}')
            setattr(cfg, name, value)
        stage_count += 1
        return cfg
    return configure_stage
_make_stage_config_parent = _make_stage_config
_make_stage_config = _bind_ori_gcg_stage_policy_factory(_make_stage_config_parent)

def _result_max_overlap(result: Any) -> float:
    status = getattr(result, 'final_status', None)
    if isinstance(status, dict):
        return float(status.get('max_overlap', float('inf')))
    if getattr(result, 'history_max_overlap', None):
        return float(result.history_max_overlap[-1])
    return float('inf')

def _geometry_summary_from_status(status: Any) -> dict[str, float]:
    if not isinstance(status, dict):
        return {}
    keys = ('geometry_energy', 'worst_geometry_violation', 'max_geometry_violation', 'max_face_violation', 'max_edge_violation', 'max_corner_violation')
    out: dict[str, float] = {}
    for key in keys:
        if key in status and status[key] is not None:
            out[key] = float(status[key])
    return out

def _compact_geometry_audit(audit: Any) -> dict[str, Any] | None:
    if not isinstance(audit, dict):
        return None
    return {key: value for key, value in audit.items() if key not in ('active_records', 'active_records_feasible')}

def _run_descent_with_optional_staged_geometry(*, particles_physical: list[EllipsoidParticle3D], config: DescentConfig, box: np.ndarray | None, collect_metrics: bool, delta_min: float, origin_shift_config: OriginShiftConfig3D | None, use_staged_geometry: bool, realization_budget: RealizationIterationBudget | None=None) -> tuple[Any, dict[str, Any]]:
    if realization_budget is None:
        realization_budget = RealizationIterationBudget()
    if not bool(config.include_geometry_distortion) or not bool(use_staged_geometry):
        return run_descent_with_constraints_3d(particles_physical=particles_physical, config=config, box=box, collect_metrics=collect_metrics, delta_min=delta_min, origin_shift_config=origin_shift_config, realization_budget=realization_budget)
    if box is None:
        raise ValueError('Staged geometry activation is periodic-box only.')
    target_geo = copy.deepcopy(config.geometry_config)
    current_particles = _clone_particles(particles_physical)
    stage_reports: list[dict[str, Any]] = []
    final_result = None
    final_constraints_report: dict[str, Any] = {}
    final_stage_index = 3
    final_stage_completed = False
    staged_failure_reason: str | None = None
    stages = [{'name': 'stage0_overlap_only', 'include_geometry': False, 'd_scale': 0.0, 'w_scale': 0.0, 'violation_tol_scale': None, 'preliminary': False, 'description': 'overlap-only search with stage0 inflated PW tolerance and required ORI; restored physical acceptance stays separate'}, {'name': 'stage1_geometry_warmup', 'include_geometry': True, 'd_scale': 0.5, 'w_scale': 0.25, 'violation_tol_scale': 0.001, 'preliminary': True, 'description': 'weak geometry: half safety bands, quarter penalty weights'}, {'name': 'stage2_geometry_intermediate', 'include_geometry': True, 'd_scale': 0.75, 'w_scale': 0.5, 'violation_tol_scale': 0.0005, 'preliminary': True, 'description': 'intermediate geometry: 0.75 safety bands, half penalty weights'}, {'name': 'stage3_geometry_final', 'include_geometry': True, 'd_scale': 1.0, 'w_scale': 1.0, 'violation_tol_scale': None, 'preliminary': False, 'description': 'target geometry: full safety bands and penalty weights'}]
    print('\n=== Geometry Distortion Staged Activation ===')
    print(f'Physical activation gate: restored PW <= {config.overlap_tol:.3e} and required ORI acceptance')
    make_stage_config = _bind_ori_gcg_stage_policy_factory(_make_stage_config_parent)
    for stage_index, stage in enumerate(stages):
        include_geo = bool(stage['include_geometry'])
        if include_geo:
            stage_geo = _scale_geometry_config(target_geo, d_scale=float(stage['d_scale']), w_scale=float(stage['w_scale']))
        else:
            stage_geo = None
        stage_config = make_stage_config(config, include_geometry=include_geo, geometry_config=stage_geo, preliminary_geometry_stop=bool(stage['preliminary']), preliminary_violation_tol_scale=stage['violation_tol_scale'])
        stage_config = _apply_stage_search_overlap_tol(stage_config, config, stage_index)
        print(f"\n--- {stage['name']} ---")
        print(str(stage['description']))
        print(f'inflated PW search tolerance={effective_inflated_search_overlap_tol(stage_config):.3e}; physical PW acceptance tolerance={stage_config.overlap_tol:.3e}')
        if include_geo:
            print(f'geometry stage values: d_face={stage_config.geometry_config.d_face:.12e}, d_edge={stage_config.geometry_config.d_edge:.12e}, d_corner={stage_config.geometry_config.d_corner:.12e}, w_face={stage_config.geometry_config.w_face:.6g}, w_edge={stage_config.geometry_config.w_edge:.6g}, w_corner={stage_config.geometry_config.w_corner:.6g}')
            print(f'geometry stage feasibility tolerances: face<={stage_config.geometry_face_violation_tol:.3e}, edge<={stage_config.geometry_edge_violation_tol:.3e}, corner<={stage_config.geometry_corner_violation_tol:.3e}')
        result, constraints_report = run_descent_with_constraints_3d(particles_physical=current_particles, config=stage_config, box=box, collect_metrics=collect_metrics, delta_min=delta_min, origin_shift_config=origin_shift_config if stage_index == len(stages) - 1 else None, realization_budget=realization_budget)
        final_result = result
        final_constraints_report = constraints_report
        max_ov = _result_max_overlap(result)
        is_final_stage = bool(stage_index == final_stage_index)
        report = {'stage_index': int(stage_index), 'name': str(stage['name']), 'stage_name': str(stage['name']), 'include_geometry': bool(include_geo), 'd_scale': float(stage['d_scale']), 'w_scale': float(stage['w_scale']), 'violation_tol_scale': None if stage['violation_tol_scale'] is None else float(stage['violation_tol_scale']), 'is_final_stage': bool(is_final_stage), 'final_stage': bool(is_final_stage), 'global_success_allowed': bool(is_final_stage), 'face_violation_tol_used': float(stage_config.geometry_face_violation_tol), 'edge_violation_tol_used': float(stage_config.geometry_edge_violation_tol), 'corner_violation_tol_used': float(stage_config.geometry_corner_violation_tol), 'steps_taken': int(result.steps_taken), 'optimization_elapsed_seconds': float(result.optimization_elapsed_seconds), 'independent_audit_elapsed_seconds': float(result.independent_audit_elapsed_seconds), 'native_final_audit_elapsed_seconds': float(result.native_final_audit_elapsed_seconds), 'physical_restore_audit_elapsed_seconds': float(result.physical_restore_audit_elapsed_seconds), 'orientation_target': stage_config.orientation_target, 'cumulative_completed_iterations': int(realization_budget.completed_iterations), 'resource_status': str(realization_budget.resource_status), 'stage_success': bool(result.success), 'stage_converged': bool(result.converged), 'stage_stop_reason': str(result.stop_reason), 'success': bool(result.success), 'converged': bool(result.converged), 'stop_reason': str(result.stop_reason), 'final_max_overlap': float(max_ov), 'inflated_search_overlap_tol': effective_inflated_search_overlap_tol(stage_config), 'physical_acceptance_overlap_tol': float(stage_config.overlap_tol), 'overlap_count_tol': float(stage_config.overlap_tol), 'physical_final_max_overlap': float(max_ov)}
        report.update(constraints_report.get('pw_search_acceptance', {}))
        report.update(_geometry_summary_from_status(getattr(result, 'final_status', None)))
        if getattr(result, 'history_energy', None):
            report['final_history_energy'] = float(result.history_energy[-1])
        stage_reports.append(report)
        print(f"[{stage['name']}] done: steps={result.steps_taken}, success={result.success}, stop_reason={result.stop_reason}, final_max_overlap={max_ov:.12e}")
        current_particles = _clone_particles(result.particles)
        if stage_index == 0 and max_ov > float(config.overlap_tol):
            staged_failure_reason = 'geometry_activation_skipped_overlap_target_not_reached'
            print('[staged-geometry] overlap-only stage did not reach activation target; geometry stages are skipped.')
            break
        if not bool(result.success):
            staged_failure_reason = f"staged_geometry_stage_failed:{stage['name']}"
            print('[staged-geometry] current stage did not succeed; remaining stages are skipped.')
            break
        if is_final_stage:
            final_stage_completed = True
        elif realization_budget.at_limit:
            realization_budget.mark_exhausted(f"next_stage_required_after:{stage['name']};native_stop={result.stop_reason}")
            staged_failure_reason = 'RESOURCE_BUDGET_EXHAUSTED'
            break
    if final_result is None:
        raise RuntimeError('No descent stage was executed.')
    if not final_stage_completed:
        if staged_failure_reason is None:
            staged_failure_reason = 'staged_geometry_final_stage_not_reached'
        final_result.success = False
        final_result.converged = False
        final_result.stop_reason = staged_failure_reason
    final_result.optimization_elapsed_seconds = sum((r['optimization_elapsed_seconds'] for r in stage_reports))
    final_result.independent_audit_elapsed_seconds = sum((r['independent_audit_elapsed_seconds'] for r in stage_reports))
    final_result.native_final_audit_elapsed_seconds = sum((r['native_final_audit_elapsed_seconds'] for r in stage_reports))
    final_result.physical_restore_audit_elapsed_seconds = sum((r['physical_restore_audit_elapsed_seconds'] for r in stage_reports))
    final_result = apply_resource_result(final_result, realization_budget)
    final_constraints_report = dict(final_constraints_report)
    final_constraints_report['independent_audit_timing'] = {'elapsed_seconds': float(final_result.independent_audit_elapsed_seconds), 'native_final_elapsed_seconds': float(final_result.native_final_audit_elapsed_seconds), 'physical_restore_elapsed_seconds': float(final_result.physical_restore_audit_elapsed_seconds), 'scope': final_result.independent_audit_timing_scope, 'aggregation': 'sum over all executed native stages'}
    final_constraints_report['resource_budget'] = {**final_constraints_report.get('resource_budget', {}), **realization_budget.snapshot(), 'generation_success': bool(final_result.generation_success), 'terminal_state_sha256': final_result.terminal_state_sha256}
    final_constraints_report['geometry_distortion_staged_activation'] = {'enabled': True, 'activation_overlap_tol': float(config.overlap_tol), 'activation_overlap_tol_role': 'restored_physical_pw_acceptance', 'stage0_inflated_search_overlap_tol': getattr(config, 'stage0_inflated_search_overlap_tol', None), 'stage123_inflated_search_overlap_tol': getattr(config, 'stage123_inflated_search_overlap_tol', None), 'final_stage_index': int(final_stage_index), 'final_stage_completed': bool(final_stage_completed), 'global_success_requires_final_stage': True, 'staged_failure_reason': staged_failure_reason, 'stages': stage_reports}
    setattr(final_result, 'stage_reports', stage_reports)
    return (final_result, final_constraints_report)

def _build_descent_config(args: argparse.Namespace, use_periodic: bool, wrap_positions: bool, pair_config: CandidatePairConfig3D, line_search_config: LineSearchConfig) -> DescentConfig:
    overlap_tol = float(args.overlap_tol)
    if overlap_tol < 0.0:
        raise ValueError(f'Expected overlap_tol >= 0, got {overlap_tol}.')
    extent_sep_tol = float(args.extent_sep_tol)
    if extent_sep_tol < 0.0:
        raise ValueError(f'Expected extent_sep_tol >= 0, got {extent_sep_tol}.')
    return DescentConfig(tau_c=float(args.tau_c), tau_p=float(args.tau_p), max_steps=int(args.steps), grad_tol=float(args.grad_tol), energy_tol=float(args.energy_tol), physical_stop=True, overlap_tol=overlap_tol, stage0_inflated_search_overlap_tol=getattr(args, 'stage0_search_overlap_tol', None), stage123_inflated_search_overlap_tol=getattr(args, 'stage123_search_overlap_tol', None), progress_window=int(args.progress_window), progress_rel_tol=float(args.progress_rel_tol), stop_on_stagnation=not bool(args.no_stop_on_stagnation), use_mic=use_periodic, attach_geometry=True, verbose=bool(args.verbose), wrap_positions=wrap_positions, include_orientation_regularization=bool(args.orientation_regularization), orientation_target=str(args.orientation_target), heartbeat_interval=int(args.heartbeat_steps), line_search=line_search_config, pair_config=pair_config, orientation_stop=True, orientation_tol=getattr(args, 'orientation_tol', None), use_extent_filter=bool(args.extent_filter), extent_sep_tol=extent_sep_tol, include_geometry_distortion=bool(args.geometry_distortion), geometry_stop=bool(args.geometry_distortion), geometry_energy_tol=float(args.geo_energy_tol), geometry_violation_tol=float(args.geo_violation_tol), geometry_face_violation_tol=float(args.geo_face_violation_tol) if args.geo_face_violation_tol is not None else float(args.geo_violation_tol), geometry_edge_violation_tol=float(args.geo_edge_violation_tol) if args.geo_edge_violation_tol is not None else float(args.geo_violation_tol), geometry_corner_violation_tol=float(args.geo_corner_violation_tol) if args.geo_corner_violation_tol is not None else float(args.geo_violation_tol), use_geometry_candidate_cache=bool(args.geo_candidate_cache), geometry_config=_build_geometry_distortion_config(args) if bool(args.geometry_distortion) else GeometryDistortionConfig3D(enabled=False))

def _build_pair_config(args: argparse.Namespace) -> CandidatePairConfig3D:
    return CandidatePairConfig3D(method=str(args.pair_method), use_bounding_sphere=not bool(args.no_bounding_sphere), sphere_padding=float(args.sphere_padding), cell_size=None if args.grid_cell_size is None else float(args.grid_cell_size), auto_cell_size=not bool(args.grid_no_auto_cell_size))

def _build_line_search_config(args: argparse.Namespace) -> LineSearchConfig:
    return LineSearchConfig(enabled=bool(args.line_search), alpha_init=float(args.ls_alpha_init), rho=float(args.ls_rho), c1=float(args.ls_c1), alpha_min=float(args.ls_alpha_min), max_trials=int(args.ls_max_trials), use_alpha_warm_start=not bool(args.ls_no_warm_start), alpha_growth=float(args.ls_alpha_growth), alpha_max=float(args.ls_alpha_max), fallback_to_safe_step=not bool(args.ls_no_fallback), safe_alpha=float(args.ls_safe_alpha), orientation_step_cap=None if args.ls_orientation_cap is None else float(args.ls_orientation_cap), position_step_cap=None if args.ls_position_cap is None else float(args.ls_position_cap), fast_energy_eval=not bool(args.ls_no_fast_energy_eval), early_abort=not bool(args.ls_no_early_abort))

def _build_origin_shift_config(args: argparse.Namespace, use_periodic: bool) -> OriginShiftConfig3D | None:
    if not bool(args.origin_shift):
        return None
    if not use_periodic:
        raise ValueError('--origin-shift is periodic-only. Remove --nonperiodic or disable origin shift.')
    eps = args.origin_shift_epsilon_b
    if eps is not None and float(eps) <= 0.0:
        raise ValueError('--origin-shift-epsilon-b must be > 0')
    beta = float(args.origin_shift_beta)
    if beta <= 0.0:
        raise ValueError('--origin-shift-beta must be > 0')
    M = int(args.origin_shift_M)
    if M <= 1:
        raise ValueError('--origin-shift-M must be > 1')
    refine = not bool(args.origin_shift_no_refine)
    refine_M = int(args.origin_shift_refine_M)
    if refine and refine_M <= 1:
        raise ValueError('--origin-shift-refine-M must be > 1 when refine is enabled')
    return OriginShiftConfig3D(enabled=True, epsilon_b=None if eps is None else float(eps), beta=float(beta), M=int(M), refine=bool(refine), refine_M=int(refine_M), objective='lexicographic_max_then_sum')

def _input_state_metadata(args, particles, size_report, *, seed_init, seed_size):
    rows = [{'id': p.id, 'a': float(p.a), 'b': float(p.b), 'c': np.asarray(p.c).tolist(), 'p': np.asarray(p.p).tolist()} for p in particles]
    raw = json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    seed_effective = size_report['input']['seed'] if size_report is not None else seed_size
    return {'seed_init': int(seed_init), 'seed_size': seed_effective, 'seed_size_requested': seed_size, 'seed_size_effective': seed_effective, 'size_gen_seed_retries': int(args.size_gen_seed_retries), 'size_distribution_mode': bool(size_report is not None), 'aspect_ratio_mode': args.aspect_ratio_mode if size_report is not None else 'native_demo_axes', 'aspect_ratio_min': args.aspect_ratio_min, 'aspect_ratio_max': args.aspect_ratio_max, 'size_report': size_report, 'physical_particle_count': len(particles), 'orientation_initialization': _initial_orientation_metadata(args, particles), 'initial_physical_state_sha256': hashlib.sha256(raw).hexdigest(), 'initial_state_hash_encoding': 'JSON sorted keys compact UTF-8; ordered id,a,b,c,p; no extra RNG'}

def _generate_sizes_for_cli(args, size_config, *, realization_index=None):
    try:
        return generate_sizes_with_seed_retries(size_config, max_seed_retries=int(args.size_gen_seed_retries))
    except RuntimeError as exc:
        report = getattr(exc, 'size_generation_retry_report', None)
        if report is not None:
            if realization_index is not None and args.batch_export_json_dir is not None:
                stem = Path(args.batch_export_json_dir) / f'real_{realization_index:03d}'
            elif args.export_json is not None:
                stem = Path(args.export_json)
            else:
                stem = Path('outputs/size_generation_failures') / f'seed_{size_config.seed}'
            path = stem.with_name(stem.name + f'.size_generation_failure.{uuid.uuid4().hex}.json')
            receipt = {'schema': 'pw-rve-size-generation-failure-receipt-v1', 'result_type': 'ADMISSION_FAILURE_RECEIPT_NOT_GEOMETRY_JSON', 'command': ' '.join(sys.argv), 'created_utc': datetime.utcnow().isoformat() + 'Z', 'realization_index': realization_index, 'status': 'SIZE_GENERATION_FAILED', 'optimization_started': False, 'optimization_elapsed_seconds': None, 'geometry_output_written_by_this_attempt': False, 'size_generation_seed_retries': report}
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('x', encoding='utf-8') as stream:
                    json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                print(f'[size generation failed] receipt: {path.resolve()}', file=sys.stderr)
            except (OSError, ValueError) as write_error:
                print(f'[size failure receipt could not be saved] {write_error}', file=sys.stderr)
        raise

def _timed_audit_call(timings, name, function, *args, **kwargs):
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timings[name] = timings.get(name, 0.0) + float(time.perf_counter() - started)

def _review_timing_json(result, main_audit_timings):
    return {'independent_review_elapsed_seconds': float(result.independent_audit_elapsed_seconds + sum(main_audit_timings.values())), 'independent_review_timing_scope': 'sum of native final audits, restored-physical acceptance checks and main independent audits; excludes generation, initialization, iteration loops, restoration/origin shift and JSON saving; main audit calls include their native report printing', 'independent_review_breakdown_seconds': {'native_final_audits': float(result.native_final_audit_elapsed_seconds), 'physical_restore_acceptance_checks': float(result.physical_restore_audit_elapsed_seconds), 'main_independent_audits': dict(main_audit_timings)}}

def _orientation_audit_json(particles, config, *, audit_result=None):
    result = audit_result if audit_result is not None else check_orientation_isotropy(particles, tolerance=float(config.orientation_tol), orientation_target=config.orientation_target)
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in {**result, 'required_for_success': bool(config.include_orientation_regularization and config.orientation_stop), 'optimization_enabled': bool(config.include_orientation_regularization)}.items()}

def _final_audit_verdicts(config, status, orientation, strict_audit, geometry_audit):
    pw_ok = bool(float(status['max_overlap']) <= float(config.overlap_tol))
    strict_ok = pw_ok if strict_audit is None else not bool(strict_audit['has_overlap'])
    geo_required = bool(config.include_geometry_distortion and config.geometry_stop)
    if geometry_audit is None:
        geo_ok = None
    else:
        geo_ok = bool(all((float(geometry_audit[f'max_{kind}_violation']) <= float(getattr(config, f'geometry_{kind}_violation_tol')) for kind in ('face', 'edge', 'corner'))))
    ori_required = bool(config.include_orientation_regularization and config.orientation_stop)
    return {'timed': False, 'physical_pw_passed': pw_ok, 'physical_acceptance_overlap_tol': float(config.overlap_tol), 'orientation_tol': float(config.orientation_tol), 'geometry_violation_tolerances': {kind: float(getattr(config, f'geometry_{kind}_violation_tol')) for kind in ('face', 'edge', 'corner')}, 'strict_periodic_audit_performed': strict_audit is not None, 'strict_periodic_passed': strict_ok, 'geometry_required': geo_required, 'geometry_audit_performed': geometry_audit is not None, 'geometry_passed': geo_ok, 'orientation_required': ori_required, 'orientation_target_passed': bool(orientation['is_target_aligned']), 'all_required_audits_passed': bool(pw_ok and strict_ok and (not geo_required or geo_ok) and (not ori_required or orientation['is_target_aligned'])), 'native_generation_success_is_separate': True}

def _run_one_realization(args: argparse.Namespace, box: np.ndarray, use_periodic: bool, size_dist_mode: bool, seed_init: int, seed_size: int, config: DescentConfig, origin_shift_config: OriginShiftConfig3D | None, realization_index: int | None=None) -> tuple[list[EllipsoidParticle3D], dict, dict | None, Any, dict[str, Any]]:
    if use_periodic is False:
        raise ValueError('This realization runner assumes periodic mode for size distribution and spatial stats.')
    size_report = None
    if size_dist_mode:
        _require_all_size_dist_args(args)
        size_cfg = SizeGenerationConfig3D(box=np.asarray(box, dtype=float), vf_target=float(args.vf_target), vf_abs_tol=float(args.vf_abs_tol), deq_dist=TruncNormalSpec(mu=float(args.deq_mu), sigma=float(args.deq_sigma), d_min=float(args.deq_min), d_max=float(args.deq_max)), aspect_ratio_k=float(args.aspect_ratio_k), aspect_ratio_mode=str(args.aspect_ratio_mode), aspect_ratio_min=args.aspect_ratio_min, aspect_ratio_max=args.aspect_ratio_max, moment_mc_samples=int(args.size_moment_mc_samples), max_adjust_iters=int(args.size_max_adjust_iters), seed=int(seed_size), scipy_truncnorm=True)
        size_res = _generate_sizes_for_cli(args, size_cfg, realization_index=realization_index)
        size_report = size_res.report
        particles0 = build_random_particles_from_size_arrays(a_arr=size_res.a, b_arr=size_res.b, box=box, seed=int(seed_init), orientation_target=str(args.orientation_target), orientation_regularization=bool(args.orientation_regularization))
    else:
        particles0 = build_random_demo_particles(n_particles=int(args.n_particles), a=float(args.a), b=float(args.b), box=box, seed=int(seed_init), periodic=True, orientation_target=str(args.orientation_target), orientation_regularization=bool(args.orientation_regularization))
    input_state = _input_state_metadata(args, particles0, size_report, seed_init=seed_init, seed_size=seed_size if size_dist_mode else None)
    descent_res, constraints_report = _run_descent_with_optional_staged_geometry(particles_physical=particles0, config=config, box=box, collect_metrics=bool(args.collect_metrics), delta_min=float(args.delta_min), origin_shift_config=origin_shift_config, use_staged_geometry=bool(args.geometry_distortion and (not args.geo_immediate)))
    constraints_report['eo_input_state'] = input_state
    final_particles = [p.clone() for p in descent_res.particles]
    if config.wrap_positions:
        final_particles = [EllipsoidParticle3D(a=p.a, b=p.b, c=wrap_position(p.c, box), p=p.p.copy(), id=p.id) for p in final_particles]
    main_audit_timings = {}
    status = _timed_audit_call(main_audit_timings, 'physical_pw_status', system_overlap_status, particles=final_particles, box=box, use_mic=True, overlap_tol=float(args.overlap_tol))
    constraints_report['eo_main_independent_audit_timings_seconds'] = main_audit_timings
    return (final_particles, status, size_report, descent_res, constraints_report)

def _overlap_tolerance_policy(args) -> dict[str, Any]:
    return {'physical_acceptance_overlap_tol': float(args.overlap_tol), 'overlap_count_tol': float(args.overlap_tol), 'stage0_inflated_search_overlap_tol': float(args.stage0_search_overlap_tol), 'stage123_inflated_search_overlap_tol': float(args.stage123_search_overlap_tol), 'stage_search_scope': 'native_inflated_particles_only', 'physical_audit_scope': 'restored_physical_particles_and_strict_periodic_audit', 'min_distance_acceptance_performed': False}

def _orientation_tolerance_policy(args) -> dict[str, Any]:
    target = str(args.orientation_target)
    explicit = getattr(args, 'orientation_tol', None)
    return {'orientation_target': target, 'model_default_tolerance': resolve_orientation_tolerance(target), 'effective_tolerance': resolve_orientation_tolerance(target, explicit), 'selection': 'model_default' if explicit is None else 'explicit_override', 'stage_scope': 'all_native_stages_and_final_audit', 'defaults_by_target': dict(ORIENTATION_TOLERANCE_DEFAULTS)}

def _effective_cli_configuration(args):
    use_periodic = not bool(args.nonperiodic)
    if bool(args.geometry_distortion) and (not use_periodic):
        raise ValueError('Geometry distortion is periodic-RVE only; use --no-geometry-distortion for nonperiodic mode.')
    cfg = _build_descent_config(args, use_periodic=use_periodic, wrap_positions=bool(use_periodic and (not args.no_wrap)), pair_config=_build_pair_config(args), line_search_config=_build_line_search_config(args))
    staged_configs = []
    if bool(args.geometry_distortion) and (not bool(args.geo_immediate)):
        factory = _bind_ori_gcg_stage_policy_factory(_make_stage_config_parent)
        for index, (d_scale, w_scale, preliminary, violation_scale) in enumerate(((None, None, False, None), (0.5, 0.25, True, 0.001), (0.75, 0.5, True, 0.0005), (1.0, 1.0, False, None))):
            geometry = None if index == 0 else _scale_geometry_config(cfg.geometry_config, d_scale=d_scale, w_scale=w_scale)
            stage = factory(cfg, include_geometry=index != 0, geometry_config=geometry, preliminary_geometry_stop=preliminary, preliminary_violation_tol_scale=violation_scale)
            stage = _apply_stage_search_overlap_tol(stage, cfg, index)
            staged_configs.append({'stage_index': index, 'config': asdict(stage)})
    return {'schema': 'pw-rve-effective-cli-config-v1', 'adaptation_revision_id': ADAPTATION_REVISION_ID, 'default_profile_id': DEFAULT_PROFILE_ID, 'scientific_execution': False, 'parsed_args': vars(args), 'descent_config': asdict(cfg), 'staged_configs': staged_configs, 'overlap_tolerance_policy': _overlap_tolerance_policy(args), 'orientation_tolerance_policy': _orientation_tolerance_policy(args), 'orientation_initialization_policy': {**_orientation_initialization_policy(args), 'configuration_only': True, 'initialization_executed': False}, 'orientation_target_tensor': orientation_target_tensor_3d(cfg.orientation_target).tolist(), 'configuration_only': True, 'geometry_stage_note': 'Configured GEO ON retains native activation after the overlap stage; this inspection does not execute a stage.'}

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.baseline_identity:
        print(json.dumps(read_current_adaptation_identity(), ensure_ascii=False, indent=2))
        return
    if args.show_effective_config:
        print(json.dumps(_effective_cli_configuration(args), ensure_ascii=False, indent=2))
        return
    if not _ensure_pypw85_available():
        print('pypw85 is not available. Please install pypw85.', file=sys.stderr)
        raise SystemExit(2)
    use_periodic = not bool(args.nonperiodic)
    use_fixed_demo = bool(args.fixed_demo)
    box = _parse_box(args.box)
    if not use_periodic:
        box = None
    if use_fixed_demo and int(args.realizations) != 1:
        raise ValueError('--fixed-demo does not support --realizations>1 (batch).')
    size_dist_mode = _size_dist_requested(args)
    if size_dist_mode and (not use_periodic):
        raise ValueError('Size-distribution mode requires periodic RVE (do not use --nonperiodic).')
    if size_dist_mode and use_fixed_demo:
        raise ValueError('Size-distribution mode is not compatible with --fixed-demo.')
    if bool(args.geometry_distortion) and (not use_periodic):
        raise ValueError('--geometry-distortion is currently periodic-RVE only; remove --nonperiodic.')
    delta_min = float(args.delta_min)
    if delta_min < 0.0:
        raise ValueError('--delta-min must be >= 0.')
    origin_shift_config = _build_origin_shift_config(args, use_periodic=use_periodic)
    M = int(args.realizations)
    if M <= 0:
        raise ValueError('--realizations must be >= 1')
    if M > 1 and (not bool(args.spatial_stats)):
        print('[warn] --realizations>1 without --spatial-stats: will run M optimizations but no aggregated stats/plots will be generated.')
    if bool(args.spatial_stats):
        _require_spatial_statistics_available()
    if M > 1 and args.export_json is not None and (args.batch_export_json_dir is None):
        raise ValueError('Batch mode: please use --batch-export-json-dir instead of --export-json.')
    if use_periodic and box is None:
        raise ValueError('Periodic mode requires --box.')
    wrap_positions = bool(use_periodic and (not args.no_wrap))
    line_search_config = _build_line_search_config(args)
    pair_config = _build_pair_config(args)
    config = _build_descent_config(args, use_periodic=use_periodic, wrap_positions=wrap_positions, pair_config=pair_config, line_search_config=line_search_config)
    if M == 1:
        size_report = None
        if use_fixed_demo:
            particles = build_demo_particles_periodic(box=box) if use_periodic else build_demo_particles_nonperiodic()
        elif use_periodic:
            if size_dist_mode:
                _require_all_size_dist_args(args)
                size_seed = int(args.seed) if args.size_seed is None else int(args.size_seed)
                size_cfg = SizeGenerationConfig3D(box=np.asarray(box, dtype=float), vf_target=float(args.vf_target), vf_abs_tol=float(args.vf_abs_tol), deq_dist=TruncNormalSpec(mu=float(args.deq_mu), sigma=float(args.deq_sigma), d_min=float(args.deq_min), d_max=float(args.deq_max)), aspect_ratio_k=float(args.aspect_ratio_k), aspect_ratio_mode=str(args.aspect_ratio_mode), aspect_ratio_min=args.aspect_ratio_min, aspect_ratio_max=args.aspect_ratio_max, moment_mc_samples=int(args.size_moment_mc_samples), max_adjust_iters=int(args.size_max_adjust_iters), seed=size_seed, scipy_truncnorm=True)
                size_res = _generate_sizes_for_cli(args, size_cfg)
                size_report = size_res.report
                particles = build_random_particles_from_size_arrays(a_arr=size_res.a, b_arr=size_res.b, box=box, seed=int(args.seed), orientation_target=str(args.orientation_target), orientation_regularization=bool(args.orientation_regularization))
            else:
                particles = build_random_demo_particles(n_particles=int(args.n_particles), a=float(args.a), b=float(args.b), box=box, seed=int(args.seed), periodic=True, orientation_target=str(args.orientation_target), orientation_regularization=bool(args.orientation_regularization))
        else:
            particles = build_random_demo_particles(n_particles=int(args.n_particles), a=float(args.a), b=float(args.b), box=None, seed=int(args.seed), periodic=False, orientation_target=str(args.orientation_target), orientation_regularization=bool(args.orientation_regularization))
        print('=== 3D Uniaxial Ellipsoid PW85 Workflow ===')
        print(f'Periodic mode: {use_periodic}')
        print(f'Size distribution mode: {size_dist_mode}')
        print(f'Number of particles: {len(particles)}')
        print(f'delta_min: {delta_min}')
        print(f'origin_shift: {bool(origin_shift_config is not None and origin_shift_config.enabled)}')
        _print_geometry_config_summary(config)
        if bool(config.include_geometry_distortion):
            print('Geometry activation mode: ' + ('immediate from step 0' if bool(args.geo_immediate) else 'staged: overlap-only -> weak/intermediate/final geometry'))
        _print_system_summary(particles=particles, box=box, use_mic=use_periodic, overlap_tol=float(args.overlap_tol), label='Initial', metrics=None, include_orientation_regularization=config.include_orientation_regularization, orientation_target=config.orientation_target)
        initial_geometry_audit = None
        if bool(args.geo_audit) or bool(config.include_geometry_distortion):
            initial_geometry_audit = _print_geometry_audit('Initial', particles, box, config)
        input_state = _input_state_metadata(args, particles, size_report, seed_init=args.seed, seed_size=(int(args.seed) if args.size_seed is None else int(args.size_seed)) if size_dist_mode else None)
        result, constraints_report = _run_descent_with_optional_staged_geometry(particles_physical=particles, config=config, box=box, collect_metrics=bool(args.collect_metrics), delta_min=delta_min, origin_shift_config=origin_shift_config, use_staged_geometry=bool(args.geometry_distortion and (not args.geo_immediate)))
        constraints_report['overlap_tolerance_policy'] = _overlap_tolerance_policy(args)
        constraints_report['orientation_tolerance_policy'] = _orientation_tolerance_policy(args)
        resource_report = constraints_report['resource_budget']
        print(f"[realization resource] G={resource_report['cumulative_completed_iterations']}/{resource_report['total_realization_iteration_budget']} resource_status={resource_report['resource_status']} generation_success={resource_report['generation_success']}")
        final_particles = [p.clone() for p in result.particles]
        final_particles_wrapped = [p.clone() for p in final_particles]
        if use_periodic and box is not None and config.wrap_positions:
            final_particles_wrapped = [EllipsoidParticle3D(a=p.a, b=p.b, c=wrap_position(p.c, box), p=p.p.copy(), id=p.id) for p in final_particles_wrapped]
        main_audit_timings = {}
        final_total_energy_exact, final_status = _timed_audit_call(main_audit_timings, 'physical_pw_and_energy', _print_system_summary, particles=final_particles_wrapped, box=box, use_mic=use_periodic, overlap_tol=float(args.overlap_tol), label='Final (physical)', metrics=None, include_orientation_regularization=config.include_orientation_regularization, orientation_target=config.orientation_target)
        final_geometry_audit = None
        if bool(args.geo_audit) or bool(config.include_geometry_distortion):
            final_geometry_audit = _timed_audit_call(main_audit_timings, 'geometry', _print_geometry_audit, 'Final', final_particles_wrapped, box, config)
        audit_res = None
        if use_periodic:
            print('\n=== Strict Periodic Overlap Audit (physical) ===')
            audit_res = _timed_audit_call(main_audit_timings, 'strict_periodic', audit_periodic_system_overlap, particles=final_particles_wrapped, box=box, overlap_tol=float(args.overlap_tol))
            print(f"Strict audit has_overlap: {audit_res['has_overlap']}")
            if not audit_res['has_overlap']:
                print('System is strictly overlap-free in 3x3x3 periodic space.')
        orientation_result = _timed_audit_call(main_audit_timings, 'orientation_target', check_orientation_isotropy, final_particles_wrapped, tolerance=float(config.orientation_tol), orientation_target=config.orientation_target)
        print()
        print(format_orientation_isotropy_report(orientation_result))
        if args.export_json is not None:
            box_tuple = (float(box[0]), float(box[1]), float(box[2])) if use_periodic and box is not None else None
            optimization_summary: dict[str, Any] = {**resource_report, **_review_timing_json(result, main_audit_timings), 'overlap_tolerance_policy': _overlap_tolerance_policy(args), 'orientation_tolerance_policy': _orientation_tolerance_policy(args), 'include_orientation_regularization': bool(config.include_orientation_regularization), 'orientation_target': config.orientation_target, 'final_orientation': _orientation_audit_json(final_particles_wrapped, config, audit_result=orientation_result), 'final_audits': _final_audit_verdicts(config, final_status, orientation_result, audit_res, final_geometry_audit), 'optimization_elapsed_seconds': float(result.optimization_elapsed_seconds), 'optimization_timing_scope': result.optimization_timing_scope, 'steps_taken': int(result.steps_taken), 'stop_reason': str(result.stop_reason), 'converged': bool(result.converged), 'success': bool(result.success), 'final_energy_total_physical': float(final_total_energy_exact), 'final_max_overlap_pw85_physical': float(final_status['max_overlap']), 'final_num_overlapping_pairs_pw85_physical': int(final_status['num_overlapping_pairs']), 'final_geometry_physical': _compact_geometry_audit(final_geometry_audit), 'overlap_tol_used': float(args.overlap_tol), 'use_mic': bool(config.use_mic), 'use_extent_filter': bool(config.use_extent_filter), 'extent_sep_tol': float(config.extent_sep_tol), 'size_distribution_mode': bool(size_dist_mode), 'geometry_distortion_enabled': bool(config.include_geometry_distortion), 'geometry_energy_tol': float(config.geometry_energy_tol), 'geometry_violation_tol': float(config.geometry_violation_tol), 'geometry_face_violation_tol': float(config.geometry_face_violation_tol), 'geometry_edge_violation_tol': float(config.geometry_edge_violation_tol), 'geometry_corner_violation_tol': float(config.geometry_corner_violation_tol), 'geometry_candidate_filter': bool(getattr(config.geometry_config, 'candidate_filter', False)), 'geometry_candidate_cache': bool(config.use_geometry_candidate_cache), 'geometry_activation_mode': 'immediate' if bool(args.geo_immediate) else 'staged' if bool(config.include_geometry_distortion) else 'off', 'geometry_stage_reports': getattr(result, 'stage_reports', None)}
            metadata = {'command': ' '.join(sys.argv), 'input_state': input_state}
            Path(args.export_json).parent.mkdir(parents=True, exist_ok=True)
            export_particles_json_3d(particles=final_particles_wrapped, path=str(args.export_json), box=box_tuple, periodic=bool(use_periodic), boundary_wrap=True, normalize_p=True, metadata=metadata, constraints=constraints_report, optimization_summary=optimization_summary, indent=int(args.export_json_indent))
            print(f'[export] wrote json: {args.export_json}')
        if bool(args.spatial_stats):
            if not use_periodic or box is None:
                raise ValueError('spatial-stats supports periodic only.')
            if args.spatial_stats_outdir is not None:
                outdir = Path(str(args.spatial_stats_outdir))
            else:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                outdir = Path('outputs') / f'spatial_stats_single_{ts}'
            outdir.mkdir(parents=True, exist_ok=True)
            plots_dir = outdir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            stats_cfg = SpatialStatsMVPConfig3D(box=np.asarray(box, dtype=float), nnoa_rank=1, r_max_factor=float(args.stats_r_max_factor), n_r_bins=int(args.stats_n_r_bins), n_K_points=int(args.stats_n_k_points), orientation_tol=float(args.stats_orientation_tol))
            stats = compute_spatial_statistics_mvp_3d(final_particles_wrapped, stats_cfg)
            stats['schema_version'] = SPATIAL_STATS_SCHEMA_VERSION
            stats['meta'] = {'command': ' '.join(sys.argv), **resource_report, 'periodic': True, 'timestamp': datetime.now().isoformat(), 'constraints': constraints_report}
            (outdir / 'stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
            if not bool(args.spatial_stats_no_plots):
                plot_nnd_pdfs([stats], plots_dir / 'nnd_pdfs.png')
                plot_nnoa_cdfs([stats], plots_dir / 'nnoa_cdfs.png', degrees=True)
                plot_ripley_and_rdf([stats], plots_dir / 'ripleyK_rdf.png')
                plot_orientation_isotropy_boxplot([stats], plots_dir / 'orientation_isotropy_boxplot.png')
            print(f'[spatial-stats] wrote to: {outdir.resolve()}')
        run_metrics = result.metrics
        if run_metrics is not None:
            run_metrics.final_energy = float(final_total_energy_exact)
            run_metrics.final_max_overlap = float(final_status['max_overlap'])
            run_metrics.converged = bool(result.success)
            if args.metrics_json is not None:
                save_run_metrics_json(run_metrics, args.metrics_json, metadata=resource_report, constraints=constraints_report)
            if args.metrics_summary_json is not None:
                save_run_summary_json(run_metrics, args.metrics_summary_json, metadata=resource_report, constraints=constraints_report)
            if args.metrics_csv is not None:
                save_step_metrics_csv(run_metrics, args.metrics_csv)
        return
    if not use_periodic or box is None:
        raise ValueError('Batch mode currently supports periodic only (provide --box, do not use --nonperiodic).')
    base_seed = int(args.seed) if args.base_seed is None else int(args.base_seed)
    if args.spatial_stats_outdir is not None:
        outdir = Path(str(args.spatial_stats_outdir))
    elif args.batch_outdir is not None:
        outdir = Path(str(args.batch_outdir))
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = Path('outputs') / f'spatial_stats_batch_{ts}'
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    export_dir = None
    if args.batch_export_json_dir is not None:
        export_dir = Path(str(args.batch_export_json_dir))
        export_dir.mkdir(parents=True, exist_ok=True)
    stats_cfg = SpatialStatsMVPConfig3D(box=np.asarray(box, dtype=float), nnoa_rank=1, r_max_factor=float(args.stats_r_max_factor), n_r_bins=int(args.stats_n_r_bins), n_K_points=int(args.stats_n_k_points), orientation_tol=float(args.stats_orientation_tol))
    stats_list: list[dict] = []
    batch_summary: dict[str, Any] = {'schema_version': SPATIAL_STATS_SCHEMA_VERSION, 'outdir': str(outdir), 'M': M, 'box': [float(x) for x in box], 'base_seed': base_seed, 'size_distribution_mode': bool(size_dist_mode), 'command': ' '.join(sys.argv), 'timestamp': datetime.now().isoformat(), 'spatial_stats': bool(args.spatial_stats), 'delta_min': float(delta_min), 'origin_shift_enabled': bool(origin_shift_config is not None and origin_shift_config.enabled), 'geometry_distortion_enabled': bool(config.include_geometry_distortion), 'geometry_energy_tol': float(config.geometry_energy_tol), 'geometry_violation_tol': float(config.geometry_violation_tol), 'geometry_face_violation_tol': float(config.geometry_face_violation_tol), 'geometry_edge_violation_tol': float(config.geometry_edge_violation_tol), 'geometry_corner_violation_tol': float(config.geometry_corner_violation_tol), 'geometry_candidate_filter': bool(getattr(config.geometry_config, 'candidate_filter', False)), 'geometry_candidate_cache': bool(config.use_geometry_candidate_cache), 'realization_resource_results': []}
    for r in range(M):
        seed_size0 = int(args.size_seed) if args.size_seed is not None else base_seed
        seed_size = seed_size0 + r
        seed_init = base_seed + 1000 + r
        real_dir = outdir / f'real_{r:03d}'
        real_dir.mkdir(parents=True, exist_ok=True)
        final_particles_wrapped, final_status, size_report, descent_res, constraints_report = _run_one_realization(args=args, box=box, use_periodic=True, size_dist_mode=size_dist_mode, seed_init=seed_init, seed_size=seed_size, realization_index=r, config=config, origin_shift_config=origin_shift_config)
        constraints_report['overlap_tolerance_policy'] = _overlap_tolerance_policy(args)
        constraints_report['orientation_tolerance_policy'] = _orientation_tolerance_policy(args)
        resource_report = constraints_report['resource_budget']
        batch_summary['realization_resource_results'].append({'realization_index': r, **resource_report, 'stop_reason': str(descent_res.stop_reason)})
        if bool(args.spatial_stats):
            stats = compute_spatial_statistics_mvp_3d(final_particles_wrapped, stats_cfg)
            stats['schema_version'] = SPATIAL_STATS_SCHEMA_VERSION
            stats['realization_index'] = r
            stats['meta'] = {**resource_report, 'seed_size': int(seed_size), 'seed_init': int(seed_init), 'stop_reason': str(descent_res.stop_reason), 'success': bool(descent_res.success), 'steps_taken': int(descent_res.steps_taken), 'final_max_overlap_pw85': float(final_status['max_overlap']), 'final_num_overlapping_pairs_pw85': int(final_status['num_overlapping_pairs']), 'constraints': constraints_report}
            if size_report is not None:
                stats['size_report'] = size_report
            (real_dir / 'stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
            stats_list.append(stats)
        if export_dir is not None:
            box_tuple = (float(box[0]), float(box[1]), float(box[2]))
            export_path = export_dir / f'real_{r:03d}.json'
            main_audit_timings = dict(constraints_report.get('eo_main_independent_audit_timings_seconds', {}))
            orientation_audit = _timed_audit_call(main_audit_timings, 'orientation_target', _orientation_audit_json, final_particles_wrapped, config)
            strict_audit = _timed_audit_call(main_audit_timings, 'strict_periodic', audit_periodic_system_overlap, final_particles_wrapped, box, overlap_tol=float(config.overlap_tol))
            geometry_audit = _timed_audit_call(main_audit_timings, 'geometry', _print_geometry_audit, 'Final batch', final_particles_wrapped, box, config)
            optimization_summary = {**resource_report, **_review_timing_json(descent_res, main_audit_timings), 'overlap_tolerance_policy': _overlap_tolerance_policy(args), 'orientation_tolerance_policy': _orientation_tolerance_policy(args), 'realization_index': r, 'seed_size': int(seed_size), 'seed_size_requested': int(seed_size), 'seed_size_effective': constraints_report['eo_input_state']['seed_size_effective'], 'seed_init': int(seed_init), 'steps_taken': int(descent_res.steps_taken), 'include_orientation_regularization': bool(config.include_orientation_regularization), 'orientation_target': config.orientation_target, 'final_orientation': orientation_audit, 'final_geometry_physical': _compact_geometry_audit(geometry_audit), 'final_audits': _final_audit_verdicts(config, final_status, orientation_audit, strict_audit, geometry_audit), 'optimization_elapsed_seconds': float(descent_res.optimization_elapsed_seconds), 'optimization_timing_scope': descent_res.optimization_timing_scope, 'stop_reason': str(descent_res.stop_reason), 'success': bool(descent_res.success), 'final_max_overlap_pw85_physical': float(final_status['max_overlap']), 'final_num_overlapping_pairs_pw85_physical': int(final_status['num_overlapping_pairs']), 'size_distribution_mode': bool(size_dist_mode), 'geometry_distortion_enabled': bool(config.include_geometry_distortion), 'geometry_energy_tol': float(config.geometry_energy_tol), 'geometry_violation_tol': float(config.geometry_violation_tol), 'geometry_face_violation_tol': float(config.geometry_face_violation_tol), 'geometry_edge_violation_tol': float(config.geometry_edge_violation_tol), 'geometry_corner_violation_tol': float(config.geometry_corner_violation_tol), 'geometry_candidate_filter': bool(getattr(config.geometry_config, 'candidate_filter', False)), 'geometry_candidate_cache': bool(config.use_geometry_candidate_cache), 'geometry_activation_mode': 'immediate' if bool(args.geo_immediate) else 'staged' if bool(config.include_geometry_distortion) else 'off', 'geometry_stage_reports': getattr(descent_res, 'stage_reports', None)}
            export_particles_json_3d(particles=final_particles_wrapped, path=str(export_path), box=box_tuple, periodic=True, boundary_wrap=True, normalize_p=True, metadata={'command': ' '.join(sys.argv), 'input_state': constraints_report['eo_input_state']}, constraints=constraints_report, optimization_summary=optimization_summary, indent=int(args.export_json_indent))
        if descent_res.metrics is not None:
            if args.collect_metrics:
                save_run_metrics_json(descent_res.metrics, str(real_dir / 'run_metrics.json'), metadata=resource_report, constraints=constraints_report)
                save_run_summary_json(descent_res.metrics, str(real_dir / 'run_summary.json'), metadata=resource_report, constraints=constraints_report)
                save_step_metrics_csv(descent_res.metrics, str(real_dir / 'step_metrics.csv'))
        print(f"[real {r:03d}] N={len(final_particles_wrapped)} max_overlap={final_status['max_overlap']:.3e} steps={int(descent_res.steps_taken)} success={bool(descent_res.success)} G={resource_report['cumulative_completed_iterations']}/{resource_report['total_realization_iteration_budget']} resource_status={resource_report['resource_status']} generation_success={resource_report['generation_success']}")
    if bool(args.spatial_stats) and (not bool(args.spatial_stats_no_plots)) and (not bool(args.batch_no_aggregate_plots)):
        plot_nnd_pdfs(stats_list, plots_dir / 'nnd_pdfs.png')
        plot_nnoa_cdfs(stats_list, plots_dir / 'nnoa_cdfs.png', degrees=True)
        plot_ripley_and_rdf(stats_list, plots_dir / 'ripleyK_rdf.png')
        plot_orientation_isotropy_boxplot(stats_list, plots_dir / 'orientation_isotropy_boxplot.png')
    (outdir / 'batch_summary.json').write_text(json.dumps(batch_summary, indent=2), encoding='utf-8')
    print(f'[batch done] outputs written to: {outdir.resolve()}')
if __name__ == '__main__':
    main()
