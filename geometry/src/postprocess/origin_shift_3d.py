from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.periodic.mic_3d import wrap_position

@dataclass(frozen=True)
class OriginShiftConfig3D:
    enabled: bool = False
    epsilon_b: float | None = None
    beta: float = 0.1
    M: int = 64
    refine: bool = True
    refine_M: int = 32
    objective: str = 'lexicographic_max_then_sum'
    offender_top_k: int = 5

def _validate_box(box) -> np.ndarray:
    box_array = np.asarray(box, dtype=float)
    if box_array.shape != (3,):
        raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
    if np.any(box_array <= 0.0):
        raise ValueError(f'Box lengths must be positive, got {box_array}.')
    return box_array

def estimate_epsilon_b_3d(particles: list[EllipsoidParticle3D], beta: float=0.1) -> float:
    if not isinstance(particles, list) or len(particles) == 0:
        raise ValueError('particles must be a non-empty list.')
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError(f'beta must be > 0, got {beta}.')
    bs = []
    for i, pt in enumerate(particles):
        if not isinstance(pt, EllipsoidParticle3D):
            raise TypeError(f'particles[{i}] must be EllipsoidParticle3D.')
        bs.append(float(pt.b))
    med_b = float(np.median(np.asarray(bs, dtype=float)))
    eps = beta * med_b
    if eps <= 0.0:
        raise ValueError(f'Estimated epsilon_b must be >0, got {eps}.')
    return float(eps)

def axis_projection_radius_3d(a: float, b: float, p: np.ndarray, axis: int) -> float:
    if axis not in (0, 1, 2):
        raise ValueError(f'axis must be 0/1/2, got {axis}.')
    p = np.asarray(p, dtype=float)
    if p.shape != (3,):
        raise ValueError(f'Expected p.shape==(3,), got {p.shape}.')
    a = float(a)
    b = float(b)
    if a < b or b <= 0.0:
        raise ValueError(f'Invalid (a,b): a={a}, b={b}. Must satisfy a>=b>0.')
    pk = float(p[axis])
    return float(np.sqrt(b * b + (a * a - b * b) * (pk * pk)))

def _axis_contact_info(pt: EllipsoidParticle3D, *, axis: int, L: float, t: float) -> dict[str, Any]:
    c = float(np.asarray(pt.c, dtype=float)[axis])
    c_prime = (c + t) % L
    d_low = c_prime
    d_high = L - c_prime
    if d_low <= d_high:
        d = d_low
        nearest_face = 'low'
    else:
        d = d_high
        nearest_face = 'high'
    h = axis_projection_radius_3d(pt.a, pt.b, pt.p, axis)
    m = float(d - h)
    abs_m = float(abs(m))
    return {'particle_index': None, 'particle_id': None if pt.id is None else int(pt.id), 'axis': int(axis), 'L': float(L), 't_axis': float(t), 'c_axis': float(c), 'c_prime': float(c_prime), 'd_low': float(d_low), 'd_high': float(d_high), 'd_nearest': float(d), 'nearest_face': nearest_face, 'h_proj': float(h), 'm': float(m), 'abs_m': float(abs_m), 'm_sign': 'crossing' if m < 0.0 else 'clearance'}

def origin_shift_axis_risk_1d(particles: list[EllipsoidParticle3D], box: np.ndarray, axis: int, t_axis: float, epsilon_b: float) -> tuple[float, float]:
    if not isinstance(particles, list):
        raise TypeError('particles must be a list.')
    box = _validate_box(box)
    if axis not in (0, 1, 2):
        raise ValueError(f'axis must be 0/1/2, got {axis}.')
    L = float(box[axis])
    t = float(t_axis) % L
    eps = float(epsilon_b)
    if eps <= 0.0:
        raise ValueError(f'epsilon_b must be > 0, got {eps}.')
    R_max = 0.0
    R_sum = 0.0
    for i, pt in enumerate(particles):
        if not isinstance(pt, EllipsoidParticle3D):
            raise TypeError(f'particles[{i}] must be EllipsoidParticle3D.')
        h = axis_projection_radius_3d(pt.a, pt.b, pt.p, axis)
        c = float(np.asarray(pt.c, dtype=float)[axis])
        c_prime = (c + t) % L
        d = min(c_prime, L - c_prime)
        abs_m = abs(float(d - h))
        r = max(0.0, eps - abs_m)
        if r > R_max:
            R_max = r
        R_sum += r * r
    return (float(R_max), float(R_sum))

def _origin_shift_axis_risk_details_1d(particles: list[EllipsoidParticle3D], box: np.ndarray, axis: int, t_axis: float, epsilon_b: float, *, top_k: int=5) -> dict[str, Any]:
    if not isinstance(particles, list):
        raise TypeError('particles must be a list.')
    box = _validate_box(box)
    if axis not in (0, 1, 2):
        raise ValueError(f'axis must be 0/1/2, got {axis}.')
    L = float(box[axis])
    t = float(t_axis) % L
    eps = float(epsilon_b)
    if eps <= 0.0:
        raise ValueError(f'epsilon_b must be > 0, got {eps}.')
    top_k = int(top_k)
    if top_k <= 0:
        top_k = 1
    R_max = 0.0
    R_sum = 0.0
    closest_abs_m = float('inf')
    closest_info: dict[str, Any] | None = None
    offenders: list[dict[str, Any]] = []
    worst_info: dict[str, Any] | None = None
    for i, pt in enumerate(particles):
        if not isinstance(pt, EllipsoidParticle3D):
            raise TypeError(f'particles[{i}] must be EllipsoidParticle3D.')
        info = _axis_contact_info(pt, axis=axis, L=L, t=t)
        info['particle_index'] = int(i)
        abs_m = float(info['abs_m'])
        r = max(0.0, eps - abs_m)
        info['epsilon_b'] = float(eps)
        info['r'] = float(r)
        if abs_m < closest_abs_m:
            closest_abs_m = abs_m
            closest_info = dict(info)
        if r > R_max:
            R_max = r
            worst_info = dict(info)
        R_sum += r * r
        if r > 0.0:
            offenders.append(info)
    offenders.sort(key=lambda d: float(d['r']), reverse=True)
    top_offenders = offenders[:top_k]
    danger_hits = int(len(offenders))
    out = {'R_max': float(R_max), 'R_sum': float(R_sum), 'danger_band_hits': danger_hits, 'epsilon_b': float(eps), 't_axis': float(t), 'axis': int(axis), 'closest_particle': closest_info, 'worst_offender': None if float(R_max) <= 0.0 else worst_info, 'top_offenders': top_offenders}
    return out

def _lexi_better(a: tuple[float, float], b: tuple[float, float]) -> bool:
    if a[0] < b[0]:
        return True
    if a[0] > b[0]:
        return False
    return a[1] < b[1]

def find_best_origin_shift_axis_1d(particles: list[EllipsoidParticle3D], box: np.ndarray, axis: int, epsilon_b: float, M: int=64, refine: bool=True, refine_M: int=32, objective: str='lexicographic_max_then_sum', *, offender_top_k: int=5) -> tuple[float, dict[str, Any]]:
    box = _validate_box(box)
    if axis not in (0, 1, 2):
        raise ValueError(f'axis must be 0/1/2, got {axis}.')
    if objective != 'lexicographic_max_then_sum':
        raise ValueError(f'Unsupported objective: {objective}')
    M = int(M)
    if M <= 1:
        raise ValueError(f'M must be > 1, got {M}.')
    refine = bool(refine)
    refine_M = int(refine_M)
    if refine and refine_M <= 1:
        raise ValueError(f'refine_M must be > 1 when refine=True, got {refine_M}.')
    L = float(box[axis])
    eps = float(epsilon_b)
    candidates = [float(k) * L / float(M) for k in range(M)]
    best_t = candidates[0]
    best_score = origin_shift_axis_risk_1d(particles, box, axis, best_t, eps)
    for t in candidates:
        score = origin_shift_axis_risk_1d(particles, box, axis, t, eps)
        if _lexi_better(score, best_score):
            best_t, best_score = (t, score)
    coarse_details = _origin_shift_axis_risk_details_1d(particles, box, axis, best_t, eps, top_k=offender_top_k)
    report: dict[str, Any] = {'axis': axis, 'L': L, 'epsilon_b': eps, 'M': M, 'objective': objective, 'coarse_best_t': float(best_t), 'coarse_best_R_max': float(coarse_details['R_max']), 'coarse_best_R_sum': float(coarse_details['R_sum']), 'coarse_best_diagnostics': coarse_details}
    if not refine:
        report['refine'] = False
        return (float(best_t), report)
    step = L / float(M)
    half = 0.5 * step
    offsets = [-half + float(k) * (2.0 * half) / float(refine_M) for k in range(refine_M)]
    refine_candidates = [float((best_t + off) % L) for off in offsets]
    refine_best_t = refine_candidates[0]
    refine_best_score = origin_shift_axis_risk_1d(particles, box, axis, refine_best_t, eps)
    for t in refine_candidates:
        score = origin_shift_axis_risk_1d(particles, box, axis, t, eps)
        if _lexi_better(score, refine_best_score):
            refine_best_t, refine_best_score = (t, score)
    refine_details = _origin_shift_axis_risk_details_1d(particles, box, axis, refine_best_t, eps, top_k=offender_top_k)
    report.update({'refine': True, 'refine_M': refine_M, 'refine_window_halfwidth': half, 'refine_best_t': float(refine_best_t), 'refine_best_R_max': float(refine_details['R_max']), 'refine_best_R_sum': float(refine_details['R_sum']), 'refine_best_diagnostics': refine_details})
    return (float(refine_best_t), report)

def choose_origin_shift_vector_3d(particles: list[EllipsoidParticle3D], box: np.ndarray, config: OriginShiftConfig3D) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(config, OriginShiftConfig3D):
        raise TypeError(f'config must be OriginShiftConfig3D, got {type(config).__name__}.')
    box = _validate_box(box)
    if not config.enabled:
        shift = np.zeros(3, dtype=float)
        report = {'enabled': False, 'shift': shift.tolist()}
        return (shift, report)
    if config.epsilon_b is not None:
        eps = float(config.epsilon_b)
        if eps <= 0.0:
            raise ValueError(f'epsilon_b must be > 0, got {eps}.')
        eps_source = 'user'
    else:
        eps = estimate_epsilon_b_3d(particles, beta=float(config.beta))
        eps_source = 'auto_beta_median_b'
    axis_reports = []
    shift = np.zeros(3, dtype=float)
    for axis in (0, 1, 2):
        t_best, axis_report = find_best_origin_shift_axis_1d(particles=particles, box=box, axis=axis, epsilon_b=eps, M=int(config.M), refine=bool(config.refine), refine_M=int(config.refine_M), objective=str(config.objective), offender_top_k=int(config.offender_top_k))
        shift[axis] = float(t_best)
        axis_reports.append(axis_report)
    report = {'enabled': True, 'epsilon_b': float(eps), 'epsilon_b_source': eps_source, 'beta': float(config.beta), 'M': int(config.M), 'refine': bool(config.refine), 'refine_M': int(config.refine_M), 'objective': str(config.objective), 'offender_top_k': int(config.offender_top_k), 'shift': shift.tolist(), 'axis_reports': axis_reports}
    return (shift, report)

def apply_origin_shift_3d(particles: list[EllipsoidParticle3D], box: np.ndarray, shift: np.ndarray) -> list[EllipsoidParticle3D]:
    box = _validate_box(box)
    shift = np.asarray(shift, dtype=float)
    if shift.shape != (3,):
        raise ValueError(f'Expected shift.shape==(3,), got {shift.shape}.')
    out: list[EllipsoidParticle3D] = []
    for i, pt in enumerate(particles):
        if not isinstance(pt, EllipsoidParticle3D):
            raise TypeError(f'particles[{i}] must be EllipsoidParticle3D.')
        new_c = wrap_position(np.asarray(pt.c, dtype=float) + shift, box)
        out.append(EllipsoidParticle3D(a=float(pt.a), b=float(pt.b), c=np.asarray(new_c, dtype=float), p=np.asarray(pt.p, dtype=float), id=pt.id))
    return out

def origin_shift_postprocess_3d(particles: list[EllipsoidParticle3D], box: np.ndarray, config: OriginShiftConfig3D) -> tuple[list[EllipsoidParticle3D], np.ndarray, dict[str, Any]]:
    box = _validate_box(box)
    if not isinstance(config, OriginShiftConfig3D):
        raise TypeError(f'config must be OriginShiftConfig3D, got {type(config).__name__}.')
    if not config.enabled:
        shift = np.zeros(3, dtype=float)
        report = {'enabled': False, 'shift': shift.tolist()}
        cloned = [EllipsoidParticle3D(a=float(pt.a), b=float(pt.b), c=np.asarray(pt.c, dtype=float), p=np.asarray(pt.p, dtype=float), id=pt.id) for pt in particles]
        return (cloned, shift, report)
    shift, report = choose_origin_shift_vector_3d(particles, box, config)
    shifted = apply_origin_shift_3d(particles, box, shift)
    return (shifted, shift, report)
