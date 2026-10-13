from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
try:
    from scipy.stats import truncnorm as _scipy_truncnorm
except Exception:
    _scipy_truncnorm = None

@dataclass(frozen=True)
class TruncNormalSpec:
    mu: float
    sigma: float
    d_min: float
    d_max: float

@dataclass(frozen=True)
class SizeGenerationConfig3D:
    box: np.ndarray
    vf_target: float
    vf_abs_tol: float
    deq_dist: TruncNormalSpec
    aspect_ratio_k: float
    moment_mc_samples: int = 200000
    max_adjust_iters: int = 20
    seed: int | None = None
    scipy_truncnorm: bool = True
    aspect_ratio_mode: str = 'fixed'
    aspect_ratio_min: float | None = None
    aspect_ratio_max: float | None = None

@dataclass
class SizeGenerationResult3D:
    a: np.ndarray
    b: np.ndarray
    deq: np.ndarray
    volumes: np.ndarray
    N: int
    vf_sample: float
    vf_error: float
    report: dict[str, Any]
    rho: np.ndarray | None = None
    aspect_ratio_mode: str = 'fixed'
    aspect_ratio_min: float | None = None
    aspect_ratio_max: float | None = None
    aspect_ratio_seed: int | None = None

def _validate_config(config: SizeGenerationConfig3D) -> None:
    box = np.asarray(config.box, dtype=float).reshape(3)
    if not np.all(np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError(f'Invalid box: {box}. Must be positive finite (Lx,Ly,Lz).')
    if not 0.0 < config.vf_target < 1.0:
        raise ValueError(f'vf_target must be in (0,1), got {config.vf_target}')
    if config.vf_abs_tol < 0.0:
        raise ValueError(f'vf_abs_tol must be >=0, got {config.vf_abs_tol}')
    dspec = config.deq_dist
    if not dspec.sigma > 0.0:
        raise ValueError('deq sigma must be > 0')
    if not dspec.d_min > 0.0:
        raise ValueError('deq d_min must be > 0')
    if not dspec.d_max > dspec.d_min:
        raise ValueError('deq d_max must be > d_min')
    if not np.isfinite(config.aspect_ratio_k) or config.aspect_ratio_k < 1.0:
        raise ValueError('aspect_ratio_k must be >= 1.0')
    if config.aspect_ratio_mode not in ('fixed', 'uniform'):
        raise ValueError('aspect_ratio_mode must be fixed or uniform')
    if config.aspect_ratio_mode == 'fixed':
        if config.aspect_ratio_min is not None or config.aspect_ratio_max is not None:
            raise ValueError('aspect_ratio_min/max are only valid in uniform mode')
    else:
        if config.aspect_ratio_min is None or config.aspect_ratio_max is None:
            raise ValueError('uniform aspect ratio requires both aspect_ratio_min and aspect_ratio_max')
        lower = float(config.aspect_ratio_min)
        upper = float(config.aspect_ratio_max)
        if not (np.isfinite(lower) and np.isfinite(upper) and (1.0 <= lower < upper)):
            raise ValueError('uniform aspect ratio requires finite 1 <= min < max')
    if config.scipy_truncnorm and _scipy_truncnorm is None:
        raise RuntimeError('scipy_truncnorm=True but SciPy is not available.')

def _sample_truncnorm_scipy(mu: float, sigma: float, d_min: float, d_max: float, size: int, rng: np.random.Generator) -> np.ndarray:
    a = (d_min - mu) / sigma
    b = (d_max - mu) / sigma
    try:
        samples = _scipy_truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rng)
    except TypeError:
        rs = np.random.RandomState(rng.integers(0, 2 ** 31 - 1))
        samples = _scipy_truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rs)
    return np.clip(np.asarray(samples, dtype=float), d_min, d_max)

def _sample_truncnorm_rejection(mu: float, sigma: float, d_min: float, d_max: float, size: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        batch = max(1024, (size - filled) * 2)
        x = rng.normal(loc=mu, scale=sigma, size=batch)
        x = x[(x >= d_min) & (x <= d_max)]
        take = min(x.size, size - filled)
        if take > 0:
            out[filled:filled + take] = x[:take]
            filled += take
    return out

def _sample_deq(config: SizeGenerationConfig3D, size: int, rng: np.random.Generator) -> np.ndarray:
    dspec = config.deq_dist
    if config.scipy_truncnorm:
        return _sample_truncnorm_scipy(dspec.mu, dspec.sigma, dspec.d_min, dspec.d_max, size, rng)
    return _sample_truncnorm_rejection(dspec.mu, dspec.sigma, dspec.d_min, dspec.d_max, size, rng)

def _estimate_E_d3(config: SizeGenerationConfig3D, rng: np.random.Generator) -> float:
    d = _sample_deq(config, config.moment_mc_samples, rng)
    return float(np.mean(d ** 3))

def _vf_from_deq(deq: np.ndarray, V_box: float) -> tuple[float, np.ndarray]:
    volumes = np.pi / 6.0 * deq ** 3
    vf = float(np.sum(volumes) / V_box)
    return (vf, volumes)

def generate_ellipsoid_sizes_from_truncnorm(config: SizeGenerationConfig3D) -> SizeGenerationResult3D:
    _validate_config(config)
    box = np.asarray(config.box, dtype=float).reshape(3)
    V_box = float(np.prod(box))
    vf_target = float(config.vf_target)
    vf_abs_tol = float(config.vf_abs_tol)
    rng_master = np.random.default_rng(config.seed)
    rng_moment = np.random.default_rng(rng_master.integers(0, 2 ** 63 - 1))
    rng_sample = np.random.default_rng(rng_master.integers(0, 2 ** 63 - 1))
    E_d3 = _estimate_E_d3(config, rng_moment)
    E_V = np.pi / 6.0 * E_d3
    V_target = vf_target * V_box
    N = int(max(1, round(V_target / E_V)))
    accepted = False
    last_deq = None
    last_volumes = None
    last_vf = None
    for it in range(config.max_adjust_iters):
        deq = _sample_deq(config, N, rng_sample)
        vf_sample, volumes = _vf_from_deq(deq, V_box)
        last_deq, last_volumes, last_vf = (deq, volumes, vf_sample)
        if vf_target - vf_abs_tol <= vf_sample <= vf_target + vf_abs_tol:
            accepted = True
            break
        if vf_sample <= 0.0:
            N_new = max(1, N * 2)
        else:
            N_new = int(max(1, round(N * (vf_target / vf_sample))))
        if N_new == N:
            N_new = N + (1 if vf_sample < vf_target else -1)
            N_new = max(1, N_new)
        N = N_new
    if not accepted:
        raise RuntimeError(f'Size generation failed to meet VF tolerance within max_adjust_iters. Last vf={last_vf}, target={vf_target}, abs_tol={vf_abs_tol}, last N={N}.')
    deq = np.asarray(last_deq, dtype=float)
    volumes = np.asarray(last_volumes, dtype=float)
    vf_sample = float(last_vf)
    vf_error = vf_sample - vf_target
    k = float(config.aspect_ratio_k)
    r_eq = 0.5 * deq
    rho_seed = None
    if config.aspect_ratio_mode == 'fixed':
        rho = np.full(deq.shape, k, dtype=float)
        b = r_eq * k ** (-1.0 / 3.0)
        a = r_eq * k ** (2.0 / 3.0)
    else:
        sequence = np.random.SeedSequence(config.seed, spawn_key=(1095917361,))
        rho_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        rho_rng = np.random.default_rng(rho_seed)
        rho = rho_rng.uniform(float(config.aspect_ratio_min), float(config.aspect_ratio_max), size=deq.size)
        b = r_eq * rho ** (-1.0 / 3.0)
        a = r_eq * rho ** (2.0 / 3.0)
    aspect_ratio_distribution = {'mode': config.aspect_ratio_mode, 'fixed_k': k if config.aspect_ratio_mode == 'fixed' else None, 'min': config.aspect_ratio_min, 'max': config.aspect_ratio_max, 'seed': rho_seed, 'input_seed': config.seed, 'stream_policy': 'NO_ADDITIONAL_RANDOM_DRAWS' if config.aspect_ratio_mode == 'fixed' else 'INDEPENDENT_SEEDSEQUENCE_SPAWN_KEY_0x41525F31_UINT64', 'rho': rho.tolist(), 'deq': deq.tolist(), 'a': a.tolist(), 'b': b.tolist(), 'N': int(deq.size), 'vf_sample': vf_sample}
    report = {'module': 'size_generator_3d', 'scheme': 'A_estimate_N_then_sample_adjust_N', 'input': {'box': box.tolist(), 'vf_target': vf_target, 'vf_abs_tol': vf_abs_tol, 'deq_dist': asdict(config.deq_dist), 'aspect_ratio_k': k, 'aspect_ratio_mode': config.aspect_ratio_mode, 'aspect_ratio_min': config.aspect_ratio_min, 'aspect_ratio_max': config.aspect_ratio_max, 'moment_mc_samples': int(config.moment_mc_samples), 'max_adjust_iters': int(config.max_adjust_iters), 'seed': config.seed, 'scipy_truncnorm': bool(config.scipy_truncnorm)}, 'result': {'N': int(deq.size), 'V_box': V_box, 'vf_sample': vf_sample, 'vf_error': vf_error, 'vf_passed': vf_target - vf_abs_tol <= vf_sample <= vf_target + vf_abs_tol, 'deq_mean': float(np.mean(deq)), 'deq_std': float(np.std(deq, ddof=1)) if deq.size > 1 else 0.0, 'deq_min': float(np.min(deq)), 'deq_max': float(np.max(deq)), 'a_min': float(np.min(a)), 'a_max': float(np.max(a)), 'b_min': float(np.min(b)), 'b_max': float(np.max(b)), 'E_d3_mc': float(E_d3), 'E_V_mc': float(E_V)}, 'aspect_ratio_distribution': aspect_ratio_distribution}
    return SizeGenerationResult3D(a=a, b=b, deq=deq, volumes=volumes, N=int(deq.size), vf_sample=vf_sample, vf_error=vf_error, report=report, rho=rho, aspect_ratio_mode=config.aspect_ratio_mode, aspect_ratio_min=config.aspect_ratio_min, aspect_ratio_max=config.aspect_ratio_max, aspect_ratio_seed=rho_seed)

def size_generation_report_dict(res: SizeGenerationResult3D) -> dict[str, Any]:
    return res.report
