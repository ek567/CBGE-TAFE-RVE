from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
ClearanceStatus3D = Literal['outside', 'inside', 'tangent', 'center_fallback']
FaceBranch3D = Literal['left', 'right']

@dataclass(frozen=True)
class ProjectedClearanceResult3D:
    s: float
    z_star: np.ndarray
    gamma: float
    inside: bool
    a: np.ndarray
    ell: float
    nu: np.ndarray
    mu: float | None
    status: ClearanceStatus3D

@dataclass(frozen=True)
class FaceClearanceResult3D:
    s: float
    branch: FaceBranch3D
    m: float
    h: float
    q_minus: float
    q_plus: float

def _validate_vector(name: str, value: np.ndarray, dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dim,):
        raise ValueError(f'Expected {name}.shape == ({dim},), got {array.shape}.')
    return array

def _validate_matrix(name: str, value: np.ndarray, dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dim, dim):
        raise ValueError(f'Expected {name}.shape == ({dim}, {dim}), got {array.shape}.')
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f'{name} must be symmetric within tolerance.')
    return array

def compute_face_signed_clearance(m: float, h: float) -> FaceClearanceResult3D:
    m_value = float(m)
    h_value = float(h)
    if h_value <= 0.0:
        raise ValueError(f'h must be > 0, got {h_value}.')
    q_minus = m_value - h_value
    q_plus = m_value + h_value
    if q_minus > 0.0:
        return FaceClearanceResult3D(s=float(q_minus), branch='left', m=m_value, h=h_value, q_minus=float(q_minus), q_plus=float(q_plus))
    if q_plus < 0.0:
        return FaceClearanceResult3D(s=float(-q_plus), branch='right', m=m_value, h=h_value, q_minus=float(q_minus), q_plus=float(q_plus))
    if -q_minus <= q_plus:
        return FaceClearanceResult3D(s=float(q_minus), branch='left', m=m_value, h=h_value, q_minus=float(q_minus), q_plus=float(q_plus))
    return FaceClearanceResult3D(s=float(-q_plus), branch='right', m=m_value, h=h_value, q_minus=float(q_minus), q_plus=float(q_plus))

def _f_mu(mu: float, q: np.ndarray, y: np.ndarray) -> float:
    denom = q + float(mu)
    if np.any(np.abs(denom) < 1e-300):
        return float('inf')
    return float(np.sum(q * y * y / (denom * denom)) - 1.0)

def _df_mu(mu: float, q: np.ndarray, y: np.ndarray) -> float:
    denom = q + float(mu)
    if np.any(np.abs(denom) < 1e-300):
        return float('nan')
    return float(-2.0 * np.sum(q * y * y / (denom * denom * denom)))

def _bisection_root(q: np.ndarray, y: np.ndarray, left: float, right: float, *, tol: float, max_iter: int) -> float | None:
    f_left = _f_mu(left, q, y)
    f_right = _f_mu(right, q, y)
    if not (np.isfinite(f_left) and np.isfinite(f_right)):
        return None
    if f_left == 0.0:
        return float(left)
    if f_right == 0.0:
        return float(right)
    if f_left * f_right > 0.0:
        return None
    lo = float(left)
    hi = float(right)
    flo = float(f_left)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        fmid = _f_mu(mid, q, y)
        if not np.isfinite(fmid):
            mid = np.nextafter(mid, hi)
            fmid = _f_mu(mid, q, y)
            if not np.isfinite(fmid):
                return None
        if abs(fmid) <= tol or abs(hi - lo) <= tol * max(1.0, abs(mid)):
            return float(mid)
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return float(0.5 * (lo + hi))

def _safeguarded_newton_root(q: np.ndarray, y: np.ndarray, left: float, right: float, *, tol: float, max_iter: int) -> float | None:
    f_left = _f_mu(left, q, y)
    f_right = _f_mu(right, q, y)
    if not (np.isfinite(f_left) and np.isfinite(f_right)):
        return _bisection_root(q, y, left, right, tol=tol, max_iter=max_iter)
    if f_left * f_right > 0.0:
        return None
    lo = float(left)
    hi = float(right)
    mu = 0.5 * (lo + hi)
    flo = float(f_left)
    for _ in range(int(max_iter)):
        fmu = _f_mu(mu, q, y)
        if abs(fmu) <= tol:
            return float(mu)
        if flo * fmu <= 0.0:
            hi = mu
        else:
            lo = mu
            flo = fmu
        dfmu = _df_mu(mu, q, y)
        if np.isfinite(dfmu) and dfmu != 0.0:
            mu_new = mu - fmu / dfmu
            if lo < mu_new < hi:
                mu = float(mu_new)
            else:
                mu = 0.5 * (lo + hi)
        else:
            mu = 0.5 * (lo + hi)
        if abs(hi - lo) <= tol * max(1.0, abs(mu)):
            return float(mu)
    return float(0.5 * (lo + hi))

def _solve_outside_mu(q: np.ndarray, y: np.ndarray, *, tol: float, max_iter: int) -> float:
    left = 0.0
    f_left = _f_mu(left, q, y)
    if f_left <= 0.0:
        return 0.0
    right = max(1.0, float(np.max(q)), float(np.linalg.norm(y) * np.max(np.sqrt(q))))
    f_right = _f_mu(right, q, y)
    expansions = 0
    while f_right > 0.0 and expansions < 100:
        right *= 2.0
        f_right = _f_mu(right, q, y)
        expansions += 1
    if f_right > 0.0:
        raise RuntimeError('Failed to bracket outside projected-clearance root.')
    root = _safeguarded_newton_root(q, y, left, right, tol=tol, max_iter=max_iter)
    if root is None:
        raise RuntimeError('Failed to solve outside projected-clearance root.')
    return float(root)

def _candidate_inside_roots(q: np.ndarray, y: np.ndarray, *, tol: float, max_iter: int) -> list[float]:
    roots: list[float] = []
    eps_scale = max(1.0, float(np.max(q)))
    eps = max(1e-14 * eps_scale, tol * eps_scale * 10.0)
    q_unique = np.unique(np.asarray(q, dtype=float))
    q_unique.sort()
    left = -float(q_unique[0]) + eps
    right = -eps
    if left < right:
        root = _safeguarded_newton_root(q, y, left, right, tol=tol, max_iter=max_iter)
        if root is not None:
            roots.append(float(root))
    singulars = sorted([-float(v) for v in q_unique])
    finite_intervals: list[tuple[float, float]] = []
    for idx in range(len(singulars) - 1):
        a = singulars[idx]
        b = singulars[idx + 1]
        left_i = a + eps
        right_i = b - eps
        if left_i < right_i:
            finite_intervals.append((left_i, right_i))
    for left_i, right_i in finite_intervals:
        samples = np.linspace(left_i, right_i, 65)
        values = np.array([_f_mu(float(mu), q, y) for mu in samples], dtype=float)
        for k in range(len(samples) - 1):
            f0 = values[k]
            f1 = values[k + 1]
            if not (np.isfinite(f0) and np.isfinite(f1)):
                continue
            if f0 == 0.0:
                roots.append(float(samples[k]))
            elif f0 * f1 < 0.0:
                root = _bisection_root(q, y, float(samples[k]), float(samples[k + 1]), tol=tol, max_iter=max_iter)
                if root is not None:
                    roots.append(float(root))
    roots_sorted = sorted(roots)
    deduped: list[float] = []
    for root in roots_sorted:
        if len(deduped) == 0 or abs(root - deduped[-1]) > 100.0 * tol * max(1.0, abs(root)):
            deduped.append(root)
    return deduped

def _center_fallback(q: np.ndarray, R: np.ndarray) -> ProjectedClearanceResult3D:
    min_index = int(np.argmin(q))
    rho_min = float(np.sqrt(q[min_index]))
    z_tilde = np.zeros_like(q, dtype=float)
    z_tilde[min_index] = rho_min
    z_star = R @ z_tilde
    a_tilde = np.zeros_like(q, dtype=float)
    a_tilde[min_index] = 1.0 / rho_min
    a = R @ a_tilde
    ell = float(np.linalg.norm(a))
    nu = a / ell
    return ProjectedClearanceResult3D(s=float(-rho_min), z_star=np.asarray(z_star, dtype=float), gamma=-1.0, inside=True, a=np.asarray(a, dtype=float), ell=float(ell), nu=np.asarray(nu, dtype=float), mu=None, status='center_fallback')

def compute_projected_signed_clearance(m: np.ndarray, S: np.ndarray, *, tol: float=1e-12, max_iter: int=80, center_tol: float=1e-14) -> ProjectedClearanceResult3D:
    m_array = np.asarray(m, dtype=float)
    if m_array.ndim != 1 or m_array.shape[0] not in (1, 2, 3):
        raise ValueError(f'Expected m.shape == (k,) with k in {{1,2,3}}, got {m_array.shape}.')
    dim = int(m_array.shape[0])
    S_array = _validate_matrix('S', S, dim)
    eigvals, R = np.linalg.eigh(S_array)
    if np.any(eigvals <= 0.0):
        raise ValueError(f'S must be positive definite; eigenvalues are {eigvals}.')
    q = np.asarray(eigvals, dtype=float)
    y = np.asarray(R.T @ m_array, dtype=float)
    gamma = float(np.sum(y * y / q) - 1.0)
    if np.linalg.norm(y) <= float(center_tol):
        return _center_fallback(q=q, R=R)
    if gamma > 0.0:
        mu = _solve_outside_mu(q, y, tol=float(tol), max_iter=int(max_iter))
        status: ClearanceStatus3D = 'outside'
        inside = False
    elif gamma < 0.0:
        roots = _candidate_inside_roots(q, y, tol=float(tol), max_iter=int(max_iter))
        if len(roots) == 0:
            return _center_fallback(q=q, R=R)
        best_mu = None
        best_dist = float('inf')
        for candidate_mu in roots:
            denom = q + candidate_mu
            if np.any(np.abs(denom) < 1e-300):
                continue
            z_tilde_candidate = candidate_mu * y / denom
            dist_candidate = float(np.linalg.norm(z_tilde_candidate))
            if dist_candidate < best_dist:
                best_dist = dist_candidate
                best_mu = float(candidate_mu)
        if best_mu is None:
            return _center_fallback(q=q, R=R)
        mu = best_mu
        status = 'inside'
        inside = True
    else:
        mu = 0.0
        status = 'tangent'
        inside = False
    denom = q + float(mu)
    z_tilde = float(mu) * y / denom
    z_star = np.asarray(R @ z_tilde, dtype=float)
    u = z_star - m_array
    a = np.linalg.solve(S_array, u)
    ell = float(np.linalg.norm(a))
    if ell <= 0.0:
        return _center_fallback(q=q, R=R)
    nu = a / ell
    unsigned_distance = float(np.linalg.norm(z_star))
    s = -unsigned_distance if inside else unsigned_distance
    return ProjectedClearanceResult3D(s=float(s), z_star=z_star, gamma=float(gamma), inside=bool(inside), a=np.asarray(a, dtype=float), ell=float(ell), nu=np.asarray(nu, dtype=float), mu=None if status == 'tangent' else float(mu), status=status)
