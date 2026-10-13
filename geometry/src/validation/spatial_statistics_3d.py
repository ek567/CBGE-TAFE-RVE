from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.spatial import cKDTree
from src.data.particle_3d import EllipsoidParticle3D
from src.periodic.mic_3d import minimum_image_vector
from src.validation.orientation_isotropy_check_3d import check_orientation_isotropy

@dataclass(frozen=True)
class SpatialStatsMVPConfig3D:
    box: np.ndarray
    use_mic: bool = True
    nnd_kmax: int = 3
    nnoa_rank: int = 1
    nnoa_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
    r_max_factor: float = 0.5
    n_r_bins: int = 120
    n_K_points: int = 60
    orientation_tol: float = 0.03
    eps: float = 1e-12

def _centers_array(particles: list[EllipsoidParticle3D]) -> np.ndarray:
    c = np.asarray([p.c for p in particles], dtype=float)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f'Invalid centers array shape {c.shape}')
    return c

def _build_periodic_tree(centers: np.ndarray, box: np.ndarray) -> cKDTree:
    return cKDTree(centers, boxsize=box)

def compute_deq_mean(particles: list[EllipsoidParticle3D]) -> float:
    if len(particles) == 0:
        raise ValueError('Need at least 1 particle to compute deq_mean.')
    deqs = np.array([2.0 * (p.a * p.b ** 2) ** (1.0 / 3.0) for p in particles], dtype=float)
    return float(np.mean(deqs))

def compute_nnd_samples_periodic(centers: np.ndarray, box: np.ndarray, kmax: int=3) -> dict[str, np.ndarray]:
    if kmax < 1:
        raise ValueError('kmax must be >= 1')
    tree = _build_periodic_tree(centers, box)
    dists, idxs = tree.query(centers, k=kmax + 1)
    out: dict[str, np.ndarray] = {}
    for k in range(1, kmax + 1):
        out[f'delta{k}'] = np.asarray(dists[:, k], dtype=float)
    return out

def compute_nnoa_polar_azimuth_periodic(centers: np.ndarray, box: np.ndarray, rank: int=1, axis: np.ndarray | None=None, eps: float=1e-12) -> dict[str, np.ndarray]:
    if rank < 1:
        raise ValueError('rank must be >= 1 (0 is self).')
    axis_v = np.array([0.0, 0.0, 1.0], dtype=float) if axis is None else np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis_v))
    if axis_norm == 0.0:
        raise ValueError('axis must be non-zero.')
    axis_v = axis_v / axis_norm
    tmp = np.array([1.0, 0.0, 0.0], dtype=float) if abs(axis_v[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = np.cross(tmp, axis_v)
    e1n = float(np.linalg.norm(e1))
    if e1n == 0.0:
        tmp = np.array([0.0, 0.0, 1.0], dtype=float)
        e1 = np.cross(tmp, axis_v)
        e1n = float(np.linalg.norm(e1))
        if e1n == 0.0:
            raise ValueError('Failed to construct azimuth basis (axis degeneracy).')
    e1 = e1 / e1n
    e2 = np.cross(axis_v, e1)
    tree = _build_periodic_tree(centers, box)
    dists, idxs = tree.query(centers, k=rank + 1)
    j_idx = idxs[:, rank]
    theta = np.empty(centers.shape[0], dtype=float)
    phi = np.empty(centers.shape[0], dtype=float)
    for i in range(centers.shape[0]):
        j = int(j_idx[i])
        rij = minimum_image_vector(centers[i], centers[j], box)
        norm = float(np.linalg.norm(rij))
        if norm <= eps:
            theta[i] = 0.0
            phi[i] = 0.0
            continue
        u = rij / norm
        cosv = float(np.clip(np.dot(u, axis_v), -1.0, 1.0))
        theta[i] = float(np.arccos(cosv))
        x = float(np.dot(u, e1))
        y = float(np.dot(u, e2))
        if abs(x) <= eps and abs(y) <= eps:
            phi[i] = 0.0
        else:
            ang = float(np.arctan2(y, x))
            if ang < 0.0:
                ang += 2.0 * np.pi
            phi[i] = ang
    return {'theta': theta, 'phi': phi}

def compute_rdf_and_ripley_periodic(centers: np.ndarray, box: np.ndarray, r_max: float, n_r_bins: int, n_K_points: int, eps: float=1e-12) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    N = centers.shape[0]
    V_box = float(np.prod(box))
    if N < 2:
        raise ValueError('Need at least 2 points for RDF/K.')
    tree = _build_periodic_tree(centers, box)
    r_max_eff = float(r_max * (1.0 - eps))
    r_edges = np.linspace(0.0, r_max_eff, int(n_r_bins) + 1, dtype=float)
    r1 = r_edges[:-1]
    r2 = r_edges[1:]
    r_centers = 0.5 * (r1 + r2)
    R_vals = np.linspace(0.0, r_max_eff, int(n_K_points), dtype=float)
    ordered_cum_r = np.asarray(tree.count_neighbors(tree, r_edges, cumulative=True), dtype=float)
    ordered_cum_R = np.asarray(tree.count_neighbors(tree, R_vals, cumulative=True), dtype=float)
    ordered_cum_r_excl_self = ordered_cum_r - float(N)
    ordered_cum_R_excl_self = ordered_cum_R - float(N)
    unordered_cum_r = 0.5 * ordered_cum_r_excl_self
    unordered_cum_R = 0.5 * ordered_cum_R_excl_self
    shell_pairs = np.diff(unordered_cum_r)
    shell_pairs = np.maximum(shell_pairs, 0.0)
    shell_vol = 4.0 / 3.0 * np.pi * (r2 ** 3 - r1 ** 3)
    g = 2.0 * V_box * shell_pairs / (float(N) * float(N - 1) * shell_vol)
    g = np.where(shell_vol > 0.0, g, 0.0)
    rdf = {'r_edges': r_edges, 'r_centers': r_centers, 'g': g, 'g_csr': np.ones_like(g), 'r_max': np.array([r_max_eff], dtype=float)}
    ordered_within_R_excl_self = 2.0 * unordered_cum_R
    K = V_box / (float(N) * float(N - 1)) * ordered_within_R_excl_self
    K_csr = 4.0 / 3.0 * np.pi * R_vals ** 3
    ripley = {'R': R_vals, 'K': K, 'K_csr': K_csr, 'r_max': np.array([r_max_eff], dtype=float)}
    return (rdf, ripley)

def compute_spatial_statistics_mvp_3d(particles: list[EllipsoidParticle3D], config: SpatialStatsMVPConfig3D) -> dict[str, Any]:
    box = np.asarray(config.box, dtype=float).reshape(3)
    centers = _centers_array(particles)
    N = int(centers.shape[0])
    V_box = float(np.prod(box))
    vols = np.array([4.0 / 3.0 * np.pi * p.a * p.b ** 2 for p in particles], dtype=float)
    vf_global = float(np.sum(vols) / V_box)
    deq_mean = compute_deq_mean(particles)
    ell0 = float(deq_mean)
    nnd = compute_nnd_samples_periodic(centers, box, kmax=config.nnd_kmax)
    nnoa = compute_nnoa_polar_azimuth_periodic(centers=centers, box=box, rank=int(config.nnoa_rank), axis=config.nnoa_axis, eps=float(config.eps))
    r_max = float(config.r_max_factor * float(np.min(box)))
    rdf, ripley = compute_rdf_and_ripley_periodic(centers=centers, box=box, r_max=r_max, n_r_bins=int(config.n_r_bins), n_K_points=int(config.n_K_points), eps=float(config.eps))
    rdf_scaled = {'r_scaled': (rdf['r_centers'] / ell0).astype(float), 'g': rdf['g'].astype(float), 'g_csr': rdf['g_csr'].astype(float)}
    ripley_scaled = {'R_scaled': (ripley['R'] / ell0).astype(float), 'K_tilde': (ripley['K'] / ell0 ** 3).astype(float), 'K_tilde_csr': (ripley['K_csr'] / ell0 ** 3).astype(float)}
    ori = check_orientation_isotropy(particles, tolerance=float(config.orientation_tol))
    return {'box': box.tolist(), 'N': N, 'vf_global': vf_global, 'deq_mean': float(deq_mean), 'nnd': {k: v.tolist() for k, v in nnd.items()}, 'nnoa': {k: v.tolist() for k, v in nnoa.items()}, 'rdf': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in rdf.items()}, 'ripley': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ripley.items()}, 'rdf_scaled': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in rdf_scaled.items()}, 'ripley_scaled': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ripley_scaled.items()}, 'orientation_isotropy': {'orientation_tensor': ori['orientation_tensor'].tolist(), 'eigenvalues': [float(x) for x in ori['eigenvalues']], 'max_abs_deviation': float(ori['max_abs_deviation']), 'is_isotropic': bool(ori['is_isotropic']), 'tolerance': float(config.orientation_tol)}}
