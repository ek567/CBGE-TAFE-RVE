from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _empirical_cdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = x.size
    if n == 0:
        return (x, x)
    y = np.arange(1, n + 1, dtype=float) / float(n)
    return (x, y)

def plot_nnd_pdfs(stats_list: list[dict[str, Any]], out_path: str | Path, *, grid_points: int=300, quantile_clip: tuple[float, float]=(0.005, 0.995), bw_method: str | float='scott') -> None:
    _ensure_parent(out_path)
    deltas_keys = ['delta1', 'delta2', 'delta3']
    pooled_scaled: dict[str, np.ndarray] = {}
    for k in deltas_keys:
        arrs = []
        for s in stats_list:
            ell0 = float(s['deq_mean'])
            data = np.asarray(s['nnd'][k], dtype=float) / ell0
            data = data[np.isfinite(data)]
            if data.size > 0:
                arrs.append(data)
        pooled_scaled[k] = np.concatenate(arrs) if len(arrs) > 0 else np.array([], dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2), dpi=150)
    for k in deltas_keys:
        pooled = pooled_scaled[k]
        if pooled.size < 2:
            continue
        qlo, qhi = np.quantile(pooled, quantile_clip)
        x_min = float(qlo)
        x_max = float(qhi)
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min = float(np.min(pooled))
            x_max = float(np.max(pooled))
        x = np.linspace(x_min, x_max, int(grid_points))
        ys = []
        for s in stats_list:
            ell0 = float(s['deq_mean'])
            data = np.asarray(s['nnd'][k], dtype=float) / ell0
            data = data[np.isfinite(data)]
            if data.size < 2:
                continue
            kde = gaussian_kde(data, bw_method=bw_method)
            ys.append(kde(x))
        if len(ys) == 0:
            continue
        ys = np.vstack(ys)
        y_mean = np.mean(ys, axis=0)
        y_std = np.std(ys, axis=0, ddof=1) if ys.shape[0] > 1 else np.zeros_like(y_mean)
        ax.plot(x, y_mean, label=f'{k} PDF')
        lo = np.maximum(0.0, y_mean - y_std)
        hi = y_mean + y_std
        ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_xlabel('Nearest-neighbor distance $\\delta / \\bar d_{eq}$')
    ax.set_ylabel('PDF (KDE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_nnoa_cdfs(stats_list: list[dict[str, Any]], out_path: str | Path, *, degrees: bool=True) -> None:
    _ensure_parent(out_path)
    theta = np.concatenate([np.asarray(s['nnoa']['theta'], dtype=float) for s in stats_list])
    phi = np.concatenate([np.asarray(s['nnoa']['phi'], dtype=float) for s in stats_list])
    theta = theta[np.isfinite(theta)]
    phi = phi[np.isfinite(phi)]
    theta = np.clip(theta, 0.0, np.pi)
    phi = np.mod(phi, 2.0 * np.pi)
    th_x, th_y = _empirical_cdf(theta)
    ph_x, ph_y = _empirical_cdf(phi)
    if degrees:
        th_x_plot = th_x * 180.0 / np.pi
        ph_x_plot = ph_x * 180.0 / np.pi
    else:
        th_x_plot = th_x
        ph_x_plot = ph_x
    th_grid = np.linspace(0.0, np.pi, 400)
    F_theta = 0.5 * (1.0 - np.cos(th_grid))
    ph_grid = np.linspace(0.0, 2.0 * np.pi, 400)
    F_phi = ph_grid / (2.0 * np.pi)
    if degrees:
        th_grid_plot = th_grid * 180.0 / np.pi
        ph_grid_plot = ph_grid * 180.0 / np.pi
    else:
        th_grid_plot = th_grid
        ph_grid_plot = ph_grid
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    ax = axes[0]
    ax.plot(th_x_plot, th_y, label='Empirical CDF (theta)')
    ax.plot(th_grid_plot, F_theta, 'k--', label='Theory $F(\\theta)=(1-\\cos\\theta)/2$')
    ax.set_xlabel('$\\theta$ (deg)' if degrees else '$\\theta$ (rad)')
    ax.set_ylabel('CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(ph_x_plot, ph_y, label='Empirical CDF (phi)')
    if degrees:
        ax.plot(ph_grid_plot, F_phi, 'k--', label='Theory $F(\\phi)=\\phi/360^\\circ$')
    else:
        ax.plot(ph_grid_plot, F_phi, 'k--', label='Theory $F(\\phi)=\\phi/2\\pi$')
    ax.set_xlabel('$\\phi$ (deg)' if degrees else '$\\phi$ (rad)')
    ax.set_ylabel('CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_ripley_and_rdf(stats_list: list[dict[str, Any]], out_path: str | Path) -> None:
    _ensure_parent(out_path)
    R_scaled = np.asarray(stats_list[0]['ripley_scaled']['R_scaled'], dtype=float)
    K_csr = np.asarray(stats_list[0]['ripley_scaled']['K_tilde_csr'], dtype=float)
    r_scaled = np.asarray(stats_list[0]['rdf_scaled']['r_scaled'], dtype=float)
    g_csr = np.ones_like(r_scaled)
    K_all = np.vstack([np.asarray(s['ripley_scaled']['K_tilde'], dtype=float) for s in stats_list])
    g_all = np.vstack([np.asarray(s['rdf_scaled']['g'], dtype=float) for s in stats_list])
    K_mean = np.mean(K_all, axis=0)
    K_std = np.std(K_all, axis=0, ddof=1) if K_all.shape[0] > 1 else np.zeros_like(K_mean)
    g_mean = np.mean(g_all, axis=0)
    g_std = np.std(g_all, axis=0, ddof=1) if g_all.shape[0] > 1 else np.zeros_like(g_mean)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    ax = axes[0]
    ax.plot(R_scaled, K_mean, label='$\\tilde K(R)$ mean')
    ax.fill_between(R_scaled, np.maximum(0.0, K_mean - K_std), K_mean + K_std, alpha=0.2)
    ax.plot(R_scaled, K_csr, 'k--', label='CSR $\\tilde K_{CSR}(R)=\\frac{4}{3}\\pi R^3$')
    ax.set_xlabel('$R / \\bar d_{eq}$')
    ax.set_ylabel('$\\tilde K(R)$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(r_scaled, g_mean, label='$g(r)$ mean')
    ax.fill_between(r_scaled, np.maximum(0.0, g_mean - g_std), g_mean + g_std, alpha=0.2)
    ax.plot(r_scaled, g_csr, 'k--', label='CSR $g(r)=1$')
    ax.set_xlabel('$r / \\bar d_{eq}$')
    ax.set_ylabel('$g(r)$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_orientation_isotropy_boxplot(stats_list: list[dict[str, Any]], out_path: str | Path) -> None:
    _ensure_parent(out_path)
    vals = [float(s['orientation_isotropy']['max_abs_deviation']) for s in stats_list]
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2), dpi=150)
    ax.boxplot(vals, vert=True, widths=0.5)
    ax.set_ylabel('max $|\\lambda(A)-1/3|$')
    ax.set_xticks([1])
    ax.set_xticklabels(['realizations'])
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
