from __future__ import annotations
import numpy as np
from src.data.pair_result_3d import PairContactResult3D
from src.data.particle_3d import EllipsoidParticle3D
from src.model.pair_potential_3d import pair_overlap_gap, pair_potential
from src.model.q_matrix_3d import build_q_sym_closed_form, build_q_sym_pw85
try:
    import pypw85 as _pypw85
except ImportError:
    _pypw85 = None

def compute_pair_contact_measure(r_ij: np.ndarray, q_i: np.ndarray, q_j: np.ndarray, out: np.ndarray | None=None, *, validate_shapes: bool=True) -> tuple[float, float]:
    r_ij_array = np.asarray(r_ij, dtype=float)
    q_i_array = np.asarray(q_i, dtype=float)
    q_j_array = np.asarray(q_j, dtype=float)
    if validate_shapes:
        if r_ij_array.shape != (3,):
            raise ValueError(f'Expected r_ij.shape == (3,), got {r_ij_array.shape}.')
        if q_i_array.shape != (6,):
            raise ValueError(f'Expected q_i.shape == (6,), got {q_i_array.shape}.')
        if q_j_array.shape != (6,):
            raise ValueError(f'Expected q_j.shape == (6,), got {q_j_array.shape}.')
    if _pypw85 is None:
        raise ImportError('pypw85 is required for compute_pair_contact_measure but is not available.')
    if out is None:
        out = np.empty(2, dtype=float)
    else:
        out = np.asarray(out, dtype=float)
        if validate_shapes and out.shape != (2,):
            raise ValueError(f'Expected out.shape == (2,), got {out.shape}.')
    try:
        _pypw85.contact_function(r_ij_array, q_i_array, q_j_array, out)
    except Exception as exc:
        raise RuntimeError('pypw85.contact_function failed.') from exc
    return (float(out[0]), float(out[1]))

def evaluate_pair_contact(particle_i: EllipsoidParticle3D, particle_j: EllipsoidParticle3D, use_pw85_q: bool=True) -> PairContactResult3D:
    r_ij = np.asarray(particle_j.c - particle_i.c, dtype=float)
    if r_ij.shape != (3,):
        raise ValueError(f'Expected r_ij.shape == (3,), got {r_ij.shape}.')
    if use_pw85_q:
        q_i = build_q_sym_pw85(a=particle_i.a, b=particle_i.b, p=particle_i.p)
        q_j = build_q_sym_pw85(a=particle_j.a, b=particle_j.b, p=particle_j.p)
    else:
        q_i = build_q_sym_closed_form(a=particle_i.a, b=particle_i.b, p=particle_i.p)
        q_j = build_q_sym_closed_form(a=particle_j.a, b=particle_j.b, p=particle_j.p)
    F_ij, lambda_star = compute_pair_contact_measure(r_ij=r_ij, q_i=q_i, q_j=q_j)
    overlap = pair_overlap_gap(F_ij=F_ij, threshold=1.0)
    psi_ij = pair_potential(F_ij=F_ij, threshold=1.0)
    return PairContactResult3D(i=-1, j=-1, r_ij=r_ij, F_ij=F_ij, lambda_star=lambda_star, overlap=overlap, psi_ij=psi_ij, Q_star=None, s_ij=None, x0_ij=None)
