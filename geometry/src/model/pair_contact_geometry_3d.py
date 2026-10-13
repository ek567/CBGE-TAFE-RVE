from __future__ import annotations
import numpy as np
from src.data.pair_result_3d import PairContactResult3D
from src.data.particle_3d import EllipsoidParticle3D
from src.model.q_matrix_3d import build_q_matrix_closed_form

def reconstruct_pair_geometry(particle_i: EllipsoidParticle3D, particle_j: EllipsoidParticle3D, lambda_star: float, r_ij: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_ij_array = np.asarray(r_ij, dtype=float)
    if r_ij_array.shape != (3,):
        raise ValueError(f'Expected r_ij.shape == (3,), got {r_ij_array.shape}.')
    lambda_value = float(lambda_star)
    Q_i = build_q_matrix_closed_form(a=particle_i.a, b=particle_i.b, p=particle_i.p)
    Q_j = build_q_matrix_closed_form(a=particle_j.a, b=particle_j.b, p=particle_j.p)
    Q_star = (1.0 - lambda_value) * Q_i + lambda_value * Q_j
    s_ij = np.linalg.solve(Q_star, r_ij_array)
    x0_ij = np.asarray(particle_i.c + (1.0 - lambda_value) * (Q_i @ s_ij), dtype=float)
    if Q_star.shape != (3, 3):
        raise ValueError(f'Expected Q_star.shape == (3, 3), got {Q_star.shape}.')
    if s_ij.shape != (3,):
        raise ValueError(f'Expected s_ij.shape == (3,), got {s_ij.shape}.')
    if x0_ij.shape != (3,):
        raise ValueError(f'Expected x0_ij.shape == (3,), got {x0_ij.shape}.')
    return (Q_star, s_ij, x0_ij)

def attach_geometry_to_pair_result(particle_i: EllipsoidParticle3D, particle_j: EllipsoidParticle3D, pair_result: PairContactResult3D) -> PairContactResult3D:
    Q_star, s_ij, x0_ij = reconstruct_pair_geometry(particle_i=particle_i, particle_j=particle_j, lambda_star=pair_result.lambda_star, r_ij=pair_result.r_ij)
    return PairContactResult3D(i=pair_result.i, j=pair_result.j, r_ij=pair_result.r_ij.copy(), F_ij=pair_result.F_ij, lambda_star=pair_result.lambda_star, overlap=pair_result.overlap, psi_ij=pair_result.psi_ij, Q_star=Q_star, s_ij=s_ij, x0_ij=x0_ij)

def reconstruct_pair_geometry_from_Q(*, c_i: np.ndarray, Q_i: np.ndarray, Q_j: np.ndarray, lambda_star: float, r_ij: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_ij_array = np.asarray(r_ij, dtype=float)
    if r_ij_array.shape != (3,):
        raise ValueError(f'Expected r_ij.shape == (3,), got {r_ij_array.shape}.')
    lam = float(lambda_star)
    Q_star = (1.0 - lam) * Q_i + lam * Q_j
    s_ij = np.linalg.solve(Q_star, r_ij_array)
    x0_ij = np.asarray(c_i + (1.0 - lam) * (Q_i @ s_ij), dtype=float)
    return (Q_star, s_ij, x0_ij)

def attach_geometry_to_pair_result_from_Q(*, pair_result: PairContactResult3D, c_i: np.ndarray, Q_i: np.ndarray, Q_j: np.ndarray) -> PairContactResult3D:
    Q_star, s_ij, x0_ij = reconstruct_pair_geometry_from_Q(c_i=c_i, Q_i=Q_i, Q_j=Q_j, lambda_star=pair_result.lambda_star, r_ij=pair_result.r_ij)
    return PairContactResult3D(i=pair_result.i, j=pair_result.j, r_ij=pair_result.r_ij.copy(), F_ij=pair_result.F_ij, lambda_star=pair_result.lambda_star, overlap=pair_result.overlap, psi_ij=pair_result.psi_ij, Q_star=Q_star, s_ij=s_ij, x0_ij=x0_ij)
