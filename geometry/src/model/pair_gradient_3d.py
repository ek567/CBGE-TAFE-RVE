from __future__ import annotations
import numpy as np
from src.data.pair_result_3d import PairContactResult3D
from src.data.particle_3d import EllipsoidParticle3D
from src.model.pair_contact_geometry_3d import reconstruct_pair_geometry
from src.model.pair_contact_measure_pw_3d import evaluate_pair_contact
from src.model.pair_potential_3d import pair_potential
from src.utils.math_utils import normalize, safe_dot, tangent_projector

def _zero_vec3() -> np.ndarray:
    return np.zeros(3, dtype=float)

def _get_pair_s_ij(particle_i: EllipsoidParticle3D, particle_j: EllipsoidParticle3D, pair_result: PairContactResult3D) -> np.ndarray:
    if pair_result.s_ij is not None:
        s_ij = np.asarray(pair_result.s_ij, dtype=float)
        if s_ij.shape != (3,):
            raise ValueError(f'Expected s_ij.shape == (3,), got {s_ij.shape}.')
        return s_ij
    Q_star, s_ij, _ = reconstruct_pair_geometry(particle_i=particle_i, particle_j=particle_j, lambda_star=pair_result.lambda_star, r_ij=pair_result.r_ij)
    if Q_star.shape != (3, 3):
        raise ValueError(f'Expected Q_star.shape == (3, 3), got {Q_star.shape}.')
    if s_ij.shape != (3,):
        raise ValueError(f'Expected s_ij.shape == (3,), got {s_ij.shape}.')
    return s_ij

def pair_position_gradients(particle_i: EllipsoidParticle3D, particle_j: EllipsoidParticle3D, pair_result: PairContactResult3D) -> tuple[np.ndarray, np.ndarray]:
    if pair_result.r_ij.shape != (3,):
        raise ValueError(f'Expected r_ij.shape == (3,), got {pair_result.r_ij.shape}.')
    if pair_result.F_ij >= 1.0:
        return (_zero_vec3(), _zero_vec3())
    s_ij = _get_pair_s_ij(particle_i=particle_i, particle_j=particle_j, pair_result=pair_result)
    prefactor = 2.0 * (1.0 - float(pair_result.F_ij)) * float(pair_result.lambda_star) * (1.0 - float(pair_result.lambda_star))
    grad_c_i = prefactor * s_ij
    grad_c_j = -grad_c_i
    if grad_c_i.shape != (3,):
        raise ValueError(f'Expected grad_c_i.shape == (3,), got {grad_c_i.shape}.')
    if grad_c_j.shape != (3,):
        raise ValueError(f'Expected grad_c_j.shape == (3,), got {grad_c_j.shape}.')
    return (grad_c_i, grad_c_j)

def pair_orientation_gradient_analytic(particle_i: EllipsoidParticle3D, particle_j: EllipsoidParticle3D, pair_result: PairContactResult3D) -> tuple[np.ndarray, np.ndarray]:
    if pair_result.r_ij.shape != (3,):
        raise ValueError(f'Expected r_ij.shape == (3,), got {pair_result.r_ij.shape}.')
    if pair_result.F_ij >= 1.0:
        return (_zero_vec3(), _zero_vec3())
    s_ij = _get_pair_s_ij(particle_i=particle_i, particle_j=particle_j, pair_result=pair_result)
    p_i = normalize(particle_i.p)
    p_j = normalize(particle_j.p)
    projector_i = tangent_projector(p_i)
    projector_j = tangent_projector(p_j)
    s_dot_p_i = safe_dot(s_ij, p_i)
    s_dot_p_j = safe_dot(s_ij, p_j)
    lambda_star = float(pair_result.lambda_star)
    one_minus_lambda = 1.0 - lambda_star
    one_minus_F = 1.0 - float(pair_result.F_ij)
    anisotropy_i = float(particle_i.a ** 2 - particle_i.b ** 2)
    anisotropy_j = float(particle_j.a ** 2 - particle_j.b ** 2)
    grad_p_i = 2.0 * one_minus_F * anisotropy_i * lambda_star * one_minus_lambda ** 2 * s_dot_p_i * (projector_i @ s_ij)
    grad_p_j = 2.0 * one_minus_F * anisotropy_j * lambda_star ** 2 * one_minus_lambda * s_dot_p_j * (projector_j @ s_ij)
    if grad_p_i.shape != (3,):
        raise ValueError(f'Expected grad_p_i.shape == (3,), got {grad_p_i.shape}.')
    if grad_p_j.shape != (3,):
        raise ValueError(f'Expected grad_p_j.shape == (3,), got {grad_p_j.shape}.')
    return (grad_p_i, grad_p_j)

def _pair_potential_for_modified_orientations(particle_i: EllipsoidParticle3D, particle_j: EllipsoidParticle3D) -> float:
    pair_result = evaluate_pair_contact(particle_i=particle_i, particle_j=particle_j, use_pw85_q=True)
    return pair_potential(pair_result.F_ij, threshold=1.0)

def pair_orientation_gradient_fd(particle_i: EllipsoidParticle3D, particle_j: EllipsoidParticle3D, eps: float=1e-06) -> tuple[np.ndarray, np.ndarray]:
    eps_value = float(eps)
    if eps_value <= 0.0:
        raise ValueError(f'Expected eps > 0, got {eps_value}.')
    if particle_i.p.shape != (3,):
        raise ValueError(f'Expected particle_i.p.shape == (3,), got {particle_i.p.shape}.')
    if particle_j.p.shape != (3,):
        raise ValueError(f'Expected particle_j.p.shape == (3,), got {particle_j.p.shape}.')
    grad_p_i = np.zeros(3, dtype=float)
    grad_p_j = np.zeros(3, dtype=float)
    basis = np.eye(3, dtype=float)
    for k in range(3):
        e_k = basis[k]
        particle_i_plus = particle_i.clone()
        particle_i_minus = particle_i.clone()
        particle_i_plus.p = normalize(particle_i_plus.p + eps_value * e_k)
        particle_i_minus.p = normalize(particle_i_minus.p - eps_value * e_k)
        psi_plus = _pair_potential_for_modified_orientations(particle_i=particle_i_plus, particle_j=particle_j)
        psi_minus = _pair_potential_for_modified_orientations(particle_i=particle_i_minus, particle_j=particle_j)
        grad_p_i[k] = (psi_plus - psi_minus) / (2.0 * eps_value)
    for k in range(3):
        e_k = basis[k]
        particle_j_plus = particle_j.clone()
        particle_j_minus = particle_j.clone()
        particle_j_plus.p = normalize(particle_j_plus.p + eps_value * e_k)
        particle_j_minus.p = normalize(particle_j_minus.p - eps_value * e_k)
        psi_plus = _pair_potential_for_modified_orientations(particle_i=particle_i, particle_j=particle_j_plus)
        psi_minus = _pair_potential_for_modified_orientations(particle_i=particle_i, particle_j=particle_j_minus)
        grad_p_j[k] = (psi_plus - psi_minus) / (2.0 * eps_value)
    if grad_p_i.shape != (3,):
        raise ValueError(f'Expected grad_p_i.shape == (3,), got {grad_p_i.shape}.')
    if grad_p_j.shape != (3,):
        raise ValueError(f'Expected grad_p_j.shape == (3,), got {grad_p_j.shape}.')
    return (grad_p_i, grad_p_j)
