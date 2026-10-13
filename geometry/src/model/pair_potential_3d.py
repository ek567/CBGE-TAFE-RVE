from __future__ import annotations

def pair_overlap_gap(F_ij: float, threshold: float=1.0) -> float:
    F_value = float(F_ij)
    threshold_value = float(threshold)
    return max(0.0, threshold_value - F_value)

def pair_potential(F_ij: float, threshold: float=1.0) -> float:
    overlap = pair_overlap_gap(F_ij=F_ij, threshold=threshold)
    return 0.5 * overlap ** 2
