from __future__ import annotations
from typing import Any
from src.model.total_potential_3d import evaluate_all_pairs, max_overlap_info

def pair_has_overlap(pair_result: Any, overlap_tol: float=1e-06) -> bool:
    tol = float(overlap_tol)
    if tol < 0.0:
        raise ValueError(f'overlap_tol must be >= 0, got {tol}.')
    if not hasattr(pair_result, 'overlap'):
        raise ValueError("pair_result must have an 'overlap' attribute.")
    return bool(float(pair_result.overlap) > tol)

def find_overlapping_pairs(pair_results: list[Any], overlap_tol: float=1e-06) -> list[tuple[int, int]]:
    tol = float(overlap_tol)
    if tol < 0.0:
        raise ValueError(f'overlap_tol must be >= 0, got {tol}.')
    if not isinstance(pair_results, list):
        raise ValueError('pair_results must be a list.')
    overlapping_pairs: list[tuple[int, int]] = []
    for idx, pair_result in enumerate(pair_results):
        if not hasattr(pair_result, 'i') or not hasattr(pair_result, 'j'):
            raise ValueError(f"pair_results[{idx}] must have 'i' and 'j' attributes.")
        if pair_has_overlap(pair_result, overlap_tol=tol):
            overlapping_pairs.append((int(pair_result.i), int(pair_result.j)))
    return overlapping_pairs

def system_overlap_status_from_pair_results(pair_results: list[Any], overlap_tol: float=1e-06) -> dict[str, Any]:
    tol = float(overlap_tol)
    if tol < 0.0:
        raise ValueError(f'overlap_tol must be >= 0, got {tol}.')
    if not isinstance(pair_results, list):
        raise ValueError('pair_results must be a list.')
    overlap_pair_results = [pair_result for pair_result in pair_results if pair_has_overlap(pair_result, overlap_tol=tol)]
    overlap_pairs = [(int(pair_result.i), int(pair_result.j)) for pair_result in overlap_pair_results]
    max_info = max_overlap_info(pair_results)
    max_overlap = float(max_info['max_overlap'])
    return {'has_overlap': bool(len(overlap_pair_results) > 0), 'max_overlap': max_overlap, 'max_overlap_pair': max_info['pair_indices'], 'num_overlapping_pairs': len(overlap_pair_results), 'overlap_pairs': overlap_pairs, 'overlap_pair_results': overlap_pair_results, 'pair_results': pair_results, 'max_info': max_info}

def system_overlap_status(particles, box=None, use_mic: bool=False, overlap_tol: float=1e-06):
    pair_results = evaluate_all_pairs(particles=particles, box=box, use_mic=use_mic, attach_geometry=True)
    return system_overlap_status_from_pair_results(pair_results=pair_results, overlap_tol=overlap_tol)
