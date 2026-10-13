from __future__ import annotations
import csv
import json
import datetime as _dt
from pathlib import Path
from typing import Any
from src.monitoring.metrics_3d import RunMetrics3D, StepMetrics3D

def _utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'

def step_metrics_to_row(step_metrics: StepMetrics3D) -> dict[str, object]:
    if not isinstance(step_metrics, StepMetrics3D):
        raise ValueError(f'step_metrics must be StepMetrics3D, got {type(step_metrics)!r}.')
    return step_metrics.to_dict()

def run_metrics_to_dict(run_metrics: RunMetrics3D) -> dict[str, object]:
    if not isinstance(run_metrics, RunMetrics3D):
        raise ValueError(f'run_metrics must be RunMetrics3D, got {type(run_metrics)!r}.')
    return run_metrics.to_dict()

def _attach_export_context(data: dict[str, Any], *, metadata: dict[str, Any] | None=None, constraints: dict[str, Any] | None=None) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError('data must be a dict.')
    out = dict(data)
    md: dict[str, Any] = {}
    existing_md = out.get('metadata', None)
    if isinstance(existing_md, dict):
        md.update(existing_md)
    md.setdefault('export_time_utc', _utc_iso())
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise TypeError(f'metadata must be dict or None, got {type(metadata).__name__}.')
        md.update(metadata)
    out['metadata'] = md
    if constraints is not None:
        if not isinstance(constraints, dict):
            raise TypeError(f'constraints must be dict or None, got {type(constraints).__name__}.')
        out['constraints'] = constraints
    return out

def save_run_metrics_json(run_metrics: RunMetrics3D, path: str | Path, *, metadata: dict[str, Any] | None=None, constraints: dict[str, Any] | None=None) -> None:
    if not isinstance(run_metrics, RunMetrics3D):
        raise ValueError(f'run_metrics must be RunMetrics3D, got {type(run_metrics)!r}.')
    output_path = Path(path)
    data = run_metrics_to_dict(run_metrics)
    data2 = _attach_export_context(data, metadata=metadata, constraints=constraints)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data2, f, ensure_ascii=False, indent=2)

def save_run_summary_json(run_metrics: RunMetrics3D, path: str | Path, *, metadata: dict[str, Any] | None=None, constraints: dict[str, Any] | None=None) -> None:
    if not isinstance(run_metrics, RunMetrics3D):
        raise ValueError(f'run_metrics must be RunMetrics3D, got {type(run_metrics)!r}.')
    output_path = Path(path)
    data = run_metrics.summary_dict()
    if not isinstance(data, dict):
        raise TypeError('run_metrics.summary_dict() must return a dict.')
    data2 = _attach_export_context(data, metadata=metadata, constraints=constraints)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data2, f, ensure_ascii=False, indent=2)

def save_step_metrics_csv(run_metrics: RunMetrics3D, path: str | Path) -> None:
    if not isinstance(run_metrics, RunMetrics3D):
        raise ValueError(f'run_metrics must be RunMetrics3D, got {type(run_metrics)!r}.')
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [step_metrics_to_row(step) for step in run_metrics.steps]
    fieldnames: list[str]
    if len(rows) == 0:
        fieldnames = ['step_index', 'num_particles', 'num_all_pairs', 'num_candidate_pairs', 'num_pw_calls', 'num_overlap_pairs', 'energy_before', 'energy_after', 'grad_norm', 'max_overlap', 'num_overlapping_pairs', 'step_alpha', 'line_search_trials', 'line_search_energy_evals', 'line_search_success', 'time_total_step', 'time_pair_generation', 'time_broadphase', 'time_neighbor_build', 'time_pw_eval', 'time_geometry_attach', 'time_gradient_eval', 'time_line_search', 'time_system_status', 'time_strict_audit', 'candidate_ratio', 'pw_call_ratio', 'energy_drop']
    else:
        fieldnames = list(rows[0].keys())
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
