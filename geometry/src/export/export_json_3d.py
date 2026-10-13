from __future__ import annotations
import json
import datetime as _dt
from typing import Any, Iterable, Optional
import numpy as np
from src.data.particle_3d import EllipsoidParticle3D
from src.periodic.mic_3d import wrap_position
from src.utils.math_utils import normalize
EXPORT_SCHEMA = 'pw85-ellipsoids-export'
EXPORT_SCHEMA_VERSION = '1.1'

def _utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'

def _vec3_to_list(v: np.ndarray | Iterable[float]) -> list[float]:
    a = np.asarray(v, dtype=float).reshape(3)
    return [float(a[0]), float(a[1]), float(a[2])]

def _ensure_particle_id_list(particles: list[EllipsoidParticle3D], *, id_policy: str='preserve_or_assign_1_to_N') -> tuple[list[int], dict[str, Any]]:
    ids = [p.id for p in particles]
    all_have_id = all((x is not None for x in ids))
    if all_have_id:
        out_ids = [int(x) for x in ids]
        return (out_ids, {'id_policy': 'preserve'})
    if id_policy == 'preserve_or_assign_1_to_N':
        out_ids = list(range(1, len(particles) + 1))
        return (out_ids, {'id_policy': 'assigned_1_to_N'})
    elif id_policy == 'preserve_or_assign_0_to_Nm1':
        out_ids = list(range(0, len(particles)))
        return (out_ids, {'id_policy': 'assigned_0_to_Nm1'})
    else:
        raise ValueError(f'Unknown id_policy: {id_policy}')

def build_export_dict_3d(particles: list[EllipsoidParticle3D], *, box: tuple[float, float, float] | None, periodic: bool, boundary_wrap: bool=True, normalize_p: bool=True, id_policy: str='preserve_or_assign_1_to_N', metadata: Optional[dict[str, Any]]=None, constraints: Optional[dict[str, Any]]=None, optimization_summary: Optional[dict[str, Any]]=None) -> dict[str, Any]:
    if periodic and box is None:
        raise ValueError('periodic=True requires box != None')
    if box is not None:
        Lx, Ly, Lz = (float(box[0]), float(box[1]), float(box[2]))
        if min(Lx, Ly, Lz) <= 0:
            raise ValueError(f'Invalid box: {box}')
        box_out = [Lx, Ly, Lz]
    else:
        box_out = None
    ids, id_meta = _ensure_particle_id_list(particles, id_policy=id_policy)
    particle_dicts: list[dict[str, Any]] = []
    for pid, p in zip(ids, particles):
        a = float(p.a)
        b = float(p.b)
        if not a >= b > 0:
            raise ValueError(f'Invalid axes for particle id={pid}: a={a}, b={b}')
        c = np.asarray(p.c, dtype=float).reshape(3)
        if periodic and boundary_wrap:
            c = np.asarray(wrap_position(c, np.asarray(box_out, dtype=float)), dtype=float)
        pdir = np.asarray(p.p, dtype=float).reshape(3)
        if normalize_p:
            pdir = np.asarray(normalize(pdir), dtype=float)
        particle_dicts.append({'id': int(pid), 'a': a, 'b': b, 'rho': a / b, 'd_eq': 2.0 * float(np.cbrt(a * b * b)), 'c': _vec3_to_list(c), 'p': _vec3_to_list(pdir)})
    out: dict[str, Any] = {'schema': EXPORT_SCHEMA, 'schema_version': EXPORT_SCHEMA_VERSION, 'periodic': bool(periodic), 'box': box_out, 'particles': particle_dicts, 'physical_summary': {'N': len(particle_dicts), 'total_particle_volume': float(sum((4.0 * np.pi / 3.0 * p.a * p.b * p.b for p in particles))), 'Vf': float(sum((4.0 * np.pi / 3.0 * p.a * p.b * p.b for p in particles))) / float(np.prod(box_out)) if box_out is not None else None}}
    md: dict[str, Any] = {}
    md.update(id_meta)
    md.update({'source_project': 'pw85-optimizer-3d', 'export_time_utc': _utc_iso(), 'wrap_policy': 'wrap_to_[0,L)' if periodic and boundary_wrap else 'none', 'p_policy': 'normalized' if normalize_p else 'as_is'})
    if metadata:
        md.update(metadata)
    if constraints is not None:
        if not isinstance(constraints, dict):
            raise TypeError(f'constraints must be dict or None, got {type(constraints).__name__}.')
        out['constraints'] = constraints
        md['constraints'] = constraints
    out['metadata'] = md
    if optimization_summary is not None:
        out['optimization_summary'] = optimization_summary
    return out

def export_particles_json_3d(particles: list[EllipsoidParticle3D], path: str, *, box: tuple[float, float, float] | None, periodic: bool=True, boundary_wrap: bool=True, normalize_p: bool=True, id_policy: str='preserve_or_assign_1_to_N', metadata: Optional[dict[str, Any]]=None, constraints: Optional[dict[str, Any]]=None, optimization_summary: Optional[dict[str, Any]]=None, indent: int=2) -> dict[str, Any]:
    data = build_export_dict_3d(particles, box=box, periodic=periodic, boundary_wrap=boundary_wrap, normalize_p=normalize_p, id_policy=id_policy, metadata=metadata, constraints=constraints, optimization_summary=optimization_summary)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    return data
