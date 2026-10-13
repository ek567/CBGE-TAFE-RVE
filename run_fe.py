from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GEOMETRY_SOURCE_SHA256 = 'ea8eec0848d9ec3b8a80c68aca8f4843828122dbeed53a62f137174583604d37'
GEOMETRY_ACCEPTANCE_ADAPTER_SHA256 = '956bc49f8a417c6dc0b940f4f7390ece2e612185842718f21c7bfdf8c6bc4b74'
BUNDLED_GEOMETRY_PATH = Path('examples/vf30_mixed_ar12_20/geometry.json')
BUNDLED_GEOMETRY_SHA256 = '07d431a299d72d4aeb7484ae59d984cea306e607183cbe42cbab79535d662e95'
BUNDLED_GEOMETRY_SIZE = 19744
MATERIAL_SCOPE = 'Original static tensile solve with nlgeom OFF, unchanged Al Johnson-Cook plastic declaration, SiC elastic material and ideal tie.'
FE_RUNTIME = {'schema': 'cbge-fe-runtime-v2.2', 'num_cpus': 10, 'memory_percent': 90, 'nout': 50, 'step_name': 'Step-1', 'job_prefix': 'RVE', 'mesh_target_ratio_default': 0.06, 'node_match_tolerance_default': '1e-6 * max(input_box)', 'launcher_environment_variable': 'CBGE_ABAQUS_COMMAND', 'memory_percent_semantics': 'Solver configuration ceiling; not a fixed allocation or measured model demand'}
FE_MATERIAL = {'schema': 'cbge-material-model-v2.2', 'name': 'al-sic', 'public_material_overrides_allowed': False, 'units': {'length': 'um', 'stress': 'MPa', 'strain': 'dimensionless'}, 'matrix': {'name': 'Al', 'E_MPa': 69000.0, 'nu': 0.33, 'plastic': {'hardening': 'JOHNSON_COOK', 'table': [[503.0, 193.0, 0.45, 0.0, 0.0, 0.0]]}}, 'particle': {'name': 'SiC', 'E_MPa': 450000.0, 'nu': 0.17}, 'interface': 'ideal_tie', 'analysis': 'StaticStep_nlgeom_OFF'}
FE_MODEL_SOURCE = (
    'import regionToolset\n'
    'from abaqusConstants import OFF, ON, COMPUTED, ANALYSIS, TET, C3D4, STANDARD, COMPLETED\n'
    'import math\n'
    'import mesh\n'
    'import json\n'
    'import os\n'
    'from itertools import product\n'
    'INPUT_JSON_PATH = None\n'
    'OUTPUT_JSON_PATH = None\n'
    '\n'
    'def _clamp(x, lo, hi):\n'
    '    return lo if x < lo else hi if x > hi else x\n'
    '\n'
    'def _norm(v):\n'
    '    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])\n'
    '\n'
    'def _dot(a, b):\n'
    '    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]\n'
    '\n'
    'def _mat_T_vec(M, v):\n'
    '    return (M[0][0] * v[0] + M[1][0] * v[1] + M[2][0] * v[2], M[0][1] * v[0] + M[1][1] * v[1] + M[2][1] * v[2], M[0][2] * v[0] + M[1][2] * v[1] + M[2][2] * v[2])\n'
    '\n'
    'def _axis_angle_y_to_p(p):\n'
    '    yhat = (0.0, 1.0, 0.0)\n'
    '    d = _clamp(_dot(yhat, p), -1.0, 1.0)\n'
    '    axis = (p[2], 0.0, -p[0])\n'
    '    na = _norm(axis)\n'
    '    if na < 1e-14:\n'
    '        if d >= 0.0:\n'
    '            return ((1.0, 0.0, 0.0), 0.0)\n'
    '        else:\n'
    '            return ((1.0, 0.0, 0.0), 180.0)\n'
    '    axis_u = (axis[0] / na, axis[1] / na, axis[2] / na)\n'
    '    ang = math.degrees(math.acos(d))\n'
    '    return (axis_u, ang)\n'
    '\n'
    'def _rotmat_from_axis_angle(axis_u, ang_deg):\n'
    '    ang = math.radians(ang_deg)\n'
    '    c = math.cos(ang)\n'
    '    s = math.sin(ang)\n'
    '    ux, uy, uz = axis_u\n'
    '    R = [[c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s], [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s], [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)]]\n'
    '    return R\n'
    '\n'
    'def _support_radius(a, b, p_comp):\n'
    '    aa = a * a\n'
    '    bb = b * b\n'
    '    return math.sqrt(bb + (aa - bb) * (p_comp * p_comp))\n'
    '\n'
    'def preprocess(data):\n'
    "    if not data.get('periodic', False):\n"
    "        raise ValueError('Input JSON periodic must be true for v1 pipeline.')\n"
    "    Lx, Ly, Lz = data['box']\n"
    '    Lmax = max(Lx, Ly, Lz)\n'
    '    tol_plane = 1e-06 * Lmax\n'
    '    particles_work = []\n'
    '    max_p_norm_err = 0.0\n'
    '    n_type1 = 0\n'
    "    for prt in data['particles']:\n"
    "        pid = int(prt['id'])\n"
    '        ID4 = str(pid).zfill(4)\n'
    "        a = float(prt['a'])\n"
    "        b = float(prt['b'])\n"
    "        cx, cy, cz = map(float, prt['c'])\n"
    "        px, py, pz = map(float, prt['p'])\n"
    '        p = (px, py, pz)\n'
    '        pn = _norm(p)\n'
    '        max_p_norm_err = max(max_p_norm_err, abs(pn - 1.0))\n'
    '        rx = _support_radius(a, b, px)\n'
    '        ry = _support_radius(a, b, py)\n'
    '        rz = _support_radius(a, b, pz)\n'
    '        x_min, x_max = (cx - rx, cx + rx)\n'
    '        y_min, y_max = (cy - ry, cy + ry)\n'
    '        z_min, z_max = (cz - rz, cz + rz)\n'
    '        hit_X0 = x_min <= tol_plane\n'
    '        hit_X1 = x_max >= Lx - tol_plane\n'
    '        hit_Y0 = y_min <= tol_plane\n'
    '        hit_Y1 = y_max >= Ly - tol_plane\n'
    '        hit_Z0 = z_min <= tol_plane\n'
    '        hit_Z1 = z_max >= Lz - tol_plane\n'
    '        if hit_X0 and hit_X1:\n'
    "            raise ValueError('Particle id=%d violates constraint: hits both X0 and X1.' % pid)\n"
    '        if hit_Y0 and hit_Y1:\n'
    "            raise ValueError('Particle id=%d violates constraint: hits both Y0 and Y1.' % pid)\n"
    '        if hit_Z0 and hit_Z1:\n'
    "            raise ValueError('Particle id=%d violates constraint: hits both Z0 and Z1.' % pid)\n"
    '        sx = -Lx if hit_X1 else 0.0\n'
    '        sy = -Ly if hit_Y1 else 0.0\n'
    '        sz = -Lz if hit_Z1 else 0.0\n'
    '        c_work = (cx + sx, cy + sy, cz + sz)\n'
    '        touch = []\n'
    '        if hit_X0 or hit_X1:\n'
    "            touch.append('X0')\n"
    '        if hit_Y0 or hit_Y1:\n'
    "            touch.append('Y0')\n"
    '        if hit_Z0 or hit_Z1:\n'
    "            touch.append('Z0')\n"
    '        ptype = 1 if len(touch) > 0 else 2\n'
    '        if ptype == 1:\n'
    '            n_type1 += 1\n'
    '        rot_axis_global, rot_angle_deg = _axis_angle_y_to_p(p)\n'
    '        R = _rotmat_from_axis_angle(rot_axis_global, rot_angle_deg)\n'
    "        tag_order = {'X0': 0, 'Y0': 1, 'Z0': 2}\n"
    '        cut_planes_local = []\n'
    '        for tag in sorted(touch, key=lambda t: tag_order[t]):\n'
    "            if tag == 'X0':\n"
    '                n = (1.0, 0.0, 0.0)\n'
    "            elif tag == 'Y0':\n"
    '                n = (0.0, 1.0, 0.0)\n'
    "            elif tag == 'Z0':\n"
    '                n = (0.0, 0.0, 1.0)\n'
    '            else:\n'
    "                raise ValueError('Unknown plane tag: %s' % tag)\n"
    '            q_minus_t = (-c_work[0], -c_work[1], -c_work[2])\n'
    '            point_local = _mat_T_vec(R, q_minus_t)\n'
    '            normal_local = _mat_T_vec(R, n)\n'
    '            nn = _norm(normal_local)\n'
    '            if nn < 1e-14:\n'
    "                raise ValueError('Degenerate normal_local for particle id=%d tag=%s' % (pid, tag))\n"
    '            normal_local = (normal_local[0] / nn, normal_local[1] / nn, normal_local[2] / nn)\n'
    "            cut_planes_local.append({'plane_tag': tag, 'point_local': [point_local[0], point_local[1], point_local[2]], 'normal_local': [normal_local[0], normal_local[1], normal_local[2]]})\n"
    "        x_opts = [0.0, Lx] if 'X0' in touch else [0.0]\n"
    "        y_opts = [0.0, Ly] if 'Y0' in touch else [0.0]\n"
    "        z_opts = [0.0, Lz] if 'Z0' in touch else [0.0]\n"
    '        tile_shifts = [[dx, dy, dz] for dx, dy, dz in product(x_opts, y_opts, z_opts)]\n'
    "        particles_work.append({'id': pid, 'ID4': ID4, 'a': a, 'b': b, 'c': [cx, cy, cz], 'p': [px, py, pz], 'hit': {'X0': hit_X0, 'X1': hit_X1, 'Y0': hit_Y0, 'Y1': hit_Y1, 'Z0': hit_Z0, 'Z1': hit_Z1}, 'touch': touch, 'type': ptype, 'shift': [sx, sy, sz], 'c_work': [c_work[0], c_work[1], c_work[2]], 'rot_axis_global': [rot_axis_global[0], rot_axis_global[1], rot_axis_global[2]], 'rot_angle_deg': rot_angle_deg, 'cut_planes_local': cut_planes_local, 'tile_shifts': tile_shifts})\n"
    "    out = {'schema': 'rve-preprocess', 'schema_version': '1.0', 'source_schema': data.get('schema'), 'source_schema_version': data.get('schema_version'), 'box': [Lx, Ly, Lz], 'Lmax': Lmax, 'tol_plane': tol_plane, 'tol_pair': tol_plane, 'particles_work': particles_work, 'notes': {'assumption_origin_shift_already_applied_to_c': True, 'origin_shift_vector_from_input': data.get('constraints', {}).get('origin_shift', {}).get('shift_vector', None)}}\n"
    "    summary = {'n_particles': len(particles_work), 'n_type1': n_type1, 'n_type2': len(particles_work) - n_type1, 'max_abs_norm_p_minus_1': max_p_norm_err, 'input_delta_min': data.get('constraints', {}).get('min_distance', {}).get('delta_min', None), 'input_final_max_overlap_pw85_physical': data.get('optimization_summary', {}).get('final_max_overlap_pw85_physical', None)}\n"
    '    return (out, summary)\n'
    'import json\n'
    'import os\n'
    'import math\n'
    'from abaqus import mdb\n'
    'from abaqusConstants import THREE_D, DEFORMABLE_BODY, OFF, UNION, DIFFERENCE\n'
    'PARTICLES_WORK_JSON = None\n'
    "MODEL_NAME = 'RVE'\n"
    'MAX_PARTICLES = None\n'
    '\n'
    'def _dot(a, b):\n'
    '    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]\n'
    '\n'
    'def _sub(a, b):\n'
    '    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])\n'
    '\n'
    'def _get_new_geom_keys(sketch, keys_before):\n'
    '    return list(set(sketch.geometry.keys()) - set(keys_before))\n'
    '\n'
    'def _build_spheroid_solid_part(model, part_name, a, b):\n'
    '    if part_name in model.parts.keys():\n'
    '        del model.parts[part_name]\n'
    '    sheet = 10.0 * max(a, b)\n'
    "    sk_name = '__SK_INC_%s' % part_name\n"
    '    if sk_name in model.sketches.keys():\n'
    '        del model.sketches[sk_name]\n'
    '    s = model.ConstrainedSketch(name=sk_name, sheetSize=sheet)\n'
    '    s.ConstructionLine(point1=(0.0, -sheet), point2=(0.0, sheet))\n'
    '    keys0 = list(s.geometry.keys())\n'
    '    s.EllipseByCenterPerimeter(center=(0.0, 0.0), axisPoint1=(b, 0.0), axisPoint2=(0.0, a))\n'
    '    new_keys = _get_new_geom_keys(s, keys0)\n'
    '    if len(new_keys) != 1:\n'
    "        raise RuntimeError('Ellipse creation unexpected: new geometry keys = %s' % str(new_keys))\n"
    '    ellipse_geom = s.geometry[new_keys[0]]\n'
    '    s.Line(point1=(0.0, a), point2=(0.0, -a))\n'
    '    s.autoTrimCurve(curve1=ellipse_geom, point1=(-0.999 * b, 0.0))\n'
    '    p = model.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)\n'
    '    p.BaseSolidRevolve(sketch=s, angle=360.0, flipRevolveDirection=OFF)\n'
    '    del model.sketches[sk_name]\n'
    '    return p\n'
    '\n'
    'def _partition_by_cut_planes(p, cut_planes_local):\n'
    '    for pl in cut_planes_local:\n'
    "        pt = tuple(pl['point_local'])\n"
    "        nn = tuple(pl['normal_local'])\n"
    '        feat_p0 = p.DatumPointByCoordinate(coords=pt)\n'
    '        dp0 = p.datums[feat_p0.id]\n'
    '        pt2 = (pt[0] + nn[0], pt[1] + nn[1], pt[2] + nn[2])\n'
    '        feat_p1 = p.DatumPointByCoordinate(coords=pt2)\n'
    '        dp1 = p.datums[feat_p1.id]\n'
    '        feat_ax = p.DatumAxisByTwoPoint(point1=dp0, point2=dp1)\n'
    '        ax = p.datums[feat_ax.id]\n'
    '        feat_pl = p.DatumPlaneByPointNormal(point=dp0, normal=ax)\n'
    '        dplane = p.datums[feat_pl.id]\n'
    '        p.PartitionCellByDatumPlane(datumPlane=dplane, cells=p.cells[:])\n'
    '\n'
    'def _remove_cells_keep_shell(p):\n'
    "    if hasattr(p, 'RemoveCells'):\n"
    '        p.RemoveCells(cellList=p.cells[:])\n'
    '        return\n'
    "    if hasattr(p, 'removeCells'):\n"
    '        p.removeCells(cellList=p.cells[:])\n'
    '        return\n'
    "    raise RuntimeError('Part object has neither RemoveCells nor removeCells: %s' % p.name)\n"
    '\n'
    'def _remove_all_cells_to_shell(p):\n'
    '    if len(p.cells) == 0:\n'
    "        raise RuntimeError('RemoveCells called but part has no cells.')\n"
    '    _remove_cells_keep_shell(p)\n'
    '\n'
    'def _set_from_face_indices(p, set_name, face_indices):\n'
    '    if set_name in p.sets.keys():\n'
    '        del p.sets[set_name]\n'
    '    idxs = sorted(set((int(i) for i in face_indices)))\n'
    '    if len(idxs) == 0:\n'
    "        raise RuntimeError('Empty face_indices for set %s on part %s' % (set_name, p.name))\n"
    '    if len(idxs) == 1:\n'
    '        i = idxs[0]\n'
    '        p.Set(name=set_name, faces=p.faces[i:i + 1])\n'
    '        return\n'
    '    tmp_names = []\n'
    '    for k, i in enumerate(idxs):\n'
    "        tmp = 'TMP_%s_%03d' % (set_name, k)\n"
    '        if tmp in p.sets.keys():\n'
    '            del p.sets[tmp]\n'
    '        p.Set(name=tmp, faces=p.faces[i:i + 1])\n'
    '        tmp_names.append(tmp)\n'
    '    p.SetByBoolean(name=set_name, sets=tuple((p.sets[nm] for nm in tmp_names)), operation=UNION)\n'
    '    for nm in tmp_names:\n'
    '        del p.sets[nm]\n'
    '\n'
    'def _build_cap_iface_sets(p, ID4, touch, cut_planes_local, tol_plane):\n'
    '    if len(p.faces) == 0:\n'
    "        raise RuntimeError('Part %s has no faces after RemoveCells.' % p.name)\n"
    "    tmp_all = 'TMP_%s_ALL_FACES' % ID4\n"
    '    if tmp_all in p.sets.keys():\n'
    '        del p.sets[tmp_all]\n'
    '    p.Set(name=tmp_all, faces=p.faces[:])\n'
    '    plane_map = {}\n'
    '    for pl in cut_planes_local:\n'
    "        plane_map[pl['plane_tag']] = (tuple(pl['point_local']), tuple(pl['normal_local']))\n"
    '    cap_set_names = []\n'
    "    for tag in ['X0', 'Y0', 'Z0']:\n"
    '        if tag not in touch:\n'
    '            continue\n'
    '        if tag not in plane_map:\n'
    "            raise RuntimeError('touch contains %s but cut_planes_local missing it for %s' % (tag, p.name))\n"
    '        p0, n = plane_map[tag]\n'
    '        cap_face_indices = []\n'
    '        for f in p.faces[:]:\n'
    '            pt = f.pointOn[0]\n'
    '            dist = (pt[0] - p0[0]) * n[0] + (pt[1] - p0[1]) * n[1] + (pt[2] - p0[2]) * n[2]\n'
    '            if abs(dist) <= tol_plane:\n'
    '                cap_face_indices.append(f.index)\n'
    "        cap_name = 'INC_%s_CAP_%s' % (ID4, tag)\n"
    '        if len(cap_face_indices) == 0:\n'
    "            raise RuntimeError('CAP empty: %s on %s (check tol_plane / partition result)' % (cap_name, p.name))\n"
    '        _set_from_face_indices(p, cap_name, cap_face_indices)\n'
    '        cap_set_names.append(cap_name)\n'
    "    iface_name = 'INC_%s_IFACE' % ID4\n"
    '    if iface_name in p.sets.keys():\n'
    '        del p.sets[iface_name]\n'
    '    if len(cap_set_names) == 0:\n'
    '        p.SetByBoolean(name=iface_name, sets=(p.sets[tmp_all],), operation=UNION)\n'
    '    else:\n'
    "        tmp_cap_union = 'TMP_%s_CAP_UNION' % ID4\n"
    '        if tmp_cap_union in p.sets.keys():\n'
    '            del p.sets[tmp_cap_union]\n'
    '        p.SetByBoolean(name=tmp_cap_union, sets=tuple((p.sets[nm] for nm in cap_set_names)), operation=UNION)\n'
    '        p.SetByBoolean(name=iface_name, sets=(p.sets[tmp_all], p.sets[tmp_cap_union]), operation=DIFFERENCE)\n'
    '        del p.sets[tmp_cap_union]\n'
    '    if len(p.sets[iface_name].faces) == 0:\n'
    "        raise RuntimeError('IFACE empty: %s on part %s' % (iface_name, p.name))\n"
    '\n'
    'def _ensure_model_rve():\n'
    '    if MODEL_NAME in mdb.models.keys():\n'
    '        del mdb.models[MODEL_NAME]\n'
    '    mdb.Model(name=MODEL_NAME)\n'
    '    return mdb.models[MODEL_NAME]\n'
    'from abaqus import mdb\n'
    'from abaqusConstants import THREE_D, DEFORMABLE_BODY, STANDALONE\n'
    "PART_NAME = 'P_MATRIX_MASTER'\n"
    '\n'
    'def _delete_part_if_exists(model, part_name):\n'
    '    if part_name in model.parts.keys():\n'
    '        del model.parts[part_name]\n'
    '\n'
    'def _delete_instance_if_exists(model, inst_name):\n'
    '    a = model.rootAssembly\n'
    '    if inst_name in a.instances.keys():\n'
    '        a.deleteInstances(instances=(a.instances[inst_name],))\n'
    '\n'
    'def _classify_face_by_pointOn(face, Lx, Ly, Lz, tol):\n'
    '    x, y, z = face.pointOn[0]\n'
    '    if abs(x - 0.0) <= tol:\n'
    "        return 'X0'\n"
    '    if abs(x - Lx) <= tol:\n'
    "        return 'X1'\n"
    '    if abs(y - 0.0) <= tol:\n'
    "        return 'Y0'\n"
    '    if abs(y - Ly) <= tol:\n'
    "        return 'Y1'\n"
    '    if abs(z - 0.0) <= tol:\n'
    "        return 'Z0'\n"
    '    if abs(z - Lz) <= tol:\n'
    "        return 'Z1'\n"
    '    return None\n'
    'import json\n'
    'import os\n'
    'from abaqus import mdb\n'
    'from abaqusConstants import ON\n'
    "MODEL_NAME = 'RVE'\n"
    '\n'
    'def delete_instance_anyway(a, name):\n'
    '    if name in a.instances.keys():\n'
    '        a.deleteInstances(instances=(a.instances[name],))\n'
    '        return\n'
    '    if name in a.features.keys():\n'
    '        del a.features[name]\n'
    '\n'
    'def encode_SXYZ(shift, Lx, Ly, Lz, tol):\n'
    '    dx, dy, dz = shift\n'
    '\n'
    '    def bit(d, L):\n'
    '        if abs(d) <= tol:\n'
    "            return '0'\n"
    '        if abs(d - L) <= tol:\n'
    "            return '1'\n"
    "        raise ValueError('tile_shift component not in {0,+L}: d=%s L=%s' % (str(d), str(L)))\n"
    "    return 'S%s%s%s' % (bit(dx, Lx), bit(dy, Ly), bit(dz, Lz))\n"
    "MERGED_INSTANCE_NAME = 'I_MERGED_GEOM'\n"
    'from abaqusConstants import GEOMETRY, SUPPRESS, ON\n'
    'import json, os, math, time\n'
    'from collections import defaultdict, deque\n'
    'import mesh\n'
    'from abaqusConstants import TRI, FREE, FINER, FIXED\n'
    'from abaqusConstants import SHAPE_FACTOR, SMALL_ANGLE, LARGE_ANGLE, ASPECT_RATIO\n'
    'try:\n'
    '    from abaqusConstants import GEOMETRIC_DEVIATION_FACTOR\n'
    '    _HAS_GEO_DEV = True\n'
    'except Exception:\n'
    '    _HAS_GEO_DEV = False\n'
    "M6_PARAM = {'mesh_target_len': None, 'mesh_target_ratio': 0.04, 'iface_refine_factor': 0.6, 'curvature_factor': 0.35, 'curvature_min_factor': 0.45, 'gradation_ratio': 1.35, 'min_size_ratio': 0.005, 'max_size_ratio': None, 'min_size_abs': None, 'max_size_abs': None, 'verify': {'shape_factor_tris_lt': 0.01, 'small_angle_lt': 5.0, 'large_angle_gt': 170.0, 'aspect_ratio_gt': 10.0, 'use_geometric_deviation': False, 'geometric_deviation_gt': 0.2}, 'log_jsonl_name': 'm6_log.jsonl'}\n"
    '\n'
    'def _v_sub(a, b):\n'
    '    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])\n'
    '\n'
    'def _v_dot(a, b):\n'
    '    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]\n'
    '\n'
    'def _v_norm(a):\n'
    '    return math.sqrt(_v_dot(a, a))\n'
    '\n'
    'def _dist(a, b):\n'
    '    return _v_norm(_v_sub(a, b))\n'
    '\n'
    'def _bbox(pts):\n'
    '    xs = [p[0] for p in pts]\n'
    '    ys = [p[1] for p in pts]\n'
    '    zs = [p[2] for p in pts]\n'
    '    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))\n'
    '\n'
    'def _in_domain_pt(p, box, tol):\n'
    '    Lx, Ly, Lz = box\n'
    '    return -tol <= p[0] <= Lx + tol and -tol <= p[1] <= Ly + tol and (-tol <= p[2] <= Lz + tol)\n'
    '\n'
    'def _canon_mod(x, L, tol):\n'
    '    if L <= 0.0:\n'
    '        return x\n'
    '    xm = x % L\n'
    '    if xm < tol or L - xm < tol:\n'
    '        xm = 0.0\n'
    '    return xm\n'
    '\n'
    'def _edge_periodic_key(edge, box, tol_plane, eps_key):\n'
    '    p = tuple(edge.pointOn[0])\n'
    '    Lx, Ly, Lz = box\n'
    '    pc = (_canon_mod(p[0], Lx, tol_plane), _canon_mod(p[1], Ly, tol_plane), _canon_mod(p[2], Lz, tol_plane))\n'
    '    return (int(round(pc[0] / eps_key)), int(round(pc[1] / eps_key)), int(round(pc[2] / eps_key)))\n'
    '\n'
    'def _build_periodic_edge_groups(inst, topo, box, tol_plane, eps_key):\n'
    '    key2eids = defaultdict(list)\n'
    "    for eid, e in enumerate(topo['edges']):\n"
    '        k = _edge_periodic_key(e, box, tol_plane, eps_key)\n'
    '        key2eids[k].append(eid)\n'
    "    rep_of_eid = [None] * len(topo['edges'])\n"
    '    groups = {}\n'
    '    for k, eids in key2eids.items():\n'
    '        rep = None\n'
    '        for eid in eids:\n'
    "            p = tuple(topo['edges'][eid].pointOn[0])\n"
    '            if _in_domain_pt(p, box, tol_plane):\n'
    '                rep = eid\n'
    '                break\n'
    '        if rep is None:\n'
    '            rep = eids[0]\n'
    '        groups[rep] = eids\n'
    '        for eid in eids:\n'
    '            rep_of_eid[eid] = rep\n'
    '    return (rep_of_eid, groups)\n'
    '\n'
    'def _sync_groups_h(h, topo, groups, r_grad):\n'
    '    for rep, eids in groups.items():\n'
    '        v = min((h[eid] for eid in eids))\n'
    '        for eid in eids:\n'
    '            h[eid] = v\n'
    '    _project_gradation(h, topo, r_grad, sweeps=10)\n'
    '    for rep, eids in groups.items():\n'
    '        v = min((h[eid] for eid in eids))\n'
    '        for eid in eids:\n'
    '            h[eid] = v\n'
    '\n'
    'def _jacobi_eig_sym_3x3(A, itmax=30):\n'
    '    V = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n'
    '\n'
    '    def max_offdiag(M):\n'
    '        m = 0.0\n'
    '        p = 0\n'
    '        q = 1\n'
    '        for i in range(3):\n'
    '            for j in range(i + 1, 3):\n'
    '                if abs(M[i][j]) > m:\n'
    '                    m = abs(M[i][j])\n'
    '                    p = i\n'
    '                    q = j\n'
    '        return (m, p, q)\n'
    '    for _ in range(itmax):\n'
    '        m, p, q = max_offdiag(A)\n'
    '        if m < 1e-12:\n'
    '            break\n'
    '        app = A[p][p]\n'
    '        aqq = A[q][q]\n'
    '        apq = A[p][q]\n'
    '        phi = 0.5 * math.atan2(2.0 * apq, aqq - app)\n'
    '        c = math.cos(phi)\n'
    '        s = math.sin(phi)\n'
    '        for k in range(3):\n'
    '            Akp = A[k][p]\n'
    '            Akq = A[k][q]\n'
    '            A[k][p] = c * Akp - s * Akq\n'
    '            A[k][q] = s * Akp + c * Akq\n'
    '        for k in range(3):\n'
    '            Apk = A[p][k]\n'
    '            Aqk = A[q][k]\n'
    '            A[p][k] = c * Apk - s * Aqk\n'
    '            A[q][k] = s * Apk + c * Aqk\n'
    '        A[p][q] = 0.0\n'
    '        A[q][p] = 0.0\n'
    '        for k in range(3):\n'
    '            Vkp = V[k][p]\n'
    '            Vkq = V[k][q]\n'
    '            V[k][p] = c * Vkp - s * Vkq\n'
    '            V[k][q] = s * Vkp + c * Vkq\n'
    '    evals = [A[0][0], A[1][1], A[2][2]]\n'
    '    evecs = [(V[0][i], V[1][i], V[2][i]) for i in range(3)]\n'
    '    return (evals, evecs)\n'
    '\n'
    'def _unit(v):\n'
    '    n = _v_norm(v)\n'
    '    if n < 1e-30:\n'
    '        return (1.0, 0.0, 0.0)\n'
    '    return (v[0] / n, v[1] / n, v[2] / n)\n'
    '\n'
    'def _pca_minor_dir(points):\n'
    '    if len(points) < 5:\n'
    '        return (0.0, 1.0, 0.0)\n'
    '    c = (sum((p[0] for p in points)) / len(points), sum((p[1] for p in points)) / len(points), sum((p[2] for p in points)) / len(points))\n'
    '    S = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]\n'
    '    for p in points:\n'
    '        d = _v_sub(p, c)\n'
    '        for i in range(3):\n'
    '            for j in range(3):\n'
    '                S[i][j] += d[i] * d[j]\n'
    '    denom = max(1, len(points) - 1)\n'
    '    for i in range(3):\n'
    '        for j in range(3):\n'
    '            S[i][j] /= denom\n'
    '    evals, evecs = _jacobi_eig_sym_3x3(S)\n'
    '    idx = sorted(range(3), key=lambda i: evals[i], reverse=True)\n'
    '    u2 = _unit(evecs[idx[1]])\n'
    '    return u2\n'
    '\n'
    'def _build_topology(inst):\n'
    '    edges = inst.edges\n'
    '    vertices = inst.vertices\n'
    '    faces = inst.faces\n'
    '    vtx_xyz = {}\n'
    '    for vid, v in enumerate(vertices):\n'
    '        vtx_xyz[vid] = tuple(v.pointOn[0])\n'
    '    edge_vids = {}\n'
    '    vtx_edges = defaultdict(set)\n'
    '    edge_len = {}\n'
    '    for eid, e in enumerate(edges):\n'
    '        vids = tuple(e.getVertices())\n'
    '        edge_vids[eid] = vids\n'
    '        for vid in vids:\n'
    '            vtx_edges[vid].add(eid)\n'
    '        try:\n'
    '            edge_len[eid] = float(e.getSize())\n'
    '        except Exception:\n'
    '            edge_len[eid] = 0.0\n'
    '    edge_adj = defaultdict(set)\n'
    '    for vid, eids in vtx_edges.items():\n'
    '        for a in eids:\n'
    '            for b in eids:\n'
    '                if a != b:\n'
    '                    edge_adj[a].add(b)\n'
    '    face_edge_ids = {}\n'
    '    edge_face_ids = defaultdict(set)\n'
    '    for fid, f in enumerate(faces):\n'
    '        eids = list(f.getEdges())\n'
    '        face_edge_ids[fid] = eids\n'
    '        for eid in eids:\n'
    '            edge_face_ids[eid].add(fid)\n'
    "    return {'edges': edges, 'faces': faces, 'vertices': vertices, 'vtx_xyz': vtx_xyz, 'edge_vids': edge_vids, 'vtx_edges': vtx_edges, 'edge_len': edge_len, 'edge_adj': edge_adj, 'face_edge_ids': face_edge_ids, 'edge_face_ids': edge_face_ids}\n"
    '\n'
    'def _edge_Rmin(edge):\n'
    '    R = []\n'
    '    for t in (0.25, 0.5, 0.75):\n'
    '        try:\n'
    '            info = edge.getCurvature(parameter=t)\n'
    "            r = float(info['radius'])\n"
    '            if r > 0:\n'
    '                R.append(r)\n'
    '        except Exception:\n'
    '            pass\n'
    '    if not R:\n'
    "        return float('inf')\n"
    '    return min(R)\n'
    '\n'
    'def _compute_vertex_proximity(vtx_xyz, edge_vids, Lchar, grid_cell_factor):\n'
    '    vtx_adj = defaultdict(set)\n'
    '    for vids in edge_vids.values():\n'
    '        if len(vids) == 2:\n'
    '            a, b = vids\n'
    '            vtx_adj[a].add(b)\n'
    '            vtx_adj[b].add(a)\n'
    '    cell = max(1e-12, Lchar / float(grid_cell_factor))\n'
    '\n'
    '    def key(p):\n'
    '        return (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)), int(math.floor(p[2] / cell)))\n'
    '    grid = defaultdict(list)\n'
    '    for vid, p in vtx_xyz.items():\n'
    '        grid[key(p)].append(vid)\n'
    '    neigh = [-1, 0, 1]\n'
    '    d_v = {}\n'
    '    for vid, p in vtx_xyz.items():\n'
    '        k0 = key(p)\n'
    "        best = float('inf')\n"
    '        for dx in neigh:\n'
    '            for dy in neigh:\n'
    '                for dz in neigh:\n'
    '                    kk = (k0[0] + dx, k0[1] + dy, k0[2] + dz)\n'
    '                    for vid2 in grid.get(kk, []):\n'
    '                        if vid2 == vid:\n'
    '                            continue\n'
    '                        if vid2 in vtx_adj[vid]:\n'
    '                            continue\n'
    '                        d = _dist(p, vtx_xyz[vid2])\n'
    '                        if d < best:\n'
    '                            best = d\n'
    '        d_v[vid] = best\n'
    '    return d_v\n'
    '\n'
    'def _compute_edge_proximity(topo, Lchar, grid_cell_factor):\n'
    "    d_v = _compute_vertex_proximity(topo['vtx_xyz'], topo['edge_vids'], Lchar, grid_cell_factor)\n"
    '    d_e = {}\n'
    "    for eid, vids in topo['edge_vids'].items():\n"
    "        d = float('inf')\n"
    '        for vid in vids:\n'
    "            d = min(d, d_v.get(vid, float('inf')))\n"
    '        d_e[eid] = d\n'
    '    return d_e\n'
    '\n'
    'def _m6_number_from_size_half_up(Le, size):\n'
    '    if size <= 0.0:\n'
    '        return 1\n'
    '    r = float(Le) / float(size)\n'
    '    n = int(math.floor(r + 0.5))\n'
    '    if n < 1:\n'
    '        n = 1\n'
    '    return n\n'
    '\n'
    'def _project_gradation(h, topo, r_grad, sweeps=10):\n'
    '    for _ in range(sweeps):\n'
    '        changed = False\n'
    "        for eid, nbrs in topo['edge_adj'].items():\n"
    '            hi = h[eid]\n'
    '            for j in nbrs:\n'
    '                hj = h[j]\n'
    '                if hj > r_grad * hi:\n'
    '                    h[j] = r_grad * hi\n'
    '                    changed = True\n'
    '                if hi > r_grad * hj:\n'
    '                    h[eid] = r_grad * hj\n'
    '                    changed = True\n'
    '        if not changed:\n'
    '            break\n'
    '\n'
    'def _worst_grad_ratio(h, topo):\n'
    '    worst = 1.0\n'
    "    for eid, nbrs in topo['edge_adj'].items():\n"
    '        hi = h[eid]\n'
    '        if hi <= 0:\n'
    '            continue\n'
    '        for j in nbrs:\n'
    '            hj = h[j]\n'
    '            if hj <= 0:\n'
    '                continue\n'
    '            worst = max(worst, max(hi, hj) / min(hi, hj))\n'
    '    return worst\n'
    '\n'
    'def _init_h(topo, R_e, d_e, Lchar, tol_plane, edge_target_len, param):\n'
    "    if param['h_min_abs'] is None:\n"
    "        param['h_min_abs'] = Lchar / 2000.0\n"
    "    if param['h_max_abs'] is None:\n"
    "        param['h_max_abs'] = Lchar / 5.0\n"
    "    hmin = param['h_min_abs']\n"
    "    hmax = param['h_max_abs']\n"
    "    if param['feature_cap_from_edge_target_len']:\n"
    '        h_feature = float(edge_target_len)\n'
    '    else:\n'
    '        h_feature = min(hmax / 4.0, Lchar / 30.0)\n'
    "    c_k = param['c_kappa']\n"
    "    c_d = param['c_d']\n"
    '    h_star = {}\n'
    "    for eid in range(len(topo['edges'])):\n"
    "        hk = c_k * R_e[eid] if math.isfinite(R_e[eid]) else float('inf')\n"
    "        hd = c_d * d_e[eid] if math.isfinite(d_e[eid]) else float('inf')\n"
    '        h0 = min(hmax, hk, hd)\n'
    "        if R_e[eid] < float('inf'):\n"
    '            h0 = min(h0, h_feature)\n'
    '        h_star[eid] = max(hmin, min(hmax, h0))\n'
    '    hs = sorted(h_star.values())\n'
    '    if hs:\n'
    '        q = 0.85\n'
    '        h_global = hs[int(q * (len(hs) - 1))]\n'
    '    else:\n'
    '        h_global = hmax\n'
    '    h_global = max(hmin, min(hmax, h_global))\n'
    '    try:\n'
    '        if global_seed_size is not None:\n'
    '            h_global = float(global_seed_size)\n'
    '    except NameError:\n'
    '        pass\n'
    '    h = {}\n'
    "    for eid in range(len(topo['edges'])):\n"
    '        h[eid] = min(h_global, h_star[eid])\n'
    "    _project_gradation(h, topo, param['r_grad'], sweeps=10)\n"
    '    return (h_global, h_star, h)\n'
    '\n'
    'def _m6_param(param, name, value):\n'
    '    if value is not None:\n'
    '        return value\n'
    '    if param is not None and name in param:\n'
    '        return param[name]\n'
    '    return M6_PARAM.get(name, None)\n'
    '\n'
    'def _m6_clamp_size(hv, hmin, hmax):\n'
    '    if hv < hmin:\n'
    '        return hmin\n'
    '    if hmax is not None and hv > hmax:\n'
    '        return hmax\n'
    '    return hv\n'
    '\n'
    'def _m6_collect_iface_edge_ids(inst, topo):\n'
    '    iface_eids = set()\n'
    '    repos = []\n'
    '    try:\n'
    '        repos.append(inst.sets)\n'
    '    except Exception:\n'
    '        pass\n'
    '    try:\n'
    '        repos.append(inst.surfaces)\n'
    '    except Exception:\n'
    '        pass\n'
    '    for repo in repos:\n'
    '        try:\n'
    '            names = list(repo.keys())\n'
    '        except Exception:\n'
    '            names = []\n'
    '        for name in names:\n'
    "            if 'IFACE' not in str(name).upper():\n"
    '                continue\n'
    '            try:\n'
    '                reg = repo[name]\n'
    '            except Exception:\n'
    '                continue\n'
    '            try:\n'
    '                for f in reg.faces:\n'
    '                    for eid in f.getEdges():\n'
    '                        iface_eids.add(int(eid))\n'
    '            except Exception:\n'
    '                pass\n'
    '            try:\n'
    '                for e in reg.edges:\n'
    "                    if hasattr(e, 'index'):\n"
    '                        iface_eids.add(int(e.index))\n'
    '            except Exception:\n'
    '                pass\n'
    '    return iface_eids\n'
    '\n'
    'def _m6_init_explicit_h(topo, R_e, iface_eids, Lchar, mesh_target_len, mesh_target_ratio, iface_refine_factor, curvature_factor, curvature_min_factor, gradation_ratio, min_size_ratio, max_size_ratio, min_size_abs, max_size_abs, param):\n'
    '    if mesh_target_len is None:\n'
    '        if mesh_target_ratio is None:\n'
    "            mesh_target_ratio = M6_PARAM['mesh_target_ratio']\n"
    '        h_global = float(mesh_target_ratio) * float(Lchar)\n'
    '    else:\n'
    '        h_global = float(mesh_target_len)\n'
    '    if h_global <= 0.0:\n'
    "        raise RuntimeError('M6: mesh target size must be positive')\n"
    '    if iface_refine_factor is None:\n'
    "        iface_refine_factor = M6_PARAM['iface_refine_factor']\n"
    '    if curvature_factor is None:\n'
    "        curvature_factor = M6_PARAM['curvature_factor']\n"
    '    if curvature_min_factor is None:\n'
    "        curvature_min_factor = M6_PARAM['curvature_min_factor']\n"
    '    if gradation_ratio is None:\n'
    "        gradation_ratio = M6_PARAM['gradation_ratio']\n"
    '    if min_size_ratio is None:\n'
    "        min_size_ratio = M6_PARAM['min_size_ratio']\n"
    '    if min_size_abs is None:\n'
    '        min_size_abs = float(min_size_ratio) * float(Lchar)\n'
    '    if max_size_abs is None and max_size_ratio is not None:\n'
    '        max_size_abs = float(max_size_ratio) * float(Lchar)\n'
    '    hmin = max(1e-12, float(min_size_abs))\n'
    '    hmax = None if max_size_abs is None else float(max_size_abs)\n'
    '    if hmax is not None and hmax < hmin:\n'
    "        raise RuntimeError('M6: max mesh size is smaller than min mesh size')\n"
    "    param['h_min_abs'] = hmin\n"
    "    param['h_max_abs'] = hmax\n"
    '    h = {}\n'
    '    n_curved = 0\n'
    '    n_iface = 0\n'
    '    n_curvature_floor = 0\n'
    "    for eid in range(len(topo['edges'])):\n"
    "        Rmin = R_e.get(eid, float('inf'))\n"
    '        is_curved = math.isfinite(Rmin)\n'
    '        is_iface = eid in iface_eids\n'
    '        hv = h_global\n'
    '        if is_curved or is_iface:\n'
    '            hv = min(hv, h_global * float(iface_refine_factor))\n'
    '            if is_curved:\n'
    '                n_curved += 1\n'
    '                h_curv_abs = float(curvature_factor) * float(Rmin)\n'
    '                h_curv_floor = float(h_global) * float(curvature_min_factor)\n'
    '                if h_curv_abs < h_curv_floor:\n'
    '                    n_curvature_floor += 1\n'
    '                hv = min(hv, max(h_curv_abs, h_curv_floor))\n'
    '            if is_iface:\n'
    '                n_iface += 1\n'
    '        h[eid] = _m6_clamp_size(hv, hmin, hmax)\n'
    '    _project_gradation(h, topo, float(gradation_ratio), sweeps=10)\n'
    "    stats = {'h_global': float(h_global), 'h_min_abs': float(hmin), 'h_max_abs': None if hmax is None else float(hmax), 'n_curved_edges': int(n_curved), 'n_iface_edges': int(n_iface), 'n_curvature_floor_edges': int(n_curvature_floor)}\n"
    '    return (h_global, h, stats)\n'
    '\n'
    'def _delete_mesh_safe(a, inst):\n'
    '    try:\n'
    '        a.deleteMesh(regions=(inst,))\n'
    '        return\n'
    '    except Exception:\n'
    '        pass\n'
    '    try:\n'
    '        inst.deleteMesh()\n'
    '        return\n'
    '    except Exception:\n'
    '        pass\n'
    '\n'
    'def _bucket_key(hv, hmin):\n'
    '    step = max(hmin, 0.02 * hv)\n'
    '    return round(hv / step) * step\n'
    '\n'
    'def _apply_seeds_and_mesh(a, inst, topo, h_global, h, param, edge_groups):\n'
    '    _delete_mesh_safe(a, inst)\n'
    '    a.setMeshControls(regions=inst.faces[:], elemShape=TRI, technique=FREE)\n'
    '    a.seedPartInstance(regions=(inst,), size=h_global, deviationFactor=0.1, minSizeFactor=0.1)\n'
    "    hmin = param['h_min_abs']\n"
    '    eid2N = {}\n'
    '    for rep, eids in edge_groups.items():\n'
    '        hv = h[rep]\n'
    '        k = _bucket_key(hv, hmin)\n'
    "        Le = float(topo['edge_len'].get(rep, 0.0))\n"
    '        Nrep = _m6_number_from_size_half_up(Le, float(k))\n'
    '        for eid in eids:\n'
    '            eid2N[eid] = Nrep\n'
    '    bucketsN = defaultdict(list)\n'
    '    for eid, N in eid2N.items():\n'
    '        bucketsN[int(N)].append(int(eid))\n'
    '    for N, eids in bucketsN.items():\n'
    '        pickedEdges = tuple((inst.edges[eid] for eid in eids))\n'
    '        a.seedEdgeByNumber(edges=pickedEdges, number=int(N), constraint=FIXED)\n'
    "    print('[M6] seedEdgeByNumber: unique N buckets =', len(bucketsN))\n"
    '    a.generateMesh(regions=(inst,))\n'
    '\n'
    'def _delete_set_if_exists(repo, name):\n'
    '    try:\n'
    '        if name in repo.keys():\n'
    '            del repo[name]\n'
    '    except Exception:\n'
    '        pass\n'
    '\n'
    'def _verify_mesh(a, inst, param):\n'
    "    v = param['verify']\n"
    '    poor = set()\n'
    '    stats = {}\n'
    "    items = [(SHAPE_FACTOR, v['shape_factor_tris_lt'], 'SF'), (SMALL_ANGLE, v['small_angle_lt'], 'SA'), (LARGE_ANGLE, v['large_angle_gt'], 'LA'), (ASPECT_RATIO, v['aspect_ratio_gt'], 'AR')]\n"
    "    if v.get('use_geometric_deviation', False) and _HAS_GEO_DEV:\n"
    "        items.append((GEOMETRIC_DEVIATION_FACTOR, v['geometric_deviation_gt'], 'GD'))\n"
    '    for crit, limit, suf in items:\n'
    "        setName = 'M6_POOR_%s' % suf\n"
    '        _delete_set_if_exists(inst.sets, setName)\n'
    '        from abaqusConstants import ANALYSIS_CHECKS, TRI\n'
    '        res = a.verifyMeshQuality(criterion=ANALYSIS_CHECKS, elemShape=TRI, regions=(inst,))\n'
    "        n_err = len(res.get('failedElements', []))\n"
    "        n_warn = len(res.get('warningElements', []))\n"
    "        print('ANALYSIS_CHECKS: errors=%d warnings=%d' % (n_err, n_warn))\n"
    '        n = 0\n'
    '        try:\n'
    '            elems = inst.sets[setName].elements\n'
    '            n = len(elems)\n'
    '            for el in elems:\n'
    '                poor.add(el.label)\n'
    '        except Exception:\n'
    '            n = 0\n'
    '        stats[suf] = n\n'
    '    return (poor, stats)\n'
    '\n'
    'def _vote_bad_edges(topo, poor_labels):\n'
    '    votes = {}\n'
    "    for eid, e in enumerate(topo['edges']):\n"
    '        cnt = 0\n'
    '        try:\n'
    '            els = e.getElements()\n'
    '            for el in els:\n'
    '                if el.label in poor_labels:\n'
    '                    cnt += 1\n'
    '        except Exception:\n'
    '            cnt = 0\n'
    '        if cnt > 0:\n'
    '            votes[eid] = cnt\n'
    '    bad_sorted = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)\n'
    '    return bad_sorted\n'
    '\n'
    'def _expand_patch(topo, seed_eids, k_ring):\n'
    '    patch = set(seed_eids)\n'
    '    q = deque([(eid, 0) for eid in seed_eids])\n'
    '    while q:\n'
    '        u, d = q.popleft()\n'
    '        if d >= k_ring:\n'
    '            continue\n'
    "        for nb in topo['edge_adj'].get(u, []):\n"
    '            if nb not in patch:\n'
    '                patch.add(nb)\n'
    '                q.append((nb, d + 1))\n'
    '    return patch\n'
    '\n'
    'def _update_h(h, topo, bad_eids, patch_eids, param):\n'
    "    beta = param['beta_bad']\n"
    "    hmin = param['h_min_abs']\n"
    "    r = param['r_grad']\n"
    "    h_bad_min = float('inf')\n"
    '    for eid in bad_eids:\n'
    '        h[eid] = max(hmin, beta * h[eid])\n'
    '        h_bad_min = min(h_bad_min, h[eid])\n'
    '    dist_ring = {eid: 0 for eid in bad_eids}\n'
    '    q = deque(bad_eids)\n'
    '    while q:\n'
    '        u = q.popleft()\n'
    "        for nb in topo['edge_adj'].get(u, []):\n"
    '            if nb in patch_eids and nb not in dist_ring:\n'
    '                dist_ring[nb] = dist_ring[u] + 1\n'
    '                q.append(nb)\n'
    '    for eid in patch_eids:\n'
    '        d = dist_ring.get(eid, None)\n'
    '        if d is None:\n'
    '            continue\n'
    '        cap = h_bad_min * r ** d\n'
    '        if h[eid] > cap:\n'
    '            h[eid] = cap\n'
    "    _project_gradation(h, topo, r_grad=param['r_grad'], sweeps=10)\n"
    '\n'
    'def _element_centroid(el):\n'
    '    nodes = el.getNodes()\n'
    '    pts = [n.coordinates for n in nodes]\n'
    '    c = (sum((p[0] for p in pts)) / len(pts), sum((p[1] for p in pts)) / len(pts), sum((p[2] for p in pts)) / len(pts))\n'
    '    return c\n'
    '\n'
    'def _pick_target_face(topo, bad_edges_sorted, top_k=20):\n'
    '    score = defaultdict(int)\n'
    '    for eid, cnt in bad_edges_sorted[:top_k]:\n'
    "        for fid in topo['edge_face_ids'].get(eid, []):\n"
    '            score[fid] += cnt\n'
    '    if not score:\n'
    '        return None\n'
    '    fid = max(score.keys(), key=lambda k: score[k])\n'
    '    return fid\n'
    '\n'
    'def _face_boundary_vertices(topo, fid):\n'
    '    vids = set()\n'
    "    for eid in topo['face_edge_ids'].get(fid, []):\n"
    "        for vid in topo['edge_vids'][eid]:\n"
    '            vids.add(vid)\n'
    '    return sorted(list(vids))\n'
    '\n'
    'def _pick_endpoints_vertices(inst, topo, fid, u_cut, center):\n'
    '    vids = _face_boundary_vertices(topo, fid)\n'
    '    if not vids:\n'
    '        return (None, None)\n'
    '    best_pos = None\n'
    '    best_neg = None\n'
    '    best_pos_s = -1e+100\n'
    '    best_neg_s = 1e+100\n'
    '    for vid in vids:\n'
    '        v = inst.vertices[vid]\n'
    '        p = tuple(v.pointOn[0])\n'
    '        s = _v_dot(_v_sub(p, center), u_cut)\n'
    '        if s > best_pos_s:\n'
    '            best_pos_s = s\n'
    '            best_pos = v\n'
    '        if s < best_neg_s:\n'
    '            best_neg_s = s\n'
    '            best_neg = v\n'
    '    if best_pos is None or best_neg is None:\n'
    '        return (None, None)\n'
    "    if hasattr(best_pos, 'index') and hasattr(best_neg, 'index') and (best_pos.index == best_neg.index):\n"
    '        return (None, None)\n'
    '    return (best_pos, best_neg)\n'
    '\n'
    'def _edge_midpoint(edge):\n'
    '    return tuple(edge.pointOn[0])\n'
    '\n'
    'def _fallback_endpoints_datums(a, inst, topo, fid, u_cut, center):\n'
    "    eids = topo['face_edge_ids'].get(fid, [])\n"
    '    if not eids:\n'
    '        return (None, None)\n'
    '    best_pos = None\n'
    '    best_neg = None\n'
    '    best_pos_s = -1e+100\n'
    '    best_neg_s = 1e+100\n'
    '    for eid in eids:\n'
    '        e = inst.edges[eid]\n'
    '        mp = _edge_midpoint(e)\n'
    '        s = _v_dot(_v_sub(mp, center), u_cut)\n'
    '        if s > best_pos_s:\n'
    '            best_pos_s = s\n'
    '            best_pos = e\n'
    '        if s < best_neg_s:\n'
    '            best_neg_s = s\n'
    '            best_neg = e\n'
    '    if best_pos is None or best_neg is None:\n'
    '        return (None, None)\n'
    '    feat1 = a.DatumPointByEdgeParam(edge=best_pos, parameter=0.5)\n'
    '    feat2 = a.DatumPointByEdgeParam(edge=best_neg, parameter=0.5)\n'
    '    p1 = a.datums[feat1.id]\n'
    '    p2 = a.datums[feat2.id]\n'
    '    return (p1, p2)\n'
    '\n'
    'def _maybe_partition(a, inst, topo, poor_labels, bad_edges_sorted, param, partition_count):\n'
    "    if not param['enable_partition'] or partition_count >= param['partition_max']:\n"
    '        return (False, partition_count)\n'
    '    fid = _pick_target_face(topo, bad_edges_sorted, top_k=30)\n'
    '    if fid is None:\n'
    '        return (False, partition_count)\n'
    '    pts = []\n'
    "    for eid in topo['face_edge_ids'].get(fid, []):\n"
    '        try:\n'
    '            for el in inst.edges[eid].getElements():\n'
    '                if el.label in poor_labels:\n'
    '                    pts.append(_element_centroid(el))\n'
    '        except Exception:\n'
    '            pass\n'
    '    if len(pts) < 10:\n'
    '        return (False, partition_count)\n'
    '    center = (sum((p[0] for p in pts)) / len(pts), sum((p[1] for p in pts)) / len(pts), sum((p[2] for p in pts)) / len(pts))\n'
    '    u_cut = _pca_minor_dir(pts)\n'
    '    p1, p2 = _pick_endpoints_vertices(inst, topo, fid, u_cut, center)\n'
    '    if p1 is None:\n'
    '        p1, p2 = _fallback_endpoints_datums(a, inst, topo, fid, u_cut, center)\n'
    '    if p1 is None:\n'
    '        return (False, partition_count)\n'
    '    try:\n'
    '        a.PartitionFaceByShortestPath(faces=(inst.faces[fid],), point1=p1, point2=p2)\n'
    '        return (True, partition_count + 1)\n'
    '    except Exception as ex:\n'
    "        print('M6 partition failed:', ex)\n"
    '        return (False, partition_count)\n'
    '\n'
    'def _h_stats(h):\n'
    '    vals = sorted(h.values())\n'
    '    if not vals:\n'
    '        return {}\n'
    '\n'
    '    def q(p):\n'
    '        return vals[int(p * (len(vals) - 1))]\n'
    "    return {'h_min': vals[0], 'h_med': q(0.5), 'h_p90': q(0.9), 'h_max': vals[-1]}\n"
    '\n'
    'def _log_jsonl(path, rec):\n'
    "    f = open(path, 'a')\n"
    "    f.write(json.dumps(rec) + '\\n')\n"
    '    f.close()\n'
    '\n'
    'def run_module6_v3(model, a, inst, particles_work_json, tol_plane, mesh_target_len=None, mesh_target_ratio=None, iface_refine_factor=None, curvature_factor=None, curvature_min_factor=None, gradation_ratio=None, min_size_ratio=None, max_size_ratio=None, min_size_abs=None, max_size_abs=None, param=None):\n'
    '    if param is None:\n'
    '        param = dict(M6_PARAM)\n'
    '    else:\n'
    '        tmp = dict(M6_PARAM)\n'
    '        tmp.update(param)\n'
    '        param = tmp\n'
    "    mesh_target_len = _m6_param(param, 'mesh_target_len', mesh_target_len)\n"
    "    mesh_target_ratio = _m6_param(param, 'mesh_target_ratio', mesh_target_ratio)\n"
    "    iface_refine_factor = _m6_param(param, 'iface_refine_factor', iface_refine_factor)\n"
    "    curvature_factor = _m6_param(param, 'curvature_factor', curvature_factor)\n"
    "    curvature_min_factor = _m6_param(param, 'curvature_min_factor', curvature_min_factor)\n"
    "    gradation_ratio = _m6_param(param, 'gradation_ratio', gradation_ratio)\n"
    "    min_size_ratio = _m6_param(param, 'min_size_ratio', min_size_ratio)\n"
    "    max_size_ratio = _m6_param(param, 'max_size_ratio', max_size_ratio)\n"
    "    min_size_abs = _m6_param(param, 'min_size_abs', min_size_abs)\n"
    "    max_size_abs = _m6_param(param, 'max_size_abs', max_size_abs)\n"
    "    with open(particles_work_json, 'r') as f:\n"
    '        _pw = json.load(f)\n'
    "    box = tuple(_pw['box'])\n"
    "    log_path = os.path.join(os.path.dirname(particles_work_json), param['log_jsonl_name'])\n"
    '    try:\n'
    '        os.remove(log_path)\n'
    '    except Exception:\n'
    '        pass\n'
    '    topo = _build_topology(inst)\n'
    "    pts = list(topo['vtx_xyz'].values())\n"
    '    bb0, bb1 = _bbox(pts)\n'
    '    Lchar_geom = max(bb1[0] - bb0[0], bb1[1] - bb0[1], bb1[2] - bb0[2])\n'
    '    Lchar = max(float(box[0]), float(box[1]), float(box[2]))\n'
    '    if Lchar <= 0:\n'
    "        raise RuntimeError('M6: invalid RVE box length')\n"
    '    eps_key = max(1e-12, float(tol_plane) / 5.0)\n'
    '    rep_of_eid, edge_groups = _build_periodic_edge_groups(inst, topo, box, tol_plane, eps_key)\n'
    "    nEdges = len(topo['edges'])\n"
    '    nGroups = len(edge_groups)\n'
    '    nMulti = 0\n'
    '    maxG = 1\n'
    '    for rep, eids in edge_groups.items():\n'
    '        if len(eids) > 1:\n'
    '            nMulti += 1\n'
    '        if len(eids) > maxG:\n'
    '            maxG = len(eids)\n'
    '    rep_eids = sorted(edge_groups.keys())\n'
    '    R_rep = {}\n'
    '    for rep in rep_eids:\n'
    "        R_rep[rep] = _edge_Rmin(topo['edges'][rep])\n"
    '    R_e = {}\n'
    "    for eid in range(len(topo['edges'])):\n"
    '        R_e[eid] = R_rep[rep_of_eid[eid]]\n'
    '    iface_eids = _m6_collect_iface_edge_ids(inst, topo)\n'
    '    h_global, h, size_stats = _m6_init_explicit_h(topo, R_e, iface_eids, Lchar, mesh_target_len, mesh_target_ratio, iface_refine_factor, curvature_factor, curvature_min_factor, gradation_ratio, min_size_ratio, max_size_ratio, min_size_abs, max_size_abs, param)\n'
    '    _sync_groups_h(h, topo, edge_groups, float(gradation_ratio))\n'
    "    _pw['mesh_target_len'] = float(h_global)\n"
    "    _pw['mesh_target_ratio'] = None if mesh_target_ratio is None else float(mesh_target_ratio)\n"
    "    _pw['iface_refine_factor'] = float(iface_refine_factor)\n"
    "    _pw['curvature_factor'] = float(curvature_factor)\n"
    "    _pw['curvature_min_factor'] = float(curvature_min_factor)\n"
    "    _pw['gradation_ratio'] = float(gradation_ratio)\n"
    "    _pw['edge_target_len'] = float(h_global)\n"
    "    with open(particles_work_json, 'w') as f:\n"
    '        json.dump(_pw, f, indent=2, sort_keys=True)\n'
    "    print('=== Module 6 v3.0 start ===')\n"
    "    print('Instance:', inst.name)\n"
    "    print('Lchar=%.6g' % float(Lchar))\n"
    "    print('Lchar_geom=%.6g' % float(Lchar_geom))\n"
    "    print('mesh_target_len=%.6g' % float(h_global))\n"
    "    print('mesh_target_ratio=%s' % str(mesh_target_ratio))\n"
    "    print('iface_refine_factor=%.6g curvature_factor=%.6g curvature_min_factor=%.6g gradation_ratio=%.6g' % (float(iface_refine_factor), float(curvature_factor), float(curvature_min_factor), float(gradation_ratio)))\n"
    "    print('[M6] periodic edge groups: edges=%d groups=%d multi_groups=%d max_group_size=%d' % (nEdges, nGroups, nMulti, maxG))\n"
    "    print('[M6] explicit sizing: curved_edges=%d iface_edges=%d curvature_floor_edges=%d h_min=%g h_max=%s' % (size_stats['n_curved_edges'], size_stats['n_iface_edges'], size_stats['n_curvature_floor_edges'], size_stats['h_min_abs'], str(size_stats['h_max_abs'])))\n"
    "    print('gradation worst ratio=%.3f (gradation_ratio=%.3f)' % (_worst_grad_ratio(h, topo), float(gradation_ratio)))\n"
    "    print('log:', log_path)\n"
    '    _apply_seeds_and_mesh(a, inst, topo, h_global, h, param, edge_groups)\n'
    "    rec = {'mode': 'explicit_size_field_one_shot', 'nElems': len(inst.elements), 'nNodes': len(inst.nodes), 'Lchar': float(Lchar), 'LcharGeom': float(Lchar_geom), 'mesh_target_len': float(h_global), 'mesh_target_ratio': None if mesh_target_ratio is None else float(mesh_target_ratio), 'iface_refine_factor': float(iface_refine_factor), 'curvature_factor': float(curvature_factor), 'curvature_min_factor': float(curvature_min_factor), 'gradation_ratio': float(gradation_ratio), 'hStats': _h_stats(h), 'worstGradRatio': _worst_grad_ratio(h, topo), 'sizingStats': size_stats, 'periodicGroups': {'nEdges': int(nEdges), 'nGroups': int(nGroups), 'multiGroups': int(nMulti), 'maxGroupSize': int(maxG)}}\n"
    '    _log_jsonl(log_path, rec)\n'
    "    hs = rec['hStats']\n"
    "    print('M6 one-shot elems=%d nodes=%d h[min/med/p90/max]=(%g,%g,%g,%g) worstGrad=%.3f' % (rec['nElems'], rec['nNodes'], hs['h_min'], hs['h_med'], hs['h_p90'], hs['h_max'], rec['worstGradRatio']))\n"
    "    print('=== Module 6 v3.0 done ===')\n"
    '\n'
    'def run_module6_v2(model, a, inst, particles_work_json, tol_plane, edge_target_len, param):\n'
    '    return run_module6_v3(model, a, inst, particles_work_json, tol_plane, mesh_target_len=float(edge_target_len), param=param)\n'
    'import json\n'
    'import os\n'
    'from abaqus import mdb\n'
    '\n'
    'def _part_from_instance_mesh(a, inst, name, copyPartSets=True):\n'
    "    if not hasattr(a, 'PartFromInstanceMesh'):\n"
    "        raise RuntimeError('rootAssembly has no PartFromInstanceMesh(). Please record a macro: Mesh -> Create Part from Mesh, then adapt the call.')\n"
    '    try:\n'
    '        a.PartFromInstanceMesh(name=name, instances=(inst,), copyPartSets=copyPartSets)\n'
    '        return\n'
    '    except TypeError:\n'
    '        pass\n'
    '    try:\n'
    '        a.PartFromInstanceMesh(name=name, partInstances=(inst,), copyPartSets=copyPartSets)\n'
    '        return\n'
    '    except TypeError as ex:\n'
    "        raise RuntimeError('PartFromInstanceMesh signature mismatch. Please paste the TypeError message.') from ex\n"
    '\n'
    'def _set_elem_count(part, set_name):\n'
    '    if set_name not in part.sets.keys():\n'
    '        return (False, 0, False)\n'
    '    s = part.sets[set_name]\n'
    '    try:\n'
    '        elems = s.elements\n'
    '    except Exception:\n'
    '        return (True, 0, False)\n'
    '    return (True, len(elems), True)\n'
    '\n'
    "def run_module7(model_name, particles_work_json, merged_instance_name, p_all_tri_name='P_ALL_TRI'):\n"
    '    if model_name not in mdb.models:\n'
    "        raise RuntimeError('Model not found: %s' % model_name)\n"
    '    model = mdb.models[model_name]\n'
    '    a = model.rootAssembly\n'
    '    if merged_instance_name not in a.instances:\n'
    "        raise RuntimeError('Missing instance: %s' % merged_instance_name)\n"
    '    inst = a.instances[merged_instance_name]\n'
    '    if len(inst.elements) == 0 or len(inst.nodes) == 0:\n'
    "        raise RuntimeError('Instance %s has no mesh. Run Module 6 generateMesh first.' % merged_instance_name)\n"
    '    if p_all_tri_name in model.parts:\n'
    '        del model.parts[p_all_tri_name]\n'
    '    _part_from_instance_mesh(a, inst, p_all_tri_name, copyPartSets=True)\n'
    '    if p_all_tri_name not in model.parts:\n'
    "        raise RuntimeError('Module 7 failed: part %s not created.' % p_all_tri_name)\n"
    '    p = model.parts[p_all_tri_name]\n'
    "    print('=== Module 7 ===')\n"
    "    print('Created part:', p_all_tri_name, 'nodes=', len(p.nodes), 'elements=', len(p.elements))\n"
    '    if not os.path.isfile(particles_work_json):\n'
    "        raise RuntimeError('particles_work.json not found: %s' % particles_work_json)\n"
    "    with open(particles_work_json, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    particles_work = data.get('particles_work', [])\n"
    '    if not particles_work:\n'
    "        raise RuntimeError('particles_work empty in %s' % particles_work_json)\n"
    '    missing = []\n'
    '    bad = []\n'
    '    ok = 0\n'
    '    req = 0\n'
    '    for pw in particles_work:\n'
    "        ID4 = pw['ID4']\n"
    "        touch = pw.get('touch', [])\n"
    "        iface = 'INC_%s_IFACE' % ID4\n"
    '        req += 1\n'
    '        exists, ne, hasElemsAttr = _set_elem_count(p, iface)\n'
    '        if not exists:\n'
    '            missing.append(iface)\n'
    '        elif not hasElemsAttr or ne == 0:\n'
    "            bad.append('%s (hasElements=%s nElems=%d)' % (iface, str(hasElemsAttr), ne))\n"
    '        else:\n'
    '            ok += 1\n'
    '        for tag in touch:\n'
    "            cap = 'INC_%s_CAP_%s' % (ID4, tag)\n"
    '            req += 1\n'
    '            exists, ne, hasElemsAttr = _set_elem_count(p, cap)\n'
    '            if not exists:\n'
    '                missing.append(cap)\n'
    '            elif not hasElemsAttr or ne == 0:\n'
    "                bad.append('%s (hasElements=%s nElems=%d)' % (cap, str(hasElemsAttr), ne))\n"
    '            else:\n'
    '                ok += 1\n'
    "    print('Set inheritance check: ok=%d / required=%d' % (ok, req))\n"
    '    if missing or bad:\n'
    "        msg = ['[Module 7 DoD FAIL] Inherited sets not usable on P_ALL_TRI.']\n"
    '        if missing:\n'
    "            msg.append('Missing sets (%d): %s' % (len(missing), ', '.join(missing[:50]) + (' ...' if len(missing) > 50 else '')))\n"
    '        if bad:\n'
    "            msg.append('Bad/empty sets (%d): %s' % (len(bad), '; '.join(bad[:20]) + (' ...' if len(bad) > 20 else '')))\n"
    "        raise RuntimeError('\\n'.join(msg))\n"
    "    print('=== Module 7 DoD OK ===')\n"
    '    return p\n'
    'import os, json, math, re\n'
    'from abaqus import mdb\n'
    'from abaqusConstants import OFF, AVERAGE\n'
    '\n'
    'def _m8p_editNode_xyz_off(part, nodeObj, xyz):\n'
    '    part.editNode(nodes=(nodeObj,), coordinates=((float(xyz[0]), float(xyz[1]), float(xyz[2])),), projectToGeometry=OFF)\n'
    '\n'
    'def _m8p_enforce_A1_B0_on_xyz(axis, Q0, xyz, Lx, Ly, Lz):\n'
    '    x, y, z = xyz\n'
    "    if axis == 'X':\n"
    '        x = Lx\n'
    "    elif axis == 'Y':\n"
    '        y = Ly\n'
    "    elif axis == 'Z':\n"
    '        z = Lz\n'
    "    if Q0 == 'X0':\n"
    '        x = 0.0\n'
    "    elif Q0 == 'Y0':\n"
    '        y = 0.0\n'
    "    elif Q0 == 'Z0':\n"
    '        z = 0.0\n'
    '    return (x, y, z)\n'
    '\n'
    'def _m8a_delete_part_if_exists(model, name):\n'
    '    if name in model.parts:\n'
    '        del model.parts[name]\n'
    '\n'
    'def _m8a_delete_set_if_exists(part, name):\n'
    '    if name in part.sets:\n'
    '        del part.sets[name]\n'
    '\n'
    'def _m8a_make_elem_set_from_labels(part, name, labels, allow_empty=True):\n'
    '    _m8a_delete_set_if_exists(part, name)\n'
    '    labs = sorted(set((int(x) for x in labels)))\n'
    '    if not labs:\n'
    '        if allow_empty:\n'
    '            return 0\n'
    "        raise RuntimeError('Empty element set not allowed: %s' % name)\n"
    "    if hasattr(part, 'SetFromElementLabels'):\n"
    '        part.SetFromElementLabels(name=name, elementLabels=tuple(labs))\n'
    '        return len(labs)\n'
    '    elems = part.elements.sequenceFromLabels(labels=labs)\n'
    '    part.Set(name=name, elements=elems)\n'
    '    return len(labs)\n'
    '\n'
    'def _m8a_make_node_set_from_labels(part, name, labels, allow_empty=True):\n'
    '    _m8a_delete_set_if_exists(part, name)\n'
    '    labs = sorted(set((int(x) for x in labels)))\n'
    '    if not labs:\n'
    '        if allow_empty:\n'
    '            return 0\n'
    "        raise RuntimeError('Empty node set not allowed: %s' % name)\n"
    "    if hasattr(part, 'SetFromNodeLabels'):\n"
    '        part.SetFromNodeLabels(name=name, nodeLabels=tuple(labs))\n'
    '        return len(labs)\n'
    '    nodes = part.nodes.sequenceFromLabels(labels=labs)\n'
    '    part.Set(name=name, nodes=nodes)\n'
    '    return len(labs)\n'
    '\n'
    'def _m8a_delete_elements_by_labels(part, labels, chunk=20000):\n'
    '    labs = sorted(set((int(x) for x in labels)))\n'
    '    if not labs:\n'
    '        return 0\n'
    '    n0 = len(labs)\n'
    '    for i in range(0, n0, chunk):\n'
    '        sub = labs[i:i + chunk]\n'
    '        elems = part.elements.sequenceFromLabels(labels=sub)\n'
    '        part.deleteElement(elements=elems)\n'
    '    return n0\n'
    '\n'
    'def _m8a_delete_unused_nodes(part, chunk=50000):\n'
    '    used = set()\n'
    '    for e in part.elements:\n'
    '        for nd in e.getNodes():\n'
    '            used.add(int(nd.label))\n'
    '    alln = [int(n.label) for n in part.nodes]\n'
    '    unused = sorted(set(alln) - used)\n'
    '    if not unused:\n'
    '        return 0\n'
    '    n0 = len(unused)\n'
    '    for i in range(0, n0, chunk):\n'
    '        sub = unused[i:i + chunk]\n'
    '        nodes = part.nodes.sequenceFromLabels(labels=sub)\n'
    '        part.deleteNode(nodes=nodes)\n'
    '    return n0\n'
    '\n'
    'def _m8a_canonicalize_xyz(xyz, Lx, Ly, Lz, tol_plane):\n'
    '    x, y, z = xyz\n'
    '    if abs(x - 0.0) <= tol_plane:\n'
    '        x = 0.0\n'
    '    elif abs(x - Lx) <= tol_plane:\n'
    '        x = float(Lx)\n'
    '    if abs(y - 0.0) <= tol_plane:\n'
    '        y = 0.0\n'
    '    elif abs(y - Ly) <= tol_plane:\n'
    '        y = float(Ly)\n'
    '    if abs(z - 0.0) <= tol_plane:\n'
    '        z = 0.0\n'
    '    elif abs(z - Lz) <= tol_plane:\n'
    '        z = float(Lz)\n'
    '    return (float(x), float(y), float(z))\n'
    '\n'
    'def _m8a_node_key_from_xyzC(xyzC, eps_key):\n'
    '    e = float(eps_key)\n'
    '    return (int(round(xyzC[0] / e)), int(round(xyzC[1] / e)), int(round(xyzC[2] / e)))\n'
    '\n'
    'def _m8a_is_in_domain_xyzC(xyzC, Lx, Ly, Lz, tol_plane):\n'
    '    x, y, z = xyzC\n'
    '    if x < -tol_plane or x > Lx + tol_plane:\n'
    '        return False\n'
    '    if y < -tol_plane or y > Ly + tol_plane:\n'
    '        return False\n'
    '    if z < -tol_plane or z > Lz + tol_plane:\n'
    '        return False\n'
    '    return True\n'
    '\n'
    'def _m8a_edge_key(kA, kB):\n'
    '    return (kA, kB) if kA <= kB else (kB, kA)\n'
    '\n'
    'def _m8a_outer_plane_of_elem_from_keys(k1, k2, k3, key2coordC, Lx, Ly, Lz):\n'
    '    x1, y1, z1 = key2coordC[k1]\n'
    '    x2, y2, z2 = key2coordC[k2]\n'
    '    x3, y3, z3 = key2coordC[k3]\n'
    '    hit = []\n'
    '    if x1 == 0.0 and x2 == 0.0 and (x3 == 0.0):\n'
    "        hit.append('X0')\n"
    '    if x1 == Lx and x2 == Lx and (x3 == Lx):\n'
    "        hit.append('X1')\n"
    '    if y1 == 0.0 and y2 == 0.0 and (y3 == 0.0):\n'
    "        hit.append('Y0')\n"
    '    if y1 == Ly and y2 == Ly and (y3 == Ly):\n'
    "        hit.append('Y1')\n"
    '    if z1 == 0.0 and z2 == 0.0 and (z3 == 0.0):\n'
    "        hit.append('Z0')\n"
    '    if z1 == Lz and z2 == Lz and (z3 == Lz):\n'
    "        hit.append('Z1')\n"
    '    if len(hit) == 0:\n'
    '        return None\n'
    '    if len(hit) > 1:\n'
    "        raise RuntimeError('[M8A REG FATAL] element lies on multiple OUTER planes: %s' % hit)\n"
    '    return hit[0]\n'
    '\n'
    'def _m8a_partition_outer_node_type_on_master_plane(P0, xyzC, Lx, Ly, Lz):\n'
    '    xC, yC, zC = xyzC\n'
    '    onX = xC == 0.0 or xC == Lx\n'
    '    onY = yC == 0.0 or yC == Ly\n'
    '    onZ = zC == 0.0 or zC == Lz\n'
    "    if P0 == 'X0':\n"
    '        return (onY or onZ, onY and onZ)\n'
    "    if P0 == 'Y0':\n"
    '        return (onX or onZ, onX and onZ)\n'
    "    if P0 == 'Z0':\n"
    '        return (onX or onY, onX and onY)\n'
    "    raise RuntimeError('Unknown master plane: %s' % P0)\n"
    '\n'
    'def _m8a_rebuild_key2nodelabel_post(part, Lx, Ly, Lz, tol_plane, eps_key):\n'
    '    key2lab = {}\n'
    '    for n in part.nodes:\n'
    '        xyzC = _m8a_canonicalize_xyz(n.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        k = _m8a_node_key_from_xyzC(xyzC, eps_key)\n'
    '        lab = int(n.label)\n'
    '        if k not in key2lab or lab < key2lab[k]:\n'
    '            key2lab[k] = lab\n'
    '    return key2lab\n'
    '\n'
    'def _m8a_rebuild_elemkey2label_post(part, Lx, Ly, Lz, tol_plane, eps_key):\n'
    '    ekey2lab = {}\n'
    '    for e in part.elements:\n'
    '        nds = e.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        ks = []\n'
    '        for nd in nds:\n'
    '            xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            ks.append(_m8a_node_key_from_xyzC(xyzC, eps_key))\n'
    '        ek = tuple(sorted((ks[0], ks[1], ks[2])))\n'
    '        lab = int(e.label)\n'
    '        if ek in ekey2lab and ekey2lab[ek] != lab:\n'
    "            raise RuntimeError('[M8A REG FATAL] duplicate elemKey maps to multiple elems after crop: %s' % str(ek))\n"
    '        ekey2lab[ek] = lab\n'
    '    return ekey2lab\n'
    '\n'
    'def _m8a_post_elem_labels_from_pre(pre_elem_labels, elemLabel2elemKey, elemKey2label_post):\n'
    '    out = []\n'
    '    miss = 0\n'
    '    for lab_pre in pre_elem_labels:\n'
    '        ek = elemLabel2elemKey.get(int(lab_pre), None)\n'
    '        if ek is None:\n'
    '            miss += 1\n'
    '            continue\n'
    '        lab_post = elemKey2label_post.get(ek, None)\n'
    '        if lab_post is None:\n'
    '            miss += 1\n'
    '            continue\n'
    '        out.append(int(lab_post))\n'
    '    return (sorted(set(out)), miss)\n'
    '\n'
    "def m8a_collect_registry_and_crop(model_name, particles_work_json, src_part_name='P_ALL_TRI', work_part_name='P_ALL_TRI_M8A', registry_json='m8_registry.json', make_sets=True, eps_key_factor=0.2):\n"
    '    if model_name not in mdb.models:\n'
    "        raise RuntimeError('Model not found: %s' % model_name)\n"
    '    model = mdb.models[model_name]\n'
    '    if not os.path.isfile(particles_work_json):\n'
    "        raise RuntimeError('particles_work.json not found: %s' % particles_work_json)\n"
    "    with open(particles_work_json, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    Lx, Ly, Lz = (float(x) for x in data['box'])\n"
    "    tol_plane = float(data['tol_plane'])\n"
    "    tol_pair = float(data.get('tol_pair', tol_plane))\n"
    '    eps_key = max(1e-18, float(eps_key_factor) * tol_pair)\n'
    '    if src_part_name not in model.parts:\n'
    "        raise RuntimeError('Missing part %s.' % src_part_name)\n"
    '    _m8a_delete_part_if_exists(model, work_part_name)\n'
    '    p_src = model.parts[src_part_name]\n'
    '    p = model.Part(name=work_part_name, objectToCopy=p_src)\n'
    "    print('=== M8A REGISTRY: created %s from %s ===' % (work_part_name, src_part_name))\n"
    "    print('[M8A REG] tol_plane=%g tol_pair=%g eps_key=%g' % (tol_plane, tol_pair, eps_key))\n"
    "    print('[M8A REG] before: nodes=%d elems=%d' % (len(p.nodes), len(p.elements)))\n"
    "    cap_set_re = re.compile('^INC_(\\\\d{4})_CAP_([XYZ]0)$')\n"
    '    inc_fams = {}\n'
    '    for sname in p.sets.keys():\n'
    '        m = cap_set_re.match(sname)\n'
    '        if m:\n'
    '            ID4 = m.group(1)\n'
    '            fam = m.group(2)\n'
    '            inc_fams.setdefault(ID4, set()).add(fam)\n'
    '    label2nodeObj_pre = dict(((int(n.label), n) for n in p.nodes))\n'
    '    key2coordC = {}\n'
    '    key2nodeLabel_pre = {}\n'
    '    elemLabel2keys = {}\n'
    '    elemLabel2elemKey = {}\n'
    '    elemLabel2inDomain = {}\n'
    '    nElems_total = len(p.elements)\n'
    '    nElems_inDomain = 0\n'
    '    for e in p.elements:\n'
    '        nds = e.getNodes()\n'
    '        if len(nds) != 3:\n'
    "            raise RuntimeError('[M8A REG FATAL] Non-TRI element found: label=%s' % e.label)\n"
    '        ks = []\n'
    '        inDom = True\n'
    '        for nd in nds:\n'
    '            xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            if not _m8a_is_in_domain_xyzC(xyzC, Lx, Ly, Lz, tol_plane):\n'
    '                inDom = False\n'
    '            k = _m8a_node_key_from_xyzC(xyzC, eps_key)\n'
    '            if k not in key2coordC:\n'
    '                key2coordC[k] = xyzC\n'
    '                key2nodeLabel_pre[k] = int(nd.label)\n'
    '            ks.append(k)\n'
    '        eLabel = int(e.label)\n'
    '        elemLabel2keys[eLabel] = (ks[0], ks[1], ks[2])\n'
    '        elemLabel2elemKey[eLabel] = tuple(sorted((ks[0], ks[1], ks[2])))\n'
    '        elemLabel2inDomain[eLabel] = bool(inDom)\n'
    '        if inDom:\n'
    '            nElems_inDomain += 1\n'
    "    print('[M8A REG:S1] nElems_total=%d nElems_inDomain=%d nNodeKeys=%d nInc=%d' % (nElems_total, nElems_inDomain, len(key2coordC), len(inc_fams)))\n"
    '    outerPlaneOfElem = {}\n'
    "    nOuterByPlane = {'X0': 0, 'X1': 0, 'Y0': 0, 'Y1': 0, 'Z0': 0, 'Z1': 0}\n"
    '    for eLabel, (k1, k2, k3) in elemLabel2keys.items():\n'
    '        if not elemLabel2inDomain[eLabel]:\n'
    '            continue\n'
    '        plane = _m8a_outer_plane_of_elem_from_keys(k1, k2, k3, key2coordC, Lx, Ly, Lz)\n'
    '        outerPlaneOfElem[eLabel] = plane\n'
    '        if plane is not None:\n'
    '            nOuterByPlane[plane] += 1\n'
    "    print('[M8A REG:S2] nOuterElemsByPlane =', nOuterByPlane)\n"
    "    master_planes = ['X0', 'Y0', 'Z0']\n"
    '    planes = {}\n'
    '    for P0 in master_planes:\n'
    '        outerElems = set([e for e, pl in outerPlaneOfElem.items() if pl == P0])\n'
    "        planes[P0] = {'outerElems': outerElems, 'planeNodes': set(), 'edgeNodes': set(), 'cornerNodes': set(), 'capSeedElems': set(), 'capElems': set(), 'capBoundaryEdges': set(), 'capBoundaryNodes_all': set(), 'capBoundaryNodes_interior': set(), 'capBoundarySharedNodes': set(), 'mustReuseKeys': set()}\n"
    '    for P0 in master_planes:\n'
    '        raw = set()\n'
    "        for eLabel in planes[P0]['outerElems']:\n"
    '            k1, k2, k3 = elemLabel2keys[eLabel]\n'
    '            raw.add(k1)\n'
    '            raw.add(k2)\n'
    '            raw.add(k3)\n'
    '        refined = set()\n'
    '        for k in raw:\n'
    '            n = label2nodeObj_pre[key2nodeLabel_pre[k]]\n'
    '            ok = False\n'
    '            for adjE in n.getElements():\n'
    '                el = int(adjE.label)\n'
    '                if not elemLabel2inDomain.get(el, False):\n'
    '                    continue\n'
    '                if outerPlaneOfElem.get(el, None) == P0:\n'
    '                    ok = True\n'
    '                    break\n'
    '            if ok:\n'
    '                refined.add(k)\n'
    "        planes[P0]['planeNodes'] = refined\n"
    "        print('[M8A REG:S4] %s planeNodes=%d' % (P0, len(refined)))\n"
    '    for P0 in master_planes:\n'
    '        edgeNodes = set()\n'
    '        cornerNodes = set()\n'
    "        for k in planes[P0]['planeNodes']:\n"
    '            isEdge, isCorner = _m8a_partition_outer_node_type_on_master_plane(P0, key2coordC[k], Lx, Ly, Lz)\n'
    '            if isEdge:\n'
    '                edgeNodes.add(k)\n'
    '            if isCorner:\n'
    '                cornerNodes.add(k)\n'
    "        planes[P0]['edgeNodes'] = edgeNodes\n"
    "        planes[P0]['cornerNodes'] = cornerNodes\n"
    "        print('[M8A REG:S5] %s edgeNodes=%d cornerNodes=%d' % (P0, len(edgeNodes), len(cornerNodes)))\n"
    '    capAllBySetElems = set()\n'
    "    sem_edge_all = {'X0Y0': set(), 'X0Z0': set(), 'Y0Z0': set()}\n"
    '    sem_corner_all = set()\n'
    '    for ID4, fams in inc_fams.items():\n'
    '        capNodesFam = {}\n'
    '        for fam in fams:\n'
    "            sname = 'INC_%s_CAP_%s' % (ID4, fam)\n"
    '            elems = [int(e.label) for e in p.sets[sname].elements]\n'
    '            elems = [e for e in elems if elemLabel2inDomain.get(e, False)]\n'
    '            capAllBySetElems |= set(elems)\n'
    '            nodes = set()\n'
    '            for eLabel in elems:\n'
    '                k1, k2, k3 = elemLabel2keys[eLabel]\n'
    '                nodes.add(k1)\n'
    '                nodes.add(k2)\n'
    '                nodes.add(k3)\n'
    '            capNodesFam[fam] = nodes\n'
    "        NX = capNodesFam.get('X0', set())\n"
    "        NY = capNodesFam.get('Y0', set())\n"
    "        NZ = capNodesFam.get('Z0', set())\n"
    '        if NX and NY:\n'
    "            sem_edge_all['X0Y0'] |= NX & NY\n"
    '        if NX and NZ:\n'
    "            sem_edge_all['X0Z0'] |= NX & NZ\n"
    '        if NY and NZ:\n'
    "            sem_edge_all['Y0Z0'] |= NY & NZ\n"
    '        if NX and NY and NZ:\n'
    '            sem_corner_all |= NX & NY & NZ\n'
    '    sem_edge_total = 0\n'
    '    for vv in sem_edge_all.values():\n'
    '        sem_edge_total += len(vv)\n'
    "    print('[M8A REG:S6] capAllBySetElems=%d semEdgeTotal=%d semCorner=%d' % (len(capAllBySetElems), sem_edge_total, len(sem_corner_all)))\n"
    '    for P0 in master_planes:\n'
    "        capSeed = set([e for e in planes[P0]['outerElems'] if e in capAllBySetElems])\n"
    "        planes[P0]['capSeedElems'] = capSeed\n"
    "        planes[P0]['capElems'] = set(capSeed)\n"
    "        print('[M8A REG:S7] %s capSeed=%d capElems=%d' % (P0, len(capSeed), len(planes[P0]['capElems'])))\n"
    '\n'
    '    def _is_outer_elem_any(eLabel):\n'
    '        return outerPlaneOfElem.get(eLabel, None) is not None\n'
    '    for P0 in master_planes:\n'
    '        edge2elems = {}\n'
    "        for eLabel in planes[P0]['outerElems']:\n"
    '            k1, k2, k3 = elemLabel2keys[eLabel]\n'
    '            for a, b in (_m8a_edge_key(k1, k2), _m8a_edge_key(k2, k3), _m8a_edge_key(k3, k1)):\n'
    '                edge2elems.setdefault((a, b), []).append(eLabel)\n'
    "        capSet = planes[P0]['capElems']\n"
    '        bEdges = set()\n'
    '        for (a, b), adj in edge2elems.items():\n'
    '            capCount = 0\n'
    '            for el in adj:\n'
    '                if el in capSet:\n'
    '                    capCount += 1\n'
    '            if capCount == 1 and len(adj) == 2:\n'
    '                bEdges.add((a, b))\n'
    '            elif capCount == 1 and len(adj) == 1 and (adj[0] in capSet):\n'
    '                bEdges.add((a, b))\n'
    '        bNodes = set()\n'
    '        for a, b in bEdges:\n'
    '            bNodes.add(a)\n'
    '            bNodes.add(b)\n'
    "        bNodes_interior = set(bNodes) - planes[P0]['edgeNodes'] - planes[P0]['cornerNodes']\n"
    '        shared = set()\n'
    '        for k in bNodes:\n'
    '            n = label2nodeObj_pre[key2nodeLabel_pre[k]]\n'
    '            ok = False\n'
    '            for adjE in n.getElements():\n'
    '                el = int(adjE.label)\n'
    '                if not elemLabel2inDomain.get(el, False):\n'
    '                    continue\n'
    '                if _is_outer_elem_any(el):\n'
    '                    continue\n'
    '                ok = True\n'
    '                break\n'
    '            if ok:\n'
    '                shared.add(k)\n'
    "        planes[P0]['capBoundaryEdges'] = bEdges\n"
    "        planes[P0]['capBoundaryNodes_all'] = bNodes\n"
    "        planes[P0]['capBoundaryNodes_interior'] = bNodes_interior\n"
    "        planes[P0]['capBoundarySharedNodes'] = shared\n"
    "        print('[M8A REG:S8] %s bEdges=%d bNodes=%d interior=%d shared=%d onEdge=%d onCorner=%d' % (P0, len(bEdges), len(bNodes), len(bNodes_interior), len(shared), len(bNodes & planes[P0]['edgeNodes']), len(bNodes & planes[P0]['cornerNodes'])))\n"
    '    sem_all = set()\n'
    "    sem_all |= sem_edge_all['X0Y0']\n"
    "    sem_all |= sem_edge_all['X0Z0']\n"
    "    sem_all |= sem_edge_all['Y0Z0']\n"
    '    sem_all |= sem_corner_all\n'
    "    sem_on_plane = {'X0': set(), 'Y0': set(), 'Z0': set()}\n"
    '    for k in sem_all:\n'
    '        if k not in key2coordC:\n'
    '            continue\n'
    '        xC, yC, zC = key2coordC[k]\n'
    '        if xC == 0.0:\n'
    "            sem_on_plane['X0'].add(k)\n"
    '        if yC == 0.0:\n'
    "            sem_on_plane['Y0'].add(k)\n"
    '        if zC == 0.0:\n'
    "            sem_on_plane['Z0'].add(k)\n"
    '    for P0 in master_planes:\n'
    '        must = set()\n'
    "        must |= planes[P0]['edgeNodes']\n"
    "        must |= planes[P0]['cornerNodes']\n"
    "        must |= planes[P0]['capBoundarySharedNodes']\n"
    '        must |= sem_on_plane[P0]\n'
    "        planes[P0]['mustReuseKeys'] = must\n"
    "        print('[M8A REG:S9] %s mustReuse=%d (edge=%d corner=%d capShared=%d sem=%d)' % (P0, len(must), len(planes[P0]['edgeNodes']), len(planes[P0]['cornerNodes']), len(planes[P0]['capBoundarySharedNodes']), len(sem_on_plane[P0])))\n"
    '    to_crop = [eLabel for eLabel, inDom in elemLabel2inDomain.items() if not inDom]\n'
    '    n_crop = _m8a_delete_elements_by_labels(p, to_crop)\n'
    '    n_unused = _m8a_delete_unused_nodes(p)\n'
    "    print('[M8A REG:CROP] deleted out-of-domain elems=%d, deleted unused nodes=%d' % (n_crop, n_unused))\n"
    "    print('[M8A REG] after crop: nodes=%d elems=%d' % (len(p.nodes), len(p.elements)))\n"
    '    key2nodeLabel_post = _m8a_rebuild_key2nodelabel_post(p, Lx, Ly, Lz, tol_plane, eps_key)\n'
    '    elemKey2label_post = _m8a_rebuild_elemkey2label_post(p, Lx, Ly, Lz, tol_plane, eps_key)\n'
    '    if make_sets:\n'
    "        print('[M8A REG:SETS] start creating visualization sets...')\n"
    '\n'
    '        def keys_to_post_labels(keys):\n'
    '            labs = []\n'
    '            for k in keys:\n'
    '                lab = key2nodeLabel_post.get(k, None)\n'
    '                if lab is not None:\n'
    '                    labs.append(int(lab))\n'
    '            return sorted(set(labs))\n'
    '        for P0 in master_planes:\n'
    "            post_outer, miss = _m8a_post_elem_labels_from_pre(planes[P0]['outerElems'], elemLabel2elemKey, elemKey2label_post)\n"
    "            print('[M8A REG:SETS] M8_OUTER_%s_ELEMS pre=%d post=%d miss=%d' % (P0, len(planes[P0]['outerElems']), len(post_outer), miss))\n"
    "            _m8a_make_elem_set_from_labels(p, 'M8_OUTER_%s_ELEMS' % P0, post_outer, allow_empty=False)\n"
    "            _m8a_make_node_set_from_labels(p, 'M8_%s_PLANENODES' % P0, keys_to_post_labels(planes[P0]['planeNodes']), allow_empty=False)\n"
    "            _m8a_make_node_set_from_labels(p, 'M8_%s_EDGENODES' % P0, keys_to_post_labels(planes[P0]['edgeNodes']), allow_empty=True)\n"
    "            _m8a_make_node_set_from_labels(p, 'M8_%s_CORNER_NODES' % P0, keys_to_post_labels(planes[P0]['cornerNodes']), allow_empty=True)\n"
    "            post_seed, _ = _m8a_post_elem_labels_from_pre(planes[P0]['capSeedElems'], elemLabel2elemKey, elemKey2label_post)\n"
    "            _m8a_make_elem_set_from_labels(p, 'M8_%s_CAP_SEED_ELEMS' % P0, post_seed, allow_empty=True)\n"
    "            post_cap, _ = _m8a_post_elem_labels_from_pre(planes[P0]['capElems'], elemLabel2elemKey, elemKey2label_post)\n"
    "            _m8a_make_elem_set_from_labels(p, 'M8_%s_CAP_ELEMS' % P0, post_cap, allow_empty=True)\n"
    "            _m8a_make_node_set_from_labels(p, 'M8_%s_CAP_BNDRY_NODES_ALL' % P0, keys_to_post_labels(planes[P0]['capBoundaryNodes_all']), allow_empty=True)\n"
    "            _m8a_make_node_set_from_labels(p, 'M8_%s_CAP_BNDRY_NODES_INTERIOR' % P0, keys_to_post_labels(planes[P0]['capBoundaryNodes_interior']), allow_empty=True)\n"
    "            _m8a_make_node_set_from_labels(p, 'M8_%s_CAP_BNDRY_SHARED_IFACE' % P0, keys_to_post_labels(planes[P0]['capBoundarySharedNodes']), allow_empty=True)\n"
    "            _m8a_make_node_set_from_labels(p, 'M8_%s_MUST_REUSE' % P0, keys_to_post_labels(planes[P0]['mustReuseKeys']), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(p, 'M8_SEM_EDGE_X0Y0', keys_to_post_labels(sem_edge_all['X0Y0']), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(p, 'M8_SEM_EDGE_X0Z0', keys_to_post_labels(sem_edge_all['X0Z0']), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(p, 'M8_SEM_EDGE_Y0Z0', keys_to_post_labels(sem_edge_all['Y0Z0']), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(p, 'M8_SEM_CORNER_X0Y0Z0', keys_to_post_labels(sem_corner_all), allow_empty=True)\n"
    "        print('[M8A REG:SETS] done.')\n"
    "    registry = {'box': [Lx, Ly, Lz], 'tol_plane': tol_plane, 'tol_pair': tol_pair, 'eps_key': eps_key, 'global': {'nElems_total': nElems_total, 'nElems_inDomain': nElems_inDomain, 'nNodeKeys_unique': len(key2coordC), 'nOuterElemsByPlane': nOuterByPlane}}\n"
    "    with open(registry_json, 'w') as f:\n"
    '        json.dump(registry, f, indent=2, sort_keys=True)\n'
    "    print('[M8A REG] registry saved:', registry_json)\n"
    '    return (p, registry)\n'
    'from collections import defaultdict\n'
    'from abaqusConstants import TRI3\n'
    '\n'
    'def _m8p_get_set_node_labels(part, set_name):\n'
    '    if set_name not in part.sets:\n'
    "        raise RuntimeError('[M8P FATAL] missing node set: %s' % set_name)\n"
    '    return set((int(n.label) for n in part.sets[set_name].nodes))\n'
    '\n'
    'def _m8p_get_set_elem_labels(part, set_name):\n'
    '    if set_name not in part.sets:\n'
    "        raise RuntimeError('[M8P FATAL] missing elem set: %s' % set_name)\n"
    '    return set((int(e.label) for e in part.sets[set_name].elements))\n'
    '\n'
    'def _m8p_plane_of_axis(axis):\n'
    "    return (axis + '0', axis + '1')\n"
    '\n'
    'def _m8p_shift(axis, Lx, Ly, Lz):\n'
    "    if axis == 'X':\n"
    '        return (Lx, 0.0, 0.0)\n'
    "    if axis == 'Y':\n"
    '        return (0.0, Ly, 0.0)\n'
    "    if axis == 'Z':\n"
    '        return (0.0, 0.0, Lz)\n'
    "    raise RuntimeError('unknown axis: %s' % axis)\n"
    '\n'
    'def _m8p_on_plane_xyzC(xyzC, plane, Lx, Ly, Lz):\n'
    '    x, y, z = xyzC\n'
    "    if plane == 'X0':\n"
    '        return x == 0.0\n'
    "    if plane == 'X1':\n"
    '        return x == Lx\n'
    "    if plane == 'Y0':\n"
    '        return y == 0.0\n'
    "    if plane == 'Y1':\n"
    '        return y == Ly\n'
    "    if plane == 'Z0':\n"
    '        return z == 0.0\n'
    "    if plane == 'Z1':\n"
    '        return z == Lz\n'
    "    raise RuntimeError('unknown plane: %s' % plane)\n"
    '\n'
    'def _m8p_outer_elem_labels_by_plane(part, plane, Lx, Ly, Lz, tol_plane):\n'
    '    labs = []\n'
    '    for el in part.elements:\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        ok = True\n'
    '        for nd in nds:\n'
    '            xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            if not _m8p_on_plane_xyzC(xyzC, plane, Lx, Ly, Lz):\n'
    '                ok = False\n'
    '                break\n'
    '        if ok:\n'
    '            labs.append(int(el.label))\n'
    '    return labs\n'
    '\n'
    'def _m8p_build_outerPlaneOfElem(part, Lx, Ly, Lz, tol_plane):\n'
    '    outerPlaneOfElem = {}\n'
    '    for el in part.elements:\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        xyzC = [_m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane) for nd in nds]\n'
    '        hit = []\n'
    '        if all((p[0] == 0.0 for p in xyzC)):\n'
    "            hit.append('X0')\n"
    '        if all((p[0] == Lx for p in xyzC)):\n'
    "            hit.append('X1')\n"
    '        if all((p[1] == 0.0 for p in xyzC)):\n'
    "            hit.append('Y0')\n"
    '        if all((p[1] == Ly for p in xyzC)):\n'
    "            hit.append('Y1')\n"
    '        if all((p[2] == 0.0 for p in xyzC)):\n'
    "            hit.append('Z0')\n"
    '        if all((p[2] == Lz for p in xyzC)):\n'
    "            hit.append('Z1')\n"
    '        if len(hit) == 0:\n'
    '            outerPlaneOfElem[int(el.label)] = None\n'
    '        elif len(hit) == 1:\n'
    '            outerPlaneOfElem[int(el.label)] = hit[0]\n'
    '        else:\n'
    "            raise RuntimeError('[M8P FATAL] element lies on multiple OUTER planes: el=%d hit=%s' % (int(el.label), str(hit)))\n"
    '    return outerPlaneOfElem\n'
    '\n'
    'def _m8p_dist2_point_to_segment_2d(px, py, ax, ay, bx, by):\n'
    '    abx = bx - ax\n'
    '    aby = by - ay\n'
    '    apx = px - ax\n'
    '    apy = py - ay\n'
    '    ab2 = abx * abx + aby * aby\n'
    '    if ab2 <= 1e-30:\n'
    '        dx = px - ax\n'
    '        dy = py - ay\n'
    '        return dx * dx + dy * dy\n'
    '    t = (apx * abx + apy * aby) / ab2\n'
    '    if t < 0.0:\n'
    '        dx = px - ax\n'
    '        dy = py - ay\n'
    '        return dx * dx + dy * dy\n'
    '    if t > 1.0:\n'
    '        dx = px - bx\n'
    '        dy = py - by\n'
    '        return dx * dx + dy * dy\n'
    '    qx = ax + t * abx\n'
    '    qy = ay + t * aby\n'
    '    dx = px - qx\n'
    '    dy = py - qy\n'
    '    return dx * dx + dy * dy\n'
    '\n'
    'def _m8p_find_meshedge_on_intersection(part, labA, labB, axis, Q0, Lx, Ly, Lz, tol_plane):\n'
    '    nA = part.nodes.getFromLabel(label=int(labA))\n'
    '    for me in nA.getElemEdges():\n'
    '        n1, n2 = me.getNodes()\n'
    '        labs = {int(n1.label), int(n2.label)}\n'
    '        if labs != {int(labA), int(labB)}:\n'
    '            continue\n'
    '        xyz1 = _m8a_canonicalize_xyz(n1.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        xyz2 = _m8a_canonicalize_xyz(n2.coordinates, Lx, Ly, Lz, tol_plane)\n'
    "        if _m8p_on_plane_xyzC(xyz1, axis + '1', Lx, Ly, Lz) and _m8p_on_plane_xyzC(xyz1, Q0, Lx, Ly, Lz) and _m8p_on_plane_xyzC(xyz2, axis + '1', Lx, Ly, Lz) and _m8p_on_plane_xyzC(xyz2, Q0, Lx, Ly, Lz):\n"
    '            return me\n'
    '    return None\n'
    '\n'
    'def _m8p_clear_image_planes(part, Lx, Ly, Lz, tol_plane):\n'
    '    total_del = 0\n'
    "    for plane in ('X1', 'Y1', 'Z1'):\n"
    '        labs = _m8p_outer_elem_labels_by_plane(part, plane, Lx, Ly, Lz, tol_plane)\n'
    '        ndel = _m8a_delete_elements_by_labels(part, labs)\n'
    '        nnd = _m8a_delete_unused_nodes(part)\n'
    "        print('[M8P] clear %s: delElems=%d delUnusedNodes=%d' % (plane, ndel, nnd))\n"
    '        total_del += ndel\n'
    "    for plane in ('X1', 'Y1', 'Z1'):\n"
    '        n = len(_m8p_outer_elem_labels_by_plane(part, plane, Lx, Ly, Lz, tol_plane))\n'
    '        if n != 0:\n'
    "            raise RuntimeError('[M8P FATAL] image plane not fully cleared: %s has outerElems=%d' % (plane, n))\n"
    "    print('[M8P] image planes cleared OK (X1/Y1/Z1 outer elems all zero). totalDeletedElems=%d' % total_del)\n"
    '\n'
    'def _m8p_build_node_maps(part, Lx, Ly, Lz, tol_plane, eps_key):\n'
    '    lab2node = {}\n'
    '    lab2xyzC = {}\n'
    '    key2lab = {}\n'
    '    for nd in part.nodes:\n'
    '        lab = int(nd.label)\n'
    '        xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        k = _m8a_node_key_from_xyzC(xyzC, eps_key)\n'
    '        lab2node[lab] = nd\n'
    '        lab2xyzC[lab] = xyzC\n'
    '        if k not in key2lab or lab < key2lab[k]:\n'
    '            key2lab[k] = lab\n'
    '    return (lab2node, lab2xyzC, key2lab)\n'
    '\n'
    'def _m8p_adj_has_nonouter(part, node_obj, Lx, Ly, Lz, tol_plane):\n'
    '    for el in node_obj.getElements():\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        hits = []\n'
    "        for plane in ('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1'):\n"
    '            ok = True\n'
    '            for nd in nds:\n'
    '                xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '                if not _m8p_on_plane_xyzC(xyzC, plane, Lx, Ly, Lz):\n'
    '                    ok = False\n'
    '                    break\n'
    '            if ok:\n'
    '                hits.append(plane)\n'
    '                break\n'
    '        if not hits:\n'
    '            return True\n'
    '    return False\n'
    '\n'
    'def _m8p_check_cap_boundary_semantics(part, Lx, Ly, Lz, tol_plane, debug_n=20):\n'
    "    print('=== [M8P] CAP boundary semantics check (hard gate) ===')\n"
    "    for P0 in ('X0', 'Y0', 'Z0'):\n"
    "        cap_all = _m8p_get_set_node_labels(part, 'M8_%s_CAP_BNDRY_NODES_ALL' % P0)\n"
    "        cap_shared = _m8p_get_set_node_labels(part, 'M8_%s_CAP_BNDRY_SHARED_IFACE' % P0)\n"
    "        edge = _m8p_get_set_node_labels(part, 'M8_%s_EDGENODES' % P0)\n"
    "        corner = _m8p_get_set_node_labels(part, 'M8_%s_CORNER_NODES' % P0)\n"
    '        ok_union = set()\n'
    '        ok_union |= cap_shared\n'
    '        ok_union |= edge\n'
    '        ok_union |= corner\n'
    '        diff = set(cap_all) - ok_union\n'
    '        cap_all_n = len(cap_all)\n'
    '        cap_shared_n = len(cap_shared & cap_all)\n'
    '        ratio = float(cap_shared_n) / float(cap_all_n) if cap_all_n > 0 else 1.0\n'
    "        print('[M8P:CAPCHK] %s CAP_ALL=%d CAP_SHARED(inCAP)=%d EDGE=%d CORNER=%d DIFF=%d shared_ratio=%.3f' % (P0, cap_all_n, cap_shared_n, len(edge), len(corner), len(diff), ratio))\n"
    '        if diff:\n'
    '            samples = []\n'
    '            for lab in list(diff)[:debug_n]:\n'
    '                nd = part.nodes.getFromLabel(label=int(lab))\n'
    '                xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '                has_nonouter = _m8p_adj_has_nonouter(part, nd, Lx, Ly, Lz, tol_plane)\n'
    '                samples.append((int(lab), xyzC, bool(has_nonouter)))\n'
    "            msg = ['[M8P FATAL] CAP boundary semantics violated on %s.' % P0, 'DIFF = CAP_ALL - (CAP_SHARED ∪ EDGE ∪ CORNER) must be empty, but DIFF=%d.' % len(diff), 'Sample DIFF nodes: (label, xyzC, adjHasNonOuter) => %s' % str(samples)]\n"
    "            raise RuntimeError('\\n'.join(msg))\n"
    "    print('=== [M8P] CAP boundary semantics OK ===')\n"
    '\n'
    'def _m8p_supported_on_image(axis, xyzC_master, done_images, Lx, Ly, Lz):\n'
    '    x, y, z = xyzC_master\n'
    "    if axis == 'X':\n"
    '        if y == 0.0 or z == 0.0:\n'
    '            return True\n'
    "        if y == Ly and 'Y' in done_images:\n"
    '            return True\n'
    "        if z == Lz and 'Z' in done_images:\n"
    '            return True\n'
    '        return False\n'
    "    if axis == 'Y':\n"
    '        if x == 0.0 or z == 0.0:\n'
    '            return True\n'
    "        if x == Lx and 'X' in done_images:\n"
    '            return True\n'
    "        if z == Lz and 'Z' in done_images:\n"
    '            return True\n'
    '        return False\n'
    "    if axis == 'Z':\n"
    '        if x == 0.0 or y == 0.0:\n'
    '            return True\n'
    "        if x == Lx and 'X' in done_images:\n"
    '            return True\n'
    "        if y == Ly and 'Y' in done_images:\n"
    '            return True\n'
    '        return False\n'
    "    raise RuntimeError('unknown axis: %s' % axis)\n"
    '\n'
    'def _m8p_build_REQ_strict(part, axis, done_images, lab2xyzC, Lx, Ly, Lz):\n'
    "    P0 = axis + '0'\n"
    "    cap_all = _m8p_get_set_node_labels(part, 'M8_%s_CAP_BNDRY_NODES_ALL' % P0)\n"
    "    cap_shared = _m8p_get_set_node_labels(part, 'M8_%s_CAP_BNDRY_SHARED_IFACE' % P0)\n"
    "    edge_all = _m8p_get_set_node_labels(part, 'M8_%s_EDGENODES' % P0)\n"
    "    corner_all = _m8p_get_set_node_labels(part, 'M8_%s_CORNER_NODES' % P0)\n"
    '    edge_full = set()\n'
    '    for lab in edge_all:\n'
    '        xyzC = lab2xyzC[int(lab)]\n'
    '        if _m8p_supported_on_image(axis, xyzC, done_images, Lx, Ly, Lz):\n'
    '            edge_full.add(int(lab))\n'
    '    corner_full = set()\n'
    '    for lab in corner_all:\n'
    '        xyzC = lab2xyzC[int(lab)]\n'
    '        if _m8p_supported_on_image(axis, xyzC, done_images, Lx, Ly, Lz):\n'
    '            corner_full.add(int(lab))\n'
    '    exclude = set()\n'
    '    for lab in cap_all:\n'
    '        lab = int(lab)\n'
    '        if lab in cap_shared:\n'
    '            continue\n'
    '        xyzC = lab2xyzC[lab]\n'
    '        if not _m8p_supported_on_image(axis, xyzC, done_images, Lx, Ly, Lz):\n'
    '            exclude.add(lab)\n'
    '    req = set(cap_all)\n'
    '    req -= exclude\n'
    '    req |= set((int(x) for x in cap_shared))\n'
    '    req |= edge_full\n'
    '    req |= corner_full\n'
    "    print('[M8P:REQ] %s done_images=%s REQ=%d EXCLUDE=%d EDGE_FULL=%d CORNER_FULL=%d CAP_ALL=%d CAP_SHARED=%d' % (P0, str(sorted(list(done_images))), len(req), len(exclude), len(edge_full), len(corner_full), len(cap_all), len(cap_shared)))\n"
    '    return req\n'
    '\n'
    'def _m8p_transverse_key(axis, xyzC, tol_pair):\n'
    '    x, y, z = xyzC\n'
    '    tp = float(tol_pair)\n'
    "    if axis == 'X':\n"
    '        return (int(math.floor(y / tp)), int(math.floor(z / tp)))\n'
    "    if axis == 'Y':\n"
    '        return (int(math.floor(x / tp)), int(math.floor(z / tp)))\n'
    "    if axis == 'Z':\n"
    '        return (int(math.floor(x / tp)), int(math.floor(y / tp)))\n'
    "    raise RuntimeError('unknown axis: %s' % axis)\n"
    '\n'
    'def _m8p_sig_1ring_shifted(node_obj, axis, shift, Lx, Ly, Lz, tol_plane, eps_key):\n'
    '    sx, sy, sz = shift\n'
    '    keys = set()\n'
    '    for el in node_obj.getElements():\n'
    '        for nd in el.getNodes():\n'
    '            xyz = nd.coordinates\n'
    '            xyzS = (xyz[0] + sx, xyz[1] + sy, xyz[2] + sz)\n'
    '            xyzC = _m8a_canonicalize_xyz(xyzS, Lx, Ly, Lz, tol_plane)\n'
    '            k = _m8a_node_key_from_xyzC(xyzC, eps_key)\n'
    '            keys.add(k)\n'
    '    return keys\n'
    '\n'
    'def _m8p_sig_1ring(node_obj, Lx, Ly, Lz, tol_plane, eps_key):\n'
    '    keys = set()\n'
    '    for el in node_obj.getElements():\n'
    '        for nd in el.getNodes():\n'
    '            xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            k = _m8a_node_key_from_xyzC(xyzC, eps_key)\n'
    '            keys.add(k)\n'
    '    return keys\n'
    '\n'
    'def _m8p_find_nearest_on_image_plane_by_scan(part, axis, xyzC_m, plane_img, Lx, Ly, Lz, tol_plane):\n'
    '    ym, zm = (xyzC_m[1], xyzC_m[2])\n'
    '    xm = xyzC_m[0]\n'
    '    best = None\n'
    '    for nd in part.nodes:\n'
    '        xyz = nd.coordinates\n'
    '        xyzC = _m8a_canonicalize_xyz(xyz, Lx, Ly, Lz, tol_plane)\n'
    "        if plane_img == 'X1':\n"
    '            if abs(xyzC[0] - Lx) > tol_plane:\n'
    '                continue\n'
    '            dy = xyzC[1] - ym\n'
    '            dz = xyzC[2] - zm\n'
    '            d2 = dy * dy + dz * dz\n'
    "        elif plane_img == 'Y1':\n"
    '            if abs(xyzC[1] - Ly) > tol_plane:\n'
    '                continue\n'
    '            dx = xyzC[0] - xyzC_m[0]\n'
    '            dz = xyzC[2] - xyzC_m[2]\n'
    '            d2 = dx * dx + dz * dz\n'
    "        elif plane_img == 'Z1':\n"
    '            if abs(xyzC[2] - Lz) > tol_plane:\n'
    '                continue\n'
    '            dx = xyzC[0] - xyzC_m[0]\n'
    '            dy = xyzC[1] - xyzC_m[1]\n'
    '            d2 = dx * dx + dy * dy\n'
    '        else:\n'
    '            return None\n'
    '        if best is None or d2 < best[0]:\n'
    '            best = (d2, int(nd.label), xyzC)\n'
    '    return best\n'
    '\n'
    'def _m8p_pair_REQ(part, axis, req, plane_img, lab2node, lab2xyzC, Lx, Ly, Lz, tol_plane, tol_pair, eps_key, shift):\n'
    '    reuseCand = []\n'
    '    for lab, xyzC in lab2xyzC.items():\n'
    '        if _m8p_on_plane_xyzC(xyzC, plane_img, Lx, Ly, Lz):\n'
    '            reuseCand.append(int(lab))\n'
    '    reuseCand_set = set(reuseCand)\n'
    '    buckets = defaultdict(list)\n'
    '    for lab in reuseCand:\n'
    '        buckets[_m8p_transverse_key(axis, lab2xyzC[lab], tol_pair)].append(lab)\n'
    '    sigM_cache = {}\n'
    '    sigI_cache = {}\n'
    '    used_img = set()\n'
    '    pair = {}\n'
    '    unmatched = []\n'
    '    for m in sorted(list(req)):\n'
    '        xyzM = lab2xyzC.get(int(m), None)\n'
    '        if xyzM is None:\n'
    '            unmatched.append(int(m))\n'
    '            continue\n'
    '        key0 = _m8p_transverse_key(axis, xyzM, tol_pair)\n'
    '        cset = set()\n'
    '        for di in (-1, 0, 1):\n'
    '            for dj in (-1, 0, 1):\n'
    '                kk = (key0[0] + di, key0[1] + dj)\n'
    '                for c in buckets.get(kk, []):\n'
    '                    if c in used_img:\n'
    '                        continue\n'
    '                    xyzI = lab2xyzC[c]\n'
    "                    if axis == 'X':\n"
    '                        if abs(xyzI[1] - xyzM[1]) <= tol_pair and abs(xyzI[2] - xyzM[2]) <= tol_pair:\n'
    '                            cset.add(int(c))\n'
    "                    elif axis == 'Y':\n"
    '                        if abs(xyzI[0] - xyzM[0]) <= tol_pair and abs(xyzI[2] - xyzM[2]) <= tol_pair:\n'
    '                            cset.add(int(c))\n'
    "                    elif axis == 'Z':\n"
    '                        if abs(xyzI[0] - xyzM[0]) <= tol_pair and abs(xyzI[1] - xyzM[1]) <= tol_pair:\n'
    '                            cset.add(int(c))\n'
    '        cands = sorted(list(cset))\n'
    '        if not cands:\n'
    '            unmatched.append(int(m))\n'
    '            continue\n'
    '        if len(cands) == 1:\n'
    '            pair[int(m)] = int(cands[0])\n'
    '            used_img.add(int(cands[0]))\n'
    '            continue\n'
    '        if int(m) not in sigM_cache:\n'
    '            sigM_cache[int(m)] = _m8p_sig_1ring_shifted(lab2node[int(m)], axis, shift, Lx, Ly, Lz, tol_plane, eps_key)\n'
    '        sigM = sigM_cache[int(m)]\n'
    '        matches = []\n'
    '        for c in cands:\n'
    '            if int(c) not in sigI_cache:\n'
    '                sigI_cache[int(c)] = _m8p_sig_1ring(lab2node[int(c)], Lx, Ly, Lz, tol_plane, eps_key)\n'
    '            if sigI_cache[int(c)] == sigM:\n'
    '                matches.append(int(c))\n'
    '        if len(matches) != 1:\n'
    "            raise RuntimeError('[M8P FATAL] ambiguous pairing unresolved: axis=%s masterNode=%d candidates=%d topoMatches=%d' % (axis, int(m), len(cands), len(matches)))\n"
    '        pair[int(m)] = int(matches[0])\n'
    '        used_img.add(int(matches[0]))\n'
    '    if unmatched:\n'
    "        P0 = axis + '0'\n"
    "        cap_all = _m8p_get_set_node_labels(part, 'M8_%s_CAP_BNDRY_NODES_ALL' % P0)\n"
    "        cap_shared = _m8p_get_set_node_labels(part, 'M8_%s_CAP_BNDRY_SHARED_IFACE' % P0)\n"
    "        edge_all = _m8p_get_set_node_labels(part, 'M8_%s_EDGENODES' % P0)\n"
    "        corner_all = _m8p_get_set_node_labels(part, 'M8_%s_CORNER_NODES' % P0)\n"
    '        unm_set = set((int(m) for m in unmatched))\n'
    '        n_in_shared = len(unm_set & cap_shared)\n'
    '        n_in_edge = len(unm_set & edge_all)\n'
    '        n_in_corner = len(unm_set & corner_all)\n'
    "        print('[M8P:UNMATCHED] axis=%s unmatched=%d inCAP_SHARED=%d inEDGE=%d inCORNER=%d' % (axis, len(unmatched), n_in_shared, n_in_edge, n_in_corner))\n"
    '        for m in unmatched[:10]:\n'
    '            xyzM = lab2xyzC[m]\n'
    '            k = _m8p_transverse_key(axis, xyzM, tol_pair)\n'
    "            print('[M8P:UNMATCHED] m=%d xyzC=%s key=%s bucketSize=%d' % (m, str(xyzM), str(k), len(buckets.get(k, []))))\n"
    '            nearest = _m8p_find_nearest_on_image_plane_by_scan(part, axis, xyzM, plane_img, Lx, Ly, Lz, tol_plane)\n'
    '            if nearest is None:\n'
    "                print('[M8P:UNMATCHED] m=%d nearest=None' % m)\n"
    '            else:\n'
    '                d2, labN, xyzCN = nearest\n'
    "                print('[M8P:UNMATCHED] m=%d nearestOn%s lab=%d xyzC=%s d_trans=%.6g' % (m, plane_img, labN, str(xyzCN), d2 ** 0.5))\n"
    "        _m8a_make_node_set_from_labels(part, 'M8P_REQ_%s0' % axis, list(req), allow_empty=False)\n"
    "        _m8a_make_node_set_from_labels(part, 'M8P_REUSECAND_%s1' % axis, list(reuseCand_set), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(part, 'M8P_UNMATCHED_%s0' % axis, list(unmatched), allow_empty=False)\n"
    "        raise RuntimeError('[M8P FATAL] REQ unmatched on %s: count=%d examples=%s' % (axis, len(unmatched), str(unmatched[:20])))\n"
    '    inv = defaultdict(int)\n'
    '    for m, i in pair.items():\n'
    '        inv[int(i)] += 1\n'
    '    bad = [i for i, cnt in inv.items() if cnt > 1]\n'
    '    if bad:\n'
    "        raise RuntimeError('[M8P FATAL] multi-to-one mapping detected on %s, example image node labels=%s' % (axis, str(bad[:10])))\n"
    "    print('[M8P:PAIR] %s paired=%d reuseCand_on_%s=%d' % (axis, len(pair), plane_img, len(reuseCand_set)))\n"
    '    return pair\n'
    '\n'
    'def _m8p_copy_outer_plane(part, axis, P0, P1, pair_map, lab2node, lab2xyzC, key2lab, Lx, Ly, Lz, tol_plane, eps_key):\n'
    '    shift = _m8p_shift(axis, Lx, Ly, Lz)\n'
    '    sx, sy, sz = shift\n'
    "    outer_master = _m8p_get_set_elem_labels(part, 'M8_OUTER_%s_ELEMS' % P0)\n"
    '    if not outer_master:\n'
    "        raise RuntimeError('[M8P FATAL] empty master outer elem set: M8_OUTER_%s_ELEMS' % P0)\n"
    '    n_created_nodes = 0\n'
    '    n_created_elems = 0\n'
    '    for eLab in sorted(list(outer_master)):\n'
    '        el = part.elements.getFromLabel(label=int(eLab))\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        img_nodes = []\n'
    '        for nd in nds:\n'
    '            mLab = int(nd.label)\n'
    '            if mLab in pair_map:\n'
    '                iLab = int(pair_map[mLab])\n'
    '                img_nodes.append(lab2node[iLab])\n'
    '                continue\n'
    '            xyzM = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            xyzI = (xyzM[0] + sx, xyzM[1] + sy, xyzM[2] + sz)\n'
    '            xyzIC = _m8a_canonicalize_xyz(xyzI, Lx, Ly, Lz, tol_plane)\n'
    '            kI = _m8a_node_key_from_xyzC(xyzIC, eps_key)\n'
    '            exist = key2lab.get(kI, None)\n'
    '            if exist is not None:\n'
    '                img_nodes.append(lab2node[int(exist)])\n'
    '                continue\n'
    '            newn = part.Node(coordinates=xyzIC)\n'
    '            newLab = int(newn.label)\n'
    '            lab2node[newLab] = newn\n'
    '            lab2xyzC[newLab] = xyzIC\n'
    '            key2lab[kI] = newLab\n'
    '            img_nodes.append(newn)\n'
    '            n_created_nodes += 1\n'
    '        part.Element(nodes=(img_nodes[0], img_nodes[2], img_nodes[1]), elemShape=TRI3)\n'
    '        n_created_elems += 1\n'
    "    print('[M8P:COPY] %s->%s createdElems=%d createdNodes=%d' % (P0, P1, n_created_elems, n_created_nodes))\n"
    '\n'
    'def _m8p_free_edge_and_nonmanifold(part):\n'
    '    edge_cnt = defaultdict(int)\n'
    '    for el in part.elements:\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        labs = [int(n.label) for n in nds]\n'
    '        e12 = (labs[0], labs[1])\n'
    '        e23 = (labs[1], labs[2])\n'
    '        e31 = (labs[2], labs[0])\n'
    '        for a, b in (e12, e23, e31):\n'
    '            if a > b:\n'
    '                a, b = (b, a)\n'
    '            edge_cnt[a, b] += 1\n'
    '    free = 0\n'
    '    nonm = 0\n'
    '    for k, c in edge_cnt.items():\n'
    '        if c == 1:\n'
    '            free += 1\n'
    '        elif c > 2:\n'
    '            nonm += 1\n'
    '    return (free, nonm)\n'
    '\n'
    'def _m8p_quick_missing_check(part, axes_done, Lx, Ly, Lz, tol_plane, tol_pair, stage_tag):\n'
    '    for ax in sorted(list(axes_done)):\n'
    '        miss, n0, n1 = _m8p_periodic_missing(part, ax, Lx, Ly, Lz, tol_plane, tol_pair)\n'
    "        print('[M8P:QMISS] %s axis=%s missing=%d (n%s0=%d n%s1=%d)' % (stage_tag, ax, miss, ax, n0, ax, n1))\n"
    '        if miss != 0:\n'
    "            print('[REGEN_JSON_REQUIRED] reason=M8P_QUICK_MISSING stage=%s axis=%s missing=%d' % (stage_tag, ax, miss))\n"
    "            raise RuntimeError('[M8P FATAL] quick missing check failed at %s: axis=%s missing=%d' % (stage_tag, ax, miss))\n"
    '\n'
    'def _m8p_periodic_missing(part, axis, Lx, Ly, Lz, tol_plane, tol_pair):\n'
    "    P0 = axis + '0'\n"
    "    P1 = axis + '1'\n"
    '    nodes0 = []\n'
    '    nodes1 = []\n'
    '    for nd in part.nodes:\n'
    '        xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        if _m8p_on_plane_xyzC(xyzC, P0, Lx, Ly, Lz):\n'
    '            nodes0.append((int(nd.label), xyzC))\n'
    '        elif _m8p_on_plane_xyzC(xyzC, P1, Lx, Ly, Lz):\n'
    '            nodes1.append((int(nd.label), xyzC))\n'
    '    buckets1 = defaultdict(list)\n'
    '    for lab1, xyz1 in nodes1:\n'
    '        buckets1[_m8p_transverse_key(axis, xyz1, tol_pair)].append((lab1, xyz1))\n'
    '    used1 = set()\n'
    '    miss0 = []\n'
    '    for lab0, xyz0 in nodes0:\n'
    '        k0 = _m8p_transverse_key(axis, xyz0, tol_pair)\n'
    '        best = None\n'
    '        for di in (-1, 0, 1):\n'
    '            for dj in (-1, 0, 1):\n'
    '                kk = (k0[0] + di, k0[1] + dj)\n'
    '                for lab1, xyz1 in buckets1.get(kk, []):\n'
    '                    if lab1 in used1:\n'
    '                        continue\n'
    "                    if axis == 'X':\n"
    '                        dy = abs(xyz1[1] - xyz0[1])\n'
    '                        dz = abs(xyz1[2] - xyz0[2])\n'
    '                        if dy > tol_pair or dz > tol_pair:\n'
    '                            continue\n'
    '                        d2 = dy * dy + dz * dz\n'
    "                    elif axis == 'Y':\n"
    '                        dx = abs(xyz1[0] - xyz0[0])\n'
    '                        dz = abs(xyz1[2] - xyz0[2])\n'
    '                        if dx > tol_pair or dz > tol_pair:\n'
    '                            continue\n'
    '                        d2 = dx * dx + dz * dz\n'
    '                    else:\n'
    '                        dx = abs(xyz1[0] - xyz0[0])\n'
    '                        dy = abs(xyz1[1] - xyz0[1])\n'
    '                        if dx > tol_pair or dy > tol_pair:\n'
    '                            continue\n'
    '                        d2 = dx * dx + dy * dy\n'
    '                    if best is None or d2 < best[0]:\n'
    '                        best = (d2, lab1)\n'
    '        if best is None:\n'
    '            miss0.append(lab0)\n'
    '        else:\n'
    '            used1.add(best[1])\n'
    '    miss1 = [lab1 for lab1, xyz1 in nodes1 if lab1 not in used1]\n'
    '    missing = len(miss0) + len(miss1)\n'
    '    return (missing, len(nodes0), len(nodes1))\n'
    '\n'
    'def _m8p_axis_free_axis(axisA, axisB):\n'
    "    for a in ('X', 'Y', 'Z'):\n"
    '        if a != axisA and a != axisB:\n'
    '            return a\n'
    "    raise RuntimeError('free axis not found')\n"
    '\n'
    'def _m8p_axis_index(ax):\n'
    "    return {'X': 0, 'Y': 1, 'Z': 2}[ax]\n"
    '\n'
    'def _m8p_is_corner_xyzC(xyzC, Lx, Ly, Lz):\n'
    '    x, y, z = xyzC\n'
    '    return x in (0.0, Lx) and y in (0.0, Ly) and (z in (0.0, Lz))\n'
    '\n'
    'def _m8p_unmatched_is_edge_or_corner(part, axis, unmatched_labels):\n'
    '    if not unmatched_labels:\n'
    '        return False\n'
    "    P0 = axis + '0'\n"
    "    edge = _m8p_get_set_node_labels(part, 'M8_%s_EDGENODES' % P0)\n"
    "    corner = _m8p_get_set_node_labels(part, 'M8_%s_CORNER_NODES' % P0)\n"
    '    u = set((int(x) for x in unmatched_labels))\n'
    '    return bool(u & (edge | corner))\n'
    '\n'
    'def _m8p_make_set_nodes_elems_from_labels(part, name, node_labels, elem_labels, allow_empty=True):\n'
    '    _m8a_delete_set_if_exists(part, name)\n'
    '    n_labs = sorted(set((int(x) for x in node_labels or [])))\n'
    '    e_labs = sorted(set((int(x) for x in elem_labels or [])))\n'
    '    if not n_labs and (not e_labs):\n'
    '        if allow_empty:\n'
    '            return (0, 0)\n'
    "        raise RuntimeError('Empty node+elem set not allowed: %s' % name)\n"
    '    nodes_seq = None\n'
    '    elems_seq = None\n'
    '    if n_labs:\n'
    '        nodes_seq = part.nodes.sequenceFromLabels(labels=n_labs)\n'
    '    if e_labs:\n'
    '        elems_seq = part.elements.sequenceFromLabels(labels=e_labs)\n'
    '    if nodes_seq is not None and elems_seq is not None:\n'
    '        part.Set(name=name, nodes=nodes_seq, elements=elems_seq)\n'
    '    elif nodes_seq is not None:\n'
    '        part.Set(name=name, nodes=nodes_seq)\n'
    '    else:\n'
    '        part.Set(name=name, elements=elems_seq)\n'
    '    return (len(n_labs), len(e_labs))\n'
    '\n'
    'def _m8p_shift_master_xyzC_to_image(axis, xyzC_master, Lx, Ly, Lz, tol_plane):\n'
    '    x, y, z = xyzC_master\n'
    "    if axis == 'X':\n"
    '        xyz = (float(Lx), float(y), float(z))\n'
    "    elif axis == 'Y':\n"
    '        xyz = (float(x), float(Ly), float(z))\n'
    "    elif axis == 'Z':\n"
    '        xyz = (float(x), float(y), float(Lz))\n'
    '    else:\n'
    "        raise RuntimeError('unknown axis: %s' % axis)\n"
    '    return _m8a_canonicalize_xyz(xyz, Lx, Ly, Lz, tol_plane)\n'
    '\n'
    'def _m8p_unmatched_hits_frozen_edge(part, axis, done_images, unmatched_master_labels, lab2xyzC, Lx, Ly, Lz, tol_plane, tol_pair, debug_n=10):\n'
    "    P1 = axis + '1'\n"
    '    samples = []\n'
    '    for mLab in list(unmatched_master_labels):\n'
    '        xyzM = lab2xyzC.get(int(mLab), None)\n'
    '        if xyzM is None:\n'
    '            continue\n'
    '        xyzT = _m8p_shift_master_xyzC_to_image(axis, xyzM, Lx, Ly, Lz, tol_plane)\n'
    '        if not _m8p_target_hits_frozen_planes(xyzT, done_images, Lx, Ly, Lz):\n'
    '            continue\n'
    '        nearest = _m8p_find_nearest_on_image_plane_by_scan(part, axis, xyzT, P1, Lx, Ly, Lz, tol_plane)\n'
    '        samples.append((int(mLab), xyzM, xyzT, nearest))\n'
    '        if len(samples) >= int(debug_n):\n'
    '            break\n'
    '    return samples\n'
    '\n'
    'def _m8p_promote_inc_cap_sets_to_nodeelem(part):\n'
    "    cap_re = re.compile('^INC_(\\\\d{4})_CAP_([XYZ]0)$')\n"
    '    names = [n for n in part.sets.keys() if cap_re.match(n)]\n'
    '    for sname in names:\n'
    '        s = part.sets[sname]\n'
    '        elem_labs = []\n'
    '        node_labs = []\n'
    '        try:\n'
    '            elem_labs = [int(e.label) for e in s.elements]\n'
    '        except Exception:\n'
    '            elem_labs = []\n'
    '        try:\n'
    '            node_labs = [int(n.label) for n in s.nodes]\n'
    '        except Exception:\n'
    '            node_labs = []\n'
    '        if not node_labs and elem_labs:\n'
    '            nds = set()\n'
    '            for eLab in elem_labs:\n'
    '                try:\n'
    '                    el = part.elements.getFromLabel(label=int(eLab))\n'
    '                except Exception:\n'
    '                    continue\n'
    '                for nd in el.getNodes():\n'
    '                    nds.add(int(nd.label))\n'
    '            node_labs = sorted(nds)\n'
    '        _m8p_make_set_nodes_elems_from_labels(part, sname, node_labs, elem_labs, allow_empty=True)\n'
    "    print('[M8P] promoted INC_*_CAP_* sets to nodes+elements: count=%d' % len(names))\n"
    '\n'
    'def _m8p_build_frozen_edge_labels_on_P1(part, P1, done_images, Lx, Ly, Lz, tol_plane):\n'
    '    frozen = set()\n'
    '    for nd in part.nodes:\n'
    '        lab = int(nd.label)\n'
    '        xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        if not _m8p_on_plane_xyzC(xyzC, P1, Lx, Ly, Lz):\n'
    '            continue\n'
    '        xC, yC, zC = xyzC\n'
    "        if 'X' in done_images and (xC == 0.0 or xC == float(Lx)):\n"
    '            frozen.add(lab)\n'
    '            continue\n'
    "        if 'Y' in done_images and (yC == 0.0 or yC == float(Ly)):\n"
    '            frozen.add(lab)\n'
    '            continue\n'
    "        if 'Z' in done_images and (zC == 0.0 or zC == float(Lz)):\n"
    '            frozen.add(lab)\n'
    '            continue\n'
    '    return frozen\n'
    '\n'
    'def _m8p_s1_update_sets_after_evt(part, evt, Lx, Ly, Lz, tol_plane):\n'
    "    kind = evt.get('kind', '')\n"
    "    created = [int(x) for x in evt.get('nodes_created', [])]\n"
    "    deleted = set((int(x) for x in evt.get('nodes_deleted', [])))\n"
    "    endpoints = evt.get('edge_endpoints', None)\n"
    "    ctx = evt.get('context', '')\n"
    '    existing_nodes = set((int(n.label) for n in part.nodes))\n'
    '    existing_elems = set((int(e.label) for e in part.elements))\n'
    "    cap_re = re.compile('^INC_(\\\\d{4})_CAP_([XYZ]0)$')\n"
    '    endpoint_labs = set()\n'
    '    if endpoints:\n'
    '        try:\n'
    '            endpoint_labs = set([int(endpoints[0]), int(endpoints[1])])\n'
    '        except Exception:\n'
    '            endpoint_labs = set()\n'
    '    survivor_labs = sorted(list(endpoint_labs - deleted & existing_nodes))\n'
    '\n'
    '    def _node_xyzC_or_none(lab):\n'
    '        try:\n'
    '            nd = part.nodes.getFromLabel(label=int(lab))\n'
    '            return _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        except Exception:\n'
    '            return None\n'
    '\n'
    '    def _planes_for_fam(fam):\n'
    "        if fam == 'X0':\n"
    "            return ('X0', 'X1')\n"
    "        if fam == 'Y0':\n"
    "            return ('Y0', 'Y1')\n"
    "        return ('Z0', 'Z1')\n"
    '\n'
    '    def _node_on_any_fam_plane(lab, fam):\n'
    '        xyzC = _node_xyzC_or_none(lab)\n'
    '        if xyzC is None:\n'
    '            return False\n'
    '        for pl in _planes_for_fam(fam):\n'
    '            if _m8p_on_plane_xyzC(xyzC, pl, Lx, Ly, Lz):\n'
    '                return True\n'
    '        return False\n'
    "    print('[M8P:S1] kind=%s ctx=%s endpoints=%s created=%s deleted=%s survivors=%s' % (str(kind), str(ctx), str(endpoints), str(created), str(sorted(list(deleted))), str(survivor_labs)))\n"
    '\n'
    '    def _node_membership_labels(set_obj):\n'
    '        labs = set()\n'
    '        try:\n'
    '            for n in set_obj.nodes:\n'
    '                labs.add(int(n.label))\n'
    '        except Exception:\n'
    '            pass\n'
    '        if labs:\n'
    '            return labs\n'
    '        try:\n'
    '            for e in set_obj.elements:\n'
    '                for nd in e.getNodes():\n'
    '                    labs.add(int(nd.label))\n'
    '        except Exception:\n'
    '            pass\n'
    '        return labs\n'
    '\n'
    '    def _is_m8_nodeset_name(sname):\n'
    "        if sname.endswith('_ELEMS'):\n"
    '            return False\n'
    "        if sname.startswith('OUTER_') or sname in ('OUTER_ALL', 'CAP_ALL', 'IFACE_ALL'):\n"
    '            return False\n'
    "        return sname.startswith('M8_') or sname.startswith('M8P_')\n"
    '    for sname in list(part.sets.keys()):\n'
    '        if not _is_m8_nodeset_name(sname):\n'
    '            continue\n'
    '        try:\n'
    '            labs = set((int(n.label) for n in part.sets[sname].nodes))\n'
    '        except Exception:\n'
    '            continue\n'
    '        labs &= existing_nodes\n'
    '        if deleted:\n'
    '            labs -= deleted\n'
    "        if kind == 'SPLIT' and endpoints and created:\n"
    '            labA, labB = (int(endpoints[0]), int(endpoints[1]))\n'
    '            newLab = int(created[0])\n'
    '            if labA in labs and labB in labs and (newLab in existing_nodes):\n'
    '                labs.add(newLab)\n'
    '        _m8a_make_node_set_from_labels(part, sname, sorted(labs), allow_empty=True)\n'
    '    for sname in list(part.sets.keys()):\n'
    '        m = cap_re.match(sname)\n'
    '        if not m:\n'
    '            continue\n'
    '        fam = m.group(2)\n'
    '        s = part.sets[sname]\n'
    '        nLabs0 = _node_membership_labels(s)\n'
    '        nLabs = set(nLabs0)\n'
    '        nLabs &= existing_nodes\n'
    '        if deleted:\n'
    '            nLabs -= deleted\n'
    '        inherited = []\n'
    "        if kind == 'COLLAPSE' and endpoint_labs and survivor_labs:\n"
    '            if endpoint_labs & set(nLabs0):\n'
    '                for labS in survivor_labs:\n'
    '                    if labS not in nLabs and _node_on_any_fam_plane(labS, fam):\n'
    '                        nLabs.add(int(labS))\n'
    "                        inherited.append(('COLLAPSE_SURVIVOR', int(labS)))\n"
    "        if kind == 'SPLIT' and endpoints and created:\n"
    '            labA, labB = (int(endpoints[0]), int(endpoints[1]))\n'
    '            newLab = int(created[0])\n'
    '            if (labA in nLabs0 or labB in nLabs0) and newLab in existing_nodes and _node_on_any_fam_plane(newLab, fam):\n'
    '                nLabs.add(newLab)\n'
    "                inherited.append(('SPLIT_NEW', int(newLab)))\n"
    '        planes = _planes_for_fam(fam)\n'
    '        eLabs = set()\n'
    '        for pl in planes:\n'
    '            for eLab in _m8p_outer_elem_labels_by_plane(part, pl, Lx, Ly, Lz, tol_plane):\n'
    '                if int(eLab) not in existing_elems:\n'
    '                    continue\n'
    '                el = part.elements.getFromLabel(label=int(eLab))\n'
    '                nds = el.getNodes()\n'
    '                if len(nds) != 3:\n'
    '                    continue\n'
    '                labs = [int(n.label) for n in nds]\n'
    '                if labs[0] in nLabs and labs[1] in nLabs and (labs[2] in nLabs):\n'
    '                    eLabs.add(int(eLab))\n'
    '        old_node_n = len(set(nLabs0) & existing_nodes)\n'
    '        old_elem_n = 0\n'
    '        try:\n'
    '            old_elem_n = len(s.elements)\n'
    '        except Exception:\n'
    '            old_elem_n = 0\n'
    '        _m8p_make_set_nodes_elems_from_labels(part, sname, sorted(nLabs), sorted(eLabs), allow_empty=True)\n'
    '        if inherited or endpoint_labs & set(nLabs0) or created or deleted:\n'
    "            print('[M8P:S1:CAP] %s fam=%s nodes %d->%d elems %d->%d inherited=%s endpointHit=%s' % (sname, fam, old_node_n, len(nLabs), old_elem_n, len(eLabs), str(inherited), str(sorted(list(endpoint_labs & set(nLabs0))))))\n"
    '    for sname, s in part.sets.items():\n'
    '        if cap_re.match(sname) or _is_m8_nodeset_name(sname):\n'
    '            try:\n'
    '                nodes_seq = list(s.nodes)\n'
    '            except Exception:\n'
    '                nodes_seq = []\n'
    '            for n in nodes_seq:\n'
    '                if int(n.label) not in existing_nodes:\n'
    "                    raise RuntimeError('[M8P FATAL] set %s has non-existing node label %d' % (sname, int(n.label)))\n"
    '            try:\n'
    '                elems_seq = list(s.elements)\n'
    '            except Exception:\n'
    '                elems_seq = []\n'
    '            for e in elems_seq:\n'
    '                if int(e.label) not in existing_elems:\n'
    "                    raise RuntimeError('[M8P FATAL] set %s has non-existing element label %d' % (sname, int(e.label)))\n"
    '\n'
    'def _m8p_pair_REQ_gate(part, axis, req, plane_img, lab2node, lab2xyzC, Lx, Ly, Lz, tol_plane, tol_pair, eps_key, shift):\n'
    "    P0 = axis + '0'\n"
    '    reuseCand = []\n'
    '    for lab, xyzC in lab2xyzC.items():\n'
    '        if _m8p_on_plane_xyzC(xyzC, plane_img, Lx, Ly, Lz):\n'
    '            reuseCand.append(int(lab))\n'
    '    buckets = defaultdict(list)\n'
    '    for lab in reuseCand:\n'
    '        buckets[_m8p_transverse_key(axis, lab2xyzC[lab], tol_pair)].append(lab)\n'
    '    outerPlaneOfElem = _m8p_build_outerPlaneOfElem(part, Lx, Ly, Lz, tol_plane)\n'
    '    sigM_full_cache = {}\n'
    '    sigM_filt_cache = {}\n'
    '    sigM_tr_cache = {}\n'
    '    sigI_3d_cache = {}\n'
    '    sigI_tr_cache = {}\n'
    '    used_img = set()\n'
    '    pair = {}\n'
    '    unmatched = []\n'
    '    for m in sorted(list(req)):\n'
    '        xyzM = lab2xyzC.get(int(m), None)\n'
    '        if xyzM is None:\n'
    '            unmatched.append(int(m))\n'
    '            continue\n'
    '        key0 = _m8p_transverse_key(axis, xyzM, tol_pair)\n'
    '        cset = set()\n'
    '        for di in (-1, 0, 1):\n'
    '            for dj in (-1, 0, 1):\n'
    '                kk = (key0[0] + di, key0[1] + dj)\n'
    '                for c in buckets.get(kk, []):\n'
    '                    if c in used_img:\n'
    '                        continue\n'
    '                    xyzI = lab2xyzC[c]\n'
    "                    if axis == 'X':\n"
    '                        if abs(xyzI[1] - xyzM[1]) <= tol_pair and abs(xyzI[2] - xyzM[2]) <= tol_pair:\n'
    '                            cset.add(int(c))\n'
    "                    elif axis == 'Y':\n"
    '                        if abs(xyzI[0] - xyzM[0]) <= tol_pair and abs(xyzI[2] - xyzM[2]) <= tol_pair:\n'
    '                            cset.add(int(c))\n'
    '                    elif abs(xyzI[0] - xyzM[0]) <= tol_pair and abs(xyzI[1] - xyzM[1]) <= tol_pair:\n'
    '                        cset.add(int(c))\n'
    '        cands = sorted(list(cset))\n'
    '        if not cands:\n'
    '            unmatched.append(int(m))\n'
    '            continue\n'
    '        if len(cands) == 1:\n'
    '            pair[int(m)] = int(cands[0])\n'
    '            used_img.add(int(cands[0]))\n'
    '            continue\n'
    '        for c in cands:\n'
    '            c = int(c)\n'
    '            if c not in sigI_3d_cache:\n'
    '                sigI_3d_cache[c] = _m8p_sig_1ring(lab2node[c], Lx, Ly, Lz, tol_plane, eps_key)\n'
    '            if c not in sigI_tr_cache:\n'
    '                sigI_tr_cache[c] = _m8p_sig_1ring_transverse_ignore_outer_plane(lab2node[c], axis, ignore_plane=None, outerPlaneOfElem=outerPlaneOfElem, Lx=Lx, Ly=Ly, Lz=Lz, tol_plane=tol_plane, eps_key=eps_key)\n'
    '        if int(m) not in sigM_full_cache:\n'
    '            sigM_full_cache[int(m)] = _m8p_sig_1ring_shifted(lab2node[int(m)], axis, shift, Lx, Ly, Lz, tol_plane, eps_key)\n'
    '        sigM = sigM_full_cache[int(m)]\n'
    '        matches = [c for c in cands if sigI_3d_cache[int(c)] == sigM]\n'
    '        if len(matches) == 0:\n'
    '            if int(m) not in sigM_filt_cache:\n'
    '                sigM_filt_cache[int(m)] = _m8p_sig_1ring_shifted_ignore_outer_plane(lab2node[int(m)], shift, ignore_plane=P0, outerPlaneOfElem=outerPlaneOfElem, Lx=Lx, Ly=Ly, Lz=Lz, tol_plane=tol_plane, eps_key=eps_key)\n'
    '            sigM2 = sigM_filt_cache[int(m)]\n'
    '            matches = [c for c in cands if sigI_3d_cache[int(c)] == sigM2]\n'
    '        if len(matches) == 0:\n'
    '            if int(m) not in sigM_tr_cache:\n'
    '                sigM_tr_cache[int(m)] = _m8p_sig_1ring_transverse_ignore_outer_plane(lab2node[int(m)], axis, ignore_plane=P0, outerPlaneOfElem=outerPlaneOfElem, Lx=Lx, Ly=Ly, Lz=Lz, tol_plane=tol_plane, eps_key=eps_key)\n'
    '            sigM3 = sigM_tr_cache[int(m)]\n'
    '            matches = [c for c in cands if sigI_tr_cache[int(c)] == sigM3]\n'
    '        if len(matches) != 1:\n'
    "            _m8a_make_node_set_from_labels(part, 'M8P_AMBIG_M_%s0' % axis, [int(m)], allow_empty=False)\n"
    "            _m8a_make_node_set_from_labels(part, 'M8P_AMBIG_C_%s1' % axis, cands, allow_empty=False)\n"
    "            raise RuntimeError('[M8P FATAL] ambiguous pairing unresolved: axis=%s masterNode=%d candidates=%d topoMatches=%d' % (axis, int(m), len(cands), len(matches)))\n"
    '        pair[int(m)] = int(matches[0])\n'
    '        used_img.add(int(matches[0]))\n'
    '    inv = defaultdict(int)\n'
    '    for m, i in pair.items():\n'
    '        inv[int(i)] += 1\n'
    '    bad = [i for i, cnt in inv.items() if cnt > 1]\n'
    '    if bad:\n'
    "        raise RuntimeError('[M8P FATAL] multi-to-one mapping detected on %s, example image node labels=%s' % (axis, str(bad[:10])))\n"
    '    return (pair, unmatched)\n'
    '\n'
    'def _m8p_bin_key_on_plane(axis, xyzC, Lx, Ly, Lz):\n'
    '    xC, yC, zC = xyzC\n'
    "    if axis == 'X':\n"
    "        yk = 'Y0' if yC == 0.0 else 'Y1' if yC == float(Ly) else 'YI'\n"
    "        zk = 'Z0' if zC == 0.0 else 'Z1' if zC == float(Lz) else 'ZI'\n"
    '        return (yk, zk)\n'
    "    if axis == 'Y':\n"
    "        xk = 'X0' if xC == 0.0 else 'X1' if xC == float(Lx) else 'XI'\n"
    "        zk = 'Z0' if zC == 0.0 else 'Z1' if zC == float(Lz) else 'ZI'\n"
    '        return (xk, zk)\n'
    "    xk = 'X0' if xC == 0.0 else 'X1' if xC == float(Lx) else 'XI'\n"
    "    yk = 'Y0' if yC == 0.0 else 'Y1' if yC == float(Ly) else 'YI'\n"
    '    return (xk, yk)\n'
    '\n'
    'def _m8p_sig_1ring_shifted_ignore_outer_plane(node_obj, shift, ignore_plane, outerPlaneOfElem, Lx, Ly, Lz, tol_plane, eps_key):\n'
    '    sx, sy, sz = shift\n'
    '    keys = set()\n'
    '    for el in node_obj.getElements():\n'
    '        eLab = int(el.label)\n'
    '        if ignore_plane is not None and outerPlaneOfElem.get(eLab, None) == ignore_plane:\n'
    '            continue\n'
    '        for nd in el.getNodes():\n'
    '            xyz = nd.coordinates\n'
    '            xyzS = (xyz[0] + sx, xyz[1] + sy, xyz[2] + sz)\n'
    '            xyzC = _m8a_canonicalize_xyz(xyzS, Lx, Ly, Lz, tol_plane)\n'
    '            k = _m8a_node_key_from_xyzC(xyzC, eps_key)\n'
    '            keys.add(k)\n'
    '    return keys\n'
    '\n'
    'def _m8p_connected_components_1d_on_plane(part, node_labels, plane_tag, Lx, Ly, Lz, tol_plane):\n'
    '    nodeset = set((int(x) for x in node_labels))\n'
    '    if not nodeset:\n'
    '        return []\n'
    '    adj = dict(((lab, set()) for lab in nodeset))\n'
    '    for lab in list(nodeset):\n'
    '        try:\n'
    '            nd = part.nodes.getFromLabel(label=int(lab))\n'
    '        except Exception:\n'
    '            continue\n'
    '        try:\n'
    '            for me in nd.getElemEdges():\n'
    '                n1, n2 = me.getNodes()\n'
    '                a = int(n1.label)\n'
    '                b = int(n2.label)\n'
    '                if a not in nodeset or b not in nodeset:\n'
    '                    continue\n'
    '                xyzA = _m8a_canonicalize_xyz(n1.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '                xyzB = _m8a_canonicalize_xyz(n2.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '                if not _m8p_on_plane_xyzC(xyzA, plane_tag, Lx, Ly, Lz) or not _m8p_on_plane_xyzC(xyzB, plane_tag, Lx, Ly, Lz):\n'
    '                    continue\n'
    '                adj[a].add(b)\n'
    '                adj[b].add(a)\n'
    '        except Exception:\n'
    '            continue\n'
    '    comps = []\n'
    '    seen = set()\n'
    '    for lab in sorted(list(nodeset)):\n'
    '        if lab in seen:\n'
    '            continue\n'
    '        q = [lab]\n'
    '        seen.add(lab)\n'
    '        comp = set([lab])\n'
    '        while q:\n'
    '            u = q.pop()\n'
    '            for v in adj.get(u, []):\n'
    '                if v in seen:\n'
    '                    continue\n'
    '                seen.add(v)\n'
    '                comp.add(v)\n'
    '                q.append(v)\n'
    '        comps.append(comp)\n'
    '    return comps\n'
    '\n'
    'def _m8p_order_component_walk(part, comp_labels, plane_tag, Lx, Ly, Lz, tol_plane):\n'
    '    comp = set((int(x) for x in comp_labels))\n'
    '    if len(comp) <= 1:\n'
    '        return sorted(list(comp))\n'
    '    adj = dict(((lab, set()) for lab in comp))\n'
    '    for lab in list(comp):\n'
    '        nd = part.nodes.getFromLabel(label=int(lab))\n'
    '        for me in nd.getElemEdges():\n'
    '            n1, n2 = me.getNodes()\n'
    '            a = int(n1.label)\n'
    '            b = int(n2.label)\n'
    '            if a not in comp or b not in comp:\n'
    '                continue\n'
    '            xyzA = _m8a_canonicalize_xyz(n1.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            xyzB = _m8a_canonicalize_xyz(n2.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            if not _m8p_on_plane_xyzC(xyzA, plane_tag, Lx, Ly, Lz) or not _m8p_on_plane_xyzC(xyzB, plane_tag, Lx, Ly, Lz):\n'
    '                continue\n'
    '            adj[a].add(b)\n'
    '            adj[b].add(a)\n'
    '    deg1 = [lab for lab in comp if len(adj.get(lab, [])) == 1]\n'
    '    if deg1:\n'
    '        start = min(deg1)\n'
    '        is_closed = False\n'
    '    else:\n'
    '        start = min(comp)\n'
    '        is_closed = True\n'
    '    order = [start]\n'
    '    prev = None\n'
    '    cur = start\n'
    '    while True:\n'
    '        nbrs = sorted(list(adj.get(cur, [])))\n'
    '        nxt = None\n'
    '        for v in nbrs:\n'
    '            if v == prev:\n'
    '                continue\n'
    '            nxt = v\n'
    '            break\n'
    '        if nxt is None:\n'
    '            break\n'
    '        if is_closed and nxt == start:\n'
    '            break\n'
    '        order.append(nxt)\n'
    '        prev, cur = (cur, nxt)\n'
    '        if len(order) > len(comp) + 5:\n'
    '            break\n'
    '    if len(order) < len(comp):\n'
    '        rem = sorted(list(comp - set(order)))\n'
    '        order.extend(rem)\n'
    '    return order\n'
    '\n'
    'def _m8p_transverse_key_eps(axis, xyzC, eps_key):\n'
    '    e = float(eps_key)\n'
    '    x, y, z = xyzC\n'
    "    if axis == 'X':\n"
    '        return (int(round(float(y) / e)), int(round(float(z) / e)))\n'
    "    if axis == 'Y':\n"
    '        return (int(round(float(x) / e)), int(round(float(z) / e)))\n'
    "    if axis == 'Z':\n"
    '        return (int(round(float(x) / e)), int(round(float(y) / e)))\n'
    "    raise RuntimeError('unknown axis: %s' % axis)\n"
    '\n'
    'def _m8p_sig_1ring_transverse_ignore_outer_plane(node_obj, axis, ignore_plane, outerPlaneOfElem, Lx, Ly, Lz, tol_plane, eps_key):\n'
    '    keys = set()\n'
    '    for el in node_obj.getElements():\n'
    '        eLab = int(el.label)\n'
    '        if ignore_plane is not None and outerPlaneOfElem.get(eLab, None) == ignore_plane:\n'
    '            continue\n'
    '        for nd in el.getNodes():\n'
    '            xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            keys.add(_m8p_transverse_key_eps(axis, xyzC, eps_key))\n'
    '    return keys\n'
    '\n'
    "def _m8p_fix_iface_curve_by_inclusions_once(part, axis, done_images, unmatched_req_master, Lx, Ly, Lz, tol_plane, tol_pair, tol_snap, eps_key, debug_prefix='M8P_IFACE'):\n"
    '    import math\n'
    '    import re\n'
    '    from collections import defaultdict\n'
    "    P0 = axis + '0'\n"
    "    P1 = axis + '1'\n"
    "    fam = axis + '0'\n"
    '    unmatched_set = set((int(x) for x in unmatched_req_master or []))\n'
    '    if not unmatched_set:\n'
    '        return False\n'
    '    lab2node, lab2xyzC, key2lab = _m8p_build_node_maps(part, Lx, Ly, Lz, tol_plane, eps_key)\n'
    '\n'
    '    def _inc_nodes_all(sname):\n'
    '        s = part.sets[sname]\n'
    '        labs = set()\n'
    '        try:\n'
    '            for n in s.nodes:\n'
    '                labs.add(int(n.label))\n'
    '        except Exception:\n'
    '            pass\n'
    '        if labs:\n'
    '            return labs\n'
    '        try:\n'
    '            for e in s.elements:\n'
    '                for nd in e.getNodes():\n'
    '                    labs.add(int(nd.label))\n'
    '        except Exception:\n'
    '            pass\n'
    '        return labs\n'
    '\n'
    '    def _shift_to_P1(xyzC_master):\n'
    '        return _m8p_shift_master_xyzC_to_image(axis, xyzC_master, Lx, Ly, Lz, tol_plane)\n'
    '\n'
    '    def _dist2_transverse(aC, bC):\n'
    "        if axis == 'X':\n"
    '            dy = aC[1] - bC[1]\n'
    '            dz = aC[2] - bC[2]\n'
    '            return dy * dy + dz * dz\n'
    "        if axis == 'Y':\n"
    '            dx = aC[0] - bC[0]\n'
    '            dz = aC[2] - bC[2]\n'
    '            return dx * dx + dz * dz\n'
    '        dx = aC[0] - bC[0]\n'
    '        dy = aC[1] - bC[1]\n'
    '        return dx * dx + dy * dy\n'
    '\n'
    '    def _center_transverse(xyz_list):\n'
    '        if not xyz_list:\n'
    '            return (0.0, 0.0)\n'
    "        if axis == 'X':\n"
    '            ys = [p[1] for p in xyz_list]\n'
    '            zs = [p[2] for p in xyz_list]\n'
    '            return (sum(ys) / len(ys), sum(zs) / len(zs))\n'
    "        if axis == 'Y':\n"
    '            xs = [p[0] for p in xyz_list]\n'
    '            zs = [p[2] for p in xyz_list]\n'
    '            return (sum(xs) / len(xs), sum(zs) / len(zs))\n'
    '        xs = [p[0] for p in xyz_list]\n'
    '        ys = [p[1] for p in xyz_list]\n'
    '        return (sum(xs) / len(xs), sum(ys) / len(ys))\n'
    '\n'
    '    def _center_dist2(ca, cb):\n'
    '        dx = ca[0] - cb[0]\n'
    '        dy = ca[1] - cb[1]\n'
    '        return dx * dx + dy * dy\n'
    '    reuseP1 = set()\n'
    '    reuseP1_iface = set()\n'
    '    for nd in part.nodes:\n'
    '        lab = int(nd.label)\n'
    '        xyzC = lab2xyzC.get(lab, None)\n'
    '        if xyzC is None:\n'
    '            continue\n'
    '        if not _m8p_on_plane_xyzC(xyzC, P1, Lx, Ly, Lz):\n'
    '            continue\n'
    '        reuseP1.add(lab)\n'
    '        if _m8p_adj_has_nonouter(part, nd, Lx, Ly, Lz, tol_plane):\n'
    '            reuseP1_iface.add(lab)\n'
    '    frozen = _m8p_build_frozen_edge_labels_on_P1(part, P1, done_images, Lx, Ly, Lz, tol_plane)\n'
    '    editable = set(reuseP1) - set(frozen)\n'
    '\n'
    '    def _is_edge_on_plane_xyzC(plane_tag, xyzC):\n'
    '        x, y, z = xyzC\n'
    "        if plane_tag in ('X0', 'X1'):\n"
    '            if not (x == 0.0 or x == float(Lx)):\n'
    '                return False\n'
    '            onY = y == 0.0 or y == float(Ly)\n'
    '            onZ = z == 0.0 or z == float(Lz)\n'
    '            return (onY or onZ) and (not (onY and onZ))\n'
    "        if plane_tag in ('Y0', 'Y1'):\n"
    '            if not (y == 0.0 or y == float(Ly)):\n'
    '                return False\n'
    '            onX = x == 0.0 or x == float(Lx)\n'
    '            onZ = z == 0.0 or z == float(Lz)\n'
    '            return (onX or onZ) and (not (onX and onZ))\n'
    "        if plane_tag in ('Z0', 'Z1'):\n"
    '            if not (z == 0.0 or z == float(Lz)):\n'
    '                return False\n'
    '            onX = x == 0.0 or x == float(Lx)\n'
    '            onY = y == 0.0 or y == float(Ly)\n'
    '            return (onX or onY) and (not (onX and onY))\n'
    '        return False\n'
    '    edgeP1 = set()\n'
    '    cornerP1 = set()\n'
    '    for lab in reuseP1:\n'
    '        xyzC = lab2xyzC[lab]\n'
    '        if _m8p_is_corner_xyzC(xyzC, Lx, Ly, Lz):\n'
    '            cornerP1.add(lab)\n'
    '        elif _is_edge_on_plane_xyzC(P1, xyzC):\n'
    '            edgeP1.add(lab)\n'
    '    ifacecurveP1_allowed_global = set(reuseP1_iface) - set(edgeP1) - set(cornerP1)\n'
    '\n'
    '    def _master_ifacecurve_nodes(P0):\n'
    "        s_interior = 'M8_%s_CAP_BNDRY_NODES_INTERIOR' % P0\n"
    '        if s_interior in part.sets:\n'
    '            return set((int(n.label) for n in part.sets[s_interior].nodes))\n'
    "        shared = _m8p_get_set_node_labels(part, 'M8_%s_CAP_BNDRY_SHARED_IFACE' % P0)\n"
    "        edge = _m8p_get_set_node_labels(part, 'M8_%s_EDGENODES' % P0)\n"
    "        corner = _m8p_get_set_node_labels(part, 'M8_%s_CORNER_NODES' % P0)\n"
    '        return set(shared) - set(edge) - set(corner)\n'
    '    master_ifacecurveP0 = _master_ifacecurve_nodes(P0)\n'
    "    cap_re = re.compile('^INC_(\\\\d{4})_CAP_%s$' % fam)\n"
    '    inc_set_names = [n for n in part.sets.keys() if cap_re.match(n)]\n'
    '    if not inc_set_names:\n'
    '        return False\n'
    '\n'
    '    def _expand_on_plane(seed_labels, allowed_labels):\n'
    '        seed = set((int(x) for x in seed_labels or []))\n'
    '        allowed = set((int(x) for x in allowed_labels or []))\n'
    '        seed &= allowed\n'
    '        if not seed:\n'
    '            return set()\n'
    '        seen = set(seed)\n'
    '        q = list(seed)\n'
    '        while q:\n'
    '            lab = q.pop()\n'
    '            try:\n'
    '                nd = part.nodes.getFromLabel(label=int(lab))\n'
    '            except Exception:\n'
    '                continue\n'
    '            try:\n'
    '                for me in nd.getElemEdges():\n'
    '                    n1, n2 = me.getNodes()\n'
    '                    a = int(n1.label)\n'
    '                    b = int(n2.label)\n'
    '                    nxt = b if a == int(lab) else a\n'
    '                    if nxt in seen:\n'
    '                        continue\n'
    '                    if nxt not in allowed:\n'
    '                        continue\n'
    '                    xyzN = lab2xyzC.get(int(nxt), None)\n'
    '                    if xyzN is None:\n'
    '                        xyzN = _m8a_canonicalize_xyz(part.nodes.getFromLabel(label=int(nxt)).coordinates, Lx, Ly, Lz, tol_plane)\n'
    '                        lab2xyzC[int(nxt)] = xyzN\n'
    '                    if not _m8p_on_plane_xyzC(xyzN, P1, Lx, Ly, Lz):\n'
    '                        continue\n'
    '                    seen.add(int(nxt))\n'
    '                    q.append(int(nxt))\n'
    '            except Exception:\n'
    '                continue\n'
    '        return seen\n'
    '\n'
    '    def _sanitize(s):\n'
    "        s = re.sub('[^A-Za-z0-9_]', '_', str(s))\n"
    '        return s.upper()\n'
    '\n'
    '    def _make_fix_sets(axis, tag, master_topo, img_core, img_anchor_edge, img_all, img_edit, img_frozen):\n'
    "        base = 'M8P_FIX_%s_%s' % (axis, _sanitize(tag))\n"
    "        names = {'MASTER_TOPO': base + '_MASTER_TOPO', 'IMG_TOPO_CORE': base + '_IMG_TOPO_CORE', 'IMG_TOPO_ANCHOR_EDGE': base + '_IMG_TOPO_ANCHOR_EDGE', 'IMG_TOPO_ALL': base + '_IMG_TOPO_ALL', 'IMG_EDIT': base + '_IMG_EDIT', 'IMG_FROZEN': base + '_IMG_FROZEN'}\n"
    '        for nm in names.values():\n'
    '            _m8a_delete_set_if_exists(part, nm)\n'
    "        _m8a_make_node_set_from_labels(part, names['MASTER_TOPO'], sorted(set((int(x) for x in master_topo or []))), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(part, names['IMG_TOPO_CORE'], sorted(set((int(x) for x in img_core or []))), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(part, names['IMG_TOPO_ANCHOR_EDGE'], sorted(set((int(x) for x in img_anchor_edge or []))), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(part, names['IMG_TOPO_ALL'], sorted(set((int(x) for x in img_all or []))), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(part, names['IMG_EDIT'], sorted(set((int(x) for x in img_edit or []))), allow_empty=True)\n"
    "        _m8a_make_node_set_from_labels(part, names['IMG_FROZEN'], sorted(set((int(x) for x in img_frozen or []))), allow_empty=True)\n"
    '    involved = []\n'
    '    for sname in sorted(inc_set_names):\n'
    '        nodes_all = _inc_nodes_all(sname)\n'
    '        inc_master_curve = set()\n'
    '        for lab in nodes_all:\n'
    '            lab = int(lab)\n'
    '            if lab not in master_ifacecurveP0:\n'
    '                continue\n'
    '            xyzC = lab2xyzC.get(lab, None)\n'
    '            if xyzC is None:\n'
    '                continue\n'
    '            if _m8p_on_plane_xyzC(xyzC, P0, Lx, Ly, Lz):\n'
    '                inc_master_curve.add(lab)\n'
    '        if inc_master_curve & unmatched_set:\n'
    '            involved.append(sname)\n'
    '    if not involved:\n'
    '        return False\n'
    '    for sname in involved:\n'
    '        m_inc = cap_re.match(sname)\n'
    "        ID4 = m_inc.group(1) if m_inc else 'XXXX'\n"
    '        nodes_all = _inc_nodes_all(sname)\n'
    '        master_curve = set()\n'
    '        for lab in nodes_all:\n'
    '            lab = int(lab)\n'
    '            if lab not in master_ifacecurveP0:\n'
    '                continue\n'
    '            xyzC = lab2xyzC.get(lab, None)\n'
    '            if xyzC is None:\n'
    '                continue\n'
    '            if _m8p_on_plane_xyzC(xyzC, P0, Lx, Ly, Lz):\n'
    '                master_curve.add(lab)\n'
    '        if not master_curve & unmatched_set:\n'
    '            continue\n'
    '        img_topo_core = set()\n'
    '        img_anchor_edge = set()\n'
    '        for lab in nodes_all:\n'
    '            lab = int(lab)\n'
    '            xyzC = lab2xyzC.get(lab, None)\n'
    '            if xyzC is None:\n'
    '                continue\n'
    '            if not _m8p_on_plane_xyzC(xyzC, P1, Lx, Ly, Lz):\n'
    '                continue\n'
    '            if lab in ifacecurveP1_allowed_global:\n'
    '                img_topo_core.add(lab)\n'
    '            elif lab in edgeP1 or lab in cornerP1:\n'
    '                img_anchor_edge.add(lab)\n'
    '        img_seed = set(img_topo_core)\n'
    '        if not img_seed:\n'
    '            best = None\n'
    '            for labm in sorted(list(master_curve & unmatched_set))[:50]:\n'
    '                tC = _shift_to_P1(lab2xyzC[int(labm)])\n'
    '                for labi in img_topo_core:\n'
    '                    xyzI = lab2xyzC.get(int(labi), None)\n'
    '                    if xyzI is None:\n'
    '                        continue\n'
    '                    d2 = _dist2_transverse(xyzI, tC)\n'
    '                    if d2 <= float(tol_snap) * float(tol_snap):\n'
    '                        cand = (d2, int(labi))\n'
    '                        if best is None or cand[0] < best[0]:\n'
    '                            best = cand\n'
    '            if best is not None:\n'
    '                img_seed.add(int(best[1]))\n'
    '        img_topo_all_allowed = set(img_topo_core) | set(img_anchor_edge)\n'
    '        img_topo_all = _expand_on_plane(img_seed, img_topo_all_allowed)\n'
    '        img_topo_all = set(img_topo_all) & set(img_topo_all_allowed)\n'
    '        img_topo_core = set(img_topo_all) & set(img_topo_core)\n'
    '        img_edit = set(img_topo_core) & set(editable)\n'
    '        if not img_topo_core:\n'
    "            _m8a_make_node_set_from_labels(part, '%s_%s_IMG_TOPO_EMPTY' % (debug_prefix, sname), sorted(list(master_curve & unmatched_set)), allow_empty=False)\n"
    "            raise RuntimeError('[M8P FATAL] IFACE-curve: IMG_TopoCore empty for %s on %s (cannot fix curve-part)' % (sname, P1))\n"
    '        bins_master = defaultdict(set)\n'
    '        for lab in master_curve:\n'
    '            bins_master[_m8p_bin_key_on_plane(axis, lab2xyzC[int(lab)], Lx, Ly, Lz)].add(int(lab))\n'
    '        bins_img_core = defaultdict(set)\n'
    '        for lab in img_topo_core:\n'
    '            bins_img_core[_m8p_bin_key_on_plane(axis, lab2xyzC[int(lab)], Lx, Ly, Lz)].add(int(lab))\n'
    '        comps_i_all = _m8p_connected_components_1d_on_plane(part, img_topo_all, P1, Lx, Ly, Lz, tol_plane)\n'
    '        for bkey in sorted(list(bins_master.keys())):\n'
    '            master_nodes_bin = bins_master[bkey]\n'
    '            if not master_nodes_bin & unmatched_set:\n'
    '                continue\n'
    '            img_nodes_bin_core = bins_img_core.get(bkey, set())\n'
    '            if not img_nodes_bin_core:\n'
    "                tag = 'I%s_%s_%s_%s' % (ID4, fam, bkey[0], bkey[1])\n"
    '                _make_fix_sets(axis, tag, master_topo=sorted(master_nodes_bin), img_core=[], img_anchor_edge=sorted(img_anchor_edge), img_all=sorted(img_topo_all), img_edit=sorted(img_edit), img_frozen=sorted(set(img_topo_all) & set(frozen)))\n'
    "                raise RuntimeError('[M8P FATAL] IFACE-curve bin missing on image (core): axis=%s incSet=%s bin=%s' % (axis, sname, str(bkey)))\n"
    '            comps_m = _m8p_connected_components_1d_on_plane(part, master_nodes_bin, P0, Lx, Ly, Lz, tol_plane)\n'
    '            comps_m = [c for c in comps_m if set(c) & unmatched_set]\n'
    '            if not comps_m:\n'
    '                continue\n'
    '            comps_m = sorted(comps_m, key=lambda c: (-len(set(c) & unmatched_set), -len(c), min(c)))\n'
    '            compM = comps_m[0]\n'
    '            cand_i = [c for c in comps_i_all if set(c) & set(img_nodes_bin_core)]\n'
    '            if not cand_i:\n'
    "                tag = 'I%s_%s_%s_%s' % (ID4, fam, bkey[0], bkey[1])\n"
    '                _make_fix_sets(axis, tag, master_topo=_m8p_order_component_walk(part, compM, P0, Lx, Ly, Lz, tol_plane), img_core=sorted(img_nodes_bin_core), img_anchor_edge=sorted(img_anchor_edge), img_all=sorted(img_topo_all), img_edit=sorted(img_edit), img_frozen=sorted(set(img_topo_all) & set(frozen)))\n'
    "                raise RuntimeError('[M8P FATAL] IFACE-curve: no image TopoAll components intersect core-bin: axis=%s incSet=%s bin=%s' % (axis, sname, str(bkey)))\n"
    '            xyzM = [_shift_to_P1(lab2xyzC[int(lab)]) for lab in compM]\n'
    '            cM = _center_transverse(xyzM)\n'
    '            bestI = None\n'
    '            for compI in cand_i:\n'
    '                xyzI = [lab2xyzC[int(lab)] for lab in compI]\n'
    '                cI = _center_transverse(xyzI)\n'
    '                d2 = _center_dist2(cM, cI)\n'
    '                cand = (d2, compI)\n'
    '                if bestI is None or cand[0] < bestI[0]:\n'
    '                    bestI = cand\n'
    '            compI = bestI[1]\n'
    '            om = _m8p_order_component_walk(part, compM, P0, Lx, Ly, Lz, tol_plane)\n'
    '            oi_all = _m8p_order_component_walk(part, compI, P1, Lx, Ly, Lz, tol_plane)\n'
    '            oi_bin = [int(lab) for lab in oi_all if int(lab) in img_nodes_bin_core]\n'
    '            img_anchor_in_comp = [int(lab) for lab in oi_all if int(lab) in edgeP1 or int(lab) in cornerP1]\n'
    '            img_edit_bin = [int(lab) for lab in oi_bin if int(lab) in img_edit]\n'
    '            img_frozen_comp = [int(lab) for lab in oi_all if int(lab) in frozen]\n'
    "            tag = 'I%s_%s_%s_%s' % (ID4, fam, bkey[0], bkey[1])\n"
    '            _make_fix_sets(axis, tag, master_topo=om, img_core=oi_bin, img_anchor_edge=img_anchor_in_comp, img_all=oi_all, img_edit=img_edit_bin, img_frozen=img_frozen_comp)\n'
    '            targets = []\n'
    '            for labm in om:\n'
    '                tC = _shift_to_P1(lab2xyzC[int(labm)])\n'
    '                if _m8p_target_hits_frozen_planes(tC, done_images, Lx, Ly, Lz):\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve target hits FrozenEdge planes (EDGE/IFACE mixing): axis=%s incSet=%s lab=%d target=%s' % (axis, sname, int(labm), str(tC)))\n"
    '                targets.append(tC)\n'
    '            if not targets:\n'
    '                continue\n'
    '            if len(oi_bin) == len(targets):\n'
    '                d_fwd = 0.0\n'
    '                d_rev = 0.0\n'
    '                for j in range(len(targets)):\n'
    '                    d_fwd += _dist2_transverse(lab2xyzC[int(oi_bin[j])], targets[j])\n'
    '                    d_rev += _dist2_transverse(lab2xyzC[int(oi_bin[-1 - j])], targets[j])\n'
    '                if d_rev < d_fwd:\n'
    '                    oi_bin = list(reversed(oi_bin))\n'
    '                    img_edit_bin = [int(lab) for lab in oi_bin if int(lab) in img_edit]\n'
    '            n = len(oi_bin)\n'
    '            m = len(targets)\n'
    '\n'
    '            def find_meshedge(a, b):\n'
    '                nA = part.nodes.getFromLabel(label=int(a))\n'
    '                for me in nA.getElemEdges():\n'
    '                    n1, n2 = me.getNodes()\n'
    '                    labs = {int(n1.label), int(n2.label)}\n'
    '                    if labs != {int(a), int(b)}:\n'
    '                        continue\n'
    '                    xyz1 = _m8a_canonicalize_xyz(n1.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '                    xyz2 = _m8a_canonicalize_xyz(n2.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '                    if _m8p_on_plane_xyzC(xyz1, P1, Lx, Ly, Lz) and _m8p_on_plane_xyzC(xyz2, P1, Lx, Ly, Lz):\n'
    '                        return me\n'
    '                return None\n'
    '            if n > m:\n'
    '                best = None\n'
    '                for idx, labi in enumerate(oi_bin):\n'
    '                    labi = int(labi)\n'
    '                    if labi not in img_edit:\n'
    '                        continue\n'
    '                    xyzI = lab2xyzC[labi]\n'
    '                    if _m8p_is_corner_xyzC(xyzI, Lx, Ly, Lz):\n'
    '                        continue\n'
    '                    d2min = None\n'
    '                    for tC in targets:\n'
    '                        d2 = _dist2_transverse(xyzI, tC)\n'
    '                        if d2min is None or d2 < d2min:\n'
    '                            d2min = d2\n'
    '                    if d2min is None:\n'
    '                        continue\n'
    '                    cand = (d2min, idx, labi)\n'
    '                    if best is None or cand[0] > best[0]:\n'
    '                        best = cand\n'
    '                if best is None:\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve collapse: no editable extra core node found: axis=%s incSet=%s bin=%s' % (axis, sname, str(bkey)))\n"
    '                lab_del = int(best[2])\n'
    '                pos = {}\n'
    '                for ii, lab in enumerate(oi_all):\n'
    '                    pos[int(lab)] = ii\n'
    '                i0 = pos.get(lab_del, None)\n'
    '                if i0 is None:\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve collapse: lab_del not in oi_all: %d' % lab_del)\n"
    '                neigh = []\n'
    '                if i0 > 0:\n'
    '                    neigh.append(int(oi_all[i0 - 1]))\n'
    '                if i0 + 1 < len(oi_all):\n'
    '                    neigh.append(int(oi_all[i0 + 1]))\n'
    '                neigh = [x for x in neigh if x in img_edit]\n'
    '                if not neigh:\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve collapse: immediate neighbors are not editable core nodes (likely anchor/frozen). del=%d' % lab_del)\n"
    '                lab_nb = int(neigh[0])\n'
    '                me = find_meshedge(lab_del, lab_nb)\n'
    '                if me is None:\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve collapse: cannot find MeshEdge on %s between %d-%d' % (P1, lab_del, lab_nb))\n"
    '                labs_before = set((int(n.label) for n in part.nodes))\n'
    '                part.collapseMeshEdge(edge=me, collapseMethod=AVERAGE)\n'
    '                labs_after = set((int(n.label) for n in part.nodes))\n'
    '                deleted = list(labs_before - labs_after)\n'
    "                ctx = 'IFACE axis=%s incSet=%s bin=%s del=%d nb=%d' % (axis, sname, str(bkey), int(lab_del), int(lab_nb))\n"
    "                evt = {'kind': 'COLLAPSE', 'nodes_created': [], 'nodes_deleted': deleted, 'edge_endpoints': (lab_del, lab_nb), 'context': ctx}\n"
    '                _m8p_s1_update_sets_after_evt(part, evt, Lx, Ly, Lz, tol_plane)\n'
    "                print('[M8P:IFACE] COLLAPSE axis=%s incSet=%s bin=%s del=%d nb=%d deleted=%s' % (axis, sname, str(bkey), lab_del, lab_nb, str(deleted)))\n"
    '                return True\n'
    '            if n < m:\n'
    '                best = None\n'
    '                for it, tC in enumerate(targets):\n'
    '                    d2min = None\n'
    '                    for labi in oi_bin:\n'
    '                        d2 = _dist2_transverse(lab2xyzC[int(labi)], tC)\n'
    '                        if d2min is None or d2 < d2min:\n'
    '                            d2min = d2\n'
    '                    if d2min is None:\n'
    '                        continue\n'
    '                    if best is None or d2min > best[0]:\n'
    '                        best = (d2min, it)\n'
    '                if best is None:\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve split: cannot choose missing target: axis=%s incSet=%s bin=%s' % (axis, sname, str(bkey)))\n"
    '                tC = targets[int(best[1])]\n'
    '                near = None\n'
    '                for lab in oi_bin:\n'
    '                    lab = int(lab)\n'
    '                    if lab not in img_edit:\n'
    '                        continue\n'
    '                    xyzI = lab2xyzC.get(lab, None)\n'
    '                    if xyzI is None:\n'
    '                        continue\n'
    '                    d2 = _dist2_transverse(xyzI, tC)\n'
    '                    if d2 <= float(tol_snap) * float(tol_snap):\n'
    '                        if near is None or d2 < near[0]:\n'
    '                            near = (d2, lab)\n'
    '                if near is not None:\n'
    '                    lab_hit = int(near[1])\n'
    '                    nd_hit = part.nodes.getFromLabel(label=int(lab_hit))\n'
    '                    part.editNode(nodes=(nd_hit,), coordinates=(tC,), projectToGeometry=OFF)\n'
    "                    ctx = 'IFACE axis=%s incSet=%s bin=%s reuse_edit hit=%d' % (axis, sname, str(bkey), int(lab_hit))\n"
    "                    evt = {'kind': 'EDIT', 'nodes_created': [], 'nodes_deleted': [], 'edge_endpoints': None, 'context': ctx}\n"
    '                    _m8p_s1_update_sets_after_evt(part, evt, Lx, Ly, Lz, tol_plane)\n'
    "                    print('[M8P:IFACE] REUSE_EDIT axis=%s incSet=%s bin=%s hit=%d target=%s' % (axis, sname, str(bkey), lab_hit, str(tC)))\n"
    '                    return True\n'
    "                if axis == 'X':\n"
    '                    px, py = (tC[1], tC[2])\n'
    '\n'
    '                    def proj(xyzC):\n'
    '                        return (xyzC[1], xyzC[2])\n'
    "                elif axis == 'Y':\n"
    '                    px, py = (tC[0], tC[2])\n'
    '\n'
    '                    def proj(xyzC):\n'
    '                        return (xyzC[0], xyzC[2])\n'
    '                else:\n'
    '                    px, py = (tC[0], tC[1])\n'
    '\n'
    '                    def proj(xyzC):\n'
    '                        return (xyzC[0], xyzC[1])\n'
    '                best_seg = None\n'
    '                for k in range(len(oi_all) - 1):\n'
    '                    a = int(oi_all[k])\n'
    '                    b = int(oi_all[k + 1])\n'
    '                    if a not in img_edit or b not in img_edit:\n'
    '                        continue\n'
    '                    ax2, ay2 = proj(lab2xyzC[a])\n'
    '                    bx2, by2 = proj(lab2xyzC[b])\n'
    '                    d2 = _m8p_dist2_point_to_segment_2d(px, py, ax2, ay2, bx2, by2)\n'
    '                    cand = (d2, a, b)\n'
    '                    if best_seg is None or cand[0] < best_seg[0]:\n'
    '                        best_seg = cand\n'
    '                if best_seg is None:\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve split: no editable core segment found for insertion (anchors/frozen blocking). axis=%s incSet=%s bin=%s' % (axis, sname, str(bkey)))\n"
    '                a = int(best_seg[1])\n'
    '                b = int(best_seg[2])\n'
    '                me = find_meshedge(a, b)\n'
    '                if me is None:\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve split: cannot find MeshEdge on %s between %d-%d' % (P1, a, b))\n"
    '                labs_before = set((int(n.label) for n in part.nodes))\n'
    '                part.splitMeshEdge(edge=me)\n'
    '                labs_after = set((int(n.label) for n in part.nodes))\n'
    '                new_labs = list(labs_after - labs_before)\n'
    '                if len(new_labs) != 1:\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve splitMeshEdge did not create exactly one new node: new=%s' % str(new_labs))\n"
    '                lab_new = int(new_labs[0])\n'
    '                nd_new = part.nodes.getFromLabel(label=int(lab_new))\n'
    '                part.editNode(nodes=(nd_new,), coordinates=(tC,), projectToGeometry=OFF)\n'
    "                ctx = 'IFACE axis=%s incSet=%s bin=%s split=%d-%d new=%d' % (axis, sname, str(bkey), int(a), int(b), int(lab_new))\n"
    "                evt = {'kind': 'SPLIT', 'nodes_created': [lab_new], 'nodes_deleted': [], 'edge_endpoints': (a, b), 'context': ctx}\n"
    '                _m8p_s1_update_sets_after_evt(part, evt, Lx, Ly, Lz, tol_plane)\n'
    "                print('[M8P:IFACE] SPLIT axis=%s incSet=%s bin=%s a=%d b=%d new=%d target=%s' % (axis, sname, str(bkey), a, b, lab_new, str(tC)))\n"
    '                return True\n'
    '            moved = []\n'
    '            for j in range(min(len(oi_bin), len(targets))):\n'
    '                labi = int(oi_bin[j])\n'
    '                if labi not in img_edit:\n'
    "                    raise RuntimeError('[M8P FATAL] IFACE-curve edit: core node not editable (unexpected for IFACEcurve core). lab=%d' % labi)\n"
    '                xyzI = lab2xyzC[labi]\n'
    '                tC = targets[j]\n'
    '                if math.sqrt(_dist2_transverse(xyzI, tC)) <= float(tol_pair):\n'
    '                    continue\n'
    '                ndI = part.nodes.getFromLabel(label=int(labi))\n'
    '                part.editNode(nodes=(ndI,), coordinates=(tC,), projectToGeometry=OFF)\n'
    '                moved.append(labi)\n'
    '            if moved:\n'
    "                ctx = 'IFACE axis=%s incSet=%s bin=%s moved=%s' % (axis, sname, str(bkey), str(moved))\n"
    "                evt = {'kind': 'EDIT', 'nodes_created': [], 'nodes_deleted': [], 'edge_endpoints': None, 'context': ctx}\n"
    '                _m8p_s1_update_sets_after_evt(part, evt, Lx, Ly, Lz, tol_plane)\n'
    "                print('[M8P:IFACE] EDIT axis=%s incSet=%s bin=%s moved=%d' % (axis, sname, str(bkey), len(moved)))\n"
    '                return True\n'
    '            return False\n'
    '    return False\n'
    '\n'
    "def _m8p_make_topology_debug_sets(part, Lx, Ly, Lz, tol_plane, prefix='M8P', max_print=20):\n"
    '    edge2elems = defaultdict(list)\n'
    '    for el in part.elements:\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        labs = [int(n.label) for n in nds]\n'
    '        eLab = int(el.label)\n'
    '        pairs = ((labs[0], labs[1]), (labs[1], labs[2]), (labs[2], labs[0]))\n'
    '        for a, b in pairs:\n'
    '            if a > b:\n'
    '                a, b = (b, a)\n'
    '            edge2elems[a, b].append(eLab)\n'
    '    free_edges = []\n'
    '    nonm_edges = []\n'
    '    for (a, b), adj in edge2elems.items():\n'
    '        if len(adj) == 1:\n'
    '            free_edges.append((a, b, adj))\n'
    '        elif len(adj) > 2:\n'
    '            nonm_edges.append((a, b, adj))\n'
    '    free_nodes = set()\n'
    '    free_elems = set()\n'
    '    for a, b, adj in free_edges:\n'
    '        free_nodes.add(int(a))\n'
    '        free_nodes.add(int(b))\n'
    '        free_elems |= set((int(x) for x in adj))\n'
    '    nonm_nodes = set()\n'
    '    nonm_elems = set()\n'
    '    for a, b, adj in nonm_edges:\n'
    '        nonm_nodes.add(int(a))\n'
    '        nonm_nodes.add(int(b))\n'
    '        nonm_elems |= set((int(x) for x in adj))\n'
    "    _m8a_make_node_set_from_labels(part, '%s_FREEEDGE_NODES' % prefix, sorted(free_nodes), allow_empty=True)\n"
    "    _m8a_make_elem_set_from_labels(part, '%s_FREEEDGE_ELEMS' % prefix, sorted(free_elems), allow_empty=True)\n"
    "    _m8a_make_node_set_from_labels(part, '%s_NONMANIF_NODES' % prefix, sorted(nonm_nodes), allow_empty=True)\n"
    "    _m8a_make_elem_set_from_labels(part, '%s_NONMANIF_ELEMS' % prefix, sorted(nonm_elems), allow_empty=True)\n"
    '\n'
    '    def _node_xyzC(lab):\n'
    '        nd = part.nodes.getFromLabel(label=int(lab))\n'
    '        return _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '\n'
    '    def _cap_sets_for_node(lab):\n'
    '        hits = []\n'
    '        for sname, s in part.sets.items():\n'
    "            if not (str(sname).startswith('INC_') and '_CAP_' in str(sname)):\n"
    '                continue\n'
    '            try:\n'
    '                for n in s.nodes:\n'
    '                    if int(n.label) == int(lab):\n'
    '                        hits.append(str(sname))\n'
    '                        break\n'
    '            except Exception:\n'
    '                pass\n'
    '        return hits\n'
    '\n'
    '    def _cap_sets_for_elem(eLab):\n'
    '        hits = []\n'
    '        for sname, s in part.sets.items():\n'
    "            if not (str(sname).startswith('INC_') and '_CAP_' in str(sname)):\n"
    '                continue\n'
    '            try:\n'
    '                for e in s.elements:\n'
    '                    if int(e.label) == int(eLab):\n'
    '                        hits.append(str(sname))\n'
    '                        break\n'
    '            except Exception:\n'
    '                pass\n'
    '        return hits\n'
    '\n'
    '    def _elem_detail(eLab):\n'
    '        el = part.elements.getFromLabel(label=int(eLab))\n'
    '        nds = el.getNodes()\n'
    '        labs = [int(n.label) for n in nds]\n'
    '        xyzs = [_node_xyzC(lab) for lab in labs]\n'
    '        node_caps = dict(((int(lab), _cap_sets_for_node(lab)) for lab in labs))\n'
    '        elem_caps = _cap_sets_for_elem(eLab)\n'
    '        return (labs, xyzs, node_caps, elem_caps)\n'
    "    print('[M8P:TOPO_DBG] freeEdges=%d nonManifoldEdges=%d' % (len(free_edges), len(nonm_edges)))\n"
    '    if free_edges:\n'
    "        print('[M8P:TOPO_DBG] Sample free edges (a,b, xyzC_a, xyzC_b, adjElems):')\n"
    '        for a, b, adj in free_edges[:int(max_print)]:\n'
    "            print('  (%d,%d) %s %s adj=%s' % (a, b, str(_node_xyzC(a)), str(_node_xyzC(b)), str(adj)))\n"
    '            for eLab in adj:\n'
    '                labs, xyzs, node_caps, elem_caps = _elem_detail(eLab)\n'
    "                print('    adjElem=%d triNodes=%s triXYZC=%s elemCapSets=%s' % (int(eLab), str(labs), str(xyzs), str(elem_caps)))\n"
    "                print('    nodeCapSets=%s' % str(node_caps))\n"
    '    if nonm_edges:\n'
    "        print('[M8P:TOPO_DBG] Sample non-manifold edges (a,b, xyzC_a, xyzC_b, adjCount):')\n"
    '        for a, b, adj in nonm_edges[:int(max_print)]:\n'
    "            print('  (%d,%d) %s %s adjCount=%d' % (a, b, str(_node_xyzC(a)), str(_node_xyzC(b)), len(adj)))\n"
    '            for eLab in adj[:5]:\n'
    '                labs, xyzs, node_caps, elem_caps = _elem_detail(eLab)\n'
    "                print('    adjElem=%d triNodes=%s triXYZC=%s elemCapSets=%s' % (int(eLab), str(labs), str(xyzs), str(elem_caps)))\n"
    '    return (len(free_edges), len(nonm_edges))\n'
    '\n'
    'def _m8p_target_hits_frozen_planes(target_xyzC, done_images, Lx, Ly, Lz):\n'
    '    x, y, z = target_xyzC\n'
    "    if 'X' in done_images and (x == 0.0 or x == float(Lx)):\n"
    '        return True\n'
    "    if 'Y' in done_images and (y == 0.0 or y == float(Ly)):\n"
    '        return True\n'
    "    if 'Z' in done_images and (z == 0.0 or z == float(Lz)):\n"
    '        return True\n'
    '    return False\n'
    '\n'
    'def _m8p_snap_intersection_edges_pre(part, axis, done_images, Lx, Ly, Lz, tol_plane, tol_pair, tol_snap, max_ops_per_edge=50):\n'
    '    import re\n'
    "    step_idx = {'X': 1, 'Y': 2, 'Z': 3}[axis]\n"
    "    P0 = axis + '0'\n"
    "    P1 = axis + '1'\n"
    "    eligible_B = [b for b in ('X', 'Y', 'Z') if b != axis and b not in done_images]\n"
    '    if not eligible_B:\n'
    '        return (False, set())\n'
    '    frozen_labels = _m8p_build_frozen_edge_labels_on_P1(part, P1, done_images, Lx, Ly, Lz, tol_plane)\n'
    '    touched = set()\n'
    '    changed_any = False\n'
    "    cap_re = re.compile('^INC_\\\\d{4}_CAP_[XYZ]0$')\n"
    '\n'
    '    def _build_cap_signature_map():\n'
    '        lab2sig = defaultdict(set)\n'
    '        for sname, s in part.sets.items():\n'
    '            sname = str(sname)\n'
    '            if not cap_re.match(sname):\n'
    '                continue\n'
    '            has_nodes = False\n'
    '            try:\n'
    '                for n in s.nodes:\n'
    '                    has_nodes = True\n'
    '                    lab2sig[int(n.label)].add(sname)\n'
    '            except Exception:\n'
    '                pass\n'
    '            if has_nodes:\n'
    '                continue\n'
    '            try:\n'
    '                for e in s.elements:\n'
    '                    for nd in e.getNodes():\n'
    '                        lab2sig[int(nd.label)].add(sname)\n'
    '            except Exception:\n'
    '                pass\n'
    '        return dict(((int(k), frozenset(v)) for k, v in lab2sig.items()))\n'
    '\n'
    '    def _fmt_sig(sig):\n'
    '        if not sig:\n'
    "            return '{}'\n"
    "        return '{' + ','.join(sorted((str(x) for x in sig))) + '}'\n"
    '\n'
    '    def _is_semantic_boundary_node(lab, cap_sig):\n'
    '        if cap_sig:\n'
    '            return True\n'
    '        try:\n'
    '            nd = part.nodes.getFromLabel(label=int(lab))\n'
    '            return bool(_m8p_adj_has_nonouter(part, nd, Lx, Ly, Lz, tol_plane))\n'
    '        except Exception:\n'
    '            return True\n'
    '\n'
    '    def _planes_for_fam(fam):\n'
    "        if fam == 'X0':\n"
    "            return ('X0', 'X1')\n"
    "        if fam == 'Y0':\n"
    "            return ('Y0', 'Y1')\n"
    "        return ('Z0', 'Z1')\n"
    '\n'
    '    def _node_on_any_fam_plane(lab, fam):\n'
    '        try:\n'
    '            xyzC = _m8a_canonicalize_xyz(part.nodes.getFromLabel(label=int(lab)).coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        except Exception:\n'
    '            return False\n'
    '        for pl in _planes_for_fam(fam):\n'
    '            if _m8p_on_plane_xyzC(xyzC, pl, Lx, Ly, Lz):\n'
    '                return True\n'
    '        return False\n'
    '\n'
    '    def _set_node_labels_fallback(s):\n'
    '        labs = set()\n'
    '        try:\n'
    '            for n in s.nodes:\n'
    '                labs.add(int(n.label))\n'
    '        except Exception:\n'
    '            pass\n'
    '        try:\n'
    '            for e in s.elements:\n'
    '                for nd in e.getNodes():\n'
    '                    labs.add(int(nd.label))\n'
    '        except Exception:\n'
    '            pass\n'
    '        return labs\n'
    '\n'
    '    def _force_node_into_cap_signature(lab, cap_signature):\n'
    '        existing_elems = set((int(e.label) for e in part.elements))\n'
    '        for sname in sorted((str(x) for x in cap_signature)):\n'
    '            if sname not in part.sets:\n'
    '                continue\n'
    "            if '_CAP_' not in sname:\n"
    '                continue\n'
    "            fam = sname.rsplit('_CAP_', 1)[1]\n"
    '            if not _node_on_any_fam_plane(lab, fam):\n'
    '                continue\n'
    '            nLabs = _set_node_labels_fallback(part.sets[sname])\n'
    '            nLabs.add(int(lab))\n'
    '            eLabs = set()\n'
    '            for pl in _planes_for_fam(fam):\n'
    '                for eLab in _m8p_outer_elem_labels_by_plane(part, pl, Lx, Ly, Lz, tol_plane):\n'
    '                    if int(eLab) not in existing_elems:\n'
    '                        continue\n'
    '                    el = part.elements.getFromLabel(label=int(eLab))\n'
    '                    nds = el.getNodes()\n'
    '                    if len(nds) != 3:\n'
    '                        continue\n'
    '                    labs = [int(n.label) for n in nds]\n'
    '                    if labs[0] in nLabs and labs[1] in nLabs and (labs[2] in nLabs):\n'
    '                        eLabs.add(int(eLab))\n'
    '            _m8p_make_set_nodes_elems_from_labels(part, sname, sorted(nLabs), sorted(eLabs), allow_empty=True)\n'
    '\n'
    '    def _find_meshedge_on_two_planes(labA, labB, planeA, planeB):\n'
    '        nA = part.nodes.getFromLabel(label=int(labA))\n'
    '        for me in nA.getElemEdges():\n'
    '            n1, n2 = me.getNodes()\n'
    '            labs = {int(n1.label), int(n2.label)}\n'
    '            if labs != {int(labA), int(labB)}:\n'
    '                continue\n'
    '            xyz1 = _m8a_canonicalize_xyz(n1.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            xyz2 = _m8a_canonicalize_xyz(n2.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            if _m8p_on_plane_xyzC(xyz1, planeA, Lx, Ly, Lz) and _m8p_on_plane_xyzC(xyz1, planeB, Lx, Ly, Lz) and _m8p_on_plane_xyzC(xyz2, planeA, Lx, Ly, Lz) and _m8p_on_plane_xyzC(xyz2, planeB, Lx, Ly, Lz):\n'
    '                return me\n'
    '        return None\n'
    '\n'
    '    def _enforce_A0_B0_on_xyz(axis, Q0, xyz):\n'
    '        x, y, z = xyz\n'
    "        if axis == 'X':\n"
    '            x = 0.0\n'
    "        elif axis == 'Y':\n"
    '            y = 0.0\n'
    "        elif axis == 'Z':\n"
    '            z = 0.0\n'
    "        if Q0 == 'X0':\n"
    '            x = 0.0\n'
    "        elif Q0 == 'Y0':\n"
    '            y = 0.0\n'
    "        elif Q0 == 'Z0':\n"
    '            z = 0.0\n'
    '        return (x, y, z)\n'
    '\n'
    '    def _node_xyzC(lab):\n'
    '        return _m8a_canonicalize_xyz(part.nodes.getFromLabel(label=int(lab)).coordinates, Lx, Ly, Lz, tol_plane)\n'
    '\n'
    '    def _elem_plane(eLab, outerPlaneOfElem):\n'
    '        return outerPlaneOfElem.get(int(eLab), None)\n'
    '\n'
    '    def _elem_nodes_labs(eLab):\n'
    '        el = part.elements.getFromLabel(label=int(eLab))\n'
    '        return [int(n.label) for n in el.getNodes()]\n'
    '\n'
    '    def _find_node_by_xyzC(xyzC_target, forbid_labels=None):\n'
    '        forbid = set((int(x) for x in forbid_labels or []))\n'
    '        best = None\n'
    '        for nd in part.nodes:\n'
    '            lab = int(nd.label)\n'
    '            if lab in forbid:\n'
    '                continue\n'
    '            xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            d2 = (float(xyzC[0]) - float(xyzC_target[0])) ** 2 + (float(xyzC[1]) - float(xyzC_target[1])) ** 2 + (float(xyzC[2]) - float(xyzC_target[2])) ** 2\n'
    '            if best is None or d2 < best[0]:\n'
    '                best = (d2, lab, xyzC)\n'
    '        if best is None:\n'
    '            return None\n'
    '        if best[0] <= float(tol_pair) * float(tol_pair):\n'
    '            return int(best[1])\n'
    '        return None\n'
    '\n'
    '    def _outer_tri_key_map():\n'
    '        mp = {}\n'
    '        for el in part.elements:\n'
    '            nds = el.getNodes()\n'
    '            if len(nds) != 3:\n'
    '                continue\n'
    '            labs = tuple(sorted((int(n.label) for n in nds)))\n'
    '            mp[labs] = int(el.label)\n'
    '        return mp\n'
    '\n'
    '    def _delete_nodes_by_labels(node_labels):\n'
    '        labs = sorted(set((int(x) for x in node_labels)))\n'
    '        if not labs:\n'
    '            return 0\n'
    '        nodes = part.nodes.sequenceFromLabels(labels=labs)\n'
    '        part.deleteNode(nodes=nodes)\n'
    '        return len(labs)\n'
    '\n'
    '    def _copy_xyz_by_axis(xyzC):\n'
    '        sx, sy, sz = _m8p_shift(axis, Lx, Ly, Lz)\n'
    '        return _m8a_canonicalize_xyz((float(xyzC[0]) + sx, float(xyzC[1]) + sy, float(xyzC[2]) + sz), Lx, Ly, Lz, tol_plane)\n'
    '\n'
    '    def _rebuild_inclusion_edge_patch_from_master(Q0, freeI, S, T, extraT, missS, outerPlaneOfElem):\n'
    '        if not extraT and (not missS):\n'
    '            return False\n'
    '        cap_sig = _build_cap_signature_map()\n'
    '        related_sig = set()\n'
    '        seed_u = None\n'
    '        for jj in extraT:\n'
    '            u, lab, xyz, prot = T[int(jj)]\n'
    '            related_sig |= set(cap_sig.get(int(lab), frozenset()))\n'
    '            if seed_u is None:\n'
    '                seed_u = float(u)\n'
    '        for ii in missS:\n'
    '            u, lab, xyz, prot = S[int(ii)]\n'
    '            related_sig |= set(cap_sig.get(int(lab), frozenset()))\n'
    '            if seed_u is None:\n'
    '                seed_u = float(u)\n'
    '        if not related_sig:\n'
    '            return False\n'
    '        related_sig = frozenset(related_sig)\n'
    '\n'
    '        def has_related_sig(lab):\n'
    '            return bool(set(cap_sig.get(int(lab), frozenset())) & set(related_sig))\n'
    '        master_edge = [(float(u), int(lab), xyz, prot) for u, lab, xyz, prot in S if has_related_sig(lab)]\n'
    '        image_edge = [(float(u), int(lab), xyz, prot) for u, lab, xyz, prot in T if has_related_sig(lab)]\n'
    '        if len(master_edge) < 2 or len(image_edge) < 2:\n'
    '            return False\n'
    '        master_edge.sort(key=lambda t: (t[0], t[1]))\n'
    '        image_edge.sort(key=lambda t: (t[0], t[1]))\n'
    '\n'
    '        def _edge_component_labs(candidates, seed_labs, seed_u0):\n'
    '            allowed = set((int(x[1]) for x in candidates))\n'
    '            if not allowed:\n'
    '                return set()\n'
    '            seeds = set((int(x) for x in seed_labs if int(x) in allowed))\n'
    '            if not seeds and seed_u0 is not None:\n'
    '                best = None\n'
    '                for u, lab, xyz, prot in candidates:\n'
    '                    cand = (abs(float(u) - float(seed_u0)), int(lab))\n'
    '                    if best is None or cand[0] < best[0]:\n'
    '                        best = cand\n'
    '                if best is not None:\n'
    '                    seeds.add(int(best[1]))\n'
    '            if not seeds:\n'
    '                return set(allowed)\n'
    '            adj = defaultdict(set)\n'
    '            for lab in allowed:\n'
    '                try:\n'
    '                    nd = part.nodes.getFromLabel(label=int(lab))\n'
    '                    for me in nd.getElemEdges():\n'
    '                        n1, n2 = me.getNodes()\n'
    '                        a = int(n1.label)\n'
    '                        b = int(n2.label)\n'
    '                        if a in allowed and b in allowed:\n'
    '                            adj[a].add(b)\n'
    '                            adj[b].add(a)\n'
    '                except Exception:\n'
    '                    pass\n'
    '            comp = set()\n'
    '            stack = list(seeds)\n'
    '            while stack:\n'
    '                lab = int(stack.pop())\n'
    '                if lab in comp:\n'
    '                    continue\n'
    '                comp.add(lab)\n'
    '                for nb in adj.get(lab, ()):\n'
    '                    if nb not in comp:\n'
    '                        stack.append(int(nb))\n'
    '            if not comp:\n'
    '                return set(allowed)\n'
    '            return comp\n'
    '        seed_master = []\n'
    '        for ii in missS:\n'
    '            u0, lab0, xyz0, prot0 = S[int(ii)]\n'
    '            if has_related_sig(lab0):\n'
    '                seed_master.append(int(lab0))\n'
    '        seed_image = []\n'
    '        for jj in extraT:\n'
    '            u0, lab0, xyz0, prot0 = T[int(jj)]\n'
    '            if has_related_sig(lab0):\n'
    '                seed_image.append(int(lab0))\n'
    '        master_comp_labs = _edge_component_labs(master_edge, seed_master, seed_u)\n'
    '        image_comp_labs = _edge_component_labs(image_edge, seed_image, seed_u)\n'
    '        master_edge = [x for x in master_edge if int(x[1]) in master_comp_labs]\n'
    '        image_edge = [x for x in image_edge if int(x[1]) in image_comp_labs]\n'
    '        master_edge.sort(key=lambda t: (t[0], t[1]))\n'
    '        image_edge.sort(key=lambda t: (t[0], t[1]))\n'
    '        if len(master_edge) < 2 or len(image_edge) < 2:\n'
    '            return False\n'
    '        master_edge_labs = [int(x[1]) for x in master_edge]\n'
    '        image_edge_labs = [int(x[1]) for x in image_edge]\n'
    '        master_anchor_labs = (master_edge_labs[0], master_edge_labs[-1])\n'
    '        image_anchor_labs = []\n'
    '        used_img_anchors = set()\n'
    '        for mLab in master_anchor_labs:\n'
    '            target = _copy_xyz_by_axis(_node_xyzC(mLab))\n'
    '            best = None\n'
    '            for labI in image_edge_labs:\n'
    '                if labI in used_img_anchors:\n'
    '                    continue\n'
    '                xyzI = _node_xyzC(labI)\n'
    '                d2 = (float(xyzI[0]) - float(target[0])) ** 2 + (float(xyzI[1]) - float(target[1])) ** 2 + (float(xyzI[2]) - float(target[2])) ** 2\n'
    '                cand = (d2, int(labI))\n'
    '                if best is None or cand[0] < best[0]:\n'
    '                    best = cand\n'
    '            if best is None:\n'
    '                return False\n'
    '            image_anchor_labs.append(int(best[1]))\n'
    '            used_img_anchors.add(int(best[1]))\n'
    '        image_anchor_labs = tuple(image_anchor_labs)\n'
    '        image_repair_labs = sorted(set(image_edge_labs) - set(image_anchor_labs))\n'
    '        if not image_repair_labs:\n'
    '            return False\n'
    '        related_cap_elems = set()\n'
    '        for sname in related_sig:\n'
    '            if sname not in part.sets:\n'
    '                continue\n'
    '            try:\n'
    '                for e in part.sets[sname].elements:\n'
    '                    related_cap_elems.add(int(e.label))\n'
    '            except Exception:\n'
    '                pass\n'
    '        if not related_cap_elems:\n'
    '            return False\n'
    '        master_patch_elems = []\n'
    '        master_edge_set = set(master_edge_labs)\n'
    '        for eLab in sorted(related_cap_elems):\n'
    '            pl = _elem_plane(eLab, outerPlaneOfElem)\n'
    '            if pl not in (P0, Q0):\n'
    '                continue\n'
    '            labs = _elem_nodes_labs(eLab)\n'
    '            if len(labs) != 3:\n'
    '                continue\n'
    '            if not set(labs) & master_edge_set:\n'
    '                continue\n'
    '            master_patch_elems.append(int(eLab))\n'
    '        if not master_patch_elems:\n'
    '            return False\n'
    '        master_patch_nodes = set()\n'
    '        for eLab in master_patch_elems:\n'
    '            master_patch_nodes |= set(_elem_nodes_labs(eLab))\n'
    '        repair_set = set(image_repair_labs)\n'
    '        image_related_elems = []\n'
    '        for eLab, pl in outerPlaneOfElem.items():\n'
    '            if pl not in (P1, Q0):\n'
    '                continue\n'
    '            labs = _elem_nodes_labs(eLab)\n'
    '            if len(labs) != 3:\n'
    '                continue\n'
    '            if set(labs) & repair_set:\n'
    '                image_related_elems.append(int(eLab))\n'
    '        if not image_related_elems:\n'
    '            return False\n'
    '        before_node_info = dict(((int(lab), _node_xyzC(lab)) for lab in image_repair_labs))\n'
    '        before_elem_info = []\n'
    '        for eLab in image_related_elems:\n'
    '            labs = _elem_nodes_labs(eLab)\n'
    '            before_elem_info.append((int(eLab), _elem_plane(eLab, outerPlaneOfElem), labs, [_node_xyzC(l) for l in labs]))\n'
    "        print('[M8P:EDGE_REBUILD] START axis=%s Q0=%s edge=%s related=%s masterEdge=%d imageEdge=%d repairNodes=%d delElems=%d masterElems=%d' % (axis, Q0, edge_tgt_tag, _fmt_sig(related_sig), len(master_edge_labs), len(image_edge_labs), len(image_repair_labs), len(image_related_elems), len(master_patch_elems)))\n"
    '        _m8a_delete_elements_by_labels(part, image_related_elems)\n'
    '        still_attached = []\n'
    '        for lab in image_repair_labs:\n'
    '            try:\n'
    '                if len(part.nodes.getFromLabel(label=int(lab)).getElements()) != 0:\n'
    '                    still_attached.append(int(lab))\n'
    '            except Exception:\n'
    '                pass\n'
    '        if still_attached:\n'
    "            raise RuntimeError('[M8P FATAL] EDGE_REBUILD repair nodes still attached after patch delete: %s' % str(still_attached[:20]))\n"
    '        _delete_nodes_by_labels(image_repair_labs)\n'
    "        evt_del = {'kind': 'COLLAPSE', 'nodes_created': [], 'nodes_deleted': image_repair_labs, 'edge_endpoints': None, 'context': 'EDGE_REBUILD delete axis=%s Q0=%s edge=%s' % (axis, Q0, edge_tgt_tag)}\n"
    '        _m8p_s1_update_sets_after_evt(part, evt_del, Lx, Ly, Lz, tol_plane)\n'
    '        mapM2I = {}\n'
    '        for mLab, iLab in zip(master_anchor_labs, image_anchor_labs):\n'
    '            target = _copy_xyz_by_axis(_node_xyzC(mLab))\n'
    '            _m8p_editNode_xyz_off(part, part.nodes.getFromLabel(label=int(iLab)), target)\n'
    '            mapM2I[int(mLab)] = int(iLab)\n'
    '        new_image_nodes = []\n'
    '        forbid = set(image_repair_labs)\n'
    '        for mLab in sorted(master_patch_nodes):\n'
    '            if int(mLab) in mapM2I:\n'
    '                continue\n'
    '            target = _copy_xyz_by_axis(_node_xyzC(mLab))\n'
    '            if int(mLab) in master_edge_set:\n'
    '                nd_new = part.Node(coordinates=target)\n'
    '                iLab = int(nd_new.label)\n'
    '                new_image_nodes.append(iLab)\n'
    '                mapM2I[int(mLab)] = iLab\n'
    '                continue\n'
    '            hit = _find_node_by_xyzC(target, forbid_labels=forbid)\n'
    '            if hit is None:\n'
    '                nd_new = part.Node(coordinates=target)\n'
    '                hit = int(nd_new.label)\n'
    '                new_image_nodes.append(hit)\n'
    '            mapM2I[int(mLab)] = int(hit)\n'
    '        missing_map = sorted(set(master_patch_nodes) - set(mapM2I.keys()))\n'
    '        if missing_map:\n'
    "            raise RuntimeError('[M8P FATAL] EDGE_REBUILD mapM2I incomplete: %s' % str(missing_map[:20]))\n"
    '        tri2elem = _outer_tri_key_map()\n'
    '        created_elems = []\n'
    '        for eLab in master_patch_elems:\n'
    '            labsM = _elem_nodes_labs(eLab)\n'
    '            labsI = [int(mapM2I[int(x)]) for x in labsM]\n'
    '            if len(set(labsI)) != 3:\n'
    "                raise RuntimeError('[M8P FATAL] EDGE_REBUILD degenerate image tri from master elem %d: %s' % (int(eLab), str(labsI)))\n"
    '            keyI = tuple(sorted(labsI))\n'
    '            if keyI in tri2elem:\n'
    '                continue\n'
    '            plM = _elem_plane(eLab, outerPlaneOfElem)\n'
    '            ndsI = [part.nodes.getFromLabel(label=int(x)) for x in labsI]\n'
    '            if plM == P0:\n'
    '                el_new = part.Element(nodes=(ndsI[0], ndsI[2], ndsI[1]), elemShape=TRI3)\n'
    '            else:\n'
    '                el_new = part.Element(nodes=(ndsI[0], ndsI[1], ndsI[2]), elemShape=TRI3)\n'
    '            created_elems.append(int(el_new.label))\n'
    '            tri2elem[keyI] = int(el_new.label)\n'
    '        for lab in new_image_nodes:\n'
    '            _force_node_into_cap_signature(lab, related_sig)\n'
    '        for lab in image_anchor_labs:\n'
    '            _force_node_into_cap_signature(lab, related_sig)\n'
    "        print('[M8P:EDGE_REBUILD] DONE axis=%s Q0=%s edge=%s deletedNodes=%d deletedElems=%d createdNodes=%d createdElems=%d beforeNodes=%s' % (axis, Q0, edge_tgt_tag, len(image_repair_labs), len(image_related_elems), len(new_image_nodes), len(created_elems), str(sorted(before_node_info.keys()))))\n"
    '        return True\n'
    '\n'
    '    def _collect_master_plane_edge_nodes(Q0, outerPlaneOfElem):\n'
    '        outerElems_Q0 = [int(eLab) for eLab, pl in outerPlaneOfElem.items() if pl == Q0]\n'
    '        if not outerElems_Q0:\n'
    '            return []\n'
    '        raw_labs = set()\n'
    '        for eLab in outerElems_Q0:\n'
    '            el = part.elements.getFromLabel(label=int(eLab))\n'
    '            for nd in el.getNodes():\n'
    '                raw_labs.add(int(nd.label))\n'
    '        out = []\n'
    '        for lab in raw_labs:\n'
    '            nd = part.nodes.getFromLabel(label=int(lab))\n'
    '            xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '            if not _m8p_on_plane_xyzC(xyzC, Q0, Lx, Ly, Lz):\n'
    '                continue\n'
    '            ok = False\n'
    '            for adjE in nd.getElements():\n'
    '                if outerPlaneOfElem.get(int(adjE.label), None) == Q0:\n'
    '                    ok = True\n'
    '                    break\n'
    '            if not ok:\n'
    '                continue\n'
    '            isEdge, isCorner = _m8a_partition_outer_node_type_on_master_plane(Q0, xyzC, Lx, Ly, Lz)\n'
    '            if isEdge:\n'
    '                out.append((int(lab), xyzC))\n'
    '        return out\n'
    '\n'
    '    def _is_protected(Q0, lab, xyzC):\n'
    '        if int(lab) in frozen_labels:\n'
    '            return True\n'
    '        if _m8p_is_corner_xyzC(xyzC, Lx, Ly, Lz):\n'
    '            return True\n'
    '        return False\n'
    '    for B in eligible_B:\n'
    "        Q0 = B + '0'\n"
    '        freeAx = _m8p_axis_free_axis(axis, B)\n'
    '        freeI = _m8p_axis_index(freeAx)\n'
    "        edge_src_tag = '%s0%s0' % (axis, B)\n"
    "        edge_tgt_tag = '%s1%s0' % (axis, B)\n"
    '        for op_it in range(int(max_ops_per_edge)):\n'
    '            outerPlaneOfElem = _m8p_build_outerPlaneOfElem(part, Lx, Ly, Lz, tol_plane)\n'
    '            edge_nodes_Q0 = _collect_master_plane_edge_nodes(Q0, outerPlaneOfElem)\n'
    '            S = []\n'
    '            T = []\n'
    '            for lab, xyzC in edge_nodes_Q0:\n'
    '                u = xyzC[freeI]\n'
    '                prot = _is_protected(Q0, lab, xyzC)\n'
    '                if _m8p_on_plane_xyzC(xyzC, P0, Lx, Ly, Lz):\n'
    '                    S.append((u, lab, xyzC, prot))\n'
    '                if _m8p_on_plane_xyzC(xyzC, P1, Lx, Ly, Lz):\n'
    '                    T.append((u, lab, xyzC, prot))\n'
    '            S.sort(key=lambda t: (t[0], t[1]))\n'
    '            T.sort(key=lambda t: (t[0], t[1]))\n'
    '            matched = []\n'
    '            usedS = set()\n'
    '            usedT = set()\n'
    '            bucketsS = defaultdict(list)\n'
    '            tp_match = max(float(tol_pair), 1e-18)\n'
    '            for ii, (us, ls, xs, ps) in enumerate(S):\n'
    '                bucketsS[int(round(float(us) / tp_match))].append(ii)\n'
    '            for jj, (ut, lt, xt, pt) in enumerate(T):\n'
    '                best_exact = None\n'
    '                keyT = int(round(float(ut) / tp_match))\n'
    '                for dk in (-1, 0, 1):\n'
    '                    for ii in bucketsS.get(keyT + dk, []):\n'
    '                        if ii in usedS:\n'
    '                            continue\n'
    '                        du = abs(float(S[ii][0]) - float(ut))\n'
    '                        if du <= float(tol_pair):\n'
    '                            cand = (du, ii)\n'
    '                            if best_exact is None or cand[0] < best_exact[0]:\n'
    '                                best_exact = cand\n'
    '                if best_exact is not None:\n'
    '                    ii = int(best_exact[1])\n'
    '                    matched.append((ii, jj))\n'
    '                    usedS.add(ii)\n'
    '                    usedT.add(jj)\n'
    '            remS = [ii for ii in range(len(S)) if ii not in usedS]\n'
    '            remT = [jj for jj in range(len(T)) if jj not in usedT]\n'
    '            i = 0\n'
    '            j = 0\n'
    '            extraT = []\n'
    '            missS = []\n'
    '            while i < len(remS) and j < len(remT):\n'
    '                ii = remS[i]\n'
    '                jj = remT[j]\n'
    '                us, ls, xs, ps = S[ii]\n'
    '                ut, lt, xt, pt = T[jj]\n'
    '                if abs(ut - us) <= tol_snap:\n'
    '                    matched.append((ii, jj))\n'
    '                    i += 1\n'
    '                    j += 1\n'
    '                elif ut < us:\n'
    '                    extraT.append(jj)\n'
    '                    j += 1\n'
    '                else:\n'
    '                    missS.append(ii)\n'
    '                    i += 1\n'
    '            while i < len(remS):\n'
    '                missS.append(remS[i])\n'
    '                i += 1\n'
    '            while j < len(remT):\n'
    '                extraT.append(remT[j])\n'
    '                j += 1\n'
    '            shiftPairs = []\n'
    '            for ii, jj in matched:\n'
    '                us, ls, xs, ps = S[ii]\n'
    '                ut, lt, xt, pt = T[jj]\n'
    '                if abs(ut - us) > tol_pair:\n'
    '                    shiftPairs.append((jj, us))\n'
    '            if extraT or missS:\n'
    '                cap_sig_probe = _build_cap_signature_map()\n'
    '                semantic_hits = []\n'
    '                for jj in extraT:\n'
    '                    ut0, lab0, xyz0, prot0 = T[int(jj)]\n'
    '                    sig0 = cap_sig_probe.get(int(lab0), frozenset())\n'
    '                    if sig0 or _is_semantic_boundary_node(lab0, sig0):\n'
    "                        semantic_hits.append(('EXTRA', int(lab0), float(ut0), _fmt_sig(sig0)))\n"
    '                for ii in missS:\n'
    '                    us0, lab0, xyz0, prot0 = S[int(ii)]\n'
    '                    sig0 = cap_sig_probe.get(int(lab0), frozenset())\n'
    '                    if sig0 or _is_semantic_boundary_node(lab0, sig0):\n'
    "                        semantic_hits.append(('MISS', int(lab0), float(us0), _fmt_sig(sig0)))\n"
    '                rebuilt = _rebuild_inclusion_edge_patch_from_master(Q0, freeI, S, T, extraT, missS, outerPlaneOfElem)\n'
    '                if rebuilt:\n'
    '                    changed_any = True\n'
    '                    touched.add(Q0)\n'
    '                    continue\n'
    '                if semantic_hits:\n'
    "                    raise RuntimeError('[M8P FATAL] inclusion/iface EDGE mismatch was detected but patch rebuild did not apply; refusing collapse/split. axis=%s Q0=%s edge=%s hits=%s extra=%s miss=%s' % (axis, Q0, edge_tgt_tag, str(semantic_hits[:20]), str(extraT[:20]), str(missS[:20])))\n"
    '            if extraT:\n'
    '                jj = extraT[0]\n'
    '                ut, lab_extra, xyz_extra, prot_extra = T[jj]\n'
    '                if prot_extra:\n'
    "                    raise RuntimeError('[M8P FATAL] extra node is protected (cannot collapse): axis=%s edge=%s lab=%d' % (axis, edge_tgt_tag, int(lab_extra)))\n"
    '                candidates = []\n'
    '                if jj > 0:\n'
    '                    candidates.append(jj - 1)\n'
    '                if jj + 1 < len(T):\n'
    '                    candidates.append(jj + 1)\n'
    '                if not candidates:\n'
    "                    raise RuntimeError('[M8P FATAL] extra node has no neighbor on edge: lab=%d' % int(lab_extra))\n"
    '                cap_sig = _build_cap_signature_map()\n'
    '                sig_extra = cap_sig.get(int(lab_extra), frozenset())\n'
    '                sem_extra = _is_semantic_boundary_node(lab_extra, sig_extra)\n'
    '                best = None\n'
    '                rejected_sem = []\n'
    '                for kk in candidates:\n'
    '                    uk, lab_k, xyz_k, prot_k = T[kk]\n'
    '                    if prot_k:\n'
    '                        continue\n'
    '                    sig_k = cap_sig.get(int(lab_k), frozenset())\n'
    '                    sem_k = _is_semantic_boundary_node(lab_k, sig_k)\n'
    '                    if sig_extra != sig_k or sem_extra or sem_k:\n'
    '                        rejected_sem.append((int(lab_k), _fmt_sig(sig_k), bool(sem_k)))\n'
    '                        continue\n'
    '                    dist = abs(uk - ut)\n'
    '                    cand = (dist, int(lab_k), uk)\n'
    '                    if best is None or cand[0] < best[0]:\n'
    '                        best = cand\n'
    '                if best is None:\n'
    '                    if rejected_sem:\n'
    "                        raise RuntimeError('[M8P FATAL] semantic EDGE collapse rejected; patch rebuild should have handled this case. axis=%s Q0=%s edge=%s extra=%d sig_extra=%s sem_extra=%s rejected_neighbors=%s' % (axis, Q0, edge_tgt_tag, int(lab_extra), _fmt_sig(sig_extra), str(bool(sem_extra)), str(rejected_sem)))\n"
    "                    raise RuntimeError('[M8P FATAL] extra node trapped by protected neighbors: lab=%d' % int(lab_extra))\n"
    '                neighbor_lab = best[1]\n'
    '                neighbor_u = best[2]\n'
    '                me = _m8p_find_meshedge_on_intersection(part, lab_extra, neighbor_lab, axis, Q0, Lx, Ly, Lz, tol_plane)\n'
    '                if me is None:\n'
    "                    raise RuntimeError('[M8P FATAL] cannot find MeshEdge to collapse on intersection: %s' % edge_tgt_tag)\n"
    '                labs_before = set((int(n.label) for n in part.nodes))\n'
    '                part.collapseMeshEdge(edge=me, collapseMethod=AVERAGE)\n'
    '                labs_after = set((int(n.label) for n in part.nodes))\n'
    '                deleted = list(labs_before - labs_after)\n'
    "                ctx = 'EDGE axis=%s Q0=%s edge=%s op_it=%d extra=%d nb=%d u_extra=%g u_nb=%g' % (axis, Q0, edge_tgt_tag, op_it, int(lab_extra), int(neighbor_lab), float(ut), float(neighbor_u))\n"
    "                print('[M8P:EDGE] COLLAPSE %s deleted=%s xyz_extra=%s' % (ctx, str(deleted), str(xyz_extra)))\n"
    "                evt = {'kind': 'COLLAPSE', 'nodes_created': [], 'nodes_deleted': deleted, 'edge_endpoints': (int(lab_extra), int(neighbor_lab)), 'context': ctx}\n"
    '                _m8p_s1_update_sets_after_evt(part, evt, Lx, Ly, Lz, tol_plane)\n'
    '                changed_any = True\n'
    '                touched.add(Q0)\n'
    '                continue\n'
    '            if missS:\n'
    '                ii = missS[0]\n'
    '                u_need, labS, xyzS, protS = S[ii]\n'
    '                if len(T) < 2:\n'
    "                    raise RuntimeError('[M8P FATAL] cannot split: target edge has <2 nodes: %s' % edge_tgt_tag)\n"
    '                kk = None\n'
    '                for t in range(len(T) - 1):\n'
    '                    if T[t][0] <= u_need <= T[t + 1][0]:\n'
    '                        kk = t\n'
    '                        break\n'
    '                if kk is None:\n'
    "                    raise RuntimeError('[M8P FATAL] missing point outside target edge range: %s' % edge_tgt_tag)\n"
    '                labA = int(T[kk][1])\n'
    '                labB = int(T[kk + 1][1])\n'
    '                if labA in frozen_labels or labB in frozen_labels:\n'
    "                    raise RuntimeError('[M8P FATAL] need split but endpoints frozen on %s' % edge_tgt_tag)\n"
    '                me = _m8p_find_meshedge_on_intersection(part, labA, labB, axis, Q0, Lx, Ly, Lz, tol_plane)\n'
    '                if me is None:\n'
    "                    raise RuntimeError('[M8P FATAL] cannot find MeshEdge to split on intersection: %s' % edge_tgt_tag)\n"
    '                labs_before = set((int(n.label) for n in part.nodes))\n'
    '                part.splitMeshEdge(edge=me)\n'
    '                labs_after = set((int(n.label) for n in part.nodes))\n'
    '                new_labs = list(labs_after - labs_before)\n'
    '                if len(new_labs) != 1:\n'
    "                    raise RuntimeError('[M8P FATAL] splitMeshEdge did not create exactly one new node: new=%s' % str(new_labs))\n"
    '                lab_new = int(new_labs[0])\n'
    '                nd_new = part.nodes.getFromLabel(label=lab_new)\n'
    '                xyz_new = list(nd_new.coordinates)\n'
    '                xyz_new[freeI] = float(u_need)\n'
    '                xyz_new = _m8p_enforce_A1_B0_on_xyz(axis, Q0, tuple(xyz_new), Lx, Ly, Lz)\n'
    '                _m8p_editNode_xyz_off(part, nd_new, xyz_new)\n'
    "                ctx = 'EDGE axis=%s Q0=%s edge=%s op_it=%d need_master=%d split=%d-%d new=%d u_need=%g' % (axis, Q0, edge_tgt_tag, op_it, int(labS), int(labA), int(labB), int(lab_new), float(u_need))\n"
    "                print('[M8P:EDGE] SPLIT %s new_xyz=%s' % (ctx, str(_m8a_canonicalize_xyz(nd_new.coordinates, Lx, Ly, Lz, tol_plane))))\n"
    "                evt = {'kind': 'SPLIT', 'nodes_created': [lab_new], 'nodes_deleted': [], 'edge_endpoints': (labA, labB), 'context': ctx}\n"
    '                _m8p_s1_update_sets_after_evt(part, evt, Lx, Ly, Lz, tol_plane)\n'
    '                changed_any = True\n'
    '                touched.add(Q0)\n'
    '                continue\n'
    '            moved = []\n'
    '            for jj, u_target in shiftPairs:\n'
    '                ut, labT, xyzT, protT = T[jj]\n'
    '                if protT:\n'
    '                    if abs(ut - u_target) > tol_pair:\n'
    "                        raise RuntimeError('[M8P FATAL] protected node shift exceeds tol_pair on %s' % edge_tgt_tag)\n"
    '                    continue\n'
    '                ndT = part.nodes.getFromLabel(label=int(labT))\n'
    '                xyz_new = list(ndT.coordinates)\n'
    '                xyz_new[freeI] = float(u_target)\n'
    '                xyz_new = _m8p_enforce_A1_B0_on_xyz(axis, Q0, tuple(xyz_new), Lx, Ly, Lz)\n'
    '                _m8p_editNode_xyz_off(part, ndT, xyz_new)\n'
    '                moved.append(int(labT))\n'
    '            if moved:\n'
    "                ctx = 'EDGE axis=%s Q0=%s edge=%s op_it=%d moved=%s' % (axis, Q0, edge_tgt_tag, op_it, str(moved))\n"
    "                print('[M8P:EDGE] EDIT %s' % ctx)\n"
    "                evt = {'kind': 'EDIT', 'nodes_created': [], 'nodes_deleted': [], 'edge_endpoints': None, 'context': ctx}\n"
    '                _m8p_s1_update_sets_after_evt(part, evt, Lx, Ly, Lz, tol_plane)\n'
    '                changed_any = True\n'
    '                touched.add(Q0)\n'
    '            break\n'
    '    return (changed_any, touched)\n'
    '\n'
    'def _m8p_refresh_master_plane_sets(part, P0, Lx, Ly, Lz, tol_plane):\n'
    '    outerPlaneOfElem = _m8p_build_outerPlaneOfElem(part, Lx, Ly, Lz, tol_plane)\n'
    '    outer_elems = [int(eLab) for eLab, pl in outerPlaneOfElem.items() if pl == P0]\n'
    "    _m8a_make_elem_set_from_labels(part, 'M8_OUTER_%s_ELEMS' % P0, outer_elems, allow_empty=False)\n"
    '    raw = set()\n'
    '    for eLab in outer_elems:\n'
    '        el = part.elements.getFromLabel(label=int(eLab))\n'
    '        for nd in el.getNodes():\n'
    '            raw.add(int(nd.label))\n'
    '    edgeLabs = set()\n'
    '    cornerLabs = set()\n'
    '    for lab in raw:\n'
    '        nd = part.nodes.getFromLabel(label=int(lab))\n'
    '        xyzC = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        if not _m8p_on_plane_xyzC(xyzC, P0, Lx, Ly, Lz):\n'
    '            continue\n'
    '        ok = False\n'
    '        for adjE in nd.getElements():\n'
    '            if outerPlaneOfElem.get(int(adjE.label), None) == P0:\n'
    '                ok = True\n'
    '                break\n'
    '        if not ok:\n'
    '            continue\n'
    '        isEdge, isCorner = _m8a_partition_outer_node_type_on_master_plane(P0, xyzC, Lx, Ly, Lz)\n'
    '        if isEdge:\n'
    '            edgeLabs.add(int(lab))\n'
    '        if isCorner:\n'
    '            cornerLabs.add(int(lab))\n'
    "    _m8a_make_node_set_from_labels(part, 'M8_%s_EDGENODES' % P0, sorted(edgeLabs), allow_empty=True)\n"
    "    _m8a_make_node_set_from_labels(part, 'M8_%s_CORNER_NODES' % P0, sorted(cornerLabs), allow_empty=True)\n"
    "    print('[M8P:REFRESH] %s outerElems=%d edgeNodes=%d cornerNodes=%d' % (P0, len(outer_elems), len(edgeLabs), len(cornerLabs)))\n"
    '\n'
    "def _m8p_patch_free_edges_fill_missing_triangles(part, Lx, Ly, Lz, tol_plane, max_new=50, prefix='M8P'):\n"
    '    edge2elems = defaultdict(list)\n'
    '    triKey2elem = {}\n'
    '    for el in part.elements:\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        labs = [int(n.label) for n in nds]\n'
    '        eLab = int(el.label)\n'
    '        triKey = tuple(sorted(labs))\n'
    '        triKey2elem[triKey] = eLab\n'
    '        pairs = ((labs[0], labs[1]), (labs[1], labs[2]), (labs[2], labs[0]))\n'
    '        for a, b in pairs:\n'
    '            if a > b:\n'
    '                a, b = (b, a)\n'
    '            edge2elems[a, b].append(eLab)\n'
    '    free_edges = [(a, b) for (a, b), adj in edge2elems.items() if len(adj) == 1]\n'
    '    free_set = set(free_edges)\n'
    '    nbr = defaultdict(set)\n'
    '    for a, b in free_edges:\n'
    '        nbr[a].add(b)\n'
    '        nbr[b].add(a)\n'
    '    tri_nodes = set()\n'
    '    for a in sorted(nbr.keys()):\n'
    '        for b in sorted(nbr[a]):\n'
    '            if b <= a:\n'
    '                continue\n'
    '            common = nbr[a].intersection(nbr[b])\n'
    '            for c in sorted(common):\n'
    '                if c <= b:\n'
    '                    continue\n'
    '                eab = (a, b) if a < b else (b, a)\n'
    '                ebc = (b, c) if b < c else (c, b)\n'
    '                eca = (a, c) if a < c else (c, a)\n'
    '                if eab in free_set and ebc in free_set and (eca in free_set):\n'
    '                    tri_nodes.add((a, b, c))\n'
    '    if not tri_nodes:\n'
    "        print('[%s:PATCH] no free-edge triangles found.' % prefix)\n"
    '        return 0\n'
    '\n'
    '    def elem_on_outer_plane(a, b, c):\n'
    '        xyzA = _m8a_canonicalize_xyz(part.nodes.getFromLabel(label=int(a)).coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        xyzB = _m8a_canonicalize_xyz(part.nodes.getFromLabel(label=int(b)).coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        xyzC = _m8a_canonicalize_xyz(part.nodes.getFromLabel(label=int(c)).coordinates, Lx, Ly, Lz, tol_plane)\n'
    '        hit = []\n'
    '        if xyzA[0] == 0.0 and xyzB[0] == 0.0 and (xyzC[0] == 0.0):\n'
    "            hit.append('X0')\n"
    '        if xyzA[0] == float(Lx) and xyzB[0] == float(Lx) and (xyzC[0] == float(Lx)):\n'
    "            hit.append('X1')\n"
    '        if xyzA[1] == 0.0 and xyzB[1] == 0.0 and (xyzC[1] == 0.0):\n'
    "            hit.append('Y0')\n"
    '        if xyzA[1] == float(Ly) and xyzB[1] == float(Ly) and (xyzC[1] == float(Ly)):\n'
    "            hit.append('Y1')\n"
    '        if xyzA[2] == 0.0 and xyzB[2] == 0.0 and (xyzC[2] == 0.0):\n'
    "            hit.append('Z0')\n"
    '        if xyzA[2] == float(Lz) and xyzB[2] == float(Lz) and (xyzC[2] == float(Lz)):\n'
    "            hit.append('Z1')\n"
    '        if len(hit) == 0:\n'
    '            return None\n'
    '        if len(hit) > 1:\n'
    "            return 'MULTI'\n"
    '        return hit[0]\n'
    '    created = 0\n'
    '    created_elems = []\n'
    '    skipped_outer = []\n'
    '    for a, b, c in sorted(list(tri_nodes)):\n'
    '        if created >= int(max_new):\n'
    '            break\n'
    '        key = tuple(sorted((a, b, c)))\n'
    '        if key in triKey2elem:\n'
    '            continue\n'
    '        outer_hit = elem_on_outer_plane(a, b, c)\n'
    '        if outer_hit is not None:\n'
    '            skipped_outer.append((a, b, c, outer_hit))\n'
    '            continue\n'
    '        nA = part.nodes.getFromLabel(label=int(a))\n'
    '        nB = part.nodes.getFromLabel(label=int(b))\n'
    '        nC = part.nodes.getFromLabel(label=int(c))\n'
    '        el_new = part.Element(nodes=(nA, nB, nC), elemShape=TRI3)\n'
    '        created += 1\n'
    '        created_elems.append(int(el_new.label))\n'
    '    if created_elems:\n'
    "        _m8a_make_elem_set_from_labels(part, '%s_PATCHED_ELEMS' % prefix, created_elems, allow_empty=False)\n"
    "    print('[%s:PATCH] createdElems=%d skippedOuter=%d (max_new=%d)' % (prefix, created, len(skipped_outer), int(max_new)))\n"
    '    if skipped_outer:\n'
    "        print('[%s:PATCH] skipped outer triangles (need periodic-consistent patch): examples=%s' % (prefix, str(skipped_outer[:5])))\n"
    '    return created\n'
    '\n'
    "def m8p_make_periodic_shell(model_name, particles_work_json, src_part_name='P_ALL_TRI_M8A', dst_part_name='P_ALL_TRI_PER', max_fix_rounds=20):\n"
    '    if model_name not in mdb.models:\n'
    "        raise RuntimeError('Model not found: %s' % model_name)\n"
    '    model = mdb.models[model_name]\n'
    "    with open(particles_work_json, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    Lx, Ly, Lz = (float(x) for x in data['box'])\n"
    "    tol_plane = float(data['tol_plane'])\n"
    "    tol_pair = float(data.get('tol_pair', tol_plane))\n"
    '    eps_key = max(1e-18, 0.2 * tol_pair)\n'
    "    edge_target_len = float(data['edge_target_len'])\n"
    '    tol_snap = 0.5 * edge_target_len\n'
    '    tol_snap = max(tol_snap, 5.0 * tol_pair)\n'
    '    tol_snap = min(tol_snap, 0.9 * edge_target_len)\n'
    "    print('[M8P] edge_target_len=%g tol_snap=%g' % (edge_target_len, tol_snap))\n"
    '    if src_part_name not in model.parts:\n'
    "        raise RuntimeError('Missing part: %s' % src_part_name)\n"
    '    if dst_part_name in model.parts:\n'
    '        del model.parts[dst_part_name]\n'
    '    p_src = model.parts[src_part_name]\n'
    '    p = model.Part(name=dst_part_name, objectToCopy=p_src)\n'
    "    print('\\n=== Module 8P START: build periodic shell %s from %s ===' % (dst_part_name, src_part_name))\n"
    "    print('[M8P] box=(%g,%g,%g) tol_plane=%g tol_pair=%g eps_key=%g' % (Lx, Ly, Lz, tol_plane, tol_pair, eps_key))\n"
    "    print('[M8P] initial nodes=%d elems=%d' % (len(p.nodes), len(p.elements)))\n"
    '    _m8p_promote_inc_cap_sets_to_nodeelem(p)\n'
    '    _m8p_check_cap_boundary_semantics(p, Lx, Ly, Lz, tol_plane, debug_n=20)\n'
    '    _m8p_clear_image_planes(p, Lx, Ly, Lz, tol_plane)\n'
    '    done_images = set()\n'
    "    for axis in ('X', 'Y', 'Z'):\n"
    '        P0, P1 = _m8p_plane_of_axis(axis)\n'
    '        shift = _m8p_shift(axis, Lx, Ly, Lz)\n'
    "        print('\\n[M8P] STEP %s: %s -> %s (done_images=%s)' % (axis, P0, P1, str(sorted(list(done_images)))))\n"
    '        pair_map = None\n'
    '        req = None\n'
    '        for fix_round in range(int(max_fix_rounds)):\n'
    '            lab2node, lab2xyzC, key2lab = _m8p_build_node_maps(p, Lx, Ly, Lz, tol_plane, eps_key)\n'
    '            req = _m8p_build_REQ_strict(p, axis, done_images, lab2xyzC, Lx, Ly, Lz)\n'
    '            pair_map, unmatched = _m8p_pair_REQ_gate(p, axis, req, P1, lab2node, lab2xyzC, Lx, Ly, Lz, tol_plane, tol_pair, eps_key, shift)\n'
    '            if not unmatched:\n'
    '                break\n'
    '            hits = _m8p_unmatched_hits_frozen_edge(p, axis, done_images, unmatched, lab2xyzC, Lx, Ly, Lz, tol_plane, tol_pair, debug_n=10)\n'
    '            if hits:\n'
    "                _m8a_make_node_set_from_labels(p, 'M8P_UNMATCHED_%s0' % axis, unmatched, allow_empty=False)\n"
    "                msg = ['[M8P FATAL] UNMATCHED_REQ contains nodes whose image targets lie on FrozenEdge(%s, done_images=%s).' % (P1, str(sorted(list(done_images)))), 'By design, FrozenEdge nodes are not editable. This indicates either missing nodes on %s or pairing conflict.' % P1, 'Samples (mLab, xyzM_C, xyzTarget_C, nearest_on_%s=(d2, lab, xyzC)): %s' % (P1, str(hits))]\n"
    "                raise RuntimeError('\\n'.join(msg))\n"
    "            print('[M8P:GATE] axis=%s UNMATCHED_REQ=%d round=%d/%d' % (axis, len(unmatched), fix_round, int(max_fix_rounds)))\n"
    '            if _m8p_unmatched_is_edge_or_corner(p, axis, unmatched):\n'
    '                changed, touched = _m8p_snap_intersection_edges_pre(p, axis, done_images, Lx, Ly, Lz, tol_plane, tol_pair, tol_snap, max_ops_per_edge=50)\n'
    '                if not changed:\n'
    "                    _m8a_make_node_set_from_labels(p, 'M8P_UNMATCHED_%s0' % axis, unmatched, allow_empty=False)\n"
    "                    raise RuntimeError('[M8P FATAL] EDGE-type unmatched but _m8p_snap_intersection_edges_pre made no change. axis=%s' % axis)\n"
    '                for Q0 in sorted(list(touched)):\n'
    '                    _m8p_refresh_master_plane_sets(p, Q0, Lx, Ly, Lz, tol_plane)\n'
    '                if done_images:\n'
    "                    _m8p_quick_missing_check(p, set(done_images), Lx, Ly, Lz, tol_plane, tol_pair, stage_tag='after EDGE-fix axis=%s round=%d' % (axis, fix_round))\n"
    '                continue\n'
    "            changed = _m8p_fix_iface_curve_by_inclusions_once(p, axis, done_images, unmatched_req_master=unmatched, Lx=Lx, Ly=Ly, Lz=Lz, tol_plane=tol_plane, tol_pair=tol_pair, tol_snap=tol_snap, eps_key=eps_key, debug_prefix='M8P_IFACE')\n"
    '            if not changed:\n'
    "                _m8a_make_node_set_from_labels(p, 'M8P_UNMATCHED_%s0' % axis, unmatched, allow_empty=False)\n"
    "                raise RuntimeError('[M8P FATAL] IFACE-curve unmatched but no mesh edit performed. axis=%s' % axis)\n"
    '            if done_images:\n'
    "                _m8p_quick_missing_check(p, set(done_images), Lx, Ly, Lz, tol_plane, tol_pair, stage_tag='after IFACE-fix axis=%s round=%d' % (axis, fix_round))\n"
    '        else:\n'
    "            raise RuntimeError('[M8P FATAL] axis=%s gate did not pass within max_fix_rounds=%d' % (axis, int(max_fix_rounds)))\n"
    "        _m8a_make_node_set_from_labels(p, 'M8P_REQ_%s' % P0, list(req), allow_empty=False)\n"
    '        lab2node, lab2xyzC, key2lab = _m8p_build_node_maps(p, Lx, Ly, Lz, tol_plane, eps_key)\n'
    '        _m8p_copy_outer_plane(p, axis, P0, P1, pair_map, lab2node, lab2xyzC, key2lab, Lx, Ly, Lz, tol_plane, eps_key)\n'
    '        n0 = len(_m8p_outer_elem_labels_by_plane(p, P0, Lx, Ly, Lz, tol_plane))\n'
    '        n1 = len(_m8p_outer_elem_labels_by_plane(p, P1, Lx, Ly, Lz, tol_plane))\n'
    "        print('[M8P:CHK] outerElems %s=%d %s=%d' % (P0, n0, P1, n1))\n"
    '        if n0 != n1:\n'
    "            raise RuntimeError('[M8P FATAL] outer elem count mismatch after copy %s: %s=%d %s=%d' % (axis, P0, n0, P1, n1))\n"
    '        done_after = set(done_images)\n'
    '        done_after.add(axis)\n'
    "        _m8p_quick_missing_check(p, done_after, Lx, Ly, Lz, tol_plane, tol_pair, stage_tag='after copy %s' % axis)\n"
    '        done_images = done_after\n'
    '    free_e, nonm = _m8p_free_edge_and_nonmanifold(p)\n'
    "    print('\\n[M8P] final closure: freeEdges=%d nonManifoldEdges=%d' % (free_e, nonm))\n"
    '    if free_e != 0 or nonm != 0:\n'
    "        _m8p_make_topology_debug_sets(p, Lx, Ly, Lz, tol_plane, prefix='M8P', max_print=20)\n"
    '    if free_e != 0:\n'
    "        nnew = _m8p_patch_free_edges_fill_missing_triangles(p, Lx, Ly, Lz, tol_plane, max_new=50, prefix='M8P')\n"
    '        if nnew > 0:\n'
    '            free_e2, nonm2 = _m8p_free_edge_and_nonmanifold(p)\n'
    "            print('[M8P] after patch: freeEdges=%d nonManifoldEdges=%d (createdElems=%d)' % (free_e2, nonm2, nnew))\n"
    "            _m8p_make_topology_debug_sets(p, Lx, Ly, Lz, tol_plane, prefix='M8P_POSTPATCH', max_print=20)\n"
    '            free_e, nonm = (free_e2, nonm2)\n'
    '    if free_e != 0:\n'
    "        print('[REGEN_JSON_REQUIRED] reason=MAIN_PLANE_ORPHAN_FREE_EDGE freeEdges=%d nonManifoldEdges=%d srcPart=%s dstPart=%s' % (free_e, nonm, src_part_name, dst_part_name))\n"
    "        raise RuntimeError('[M8P FATAL] shell not closed: freeEdges=%d (nonManifoldEdges=%d)' % (free_e, nonm))\n"
    '    if nonm != 0:\n'
    "        print('[M8P WARNING] nonManifoldEdges=%d (allowed by current DoD; investigate if unexpected)' % nonm)\n"
    "    mx, nX0, nX1 = _m8p_periodic_missing(p, 'X', Lx, Ly, Lz, tol_plane, tol_pair)\n"
    "    my, nY0, nY1 = _m8p_periodic_missing(p, 'Y', Lx, Ly, Lz, tol_plane, tol_pair)\n"
    "    mz, nZ0, nZ1 = _m8p_periodic_missing(p, 'Z', Lx, Ly, Lz, tol_plane, tol_pair)\n"
    "    print('[M8P] periodic missing: X=%d (nX0=%d nX1=%d) Y=%d (nY0=%d nY1=%d) Z=%d (nZ0=%d nZ1=%d)' % (mx, nX0, nX1, my, nY0, nY1, mz, nZ0, nZ1))\n"
    '    if mx != 0 or my != 0 or mz != 0:\n'
    "        raise RuntimeError('[M8P FATAL] periodic node pairing not strict: missing_x=%d missing_y=%d missing_z=%d' % (mx, my, mz))\n"
    '    m8p_rebuild_sets_P_ALL_TRI_PER(p, Lx, Ly, Lz, tol_plane, tol_pair)\n'
    "    print('=== Module 8P OK: created %s nodes=%d elems=%d ===\\n' % (dst_part_name, len(p.nodes), len(p.elements)))\n"
    '    return p\n'
    'import re\n'
    'from collections import defaultdict\n'
    '\n'
    'def _m8p_union_nodes_of_elem_labels(part, elem_labels):\n'
    '    s = set()\n'
    '    for eLab in elem_labels:\n'
    '        el = part.elements.getFromLabel(label=int(eLab))\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        for n in nds:\n'
    '            s.add(int(n.label))\n'
    '    return s\n'
    '\n'
    'def _m8p_build_outer_tri_lookup_by_node_triple(part, outer_elem_labels):\n'
    '    mp = {}\n'
    '    for eLab in outer_elem_labels:\n'
    '        el = part.elements.getFromLabel(label=int(eLab))\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        key = tuple(sorted((int(n.label) for n in nds)))\n'
    '        if key in mp and mp[key] != int(eLab):\n'
    "            raise RuntimeError('[M8P FATAL] duplicate outer tri node-triple maps to multiple elems: %s' % str(key))\n"
    '        mp[key] = int(eLab)\n'
    '    return mp\n'
    '\n'
    'def _m8p_get_master_cap_elems(part, P0, outer_elem_labels_P0):\n'
    '    outerP0 = set((int(x) for x in outer_elem_labels_P0))\n'
    "    s_m8 = 'M8_%s_CAP_ELEMS' % P0\n"
    '    if s_m8 in part.sets:\n'
    '        labs = set((int(e.label) for e in part.sets[s_m8].elements))\n'
    '        labs &= outerP0\n'
    "        print('[M8P:CAPMASTER] %s from %s: %d' % (P0, s_m8, len(labs)))\n"
    '        return labs\n'
    "    pat = re.compile('^INC_\\\\d{4}_CAP_%s$' % P0)\n"
    '    labs = set()\n'
    '    for sname, s in part.sets.items():\n'
    '        if not pat.match(sname):\n'
    '            continue\n'
    '        try:\n'
    '            for e in s.elements:\n'
    '                labs.add(int(e.label))\n'
    '        except Exception:\n'
    '            pass\n'
    '    labs &= outerP0\n'
    "    print('[M8P:CAPMASTER] %s from INC_* fallback: %d' % (P0, len(labs)))\n"
    '    return labs\n'
    '\n'
    'def _m8p_build_cap_image_from_master(part, axis, cap_master_elems, outer_master_elems, outer_image_elems, lab2node, lab2xyzC, Lx, Ly, Lz, tol_plane, tol_pair):\n'
    '    if not cap_master_elems:\n'
    '        return set()\n'
    "    P0 = axis + '0'\n"
    "    P1 = axis + '1'\n"
    '    shift = _m8p_shift(axis, Lx, Ly, Lz)\n'
    '    req_nodes = _m8p_union_nodes_of_elem_labels(part, cap_master_elems)\n'
    '    eps_sig = float(tol_pair)\n'
    '    pair_map = _m8p_pair_REQ(part, axis, req_nodes, P1, lab2node, lab2xyzC, Lx, Ly, Lz, tol_plane, tol_pair, eps_sig, shift)\n'
    '    tri2e_img = _m8p_build_outer_tri_lookup_by_node_triple(part, outer_image_elems)\n'
    '    cap_img = set()\n'
    '    miss = []\n'
    '    for eLab_m in cap_master_elems:\n'
    '        el = part.elements.getFromLabel(label=int(eLab_m))\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 3:\n'
    '            continue\n'
    '        img_nodes = []\n'
    '        for n in nds:\n'
    '            mLab = int(n.label)\n'
    '            iLab = pair_map.get(mLab, None)\n'
    '            if iLab is None:\n'
    '                miss.append((int(eLab_m), mLab))\n'
    '                break\n'
    '            img_nodes.append(int(iLab))\n'
    '        if len(img_nodes) != 3:\n'
    '            continue\n'
    '        key = tuple(sorted(img_nodes))\n'
    '        eLab_i = tri2e_img.get(key, None)\n'
    '        if eLab_i is None:\n'
    '            miss.append((int(eLab_m), key))\n'
    '            continue\n'
    '        cap_img.add(int(eLab_i))\n'
    '    if miss:\n'
    "        raise RuntimeError('[M8P FATAL] CAP master->image elem lookup failed on %s: miss=%d examples=%s' % (axis, len(miss), str(miss[:10])))\n"
    "    print('[M8P:CAPIMG] %s %s->%s capMaster=%d capImage=%d' % (axis, P0, P1, len(cap_master_elems), len(cap_img)))\n"
    '    return cap_img\n'
    '\n'
    'def m8p_rebuild_sets_P_ALL_TRI_PER(part, Lx, Ly, Lz, tol_plane, tol_pair):\n'
    "    print('\\n=== [M8P] rebuild final sets on P_ALL_TRI_PER (v2) ===')\n"
    '    outer_by_plane = {}\n'
    '    outer_all = set()\n'
    "    for plane in ('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1'):\n"
    '        labs = set(_m8p_outer_elem_labels_by_plane(part, plane, Lx, Ly, Lz, tol_plane))\n'
    '        outer_by_plane[plane] = labs\n'
    '        outer_all |= labs\n'
    "        _m8a_make_elem_set_from_labels(part, 'OUTER_%s' % plane, labs, allow_empty=False)\n"
    "        print('[M8P:SET] OUTER_%s elems=%d' % (plane, len(labs)))\n"
    "    _m8a_make_elem_set_from_labels(part, 'OUTER_ALL', outer_all, allow_empty=False)\n"
    "    print('[M8P:SET] OUTER_ALL elems=%d' % len(outer_all))\n"
    '    cap_master = {}\n'
    "    for P0 in ('X0', 'Y0', 'Z0'):\n"
    '        cap_master[P0] = _m8p_get_master_cap_elems(part, P0, outer_by_plane[P0])\n'
    '    lab2node = {}\n'
    '    lab2xyzC = {}\n'
    '    for nd in part.nodes:\n'
    '        lab = int(nd.label)\n'
    '        lab2node[lab] = nd\n'
    '        lab2xyzC[lab] = _m8a_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '    cap_all = set()\n'
    "    for axis in ('X', 'Y', 'Z'):\n"
    "        P0 = axis + '0'\n"
    "        P1 = axis + '1'\n"
    '        cap_all |= cap_master[P0]\n'
    '        cap_img = _m8p_build_cap_image_from_master(part, axis, cap_master[P0], outer_by_plane[P0], outer_by_plane[P1], lab2node, lab2xyzC, Lx, Ly, Lz, tol_plane, tol_pair)\n'
    '        cap_all |= cap_img\n'
    "    _m8a_make_elem_set_from_labels(part, 'CAP_ALL', cap_all, allow_empty=True)\n"
    "    print('[M8P:SET] CAP_ALL elems=%d' % len(cap_all))\n"
    '    all_e = set((int(e.label) for e in part.elements))\n'
    '    iface = all_e - outer_all\n'
    "    _m8a_make_elem_set_from_labels(part, 'IFACE_ALL', iface, allow_empty=True)\n"
    "    print('[M8P:SET] IFACE_ALL elems=%d' % len(iface))\n"
    '    if not cap_all <= outer_all:\n'
    "        raise RuntimeError('[M8P FATAL] CAP_ALL not subset of OUTER_ALL: diff=%d' % len(cap_all - outer_all))\n"
    '    if iface & outer_all:\n'
    "        raise RuntimeError('[M8P FATAL] IFACE_ALL intersects OUTER_ALL: inter=%d' % len(iface & outer_all))\n"
    "    print('=== [M8P] rebuild final sets OK ===\\n')\n"
    '\n'
    'def _m8_get_elem_labels_from_set(part, set_name):\n'
    '    if set_name not in part.sets:\n'
    "        raise RuntimeError('[M8B FATAL] missing elem set: %s' % set_name)\n"
    '    return set((int(e.label) for e in part.sets[set_name].elements))\n'
    '\n'
    'def _m8_all_elem_labels(part):\n'
    '    return set((int(e.label) for e in part.elements))\n'
    '\n'
    'def _m8_check_closed_and_manifold(part, tag):\n'
    '    free_e, nonm = _m8p_free_edge_and_nonmanifold(part)\n'
    "    print('[M8CHK] %s: elems=%d nodes=%d freeEdges=%d nonManifold=%d' % (tag, len(part.elements), len(part.nodes), free_e, nonm))\n"
    '    if free_e != 0:\n'
    "        raise RuntimeError('[M8B FATAL] %s not closed: freeEdges=%d' % (tag, free_e))\n"
    '    if nonm != 0:\n'
    "        raise RuntimeError('[M8B FATAL] %s non-manifold: nonManifoldEdges=%d' % (tag, nonm))\n"
    '\n'
    "def m8b_split_shells(model_name, src_part_name='P_ALL_TRI_PER', inc_part_name='P_INC_SHELL', mat_part_name='P_MAT_SHELL', keep_debug_sets=True):\n"
    '    if model_name not in mdb.models:\n'
    "        raise RuntimeError('[M8B FATAL] model not found: %s' % model_name)\n"
    '    model = mdb.models[model_name]\n'
    '    if src_part_name not in model.parts:\n'
    "        raise RuntimeError('[M8B FATAL] missing part: %s' % src_part_name)\n"
    '    p_src = model.parts[src_part_name]\n'
    "    outer_all = _m8_get_elem_labels_from_set(p_src, 'OUTER_ALL')\n"
    "    cap_all = _m8_get_elem_labels_from_set(p_src, 'CAP_ALL')\n"
    "    iface_all = _m8_get_elem_labels_from_set(p_src, 'IFACE_ALL')\n"
    '    if not cap_all.issubset(outer_all):\n'
    "        raise RuntimeError('[M8B FATAL] CAP_ALL not subset of OUTER_ALL: diff=%d' % len(cap_all - outer_all))\n"
    '    if iface_all & outer_all:\n'
    "        raise RuntimeError('[M8B FATAL] IFACE_ALL intersects OUTER_ALL: inter=%d' % len(iface_all & outer_all))\n"
    '    keep_inc = set(iface_all) | set(cap_all)\n'
    '    keep_mat = set(outer_all) - set(cap_all) | set(iface_all)\n'
    '    if not keep_inc:\n'
    "        raise RuntimeError('[M8B FATAL] E_INC empty (IFACE_ALL∪CAP_ALL).')\n"
    '    if not keep_mat:\n'
    "        raise RuntimeError('[M8B FATAL] E_MAT empty ((OUTER_ALL-CAP_ALL)∪IFACE_ALL).')\n"
    "    print('[M8B] keep_inc=%d keep_mat=%d outer=%d cap=%d iface=%d' % (len(keep_inc), len(keep_mat), len(outer_all), len(cap_all), len(iface_all)))\n"
    '    _m8a_delete_part_if_exists(model, inc_part_name)\n'
    '    _m8a_delete_part_if_exists(model, mat_part_name)\n'
    '    p_inc = model.Part(name=inc_part_name, objectToCopy=p_src)\n'
    '    p_mat = model.Part(name=mat_part_name, objectToCopy=p_src)\n'
    '    all_inc = _m8_all_elem_labels(p_inc)\n'
    '    del_inc = sorted(all_inc - keep_inc)\n'
    '    ndel_e_inc = _m8a_delete_elements_by_labels(p_inc, del_inc)\n'
    '    ndel_n_inc = _m8a_delete_unused_nodes(p_inc)\n'
    "    print('[M8B] %s: delElems=%d delUnusedNodes=%d' % (inc_part_name, ndel_e_inc, ndel_n_inc))\n"
    '    all_mat = _m8_all_elem_labels(p_mat)\n'
    '    del_mat = sorted(all_mat - keep_mat)\n'
    '    ndel_e_mat = _m8a_delete_elements_by_labels(p_mat, del_mat)\n'
    '    ndel_n_mat = _m8a_delete_unused_nodes(p_mat)\n'
    "    print('[M8B] %s: delElems=%d delUnusedNodes=%d' % (mat_part_name, ndel_e_mat, ndel_n_mat))\n"
    '    if keep_debug_sets:\n'
    "        _m8a_make_elem_set_from_labels(p_inc, 'E_KEEP_INC', sorted(keep_inc), allow_empty=False)\n"
    "        _m8a_make_elem_set_from_labels(p_mat, 'E_KEEP_MAT', sorted(keep_mat), allow_empty=False)\n"
    "    _m8a_make_elem_set_from_labels(p_inc, 'IFACE_ALL', sorted(set(iface_all) & set(_m8_all_elem_labels(p_inc))), allow_empty=True)\n"
    "    _m8a_make_elem_set_from_labels(p_mat, 'IFACE_ALL', sorted(set(iface_all) & set(_m8_all_elem_labels(p_mat))), allow_empty=True)\n"
    '    if len(p_inc.elements) <= 0 or len(p_mat.elements) <= 0:\n'
    "        raise RuntimeError('[M8B FATAL] empty output part after deletion: incElems=%d matElems=%d' % (len(p_inc.elements), len(p_mat.elements)))\n"
    '    _m8_check_closed_and_manifold(p_inc, inc_part_name)\n'
    '    _m8_check_closed_and_manifold(p_mat, mat_part_name)\n'
    "    print('=== Module 8B OK: created %s and %s ===' % (inc_part_name, mat_part_name))\n"
    '    return (p_inc, p_mat)\n'
    'import regionToolset\n'
    'from abaqusConstants import TRI3\n'
    '\n'
    "def m8c_flip_matrix_iface(model_name, mat_part_name='P_MAT_SHELL', iface_set_name='IFACE_ALL', keep_debug_sets=True):\n"
    '    if model_name not in mdb.models:\n'
    "        raise RuntimeError('[M8C FATAL] model not found: %s' % model_name)\n"
    '    model = mdb.models[model_name]\n'
    '    if mat_part_name not in model.parts:\n'
    "        raise RuntimeError('[M8C FATAL] missing part: %s' % mat_part_name)\n"
    '    p = model.parts[mat_part_name]\n'
    '    if iface_set_name not in p.sets:\n'
    "        raise RuntimeError('[M8C FATAL] missing elem set on %s: %s' % (mat_part_name, iface_set_name))\n"
    '    iface_elems = p.sets[iface_set_name].elements\n'
    '    if len(iface_elems) == 0:\n'
    "        raise RuntimeError('[M8C FATAL] %s is empty on %s, cannot flip.' % (iface_set_name, mat_part_name))\n"
    '    if keep_debug_sets:\n'
    '        labs = [int(e.label) for e in iface_elems]\n'
    "        _m8a_make_elem_set_from_labels(p, 'M8C_IFACE_TO_FLIP', labs, allow_empty=False)\n"
    '    region = regionToolset.Region(elements=iface_elems)\n'
    "    print('[M8C] flipNormal: part=%s set=%s elems=%d' % (mat_part_name, iface_set_name, len(iface_elems)))\n"
    '    p.flipNormal(regions=region)\n'
    "    _m8_check_closed_and_manifold(p, mat_part_name + ' (after flipNormal)')\n"
    "    print('=== Module 8C OK: flipped normals on %s elems=%d ===' % (iface_set_name, len(iface_elems)))\n"
    '    return p\n'
    'import regionToolset\n'
    'import mesh\n'
    'from abaqusConstants import TET, C3D4, STANDARD, JOHNSON_COOK\n'
    '\n'
    "def _m9_ensure_materials_and_sections(model, mat_matrix='Al_matrix', mat_inc='SiC', sec_matrix='SecMatrix', sec_inc='SecSiC'):\n"
    '    if mat_matrix not in model.materials:\n'
    '        model.Material(name=mat_matrix)\n'
    '        model.materials[mat_matrix].Elastic(table=((69000.0, 0.33),))\n'
    '        model.materials[mat_matrix].Plastic(hardening=JOHNSON_COOK, scaleStress=None, table=((503.0, 193.0, 0.45, 0.0, 0.0, 0.0),))\n'
    '    if mat_inc not in model.materials:\n'
    '        model.Material(name=mat_inc)\n'
    '        model.materials[mat_inc].Elastic(table=((450000.0, 0.17),))\n'
    '    if sec_matrix not in model.sections:\n'
    '        model.HomogeneousSolidSection(name=sec_matrix, material=mat_matrix, thickness=None)\n'
    '    if sec_inc not in model.sections:\n'
    '        model.HomogeneousSolidSection(name=sec_inc, material=mat_inc, thickness=None)\n'
    '\n'
    'def _m9_ensure_section_exists(model, section_name):\n'
    '    if section_name not in model.sections:\n'
    "        raise RuntimeError('[M9 FATAL] missing section: %s (create it before Module 9 or pass correct name)' % section_name)\n"
    '\n'
    'def _m9_set_all_elements_C3D4(part):\n'
    '    if len(part.elements) == 0:\n'
    "        raise RuntimeError('[M9 FATAL] no elements in part=%s' % part.name)\n"
    '    elemType = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)\n'
    '    region = regionToolset.Region(elements=part.elements)\n'
    '    part.setElementType(regions=region, elemTypes=(elemType,))\n'
    '\n'
    'def _m9_check_only_C3D4(part, tag):\n'
    '    type_counts = {}\n'
    '    for e in part.elements:\n'
    "        et = getattr(e, 'type', None)\n"
    '        ets = str(et)\n'
    '        type_counts[ets] = type_counts.get(ets, 0) + 1\n'
    "    if not (len(type_counts) == 1 and 'C3D4' in type_counts):\n"
    "        raise RuntimeError('[M9 FATAL] %s element types not pure C3D4: %s' % (tag, str(type_counts)))\n"
    "    print('[M9] %s elementTypes=%s' % (tag, str(type_counts)))\n"
    '\n'
    'def _m9_make_all_elem_set(part, set_name):\n'
    '    labs = [int(e.label) for e in part.elements]\n'
    '    _m8a_make_elem_set_from_labels(part, set_name, labs, allow_empty=False)\n'
    '    return len(labs)\n'
    '\n'
    'def _m9_assign_section_by_set(part, set_name, section_name):\n'
    '    if set_name not in part.sets:\n'
    "        raise RuntimeError('[M9 FATAL] missing elem set on %s: %s' % (part.name, set_name))\n"
    '    region = regionToolset.Region(elements=part.sets[set_name].elements)\n'
    '    part.SectionAssignment(region=region, sectionName=section_name)\n'
    '\n'
    'def _m9_shell_to_tet_part(model, src_shell_part_name, dst_tet_part_name, elem_set_name, section_name):\n'
    '    if src_shell_part_name not in model.parts:\n'
    "        raise RuntimeError('[M9 FATAL] missing shell part: %s' % src_shell_part_name)\n"
    '    _m8a_delete_part_if_exists(model, dst_tet_part_name)\n'
    '    p_shell = model.parts[src_shell_part_name]\n'
    '    p_tet = model.Part(name=dst_tet_part_name, objectToCopy=p_shell)\n'
    "    print('\\n=== [M9] START shell->tet: %s -> %s ===' % (src_shell_part_name, dst_tet_part_name))\n"
    "    print('[M9] before generateMesh: nodes=%d elems=%d' % (len(p_tet.nodes), len(p_tet.elements)))\n"
    '    p_tet.generateMesh(elemShape=TET)\n'
    "    print('[M9] after generateMesh: nodes=%d elems=%d' % (len(p_tet.nodes), len(p_tet.elements)))\n"
    '    if len(p_tet.elements) == 0:\n'
    "        raise RuntimeError('[M9 FATAL] generateMesh produced zero elements on %s' % dst_tet_part_name)\n"
    '    _m9_set_all_elements_C3D4(p_tet)\n'
    '    _m9_check_only_C3D4(p_tet, dst_tet_part_name)\n'
    '    nset = _m9_make_all_elem_set(p_tet, elem_set_name)\n'
    "    print('[M9] created %s on %s: elems=%d' % (elem_set_name, dst_tet_part_name, nset))\n"
    '    _m9_assign_section_by_set(p_tet, elem_set_name, section_name)\n'
    "    print('[M9] section assigned: %s -> %s.%s' % (section_name, dst_tet_part_name, elem_set_name))\n"
    '    return p_tet\n'
    '\n'
    "def m9_make_tet_and_assign(model_name, inc_shell_part='P_INC_SHELL', mat_shell_part='P_MAT_SHELL', inc_tet_part='P_INC_TET', mat_tet_part='P_MAT_TET', eset_inc='ESET_INC', eset_mat='ESET_MAT', sec_inc='SEC_SIC', sec_mat='SEC_MATRIX'):\n"
    '    if model_name not in mdb.models:\n'
    "        raise RuntimeError('[M9 FATAL] model not found: %s' % model_name)\n"
    '    model = mdb.models[model_name]\n'
    "    _m9_ensure_materials_and_sections(model, mat_matrix='Al_matrix', mat_inc='SiC', sec_matrix=sec_mat, sec_inc=sec_inc)\n"
    '    _m8_check_closed_and_manifold(model.parts[inc_shell_part], inc_shell_part)\n'
    '    _m8_check_closed_and_manifold(model.parts[mat_shell_part], mat_shell_part)\n'
    '    _m9_ensure_section_exists(model, sec_inc)\n'
    '    _m9_ensure_section_exists(model, sec_mat)\n'
    '    p_inc_tet = _m9_shell_to_tet_part(model, src_shell_part_name=inc_shell_part, dst_tet_part_name=inc_tet_part, elem_set_name=eset_inc, section_name=sec_inc)\n'
    '    p_mat_tet = _m9_shell_to_tet_part(model, src_shell_part_name=mat_shell_part, dst_tet_part_name=mat_tet_part, elem_set_name=eset_mat, section_name=sec_mat)\n'
    "    print('\\n=== Module 9 OK: created %s and %s ===' % (inc_tet_part, mat_tet_part))\n"
    '    return (p_inc_tet, p_mat_tet)\n'
    'import json, math\n'
    'import regionToolset\n'
    'from collections import defaultdict\n'
    'from abaqus import mdb\n'
    'from abaqusConstants import ON, OFF, COMPUTED\n'
    "_M10_TPL_PBC_E = {'E1': {1: [('RP4', +1)], 2: [], 3: []}, 'E2': {1: [('RP4', -1)], 2: [], 3: []}, 'E3': {1: [], 2: [('RP5', -1)], 3: []}, 'E4': {1: [], 2: [], 3: [('RP6', -1)]}, 'E5': {1: [('RP4', -1)], 2: [], 3: []}, 'E6': {1: [], 2: [], 3: [('RP6', -1)]}, 'E7': {1: [], 2: [('RP5', -1)], 3: []}, 'E8': {1: [('RP4', +1)], 2: [], 3: []}, 'E9': {1: [], 2: [('RP5', +1)], 3: []}, 'F-x': {1: [('RP4', -1)], 2: [], 3: []}, 'F-y': {1: [], 2: [('RP5', -1)], 3: []}, 'F-z': {1: [], 2: [], 3: [('RP6', -1)]}, 'C62': {1: [], 2: [('RP5', +1)], 3: []}, 'C23': {1: [], 2: [], 3: [('RP6', -1)]}, 'C34': {1: [('RP4', +1)], 2: [], 3: []}, 'C48': {1: [], 2: [('RP5', -1)], 3: []}, 'C85': {1: [], 2: [], 3: [('RP6', +1)]}, 'C51': {1: [], 2: [('RP5', +1)], 3: []}, 'C17': {1: [('RP4', -1)], 2: [('RP5', -1)], 3: [('RP6', -1)]}}\n"
    "_M10_TPL_PBC_G = {'E1': {1: [('RP3', +1)], 2: [('RP4', +1)], 3: [('RP5', +1)]}, 'E2': {1: [('RP3', -1)], 2: [('RP4', -1)], 3: [('RP5', -1)]}, 'E3': {1: [('RP4', -1)], 2: [('RP1', -1)], 3: [('RP6', -1)]}, 'E4': {1: [('RP5', -1)], 2: [('RP6', -1)], 3: [('RP2', -1)]}, 'E5': {1: [('RP3', -1)], 2: [('RP4', -1)], 3: [('RP5', -1)]}, 'E6': {1: [('RP5', -1)], 2: [('RP6', -1)], 3: [('RP2', -1)]}, 'E7': {1: [('RP4', -1)], 2: [('RP1', -1)], 3: [('RP6', -1)]}, 'E8': {1: [('RP3', +1)], 2: [('RP4', +1)], 3: [('RP5', +1)]}, 'E9': {1: [('RP4', +1)], 2: [('RP1', +1)], 3: [('RP6', +1)]}, 'F-x': {1: [('RP3', -1)], 2: [('RP4', -1)], 3: [('RP5', -1)]}, 'F-y': {1: [('RP4', -1)], 2: [('RP1', -1)], 3: [('RP6', -1)]}, 'F-z': {1: [('RP5', -1)], 2: [('RP6', -1)], 3: [('RP2', -1)]}, 'C62': {1: [('RP4', +1)], 2: [('RP1', +1)], 3: [('RP6', +1)]}, 'C23': {1: [('RP5', -1)], 2: [('RP6', -1)], 3: [('RP2', -1)]}, 'C34': {1: [('RP3', +1)], 2: [('RP4', +1)], 3: [('RP5', +1)]}, 'C48': {1: [('RP4', -1)], 2: [('RP1', -1)], 3: [('RP6', -1)]}, 'C85': {1: [('RP5', +1)], 2: [('RP6', +1)], 3: [('RP2', +1)]}, 'C51': {1: [('RP4', +1)], 2: [('RP1', +1)], 3: [('RP6', +1)]}, 'C17': {1: [('RP3', -1), ('RP4', -1), ('RP5', -1)], 2: [('RP1', -1), ('RP4', -1), ('RP6', -1)], 3: [('RP2', -1), ('RP5', -1), ('RP6', -1)]}}\n"
    "_M10_EDGE_GROUPS = [('E1', 'x0y0', 'x1y0', 'z'), ('E2', 'x1y1', 'x0y1', 'z'), ('E3', 'x0y1', 'x0y0', 'z'), ('E4', 'x0z1', 'x0z0', 'y'), ('E5', 'x1z1', 'x0z1', 'y'), ('E6', 'y0z1', 'y0z0', 'x'), ('E7', 'y1z1', 'y0z1', 'x'), ('E8', 'x0z0', 'x1z0', 'y'), ('E9', 'y0z0', 'y1z0', 'x')]\n"
    "_M10_FACE_GROUPS = [('F-x', 'x1', 'x0', ('y', 'z')), ('F-y', 'y1', 'y0', ('x', 'z')), ('F-z', 'z1', 'z0', ('x', 'y'))]\n"
    "_M10_CORNER_GROUPS = [('C62', 'C6', 'C2'), ('C23', 'C2', 'C3'), ('C34', 'C3', 'C4'), ('C48', 'C4', 'C8'), ('C85', 'C8', 'C5'), ('C51', 'C5', 'C1'), ('C17', 'C1', 'C7')]\n"
    '\n'
    'def _m10_get_tpl(pbc_mode):\n'
    "    if pbc_mode == 'PBC_E':\n"
    '        return _M10_TPL_PBC_E\n'
    "    if pbc_mode == 'PBC_G':\n"
    '        return _M10_TPL_PBC_G\n'
    "    raise RuntimeError('[M10 FATAL] unknown pbc_mode: %s' % str(pbc_mode))\n"
    '\n'
    'def _m10_del_if_exists(dct, name):\n'
    '    if name in dct:\n'
    '        del dct[name]\n'
    '\n'
    'def _m10_delete_constraints_by_prefix(model, prefix):\n'
    '    keys = list(model.constraints.keys())\n'
    '    for k in keys:\n'
    '        if str(k).startswith(prefix):\n'
    '            del model.constraints[k]\n'
    '\n'
    'def _m10_delete_bcs_by_prefix(model, prefix):\n'
    '    keys = list(model.boundaryConditions.keys())\n'
    '    for k in keys:\n'
    '        if str(k).startswith(prefix):\n'
    '            del model.boundaryConditions[k]\n'
    '\n'
    'def _m10_delete_interactions_by_prefix(model, prefix):\n'
    '    keys = list(model.interactions.keys())\n'
    '    for k in keys:\n'
    '        if str(k).startswith(prefix):\n'
    '            del model.interactions[k]\n'
    '\n'
    'def _m10_delete_surface_if_exists(part, name):\n'
    "    if hasattr(part, 'surfaces') and name in part.surfaces:\n"
    '        del part.surfaces[name]\n'
    '\n'
    'def _m10_canonicalize_xyz(xyz, Lx, Ly, Lz, tol_plane):\n'
    '    return _m8a_canonicalize_xyz(xyz, Lx, Ly, Lz, tol_plane)\n'
    '\n'
    'def _m10_plane_tag_of_tri_by_nodes(n1, n2, n3, lab2xyzC, Lx, Ly, Lz):\n'
    '    x1, y1, z1 = lab2xyzC[n1]\n'
    '    x2, y2, z2 = lab2xyzC[n2]\n'
    '    x3, y3, z3 = lab2xyzC[n3]\n'
    '    if x1 == 0.0 and x2 == 0.0 and (x3 == 0.0):\n'
    "        return 'X0'\n"
    '    if x1 == Lx and x2 == Lx and (x3 == Lx):\n'
    "        return 'X1'\n"
    '    if y1 == 0.0 and y2 == 0.0 and (y3 == 0.0):\n'
    "        return 'Y0'\n"
    '    if y1 == Ly and y2 == Ly and (y3 == Ly):\n'
    "        return 'Y1'\n"
    '    if z1 == 0.0 and z2 == 0.0 and (z3 == 0.0):\n'
    "        return 'Z0'\n"
    '    if z1 == Lz and z2 == Lz and (z3 == Lz):\n'
    "        return 'Z1'\n"
    '    return None\n'
    '\n'
    'def _m10_build_boundary_faces_map_for_tet_part(p_tet):\n'
    '    triCount = {}\n'
    '    triFirst = {}\n'
    '    for el in p_tet.elements:\n'
    '        nds = el.getNodes()\n'
    '        if len(nds) != 4:\n'
    '            continue\n'
    '        n = [int(nd.label) for nd in nds]\n'
    '        faces = {1: (n[0], n[1], n[2]), 2: (n[0], n[1], n[3]), 3: (n[1], n[2], n[3]), 4: (n[0], n[2], n[3])}\n'
    '        for fidx, fnodes in faces.items():\n'
    '            triKey = tuple(sorted(fnodes))\n'
    '            triCount[triKey] = triCount.get(triKey, 0) + 1\n'
    '            if triKey not in triFirst:\n'
    '                triFirst[triKey] = (int(el.label), int(fidx))\n'
    '    boundaryTri2face = {}\n'
    '    for triKey, cnt in triCount.items():\n'
    '        if cnt == 1:\n'
    '            boundaryTri2face[triKey] = triFirst[triKey]\n'
    '        elif cnt > 2:\n'
    "            raise RuntimeError('[M10 FATAL] non-manifold face triKey=%s cnt=%d on part=%s' % (str(triKey), cnt, p_tet.name))\n"
    '    return boundaryTri2face\n'
    '\n'
    'def _m10_debug_pair_failure(a, inst_name, A_labs, B_labs, lab2xyzC, axes, tol_pair, tag):\n'
    "    _m10_del_if_exists(a.sets, 'M10_DBG_%s_UNMATCHED_A' % tag)\n"
    "    _m10_del_if_exists(a.sets, 'M10_DBG_%s_UNUSED_B' % tag)\n"
    "    _m10_del_if_exists(a.sets, 'M10_DBG_%s_ALL_B' % tag)\n"
    '\n'
    '    def dist2(x1, x2):\n'
    '        d = 0.0\n'
    '        for ax in axes:\n'
    "            i = {'x': 0, 'y': 1, 'z': 2}[ax]\n"
    '            dv = float(x1[i] - x2[i])\n'
    '            d += dv * dv\n'
    '        return d\n'
    '    A = sorted(set((int(x) for x in A_labs)))\n'
    '    B = sorted(set((int(x) for x in B_labs)))\n'
    '    usedB = set()\n'
    '    unmatchedA = []\n'
    '    tp = float(tol_pair)\n'
    '    for aLab in A:\n'
    '        xA = lab2xyzC[aLab]\n'
    '        hit = None\n'
    '        for bLab in B:\n'
    '            if bLab in usedB:\n'
    '                continue\n'
    '            xB = lab2xyzC[bLab]\n'
    '            ok = True\n'
    '            for ax in axes:\n'
    "                i = {'x': 0, 'y': 1, 'z': 2}[ax]\n"
    '                if abs(float(xB[i] - xA[i])) > tp:\n'
    '                    ok = False\n'
    '                    break\n'
    '            if ok:\n'
    '                hit = bLab\n'
    '                break\n'
    '        if hit is None:\n'
    '            unmatchedA.append(aLab)\n'
    '        else:\n'
    '            usedB.add(hit)\n'
    '    unusedB = [b for b in B if b not in usedB]\n'
    "    if hasattr(a, 'SetFromNodeLabels'):\n"
    '        try:\n'
    '            if unmatchedA:\n'
    "                a.SetFromNodeLabels(name='M10_DBG_%s_UNMATCHED_A' % tag, nodeLabels=((inst_name, tuple(unmatchedA)),))\n"
    '        except Exception:\n'
    '            pass\n'
    '        try:\n'
    '            if unusedB:\n'
    "                a.SetFromNodeLabels(name='M10_DBG_%s_UNUSED_B' % tag, nodeLabels=((inst_name, tuple(unusedB)),))\n"
    '        except Exception:\n'
    '            pass\n'
    '        try:\n'
    "            a.SetFromNodeLabels(name='M10_DBG_%s_ALL_B' % tag, nodeLabels=((inst_name, tuple(B)),))\n"
    '        except Exception:\n'
    '            pass\n'
    '    else:\n'
    '        try:\n'
    '            if unmatchedA:\n'
    '                nds = [a.instances[inst_name].nodes.getFromLabel(label=int(x)) for x in unmatchedA]\n'
    "                a.Set(name='M10_DBG_%s_UNMATCHED_A' % tag, nodes=nds)\n"
    '        except Exception:\n'
    '            pass\n'
    '        try:\n'
    '            if unusedB:\n'
    '                nds = [a.instances[inst_name].nodes.getFromLabel(label=int(x)) for x in unusedB]\n'
    "                a.Set(name='M10_DBG_%s_UNUSED_B' % tag, nodes=nds)\n"
    '        except Exception:\n'
    '            pass\n'
    '        try:\n'
    '            nds = [a.instances[inst_name].nodes.getFromLabel(label=int(x)) for x in B]\n'
    "            a.Set(name='M10_DBG_%s_ALL_B' % tag, nodes=nds)\n"
    '        except Exception:\n'
    '            pass\n'
    '    part_name = a.instances[inst_name].partName\n'
    '    if part_name in a.model.parts:\n'
    '        part = a.model.parts[part_name]\n'
    "        for sname in (f'M10_DBG_{tag}_UNMATCHED_A', f'M10_DBG_{tag}_UNUSED_B'):\n"
    '            if sname in part.sets:\n'
    '                del part.sets[sname]\n'
    '        try:\n'
    '            if unusedB:\n'
    "                part.SetFromNodeLabels(name=f'M10_DBG_{tag}_UNUSED_B', nodeLabels=tuple(unusedB))\n"
    '        except Exception:\n'
    '            try:\n'
    '                if unusedB:\n'
    '                    nds = part.nodes.sequenceFromLabels(labels=unusedB)\n'
    "                    part.Set(name=f'M10_DBG_{tag}_UNUSED_B', nodes=nds)\n"
    '            except Exception:\n'
    '                pass\n'
    '        if unmatchedA:\n'
    '            try:\n'
    "                part.SetFromNodeLabels(name=f'M10_DBG_{tag}_UNMATCHED_A', nodeLabels=tuple(unmatchedA))\n"
    '            except Exception:\n'
    '                try:\n'
    '                    nds = part.nodes.sequenceFromLabels(labels=unmatchedA)\n'
    "                    part.Set(name=f'M10_DBG_{tag}_UNMATCHED_A', nodes=nds)\n"
    '                except Exception:\n'
    '                    pass\n'
    "    print('[M10:DBG] %s unmatchedA=%d unusedB=%d' % (tag, len(unmatchedA), len(unusedB)))\n"
    '    if unusedB:\n'
    "        print('[M10:DBG] === UNUSED B nodes for tag=%s ===' % tag)\n"
    '        for bLab in unusedB:\n'
    '            xB = lab2xyzC[bLab]\n'
    '            bestA = None\n'
    '            for aLab in A:\n'
    '                xA = lab2xyzC[aLab]\n'
    '                d2 = dist2(xA, xB)\n'
    '                if bestA is None or d2 < bestA[0]:\n'
    '                    bestA = (d2, aLab, xA)\n'
    '            if bestA is not None:\n'
    "                print('[M10:DBG] UNUSED_B node=%d xyzC=%s  nearest_A=%d xyzC=%s  dist=%.6g' % (bLab, str(xB), bestA[1], str(bestA[2]), math.sqrt(bestA[0])))\n"
    '            else:\n'
    "                print('[M10:DBG] UNUSED_B node=%d xyzC=%s  (no A nodes to compare)' % (bLab, str(xB)))\n"
    '    for aLab in unmatchedA[:10]:\n'
    '        xA = lab2xyzC[aLab]\n'
    '        best = None\n'
    '        for bLab in B:\n'
    '            xB = lab2xyzC[bLab]\n'
    '            d2 = dist2(xA, xB)\n'
    '            if best is None or d2 < best[0]:\n'
    '                best = (d2, bLab, xB)\n'
    "        print('[M10:DBG] UNMATCHED_A=%d xyzC=%s nearestB=%d xyzC=%s d=%.6g (tol_pair=%.6g)' % (aLab, str(xA), best[1], str(best[2]), math.sqrt(best[0]), tp))\n"
    '\n'
    'def _m10_build_surfaces_and_plane_nodes_from_tet(model, tet_part_name, phase_tag, Lx, Ly, Lz, tol_plane, make_debug_outer_surfaces=True, make_debug_plane_node_sets=True):\n'
    '    model = mdb.models[model.name]\n'
    '    if tet_part_name not in model.parts:\n'
    "        raise RuntimeError('[M10 FATAL] missing tet part: %s' % tet_part_name)\n"
    '    p = model.parts[tet_part_name]\n'
    '    lab2xyzC = {}\n'
    '    for nd in p.nodes:\n'
    '        lab2xyzC[int(nd.label)] = _m10_canonicalize_xyz(nd.coordinates, Lx, Ly, Lz, tol_plane)\n'
    '    boundaryTri2face = _m10_build_boundary_faces_map_for_tet_part(p)\n'
    '    ifaceBuckets = {1: set(), 2: set(), 3: set(), 4: set()}\n'
    "    outerBuckets = {pl: {1: set(), 2: set(), 3: set(), 4: set()} for pl in ('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1')}\n"
    "    planeNodes = {pl: set() for pl in ('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1')}\n"
    '    ifaceElemLabs = set()\n'
    '    ifaceNodeLabs = set()\n'
    '    for triKey, (eLab, fidx) in boundaryTri2face.items():\n'
    '        n1, n2, n3 = triKey\n'
    '        pl = _m10_plane_tag_of_tri_by_nodes(n1, n2, n3, lab2xyzC, Lx, Ly, Lz)\n'
    '        if pl is None:\n'
    '            ifaceBuckets[fidx].add(int(eLab))\n'
    '            ifaceElemLabs.add(int(eLab))\n'
    '            ifaceNodeLabs.add(int(n1))\n'
    '            ifaceNodeLabs.add(int(n2))\n'
    '            ifaceNodeLabs.add(int(n3))\n'
    '        else:\n'
    '            outerBuckets[pl][fidx].add(int(eLab))\n'
    '            planeNodes[pl].add(int(n1))\n'
    '            planeNodes[pl].add(int(n2))\n'
    '            planeNodes[pl].add(int(n3))\n'
    "    print('[M10:IFACE2] %s part=%s boundaryFaces=%d IFACE_nodes=%d IFACE_tets=%d' % (phase_tag, tet_part_name, len(boundaryTri2face), len(ifaceNodeLabs), len(ifaceElemLabs)))\n"
    "    surf_iface = 'SURF_%s_IFACE' % phase_tag\n"
    '    _m10_delete_surface_if_exists(p, surf_iface)\n'
    '    kwargs = {}\n'
    '    for fidx in (1, 2, 3, 4):\n'
    '        labs = sorted(ifaceBuckets[fidx])\n'
    '        if labs:\n'
    "            kwargs['face%dElements' % fidx] = p.elements.sequenceFromLabels(labels=labs)\n"
    '    if not kwargs:\n'
    "        raise RuntimeError('[M10 FATAL] %s IFACE surface empty (no iface boundary faces found).' % phase_tag)\n"
    '    p.Surface(name=surf_iface, **kwargs)\n'
    "    _m8a_make_node_set_from_labels(p, 'NSET_%s_IFACE' % phase_tag, sorted(ifaceNodeLabs), allow_empty=False)\n"
    "    _m8a_make_elem_set_from_labels(p, 'ESET_%s_IFACE_TETS' % phase_tag, sorted(ifaceElemLabs), allow_empty=False)\n"
    '    if make_debug_outer_surfaces:\n'
    "        for pl in ('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1'):\n"
    "            sname = 'SURF_%s_OUTER_%s' % (phase_tag, pl)\n"
    '            _m10_delete_surface_if_exists(p, sname)\n'
    '            okw = {}\n'
    '            for fidx in (1, 2, 3, 4):\n'
    '                labs = sorted(outerBuckets[pl][fidx])\n'
    '                if labs:\n'
    "                    okw['face%dElements' % fidx] = p.elements.sequenceFromLabels(labels=labs)\n"
    '            if okw:\n'
    '                p.Surface(name=sname, **okw)\n'
    '    if make_debug_plane_node_sets:\n'
    "        for pl in ('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1'):\n"
    "            _m8a_make_node_set_from_labels(p, 'NSET_%s_PLANE_%s' % (phase_tag, pl), sorted(planeNodes[pl]), allow_empty=True)\n"
    '    return (planeNodes, lab2xyzC)\n'
    '\n'
    'def _m10_bins_from_plane_nodes(planeNodes):\n'
    "    X0, X1 = (planeNodes['X0'], planeNodes['X1'])\n"
    "    Y0, Y1 = (planeNodes['Y0'], planeNodes['Y1'])\n"
    "    Z0, Z1 = (planeNodes['Z0'], planeNodes['Z1'])\n"
    "    cornerSets = {'C7': X0 & Y0 & Z0, 'C8': X1 & Y0 & Z0, 'C3': X0 & Y1 & Z0, 'C4': X1 & Y1 & Z0, 'C6': X0 & Y0 & Z1, 'C5': X1 & Y0 & Z1, 'C2': X0 & Y1 & Z1, 'C1': X1 & Y1 & Z1}\n"
    '    corners = {}\n'
    '    for ck, s in cornerSets.items():\n'
    '        if not s:\n'
    '            corners[ck] = None\n'
    '        else:\n'
    '            corners[ck] = min((int(x) for x in s))\n'
    '\n'
    '    def rm_corner(s, cA, cB):\n'
    '        out = set(s)\n'
    '        if corners[cA] is not None:\n'
    '            out.discard(int(corners[cA]))\n'
    '        if corners[cB] is not None:\n'
    '            out.discard(int(corners[cB]))\n'
    '        return out\n'
    '    edges = {}\n'
    "    edges['x0y0'] = rm_corner(X0 & Y0, 'C7', 'C6')\n"
    "    edges['x1y0'] = rm_corner(X1 & Y0, 'C8', 'C5')\n"
    "    edges['x0y1'] = rm_corner(X0 & Y1, 'C3', 'C2')\n"
    "    edges['x1y1'] = rm_corner(X1 & Y1, 'C4', 'C1')\n"
    "    edges['x0z0'] = rm_corner(X0 & Z0, 'C7', 'C3')\n"
    "    edges['x1z0'] = rm_corner(X1 & Z0, 'C8', 'C4')\n"
    "    edges['x0z1'] = rm_corner(X0 & Z1, 'C6', 'C2')\n"
    "    edges['x1z1'] = rm_corner(X1 & Z1, 'C5', 'C1')\n"
    "    edges['y0z0'] = rm_corner(Y0 & Z0, 'C7', 'C8')\n"
    "    edges['y1z0'] = rm_corner(Y1 & Z0, 'C3', 'C4')\n"
    "    edges['y0z1'] = rm_corner(Y0 & Z1, 'C6', 'C5')\n"
    "    edges['y1z1'] = rm_corner(Y1 & Z1, 'C2', 'C1')\n"
    "    faces_plane = {'x0': set(X0), 'x1': set(X1), 'y0': set(Y0), 'y1': set(Y1), 'z0': set(Z0), 'z1': set(Z1)}\n"
    "    x0_edges = edges['x0y0'] | edges['x0y1'] | edges['x0z0'] | edges['x0z1']\n"
    "    x1_edges = edges['x1y0'] | edges['x1y1'] | edges['x1z0'] | edges['x1z1']\n"
    "    y0_edges = edges['x0y0'] | edges['x1y0'] | edges['y0z0'] | edges['y0z1']\n"
    "    y1_edges = edges['x0y1'] | edges['x1y1'] | edges['y1z0'] | edges['y1z1']\n"
    "    z0_edges = edges['x0z0'] | edges['x1z0'] | edges['y0z0'] | edges['y1z0']\n"
    "    z1_edges = edges['x0z1'] | edges['x1z1'] | edges['y0z1'] | edges['y1z1']\n"
    "    plane_edge = {'x0': x0_edges, 'x1': x1_edges, 'y0': y0_edges, 'y1': y1_edges, 'z0': z0_edges, 'z1': z1_edges}\n"
    "    plane_corner = {'x0': set((x for x in (corners['C7'], corners['C6'], corners['C3'], corners['C2']) if x is not None)), 'x1': set((x for x in (corners['C8'], corners['C5'], corners['C4'], corners['C1']) if x is not None)), 'y0': set((x for x in (corners['C7'], corners['C6'], corners['C8'], corners['C5']) if x is not None)), 'y1': set((x for x in (corners['C3'], corners['C2'], corners['C4'], corners['C1']) if x is not None)), 'z0': set((x for x in (corners['C7'], corners['C8'], corners['C3'], corners['C4']) if x is not None)), 'z1': set((x for x in (corners['C6'], corners['C5'], corners['C2'], corners['C1']) if x is not None))}\n"
    '    faces = {}\n'
    "    for fk in ('x0', 'x1', 'y0', 'y1', 'z0', 'z1'):\n"
    '        faces[fk] = faces_plane[fk] - plane_edge[fk] - plane_corner[fk]\n'
    '    return (faces, edges, corners)\n'
    '\n'
    'def _m10_pair_by_axes(A_labs, B_labs, lab2xyzC, axes, tol_pair, group_id, phase_tag):\n'
    '    A = sorted(set((int(x) for x in A_labs)))\n'
    '    B = sorted(set((int(x) for x in B_labs)))\n'
    '    if not A and (not B):\n'
    '        return []\n'
    '    if not A and B:\n'
    "        raise RuntimeError('[M10 FATAL] %s pairing %s: A empty but B=%d' % (phase_tag, group_id, len(B)))\n"
    '    if A and (not B):\n'
    "        raise RuntimeError('[M10 FATAL] %s pairing %s: B empty but A=%d' % (phase_tag, group_id, len(A)))\n"
    '    tp = float(tol_pair)\n'
    '    eps = max(1e-18, 0.2 * tp)\n'
    '    rad = int(math.ceil(tp / eps)) + 1\n'
    '\n'
    '    def q(v):\n'
    '        return int(math.floor(float(v) / eps + 0.5))\n'
    '    buckets = defaultdict(list)\n'
    '    for b in B:\n'
    '        xyz = lab2xyzC[b]\n'
    '        if len(axes) == 1:\n'
    '            ax = axes[0]\n'
    "            val = xyz[{'x': 0, 'y': 1, 'z': 2}[ax]]\n"
    '            buckets[q(val),].append(b)\n'
    '        else:\n'
    '            a1, a2 = axes\n'
    "            v1 = xyz[{'x': 0, 'y': 1, 'z': 2}[a1]]\n"
    "            v2 = xyz[{'x': 0, 'y': 1, 'z': 2}[a2]]\n"
    '            buckets[q(v1), q(v2)].append(b)\n'
    '    usedB = set()\n'
    '    pairs = []\n'
    '    unmatchedA = []\n'
    '    for a in A:\n'
    '        xyzA = lab2xyzC[a]\n'
    '        if len(axes) == 1:\n'
    '            ax = axes[0]\n'
    "            vA = xyzA[{'x': 0, 'y': 1, 'z': 2}[ax]]\n"
    '            k0 = (q(vA),)\n'
    '            neigh = [(k0[0] + di,) for di in range(-rad, rad + 1)]\n'
    '        else:\n'
    '            a1, a2 = axes\n'
    "            v1 = xyzA[{'x': 0, 'y': 1, 'z': 2}[a1]]\n"
    "            v2 = xyzA[{'x': 0, 'y': 1, 'z': 2}[a2]]\n"
    '            k0 = (q(v1), q(v2))\n'
    '            neigh = []\n'
    '            for di in range(-rad, rad + 1):\n'
    '                for dj in range(-rad, rad + 1):\n'
    '                    neigh.append((k0[0] + di, k0[1] + dj))\n'
    '        cand = []\n'
    '        for kk in neigh:\n'
    '            for b in buckets.get(kk, []):\n'
    '                if b in usedB:\n'
    '                    continue\n'
    '                xyzB = lab2xyzC[b]\n'
    '                ok = True\n'
    '                d2 = 0.0\n'
    '                for ax in axes:\n'
    "                    ia = {'x': 0, 'y': 1, 'z': 2}[ax]\n"
    '                    dv = float(xyzB[ia] - xyzA[ia])\n'
    '                    if abs(dv) > tp:\n'
    '                        ok = False\n'
    '                        break\n'
    '                    d2 += dv * dv\n'
    '                if ok:\n'
    '                    cand.append((d2, b))\n'
    '        if not cand:\n'
    '            unmatchedA.append(a)\n'
    '            continue\n'
    '        cand.sort(key=lambda t: t[0])\n'
    '        if len(cand) >= 2 and abs(cand[1][0] - cand[0][0]) < 1e-30:\n'
    "            raise RuntimeError('[M10 FATAL] %s pairing %s ambiguous: nodeA=%d has >=2 equal-distance candidates' % (phase_tag, group_id, a))\n"
    '        b = cand[0][1]\n'
    '        usedB.add(b)\n'
    '        pairs.append((a, b))\n'
    '    if unmatchedA or len(usedB) != len(B):\n'
    "        raise RuntimeError('[M10 FATAL] %s pairing failed group=%s: A=%d B=%d paired=%d unmatchedA=%d unusedB=%d' % (phase_tag, group_id, len(A), len(B), len(pairs), len(unmatchedA), len(B) - len(usedB)))\n"
    '    return pairs\n'
    '\n'
    'def _m10_get_or_make_single_node_set(a, set_name, inst_name, node_label):\n'
    '    if set_name in a.sets:\n'
    '        return set_name\n'
    "    if hasattr(a, 'SetFromNodeLabels'):\n"
    '        a.SetFromNodeLabels(name=set_name, nodeLabels=((inst_name, (int(node_label),)),))\n'
    '        return set_name\n'
    '    nd = a.instances[inst_name].nodes.getFromLabel(label=int(node_label))\n'
    '    a.Set(name=set_name, nodes=(nd,))\n'
    '    return set_name\n'
    '\n'
    'def _m10_apply_equations_for_pairs(model, a, pbc_mode, tpl, inst_name, group_id, pairs, rp_sets, ns_prefix):\n'
    '    nEq = 0\n'
    '    for i, (aLab, bLab) in enumerate(pairs):\n'
    "        setA = _m10_get_or_make_single_node_set(a, '%s%d' % (ns_prefix, int(aLab)), inst_name, int(aLab))\n"
    "        setB = _m10_get_or_make_single_node_set(a, '%s%d' % (ns_prefix, int(bLab)), inst_name, int(bLab))\n"
    '        for dof in (1, 2, 3):\n'
    '            terms = [(1.0, setA, dof), (-1.0, setB, dof)]\n'
    '            for rpn, sgn in tpl[group_id][dof]:\n'
    '                terms.append((float(sgn), rp_sets[rpn], dof))\n'
    "            eq_name = 'M10_EQ_%s_%s_%s_U%d_%06d' % (pbc_mode, inst_name, group_id, dof, i)\n"
    '            model.Equation(name=eq_name, terms=tuple(terms))\n'
    '            nEq += 1\n'
    '    return nEq\n'
    '\n'
    'def _m10_safe_inst_tag(inst_name):\n'
    "    return str(inst_name).replace('-', '_').replace('.', '_')\n"
    '\n'
    'def _m10_apply_equations_for_node_refs(model, a, pbc_mode, tpl, group_id, pairs, rp_sets, ns_prefix):\n'
    '    nEq = 0\n'
    '    for i, (refA, refB) in enumerate(pairs):\n'
    '        instA, aLab = refA\n'
    '        instB, bLab = refB\n'
    "        setA = _m10_get_or_make_single_node_set(a, '%s%s_%d' % (ns_prefix, _m10_safe_inst_tag(instA), int(aLab)), instA, int(aLab))\n"
    "        setB = _m10_get_or_make_single_node_set(a, '%s%s_%d' % (ns_prefix, _m10_safe_inst_tag(instB), int(bLab)), instB, int(bLab))\n"
    '        for dof in (1, 2, 3):\n'
    '            terms = [(1.0, setA, dof), (-1.0, setB, dof)]\n'
    '            for rpn, sgn in tpl[group_id][dof]:\n'
    '                terms.append((float(sgn), rp_sets[rpn], dof))\n'
    "            eq_name = 'M10_EQ_%s_MIXED_%s_U%d_%06d' % (pbc_mode, group_id, dof, i)\n"
    '            model.Equation(name=eq_name, terms=tuple(terms))\n'
    '            nEq += 1\n'
    '    return nEq\n'
    '\n'
    'def _m10_build_coord_to_label_map(lab2xyzC, tol_pair):\n'
    '    eps = max(1e-18, 0.2 * float(tol_pair))\n'
    '\n'
    '    def q(xyz):\n'
    '        return tuple((int(math.floor(float(v) / eps + 0.5)) for v in xyz))\n'
    '    buckets = defaultdict(list)\n'
    '    for lab, xyz in lab2xyzC.items():\n'
    '        buckets[q(xyz)].append(int(lab))\n'
    '    return (buckets, eps)\n'
    '\n'
    'def _m10_find_same_xyz_label(lab, xyz, target_buckets, target_lab2xyzC, eps, tol_pair):\n'
    '    key0 = tuple((int(math.floor(float(v) / eps + 0.5)) for v in xyz))\n'
    '    rad = int(math.ceil(float(tol_pair) / float(eps))) + 1\n'
    '    best = None\n'
    '    for dx in range(-rad, rad + 1):\n'
    '        for dy in range(-rad, rad + 1):\n'
    '            for dz in range(-rad, rad + 1):\n'
    '                kk = (key0[0] + dx, key0[1] + dy, key0[2] + dz)\n'
    '                for cand in target_buckets.get(kk, []):\n'
    '                    xyzC = target_lab2xyzC[int(cand)]\n'
    '                    ok = True\n'
    '                    d2 = 0.0\n'
    '                    for i in (0, 1, 2):\n'
    '                        dv = float(xyzC[i] - xyz[i])\n'
    '                        if abs(dv) > float(tol_pair):\n'
    '                            ok = False\n'
    '                            break\n'
    '                        d2 += dv * dv\n'
    '                    if ok and (best is None or d2 < best[0]):\n'
    '                        best = (d2, int(cand))\n'
    '    if best is None:\n'
    "        raise RuntimeError('[M10 FATAL] tied MAT node %d has no same-coordinate INC main node' % int(lab))\n"
    '    return int(best[1])\n'
    '\n'
    'def _m10_build_mat_tied_to_inc_map(mat_labels, lab2xyzM, lab2xyzI, tol_pair):\n'
    '    inc_buckets, eps = _m10_build_coord_to_label_map(lab2xyzI, tol_pair)\n'
    '    out = {}\n'
    '    for lab in sorted(set((int(x) for x in mat_labels))):\n'
    '        if lab not in lab2xyzM:\n'
    '            continue\n'
    '        out[int(lab)] = _m10_find_same_xyz_label(int(lab), lab2xyzM[int(lab)], inc_buckets, lab2xyzI, eps, tol_pair)\n'
    '    return out\n'
    '\n'
    "def m10_apply_tie_and_pbc(model_name, particles_work_json, pbc_mode='PBC_E', mat_tet_part='P_MAT_TET', inc_tet_part='P_INC_TET', inst_mat='I_MAT_TET', inst_inc='I_INC_TET', apply_rp_bc=False, rp_disp=None, step_name='Step-1'):\n"
    '    if model_name not in mdb.models:\n'
    "        raise RuntimeError('[M10 FATAL] model not found: %s' % model_name)\n"
    '    model = mdb.models[model_name]\n'
    '    a = model.rootAssembly\n'
    "    with open(particles_work_json, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    Lx, Ly, Lz = (float(x) for x in data['box'])\n"
    "    tol_plane = float(data['tol_plane'])\n"
    "    tol_pair = float(data.get('tol_pair', tol_plane))\n"
    '    tpl = _m10_get_tpl(pbc_mode)\n'
    "    _m10_delete_constraints_by_prefix(model, 'M10_EQ_')\n"
    "    _m10_delete_bcs_by_prefix(model, 'M10_BC_')\n"
    "    _m10_delete_interactions_by_prefix(model, 'M10_TIE_')\n"
    '    if inst_mat not in a.instances:\n'
    '        a.Instance(name=inst_mat, part=model.parts[mat_tet_part], dependent=ON)\n'
    '    if inst_inc not in a.instances:\n'
    '        a.Instance(name=inst_inc, part=model.parts[inc_tet_part], dependent=ON)\n'
    "    planeNodes_M, lab2xyzM = _m10_build_surfaces_and_plane_nodes_from_tet(model, mat_tet_part, 'MAT', Lx, Ly, Lz, tol_plane, make_debug_outer_surfaces=True, make_debug_plane_node_sets=True)\n"
    "    planeNodes_I, lab2xyzI = _m10_build_surfaces_and_plane_nodes_from_tet(model, inc_tet_part, 'INC', Lx, Ly, Lz, tol_plane, make_debug_outer_surfaces=True, make_debug_plane_node_sets=True)\n"
    '    a.regenerate()\n'
    "    tie_name = 'M10_TIE_IFACE_INC_TO_MAT'\n"
    "    main_surf = a.instances[inst_inc].surfaces['SURF_INC_IFACE']\n"
    "    sec_surf = a.instances[inst_mat].surfaces['SURF_MAT_IFACE']\n"
    '    model.Tie(name=tie_name, main=main_surf, secondary=sec_surf, positionToleranceMethod=COMPUTED, adjust=OFF, tieRotations=ON, thickness=ON)\n'
    "    print('[M10] created tie:', tie_name)\n"
    "    rp_pos = {'RP1': (Lx / 2.0, Ly / 2.0, 1.2 * Lz), 'RP2': (Lx / 2.0, 1.2 * Ly, Lz / 2.0), 'RP3': (1.2 * Lx, Ly / 2.0, Lz / 2.0), 'RP4': (1.4 * Lx, Ly / 2.0, Lz / 2.0), 'RP5': (1.6 * Lx, Ly / 2.0, Lz / 2.0), 'RP6': (1.8 * Lx, Ly / 2.0, Lz / 2.0)}\n"
    '    rp_sets = {}\n'
    "    if pbc_mode == 'PBC_E':\n"
    "        rp_list = ('RP4', 'RP5', 'RP6')\n"
    '    else:\n'
    "        rp_list = ('RP1', 'RP2', 'RP3', 'RP4', 'RP5', 'RP6')\n"
    '    rp_sets = {}\n'
    '    for rpn in rp_list:\n'
    "        sname = 'SET_%s' % rpn\n"
    '        rp_sets[rpn] = sname\n'
    '        if sname in a.sets:\n'
    '            continue\n'
    '        rp_feat = a.ReferencePoint(point=rp_pos[rpn])\n'
    '        rp_obj = a.referencePoints[rp_feat.id]\n'
    '        a.Set(name=sname, referencePoints=(rp_obj,))\n'
    "        print('[M10] created', sname, 'at', rp_pos[rpn])\n"
    '    if apply_rp_bc:\n'
    '        if step_name not in model.steps:\n'
    "            model.StaticStep(name=step_name, previous='Initial')\n"
    '        if rp_disp is None:\n'
    "            raise RuntimeError('[M10 FATAL] apply_rp_bc=True but rp_disp is None')\n"
    '        for rpn, vals in rp_disp.items():\n'
    '            if rpn not in rp_sets:\n'
    '                continue\n'
    '            u1, u2, u3 = vals\n'
    "            model.DisplacementBC(name='M10_BC_%s' % rpn, createStepName=step_name, region=a.sets[rp_sets[rpn]], u1=u1, u2=u2, u3=u3)\n"
    "        print('[M10] RP displacement BCs applied on step:', step_name)\n"
    '    p_mat = model.parts[mat_tet_part]\n'
    "    if 'NSET_MAT_IFACE' not in p_mat.sets:\n"
    "        raise RuntimeError('[M10 FATAL] missing NSET_MAT_IFACE on %s' % mat_tet_part)\n"
    "    mat_iface_nodes = set((int(n.label) for n in p_mat.sets['NSET_MAT_IFACE'].nodes))\n"
    "    _m8a_make_node_set_from_labels(p_mat, 'NSET_MAT_PBC_EXCLUDED_TIED', sorted(mat_iface_nodes), allow_empty=True)\n"
    '    facesM, edgesM, cornersM = _m10_bins_from_plane_nodes(planeNodes_M)\n'
    '    facesI, edgesI, cornersI = _m10_bins_from_plane_nodes(planeNodes_I)\n'
    '    removed_faces = {}\n'
    "    for fk in ('x0', 'x1', 'y0', 'y1', 'z0', 'z1'):\n"
    '        before = len(facesM[fk])\n'
    '        facesM[fk] = set(facesM[fk]) - mat_iface_nodes\n'
    '        removed_faces[fk] = before - len(facesM[fk])\n'
    '    mat_edge_nodes = set()\n'
    '    for ek in edgesM:\n'
    '        mat_edge_nodes |= set(edgesM[ek])\n'
    '    mat_corner_nodes = set((int(v) for v in cornersM.values() if v is not None))\n'
    '    mat_tied_edge_nodes = mat_iface_nodes & mat_edge_nodes\n'
    '    mat_tied_corner_nodes = mat_iface_nodes & mat_corner_nodes\n'
    '    mat_tied_pbc_nodes = mat_tied_edge_nodes | mat_tied_corner_nodes\n'
    '    mat_tied_to_inc = {}\n'
    '    if mat_tied_pbc_nodes:\n'
    '        mat_tied_to_inc = _m10_build_mat_tied_to_inc_map(mat_tied_pbc_nodes, lab2xyzM, lab2xyzI, tol_pair)\n'
    "        _m8a_make_node_set_from_labels(p_mat, 'NSET_MAT_PBC_REDIRECT_TO_INC', sorted(mat_tied_pbc_nodes), allow_empty=True)\n"
    '    redirected_by_edge = {}\n'
    '    for ek in sorted(edgesM.keys()):\n'
    '        redirected_by_edge[ek] = len(set(edgesM[ek]) & mat_tied_edge_nodes)\n'
    "    print('[M10] MAT PBC exclude faces: removed_by_face=%s total=%d' % (str(removed_faces), sum(list(removed_faces.values()))))\n"
    "    print('[M10] MAT PBC redirect tied edge/corner nodes to INC: edge_by_set=%s edgeTotal=%d cornerTotal=%d' % (str(redirected_by_edge), len(mat_tied_edge_nodes), len(mat_tied_corner_nodes)))\n"
    '    corner_owner = {}\n'
    "    for ck in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'):\n"
    '        mLab = cornersM.get(ck, None)\n'
    '        iLab = cornersI.get(ck, None)\n'
    '        if mLab is not None and int(mLab) in mat_tied_to_inc:\n'
    '            mLab = None\n'
    '            iLab = int(mat_tied_to_inc[int(cornersM.get(ck))])\n'
    '        if mLab is None and iLab is None:\n'
    "            raise RuntimeError('[M10 FATAL] corner %s missing in both phases' % ck)\n"
    '        if mLab is not None and iLab is not None:\n'
    "            raise RuntimeError('[M10 FATAL] corner %s exists in both phases (overlap on OUTER faces?)' % ck)\n"
    '        if mLab is not None:\n'
    '            corner_owner[ck] = (inst_mat, int(mLab))\n'
    '        else:\n'
    '            corner_owner[ck] = (inst_inc, int(iLab))\n'
    "    c7_inst, c7_lab = corner_owner['C7']\n"
    "    set_c7 = _m10_get_or_make_single_node_set(a, 'M10_C7_ANCHOR', c7_inst, c7_lab)\n"
    "    model.DisplacementBC(name='M10_BC_FIX_C7', createStepName='Initial', region=a.sets[set_c7], u1=0.0, u2=0.0, u3=0.0)\n"
    "    print('[M10] fixed C7 on', c7_inst, 'nodeLabel=', c7_lab)\n"
    '\n'
    '    def apply_faces_edges(phase_tag, inst_name, faces, edges, lab2xyz):\n'
    '        nEq = 0\n'
    '        nMixedPairs = 0\n'
    '        nSkipBothTied = 0\n'
    "        ns_prefix = 'NS_%s_' % phase_tag\n"
    '        for gid, tagA, tagB, axes in _M10_FACE_GROUPS:\n'
    '            try:\n'
    '                pairs = _m10_pair_by_axes(faces[tagA], faces[tagB], lab2xyz, axes, tol_pair, gid, phase_tag)\n'
    '            except Exception:\n'
    "                dbg_tag = ('%s_%s' % (phase_tag, gid)).replace('-', '_')\n"
    '                _m10_debug_pair_failure(a, inst_name, faces[tagA], faces[tagB], lab2xyz, axes, tol_pair, dbg_tag)\n'
    '                raise\n'
    '            nEq += _m10_apply_equations_for_pairs(model, a, pbc_mode, tpl, inst_name, gid, pairs, rp_sets, ns_prefix)\n'
    '        for gid, tagA, tagB, freeAxis in _M10_EDGE_GROUPS:\n'
    '            try:\n'
    '                pairs = _m10_pair_by_axes(edges[tagA], edges[tagB], lab2xyz, (freeAxis,), tol_pair, gid, phase_tag)\n'
    '            except Exception:\n'
    "                dbg_tag = ('%s_%s' % (phase_tag, gid)).replace('-', '_')\n"
    '                _m10_debug_pair_failure(a, inst_name, edges[tagA], edges[tagB], lab2xyz, (freeAxis,), tol_pair, dbg_tag)\n'
    '                raise\n'
    "            if phase_tag == 'MAT' and mat_tied_to_inc:\n"
    '                ref_pairs = []\n'
    '                for aLab, bLab in pairs:\n'
    '                    if int(aLab) in mat_tied_to_inc:\n'
    '                        refA = (inst_inc, int(mat_tied_to_inc[int(aLab)]))\n'
    '                    else:\n'
    '                        refA = (inst_mat, int(aLab))\n'
    '                    if int(bLab) in mat_tied_to_inc:\n'
    '                        refB = (inst_inc, int(mat_tied_to_inc[int(bLab)]))\n'
    '                    else:\n'
    '                        refB = (inst_mat, int(bLab))\n'
    '                    if refA[0] == inst_inc and refB[0] == inst_inc:\n'
    '                        nSkipBothTied += 1\n'
    '                        continue\n'
    '                    if refA[0] != inst_mat or refB[0] != inst_mat:\n'
    '                        nMixedPairs += 1\n'
    '                    ref_pairs.append((refA, refB))\n'
    "                nEq += _m10_apply_equations_for_node_refs(model, a, pbc_mode, tpl, gid, ref_pairs, rp_sets, 'NS_%s_REF_' % phase_tag)\n"
    '            else:\n'
    '                nEq += _m10_apply_equations_for_pairs(model, a, pbc_mode, tpl, inst_name, gid, pairs, rp_sets, ns_prefix)\n'
    "        if phase_tag == 'MAT':\n"
    "            print('[M10] %s edge redirect pairs=%d skipped_both_tied_pairs=%d' % (phase_tag, nMixedPairs, nSkipBothTied))\n"
    "        print('[M10] %s equations (faces+edges) = %d' % (phase_tag, nEq))\n"
    '        return nEq\n'
    "    nEq_M = apply_faces_edges('MAT', inst_mat, facesM, edgesM, lab2xyzM)\n"
    "    nEq_I = apply_faces_edges('INC', inst_inc, facesI, edgesI, lab2xyzI)\n"
    '    corner_sets = {}\n'
    '    for ck, (instn, lab) in corner_owner.items():\n'
    "        sname = 'M10_%s' % ck\n"
    '        corner_sets[ck] = _m10_get_or_make_single_node_set(a, sname, instn, lab)\n'
    '    nEq_C = 0\n'
    '    for gid, cA, cB in _M10_CORNER_GROUPS:\n'
    '        setA = corner_sets[cA]\n'
    '        setB = corner_sets[cB]\n'
    '        for dof in (1, 2, 3):\n'
    '            terms = [(1.0, setA, dof), (-1.0, setB, dof)]\n'
    '            for rpn, sgn in tpl[gid][dof]:\n'
    '                terms.append((float(sgn), rp_sets[rpn], dof))\n'
    "            eq_name = 'M10_EQ_%s_CORNER_%s_U%d' % (pbc_mode, gid, dof)\n"
    '            model.Equation(name=eq_name, terms=tuple(terms))\n'
    '            nEq_C += 1\n'
    "    print('[M10] corner equations = %d' % nEq_C)\n"
    "    print('=== Module 10 OK ===')\n"
    "    print('[M10] summary: nEq_MAT=%d nEq_INC=%d nEq_CORNER=%d tie=%s anchor=%s mode=%s' % (nEq_M, nEq_I, nEq_C, tie_name, set_c7, pbc_mode))\n"
    "    return {'nEq_MAT': nEq_M, 'nEq_INC': nEq_I, 'nEq_CORNER': nEq_C, 'C7': (c7_inst, c7_lab)}\n"
    'import os\n'
    'from abaqusConstants import UNSET, PERCENTAGE\n'
    '\n'
    'def _m11_delete_field_output_by_prefix(model, prefix):\n'
    '    for k in list(model.fieldOutputRequests.keys()):\n'
    '        if str(k).startswith(prefix):\n'
    '            del model.fieldOutputRequests[k]\n'
    '\n'
    'def _m11_delete_history_output_by_prefix(model, prefix):\n'
    '    for k in list(model.historyOutputRequests.keys()):\n'
    '        if str(k).startswith(prefix):\n'
    '            del model.historyOutputRequests[k]\n'
    '\n'
    'def _m11_delete_bcs_by_prefix(model, prefix):\n'
    '    for k in list(model.boundaryConditions.keys()):\n'
    '        if str(k).startswith(prefix):\n'
    '            del model.boundaryConditions[k]\n'
    '\n'
    'def _m11_used_rp_dofs_from_template(tpl):\n'
    '    used = set()\n'
    '    for gid, dof_map in tpl.items():\n'
    '        for dof, terms in dof_map.items():\n'
    '            for rp, sgn in terms:\n'
    '                used.add((str(rp), int(dof)))\n'
    '    return used\n'
    '\n'
    'def _m11_set_dispBC(model, name, step, region, u1=UNSET, u2=UNSET, u3=UNSET):\n'
    '    if name in model.boundaryConditions:\n'
    '        del model.boundaryConditions[name]\n'
    '    model.DisplacementBC(name=name, createStepName=step, region=region, u1=u1, u2=u2, u3=u3)\n'
    '\n'
    'def _m11_ensure_static_step(model, step_name, timePeriod=1.0):\n'
    '    if step_name not in model.steps:\n'
    "        model.StaticStep(name=step_name, previous='Initial', timePeriod=timePeriod, nlgeom=OFF)\n"
    '    else:\n'
    '        st = model.steps[step_name]\n'
    '        try:\n'
    '            st.setValues(timePeriod=timePeriod, nlgeom=OFF)\n'
    '        except Exception:\n'
    '            pass\n'
    '\n'
    'def _m11_make_outputs(model, a, step_name, Nout, rp_list):\n'
    '    for k in list(model.fieldOutputRequests.keys()):\n'
    '        del model.fieldOutputRequests[k]\n'
    '    for k in list(model.historyOutputRequests.keys()):\n'
    '        del model.historyOutputRequests[k]\n'
    "    model.FieldOutputRequest(name='M11_FO_AVG', createStepName=step_name, variables=('S', 'EVOL'), numIntervals=int(Nout))\n"
    '    for rpn in rp_list:\n'
    "        sname = 'SET_%s' % rpn\n"
    '        if sname not in a.sets:\n'
    '            continue\n'
    "        model.HistoryOutputRequest(name='M11_HO_%s' % rpn, createStepName=step_name, variables=('U1', 'U2', 'U3'), region=a.sets[sname], numIntervals=int(Nout))\n"
    '\n'
    'def _m11_apply_rp_bcs(model, a, step_name, tpl, case, Lx, Ly, Lz, rp_list):\n'
    "    _m11_delete_bcs_by_prefix(model, 'M11_BC_')\n"
    "    case_type = case['type'].upper()\n"
    "    sign = float(case.get('sign', +1.0))\n"
    "    strain_pct = float(case['strain_pct'])\n"
    '    eps = sign * (strain_pct / 100.0)\n'
    '    used = _m11_used_rp_dofs_from_template(tpl)\n'
    '    for rpn in rp_list:\n'
    "        sname = 'SET_%s' % rpn\n"
    '        if sname not in a.sets:\n'
    '            continue\n'
    '        u = {1: UNSET, 2: UNSET, 3: UNSET}\n'
    '        for dof in (1, 2, 3):\n'
    '            if (rpn, dof) not in used:\n'
    '                u[dof] = 0.0\n'
    '        if u[1] is not UNSET or u[2] is not UNSET or u[3] is not UNSET:\n'
    "            _m11_set_dispBC(model, name='M11_BC_FIX_UNUSED_%s' % rpn, step='Initial', region=a.sets[sname], u1=u[1], u2=u[2], u3=u[3])\n"
    '    desired = {}\n'
    '    for rp, dof in used:\n'
    '        desired[rp, dof] = UNSET\n'
    "    if case_type == 'AXIAL':\n"
    "        axial_dir = case['axial_dir'].upper()\n"
    "        if axial_dir == 'X':\n"
    "            desired['RP4', 1] = eps * Lx\n"
    "        elif axial_dir == 'Y':\n"
    "            desired['RP5', 2] = eps * Ly\n"
    "        elif axial_dir == 'Z':\n"
    "            desired['RP6', 3] = eps * Lz\n"
    '        else:\n'
    "            raise RuntimeError('[M11 FATAL] unknown axial_dir: %s' % axial_dir)\n"
    "    elif case_type == 'SHEAR':\n"
    '        gamma = eps\n'
    '        eij = 0.5 * gamma\n'
    "        shear_plane = case['shear_plane'].upper()\n"
    "        if shear_plane == 'XY':\n"
    "            desired['RP4', 2] = eij * Lx\n"
    "            desired['RP4', 1] = eij * Ly\n"
    "        elif shear_plane == 'XZ':\n"
    "            desired['RP5', 3] = eij * Lx\n"
    "            desired['RP5', 1] = eij * Lz\n"
    "        elif shear_plane == 'YZ':\n"
    "            desired['RP6', 3] = eij * Ly\n"
    "            desired['RP6', 2] = eij * Lz\n"
    '        else:\n'
    "            raise RuntimeError('[M11 FATAL] unknown shear_plane: %s' % shear_plane)\n"
    '        for rp, dof in used:\n'
    '            if desired[rp, dof] is UNSET:\n'
    '                desired[rp, dof] = 0.0\n'
    '    else:\n'
    "        raise RuntimeError('[M11 FATAL] unknown case type: %s' % case_type)\n"
    '    for rpn in rp_list:\n'
    "        sname = 'SET_%s' % rpn\n"
    '        if sname not in a.sets:\n'
    '            continue\n'
    '        u = {1: UNSET, 2: UNSET, 3: UNSET}\n'
    '        for dof in (1, 2, 3):\n'
    '            if (rpn, dof) in used:\n'
    '                u[dof] = desired[rpn, dof]\n'
    '        if u[1] is not UNSET or u[2] is not UNSET or u[3] is not UNSET:\n'
    "            _m11_set_dispBC(model, name='M11_BC_MACRO_%s' % rpn, step=step_name, region=a.sets[sname], u1=u[1], u2=u[2], u3=u[3])\n"
    "    print('[M11] Macro BC summary: type=%s strain_pct=%g sign=%+g' % (case_type, strain_pct, sign))\n"
    '    for (rp, dof), val in sorted(desired.items()):\n'
    '        if val is UNSET:\n'
    '            continue\n'
    "        comp = 'U%d' % int(dof)\n"
    "        print('  %s.%s = %g' % (rp, comp, float(val)))\n"
    '\n'
    "def _m11_make_job_name(case, prefix='RVE'):\n"
    "    t = case['type'].upper()\n"
    "    sign = 'P' if float(case.get('sign', +1.0)) >= 0 else 'N'\n"
    "    sp = ('%.3f' % float(case['strain_pct'])).rstrip('0').rstrip('.').replace('.', 'p')\n"
    "    if t == 'AXIAL':\n"
    "        return '%s_AXIAL_%s_%s_%s' % (prefix, case['axial_dir'].upper(), sign, sp)\n"
    '    else:\n'
    "        return '%s_SHEAR_%s_%s_%s' % (prefix, case['shear_plane'].upper(), sign, sp)\n"
    '\n'
    "def m11_run_case(model_name, particles_work_json, case, mat_tet_part='P_MAT_TET', inc_tet_part='P_INC_TET', inst_mat='I_MAT_TET', inst_inc='I_INC_TET', step_name='Step-1', numCpus=8, memoryPercent=80, Nout=100, job_prefix='RVE', wait=True):\n"
    '    if model_name not in mdb.models:\n'
    "        raise RuntimeError('[M11 FATAL] model not found: %s' % model_name)\n"
    '    model = mdb.models[model_name]\n'
    '    a = model.rootAssembly\n'
    "    with open(particles_work_json, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    Lx, Ly, Lz = (float(x) for x in data['box'])\n"
    "    case_type = case['type'].upper()\n"
    "    if case_type == 'AXIAL':\n"
    "        pbc_mode = 'PBC_E'\n"
    "    elif case_type == 'SHEAR':\n"
    "        pbc_mode = 'PBC_G'\n"
    '    else:\n'
    "        raise RuntimeError('[M11 FATAL] unknown case type: %s' % case_type)\n"
    "    if pbc_mode == 'PBC_E':\n"
    "        rp_list = ('RP4', 'RP5', 'RP6')\n"
    '    else:\n'
    "        rp_list = ('RP1', 'RP2', 'RP3', 'RP4', 'RP5', 'RP6')\n"
    '    m10_apply_tie_and_pbc(model_name=model_name, particles_work_json=particles_work_json, pbc_mode=pbc_mode, mat_tet_part=mat_tet_part, inc_tet_part=inc_tet_part, inst_mat=inst_mat, inst_inc=inst_inc, apply_rp_bc=False)\n'
    '    _m11_ensure_static_step(model, step_name, timePeriod=1.0)\n'
    '    _m11_make_outputs(model, a, step_name, Nout, rp_list)\n'
    '    tpl = _m10_get_tpl(pbc_mode)\n'
    '    _m11_apply_rp_bcs(model, a, step_name, tpl, case, Lx, Ly, Lz, rp_list)\n'
    '    job_name = _m11_make_job_name(case, prefix=job_prefix)\n'
    '    if job_name in mdb.jobs:\n'
    '        del mdb.jobs[job_name]\n'
    '    job = mdb.Job(name=job_name, model=model_name, type=ANALYSIS, numCpus=int(numCpus), numDomains=int(numCpus), memory=int(memoryPercent), memoryUnits=PERCENTAGE)\n'
    "    print('[M11] submitting job:', job_name, 'cpus=', numCpus, 'mem%=', memoryPercent)\n"
    '    job.submit()\n'
    '    if wait:\n'
    '        job.waitForCompletion()\n'
    "        print('[M11] job status:', job.status)\n"
    '        if job.status != COMPLETED:\n'
    "            raise RuntimeError('[M11 FATAL] job not completed: %s status=%s' % (job_name, str(job.status)))\n"
    '    return job_name\n'
    '\n'
    'def main():\n'
    '    if not os.path.isfile(INPUT_JSON_PATH):\n'
    "        raise RuntimeError('INPUT_JSON_PATH not found: %s' % INPUT_JSON_PATH)\n"
    "    with open(INPUT_JSON_PATH, 'r') as f:\n"
    '        data = json.load(f)\n'
    '    out, summary = preprocess(data)\n'
    '    if OUTPUT_JSON_PATH is None:\n'
    "        out_path = os.path.join(os.path.dirname(INPUT_JSON_PATH), 'particles_work.json')\n"
    '    else:\n'
    '        out_path = OUTPUT_JSON_PATH\n'
    "    with open(out_path, 'w') as f:\n"
    '        json.dump(out, f, indent=2)\n'
    "    print('=== Preprocess summary ===')\n"
    '    for k in sorted(summary.keys()):\n'
    "        print('%s: %s' % (k, summary[k]))\n"
    "    print('tol_plane = %.6e' % out['tol_plane'])\n"
    "    print('Wrote:', out_path)\n"
    '    if not os.path.isfile(PARTICLES_WORK_JSON):\n'
    "        raise RuntimeError('particles_work.json not found: %s' % PARTICLES_WORK_JSON)\n"
    "    with open(PARTICLES_WORK_JSON, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    tol_plane = float(data['tol_plane'])\n"
    "    particles_work = data['particles_work']\n"
    '    model = _ensure_model_rve()\n'
    '    n = len(particles_work) if MAX_PARTICLES is None else min(int(MAX_PARTICLES), len(particles_work))\n'
    "    print('=== Module 3 start ===')\n"
    "    print('Model:', MODEL_NAME)\n"
    "    print('tol_plane:', tol_plane)\n"
    "    print('particles:', n, '/', len(particles_work))\n"
    '    for i in range(n):\n'
    '        pw = particles_work[i]\n'
    "        ID4 = pw['ID4']\n"
    "        a = float(pw['a'])\n"
    "        b = float(pw['b'])\n"
    "        touch = pw['touch']\n"
    "        cut_planes_local = pw['cut_planes_local']\n"
    "        part_name = 'P_INC_%s' % ID4\n"
    '        p = _build_spheroid_solid_part(model, part_name, a, b)\n'
    '        if len(cut_planes_local) > 0:\n'
    '            _partition_by_cut_planes(p, cut_planes_local)\n'
    '        _remove_all_cells_to_shell(p)\n'
    '        if len(p.cells) != 0 or len(p.faces) == 0:\n'
    "            raise RuntimeError('Shell conversion failed on %s: cells=%d faces=%d' % (p.name, len(p.cells), len(p.faces)))\n"
    '        _build_cap_iface_sets(p, ID4, touch, cut_planes_local, tol_plane)\n'
    "        iface_name = 'INC_%s_IFACE' % ID4\n"
    "        print('Built', part_name, 'faces=', len(p.faces), 'touch=', touch, 'IFACE_faces=', len(p.sets[iface_name].faces))\n"
    "    print('=== Module 3 done ===')\n"
    '    if MODEL_NAME not in mdb.models.keys():\n'
    '        mdb.Model(name=MODEL_NAME)\n'
    '    model = mdb.models[MODEL_NAME]\n'
    '    if not os.path.isfile(PARTICLES_WORK_JSON):\n'
    "        raise RuntimeError('particles_work.json not found: %s' % PARTICLES_WORK_JSON)\n"
    "    with open(PARTICLES_WORK_JSON, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    Lx, Ly, Lz = map(float, data['box'])\n"
    "    tol_plane = float(data['tol_plane'])\n"
    "    print('=== Module 2 start ===')\n"
    "    print('Model:', MODEL_NAME)\n"
    "    print('Box:', (Lx, Ly, Lz), 'tol_plane:', tol_plane)\n"
    "    _delete_instance_if_exists(model, 'I_MATRIX_MASTER')\n"
    '    _delete_part_if_exists(model, PART_NAME)\n'
    "    sk = model.ConstrainedSketch(name='__SK_MATRIX_MASTER__', sheetSize=10.0 * max(Lx, Ly, Lz))\n"
    '    sk.setPrimaryObject(option=STANDALONE)\n'
    '    sk.rectangle(point1=(0.0, 0.0), point2=(Lx, Ly))\n'
    '    p = model.Part(name=PART_NAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)\n'
    '    p.BaseSolidExtrude(sketch=sk, depth=Lz)\n'
    '    sk.unsetPrimaryObject()\n'
    "    del model.sketches['__SK_MATRIX_MASTER__']\n"
    '    if len(p.cells) == 0:\n'
    "        raise RuntimeError('Box solid has no cells; extrude failed.')\n"
    '    _remove_cells_keep_shell(p)\n'
    '    if len(p.cells) != 0:\n'
    "        raise RuntimeError('RemoveCells failed: cells still exist.')\n"
    '    if len(p.faces) != 6:\n'
    "        print('[WARN] After RemoveCells expected 6 faces, got', len(p.faces))\n"
    '    tags = []\n'
    '    for fce in p.faces[:]:\n'
    '        tag = _classify_face_by_pointOn(fce, Lx, Ly, Lz, tol_plane)\n'
    '        tags.append(tag)\n'
    '    tags_sorted = sorted([t for t in tags if t is not None])\n'
    "    expected = ['X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1']\n"
    '    if tags_sorted != expected:\n'
    "        raise RuntimeError('P_MATRIX_MASTER faces not complete 6 planes. got=%s expected=%s' % (str(tags_sorted), str(expected)))\n"
    "    print('Built part:', PART_NAME, 'faces=', len(p.faces), 'planes=', tags_sorted)\n"
    "    print('=== Module 2 done ===')\n"
    '    if MODEL_NAME not in mdb.models.keys():\n'
    '        raise RuntimeError("Model \'%s\' not found. Run Module 2/3 first in same CAE session." % MODEL_NAME)\n'
    '    model = mdb.models[MODEL_NAME]\n'
    '    a = model.rootAssembly\n'
    '    if not os.path.isfile(PARTICLES_WORK_JSON):\n'
    "        raise RuntimeError('particles_work.json not found: %s' % PARTICLES_WORK_JSON)\n"
    "    with open(PARTICLES_WORK_JSON, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    Lx, Ly, Lz = map(float, data['box'])\n"
    "    tol_plane = float(data['tol_plane'])\n"
    "    particles_work = data['particles_work']\n"
    "    print('=== Module 4 start ===')\n"
    "    print('Model:', MODEL_NAME)\n"
    "    print('Box:', (Lx, Ly, Lz), 'tol_plane:', tol_plane)\n"
    "    print('particles:', len(particles_work))\n"
    "    if 'P_MATRIX_MASTER' in model.parts.keys():\n"
    "        delete_instance_anyway(a, 'I_MATRIX_MASTER')\n"
    "        a.Instance(name='I_MATRIX_MASTER', part=model.parts['P_MATRIX_MASTER'], dependent=ON)\n"
    "        print('Created instance: I_MATRIX_MASTER')\n"
    '    else:\n'
    "        print('[WARN] Part P_MATRIX_MASTER not found. Skip I_MATRIX_MASTER.')\n"
    '    for pw in particles_work:\n'
    "        ID4 = pw['ID4']\n"
    "        touch = pw['touch']\n"
    "        tile_shifts = pw['tile_shifts']\n"
    '        expected_n = 2 ** len(touch)\n'
    '        if len(tile_shifts) != expected_n:\n'
    "            raise RuntimeError('tile_shifts count mismatch ID4=%s expected=%d got=%d' % (ID4, expected_n, len(tile_shifts)))\n"
    "        part_name = 'P_INC_%s' % ID4\n"
    '        if part_name not in model.parts.keys():\n'
    "            raise RuntimeError('Missing part %s (Module 3 output)' % part_name)\n"
    "        axis = tuple(pw['rot_axis_global'])\n"
    "        ang = float(pw['rot_angle_deg'])\n"
    "        c_work = tuple(pw['c_work'])\n"
    '        for sh in tile_shifts:\n'
    '            Sxyz = encode_SXYZ(sh, Lx, Ly, Lz, tol_plane)\n'
    "            inst_name = 'I_INC_%s_%s' % (ID4, Sxyz)\n"
    '            delete_instance_anyway(a, inst_name)\n'
    '            a.Instance(name=inst_name, part=model.parts[part_name], dependent=ON)\n'
    '            inst = a.instances[inst_name]\n'
    '            inst.rotateAboutAxis(axisPoint=(0.0, 0.0, 0.0), axisDirection=axis, angle=ang)\n'
    '            vec = (c_work[0] + sh[0], c_work[1] + sh[1], c_work[2] + sh[2])\n'
    '            inst.translate(vector=vec)\n'
    "        print('Placed', part_name, 'touch=', touch, 'instances=', len(tile_shifts))\n"
    "    print('=== Module 4 done ===')\n"
    '    if MODEL_NAME not in mdb.models.keys():\n'
    '        raise RuntimeError("Model \'%s\' not found. Must reuse same RVE model (run Module 2/3/4 first)." % MODEL_NAME)\n'
    '    model = mdb.models[MODEL_NAME]\n'
    '    a = model.rootAssembly\n'
    '    if not os.path.isfile(PARTICLES_WORK_JSON):\n'
    "        raise RuntimeError('particles_work.json not found: %s' % PARTICLES_WORK_JSON)\n"
    "    with open(PARTICLES_WORK_JSON, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    Lx, Ly, Lz = map(float, data['box'])\n"
    "    tol_plane = float(data['tol_plane'])\n"
    "    particles_work = data['particles_work']\n"
    "    print('=== Module 5 start ===')\n"
    "    print('Model:', MODEL_NAME)\n"
    "    print('Box:', (Lx, Ly, Lz), 'tol_plane:', tol_plane)\n"
    '    delete_instance_anyway(a, MERGED_INSTANCE_NAME)\n'
    "    if 'I_MATRIX_MASTER' not in a.instances.keys():\n"
    "        raise RuntimeError('Missing instance I_MATRIX_MASTER. Run Module 4 first (and ensure P_MATRIX_MASTER exists).')\n"
    "    inst_list = [a.instances['I_MATRIX_MASTER']]\n"
    '    for pw in particles_work:\n'
    "        ID4 = pw['ID4']\n"
    "        tile_shifts = pw['tile_shifts']\n"
    '        for sh in tile_shifts:\n'
    '            Sxyz = encode_SXYZ(sh, Lx, Ly, Lz, tol_plane)\n'
    "            inst_name = 'I_INC_%s_%s' % (ID4, Sxyz)\n"
    '            if inst_name not in a.instances.keys():\n'
    "                raise RuntimeError('Missing instance %s. Run Module 4 first.' % inst_name)\n"
    '            inst_list.append(a.instances[inst_name])\n'
    "    print('Merging instances count =', len(inst_list))\n"
    '    merged_inst = a.InstanceFromBooleanMerge(name=MERGED_INSTANCE_NAME, instances=tuple(inst_list), keepIntersections=ON, originalInstances=SUPPRESS, domain=GEOMETRY)\n'
    '    created_name = merged_inst.name\n'
    '    if created_name != MERGED_INSTANCE_NAME:\n'
    '        delete_instance_anyway(a, MERGED_INSTANCE_NAME)\n'
    '        a.instances.changeKey(fromName=created_name, toName=MERGED_INSTANCE_NAME)\n'
    '    if MERGED_INSTANCE_NAME not in a.instances.keys():\n'
    "        raise RuntimeError('Boolean merge failed: cannot find instance %s after rename.' % MERGED_INSTANCE_NAME)\n"
    '    a.makeIndependent(instances=(a.instances[MERGED_INSTANCE_NAME],))\n'
    "    print('Created merged independent instance:', MERGED_INSTANCE_NAME)\n"
    '    suppressed_cnt = 0\n'
    '    for inst in inst_list:\n'
    "        if hasattr(inst, 'suppressed') and inst.suppressed:\n"
    '            suppressed_cnt += 1\n'
    "    print('Original instances suppressed (counted) =', suppressed_cnt, '/', len(inst_list))\n"
    "    print('=== Module 5 done ===')\n"
    '    model = mdb.models[MODEL_NAME]\n'
    '    a = model.rootAssembly\n'
    '    inst = a.instances[MERGED_INSTANCE_NAME]\n'
    "    with open(PARTICLES_WORK_JSON, 'r') as f:\n"
    '        data = json.load(f)\n'
    "    tol_plane_m6 = float(data['tol_plane'])\n"
    "    Lx, Ly, Lz = map(float, data['box'])\n"
    '    Lchar = max(Lx, Ly, Lz)\n'
    '    run_module6_v3(model=model, a=a, inst=inst, particles_work_json=PARTICLES_WORK_JSON, tol_plane=tol_plane_m6, mesh_target_len=None, mesh_target_ratio=None, iface_refine_factor=0.6, curvature_factor=0.35, curvature_min_factor=0.45, gradation_ratio=1.35, min_size_ratio=0.005, param=M6_PARAM)\n'
    "    run_module7(model_name=MODEL_NAME, particles_work_json=PARTICLES_WORK_JSON, merged_instance_name=MERGED_INSTANCE_NAME, p_all_tri_name='P_ALL_TRI')\n"
    "    p, reg = m8a_collect_registry_and_crop(model_name=MODEL_NAME, particles_work_json=PARTICLES_WORK_JSON, src_part_name='P_ALL_TRI', work_part_name='P_ALL_TRI_M8A', registry_json='m8_registry.json', make_sets=True, eps_key_factor=0.2)\n"
    "    p_per = m8p_make_periodic_shell(model_name=MODEL_NAME, particles_work_json=PARTICLES_WORK_JSON, src_part_name='P_ALL_TRI_M8A', dst_part_name='P_ALL_TRI_PER')\n"
    "    p_inc, p_mat = m8b_split_shells(MODEL_NAME, src_part_name='P_ALL_TRI_PER', inc_part_name='P_INC_SHELL', mat_part_name='P_MAT_SHELL', keep_debug_sets=True)\n"
    "    m8c_flip_matrix_iface(MODEL_NAME, mat_part_name='P_MAT_SHELL', iface_set_name='IFACE_ALL', keep_debug_sets=True)\n"
    "    a = mdb.models['RVE'].rootAssembly\n"
    "    a.features['I_MERGED_GEOM'].suppress()\n"
    "    p_inc_tet, p_mat_tet = m9_make_tet_and_assign(model_name='RVE', inc_shell_part='P_INC_SHELL', mat_shell_part='P_MAT_SHELL', inc_tet_part='P_INC_TET', mat_tet_part='P_MAT_TET', eset_inc='ESET_INC', eset_mat='ESET_MAT', sec_inc='SecSiC', sec_mat='SecMatrix')\n"
    "    m10_apply_tie_and_pbc(model_name='RVE', particles_work_json=PARTICLES_WORK_JSON, pbc_mode='PBC_E', mat_tet_part='P_MAT_TET', inc_tet_part='P_INC_TET', inst_mat='I_MAT_TET', inst_inc='I_INC_TET', apply_rp_bc=False)\n"
    "    case = {'type': 'AXIAL', 'axial_dir': 'X', 'strain_pct': 0.5, 'sign': +1}\n"
    "    job_name = m11_run_case('RVE', PARTICLES_WORK_JSON, case, Nout=50, numCpus=10, memoryPercent=90, job_prefix='RVE')\n"
    "if __name__ == '__main__':\n"
    "    raise RuntimeError('Use run_fe.py with explicit input and output configuration.')\n"
)
FE_HELPER_SOURCE = (
    'import re\n'
    'from pathlib import Path\n'
    '\n'
    'def _classify_regen_required(run_dir: Path, since_mtime=None):\n'
    "    text = '\\n'.join([_read_text_if_exists(run_dir / 'abaqus_build.log'), _read_text_if_exists(run_dir / 'error_traceback.txt'), _read_abaqus_rpy_text(run_dir, since_mtime=since_mtime)])\n"
    "    if '[REGEN_JSON_REQUIRED]' in text:\n"
    "        m = re.search('\\\\[REGEN_JSON_REQUIRED\\\\]\\\\s+reason=([A-Za-z0-9_:-]+)', text)\n"
    '        if m:\n'
    '            return (True, m.group(1).lower())\n'
    "        return (True, 'regen_json_required')\n"
    "    if '[M8P FATAL] shell not closed' in text and 'freeEdges=' in text:\n"
    "        return (True, 'm8p_shell_free_edges')\n"
    "    if '[M8P FATAL] quick missing check failed' in text or 'quick missing check failed' in text:\n"
    "        return (True, 'm8p_quick_missing')\n"
    "    if '[M8P FATAL] periodic node pairing not strict' in text:\n"
    "        return (True, 'm8p_periodic_pairing')\n"
    "    if '[M8P FATAL]' in text:\n"
    "        return (True, 'm8p_fatal')\n"
    "    return (False, '')\n"
    '\n'
    "def _classify_abaqus_lock_failure(run_dir: Path, job_name: str='', since_mtime=None):\n"
    "    chunks = [_read_text_if_exists(run_dir / 'abaqus_build.log'), _read_text_if_exists(run_dir / 'error_traceback.txt'), _read_abaqus_rpy_text(run_dir, since_mtime=since_mtime)]\n"
    '    if job_name:\n'
    "        for suffix in ('.msg', '.log', '.sta'):\n"
    "            path = run_dir / ('%s%s' % (job_name, suffix))\n"
    '            if path.is_file():\n'
    '                if since_mtime is None or path.stat().st_mtime >= since_mtime:\n'
    '                    chunks.append(_read_text_if_exists(path))\n'
    "    text = '\\n'.join(chunks)\n"
    "    if 'Detected lock file' in text or 'Cannot create the lock file' in text:\n"
    '        return True\n'
    '    return False\n'
)
FE_MESH_SOURCE = (
    'from __future__ import annotations\n'
    'import copy\n'
    "ORIGINAL_MODEL_SHA256 = '26b575897547ef45aa9a7de893c5affea2c57efb12c13feb0d2041b6037a2ce7'\n"
    'import ast\n'
    'import hashlib\n'
    'import inspect\n'
    'import json\n'
    'import math\n'
    'import re\n'
    'import textwrap\n'
    'from pathlib import Path\n'
    "MODEL_SHA256 = '96f56c421feebd899ecec5b20734a3a536d24748e7120e66d693cd9a785487ad'\n"
    "HELPER_SHA256 = '04f236c1c7b41164415ac2b1006fe065630df940633a0ef1212190b27117a551'\n"
    "RECEIPT_SCHEMA = 'pw-rve-m9-cae-completion-v1'\n"
    "FAILURE_SCHEMA = 'pw-rve-m9-cae-failure-v1'\n"
    "NATIVE_PHASES = ('M1_PREPROCESS', 'M3', 'M2', 'M4', 'M5', 'M6', 'M7', 'M8A', 'M8P', 'M8B', 'M8C', 'M9')\n"
    "BOOTSTRAP_IMPORTS = 'from abaqus import *\\nfrom caeModules import *\\nimport part\\n'\n"
    '\n'
    'def _sha(path):\n'
    '    h = hashlib.sha256()\n'
    "    with Path(path).open('rb') as handle:\n"
    "        for chunk in iter(lambda: handle.read(1024 * 1024), b''):\n"
    '            h.update(chunk)\n'
    '    return h.hexdigest()\n'
    '\n'
    'def _bound_sources(fe_root):\n'
    '    root = Path(fe_root).resolve()\n'
    "    model = root / '602collaspse.py'\n"
    "    helpers = root / 'adapter_helpers.py'\n"
    '    for path, wanted in ((model, MODEL_SHA256), (helpers, HELPER_SHA256)):\n'
    '        if not path.is_file():\n'
    "            raise ValueError('Missing bound Stage 2 source: %s' % path)\n"
    '        if _sha(path) != wanted:\n'
    "            raise ValueError('FE source identity mismatch: %s' % path)\n"
    "    return (root, model.read_text(encoding='utf-8-sig'), helpers.read_text(encoding='utf-8-sig'))\n"
    '\n'
    'def inspect_native_mesh_defaults(fe_root):\n'
    '    root, src, _ = _bound_sources(fe_root)\n'
    '    tree = ast.parse(src)\n'
    '    values = []\n'
    '    for node in tree.body:\n'
    "        if isinstance(node, ast.Assign) and any((isinstance(target, ast.Name) and target.id == 'M6_PARAM' for target in node.targets)):\n"
    '            values.append(ast.literal_eval(node.value))\n'
    '    if len(values) != 1:\n'
    "        raise ValueError('Cannot bind unique native M6_PARAM')\n"
    "    main = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main'))\n"
    "    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'run_module6_v3')]\n"
    '    if len(calls) != 1:\n'
    "        raise ValueError('Cannot bind unique native main M6 invocation')\n"
    '    literal_args = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords if isinstance(kw.value, (ast.Constant, ast.UnaryOp))}\n'
    "    ratio = literal_args.get('mesh_target_ratio')\n"
    "    effective_ratio = values[0]['mesh_target_ratio'] if ratio is None else ratio\n"
    "    return {'fe_root': str(root), 'model_sha256': MODEL_SHA256, 'helper_sha256': HELPER_SHA256, 'native_M6_PARAM': values[0], 'native_main_M6_literal_arguments': literal_args, 'mesh_target_ratio_main_argument': ratio, 'mesh_target_ratio_effective': effective_ratio, 'mesh_target_len_main_argument': literal_args.get('mesh_target_len'), 'override_applied': False, 'source_inspection_only': True}\n"
    '\n'
    'def _native_segments(src):\n'
    '    tree = ast.parse(src)\n'
    "    mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']\n"
    "    terminals = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'm9_make_tet_and_assign']\n"
    '    if len(mains) != 1 or len(terminals) != 1:\n'
    "        raise ValueError('Ambiguous structural M9 boundary')\n"
    '    main = mains[0]\n'
    "    statements = [i for i, statement in enumerate(main.body) if any((isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and (call.func.id == 'm9_make_tet_and_assign') for call in ast.walk(statement)))]\n"
    '    if len(statements) != 1:\n'
    "        raise ValueError('Native main needs exactly one M9 invocation')\n"
    '    prefix = ast.Module(body=[copy.deepcopy(node) for node in tree.body if node.end_lineno <= terminals[0].end_lineno], type_ignores=[])\n'
    '    main = copy.deepcopy(main)\n'
    '    main.body = main.body[:statements[0] + 1]\n'
    '    return (_public_source(ast.unparse(prefix)), _public_source(ast.unparse(main)))\n'
    '\n'
    'def _configure_json_paths(prefix, json_path, particles_work_json):\n'
    '    tree = ast.parse(prefix)\n'
    "    values = {'INPUT_JSON_PATH': str(json_path), 'OUTPUT_JSON_PATH': str(particles_work_json), 'PARTICLES_WORK_JSON': str(particles_work_json)}\n"
    '    for name, value in values.items():\n'
    '        matches = [node for node in tree.body if isinstance(node, ast.Assign) and any((isinstance(target, ast.Name) and target.id == name for target in node.targets))]\n'
    '        if len(matches) != 1:\n'
    "            raise ValueError('Cannot bind unique native path configuration: ' + name)\n"
    '        matches[0].value = ast.Constant(value)\n'
    "    return ast.unparse(ast.fix_missing_locations(tree)) + '\\n'\n"
    '\n'
    'def _validate_actual_mesh_parameters(data, expected_ratio, expected_box=None, expected_node_match_tol=None):\n'
    '    if not isinstance(data, dict):\n'
    "        raise ValueError('Native M6 work JSON must be an object')\n"
    "    box = data.get('box')\n"
    '    if not isinstance(box, (list, tuple)) or len(box) != 3:\n'
    "        raise ValueError('Native M6 work JSON lacks a three-dimensional box')\n"
    "    numbers = list(box) + [data.get(key) for key in ('mesh_target_ratio', 'mesh_target_len', 'edge_target_len')]\n"
    '    if any((isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value <= 0) for value in numbers)):\n'
    "        raise ValueError('Native M6 box and mesh parameters must be finite positive numbers')\n"
    '    box = [float(value) for value in box]\n'
    '    if expected_box is not None and box != expected_box:\n'
    "        raise ValueError('Native M6 work JSON box differs from the bound input geometry')\n"
    '    ratio, length, edge = [float(value) for value in numbers[3:]]\n'
    '    characteristic = max(box)\n'
    '    if not math.isclose(ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-15):\n'
    "        raise ValueError('Persisted native M6 ratio differs from the requested/default ratio')\n"
    '    if not math.isclose(length, expected_ratio * characteristic, rel_tol=1e-12, abs_tol=1e-15):\n'
    "        raise ValueError('Persisted native M6 target length differs from ratio times max(box)')\n"
    '    if not math.isclose(edge, length, rel_tol=1e-12, abs_tol=1e-15):\n'
    "        raise ValueError('Persisted native M6 edge_target_len alias differs from target length')\n"
    '    if expected_node_match_tol is not None:\n'
    "        observed_tol = data.get('tol_pair')\n"
    '        if isinstance(observed_tol, bool) or not isinstance(observed_tol, (int, float)) or (not math.isfinite(observed_tol)) or (observed_tol <= 0) or (not math.isclose(observed_tol, expected_node_match_tol, rel_tol=1e-12, abs_tol=1e-18)):\n'
    "            raise ValueError('Native work tol_pair differs from the bound coordinate node-match tolerance')\n"
    "    return {'box': box, 'Lchar': characteristic, 'mesh_target_ratio': ratio, 'mesh_target_len': length, 'edge_target_len': edge, 'mesh_target_len_formula': 'mesh_target_ratio * max(box)', 'source': 'native_M6_persisted_work_json', 'node_match_tol': data.get('tol_pair')}\n"
    '\n'
    'def _runtime_prelude(success_cae, receipt_path, debug_cae, script_path, run_info_path, effective_ratio, failure_receipt_path, module_events_path, *, json_path, requested_ratio=None, particles_work_json=None, node_match_tol=None):\n'
    "    settings = {'success_cae': str(success_cae), 'receipt_path': str(receipt_path), 'debug_cae': None if debug_cae is None else str(debug_cae), 'script_path': str(script_path), 'run_info_path': str(run_info_path), 'input_json_path': str(json_path), 'mesh_target_ratio': effective_ratio, 'particles_work_json': str(Path(script_path).parent / 'particles_work.json' if particles_work_json is None else particles_work_json), 'mesh_target_ratio_call_literal': requested_ratio, 'failure_receipt_path': str(failure_receipt_path), 'module_events_path': str(module_events_path), 'completion_schema': RECEIPT_SCHEMA, 'failure_schema': FAILURE_SCHEMA, 'native_phases': list(NATIVE_PHASES), 'node_match_tol': node_match_tol}\n"
    '    return \'_mesh_settings = \' + repr(settings) + \'\\n\' + inspect.getsource(_validate_actual_mesh_parameters) + \'\\n\' + "\\nimport sys\\nimport os\\nimport json\\nimport traceback\\nimport hashlib\\nimport math\\nimport time\\n\\nINPUT_JSON_PATH = _mesh_settings[\'input_json_path\']\\n_mesh_success_cae = _mesh_settings[\'success_cae\']\\n_mesh_receipt_path = _mesh_settings[\'receipt_path\']\\n_mesh_debug_cae = _mesh_settings[\'debug_cae\']\\n_mesh_script_path = _mesh_settings[\'script_path\']\\n_mesh_run_info_path = _mesh_settings[\'run_info_path\']\\n_mesh_target_ratio = _mesh_settings[\'mesh_target_ratio\']\\n_mesh_closed = False\\n_mesh_saved = False\\n_mesh_rc = 1\\n_mesh_phase = \'BOOTSTRAP\'\\n_mesh_completed_phases = []\\n\\ndef _mesh_file_sha(path):\\n    h = hashlib.sha256()\\n    with open(path, \'rb\') as handle:\\n        for chunk in iter(lambda: handle.read(1048576), b\'\'):\\n            h.update(chunk)\\n    return h.hexdigest()\\n\\ndef _mesh_write_json(path, value):\\n    temporary = path + \'.tmp\'\\n    with open(temporary, \'x\', encoding=\'utf-8\') as handle:\\n        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)\\n        handle.flush()\\n        os.fsync(handle.fileno())\\n    os.replace(temporary, path)\\n\\ndef _fe_event(phase, state):\\n    event = {\'schema\': \'pw-rve-fe-module-event-v1\', \'phase\': phase,\\n             \'state\': state, \'time_unix\': time.time()}\\n    with open(_mesh_settings[\'module_events_path\'], \'a\', encoding=\'utf-8\') as handle:\\n        handle.write(json.dumps(event, sort_keys=True) + \'\\\\n\')\\n        handle.flush()\\n        os.fsync(handle.fileno())\\n    print(\'[FE_PHASE_%s] %s\' % (state, phase), flush=True)\\n\\ndef _fe_phase_begin(phase):\\n    global _mesh_phase\\n    _mesh_phase = phase\\n    _fe_event(phase, \'BEGIN\')\\n\\ndef _fe_phase_end(phase):\\n    if _mesh_phase != phase:\\n        raise RuntimeError(\'FE phase ledger mismatch: %s != %s\' % (_mesh_phase, phase))\\n    _mesh_completed_phases.append(phase)\\n    _fe_event(phase, \'END\')\\n\\ndef _mesh_bound_fields():\\n    return {\\n        \'input_json_path\': INPUT_JSON_PATH,\\n        \'input_json_sha256\': _mesh_file_sha(INPUT_JSON_PATH),\\n        \'script_path\': _mesh_script_path,\\n        \'script_sha256\': _mesh_file_sha(_mesh_script_path),\\n        \'run_info_path\': _mesh_run_info_path,\\n        \'run_info_sha256\': _mesh_file_sha(_mesh_run_info_path),\\n        \'mesh_target_ratio\': _mesh_target_ratio,\\n        \'mesh_target_ratio_call_literal\': _mesh_settings[\'mesh_target_ratio_call_literal\'],\\n        \'node_match_tol\': _mesh_settings[\'node_match_tol\'],\\n    }\\n\\ndef _mesh_record_failure(exc):\\n    raw_traceback = traceback.format_exc()\\n    failed_phase = _mesh_phase\\n    print(\'[MESH_ONLY_FAILURE] phase=%s %s: %s\' %\\n          (failed_phase, type(exc).__name__, str(exc)), flush=True)\\n    traceback.print_exc()\\n    trace_path = os.path.join(os.path.dirname(_mesh_script_path), \'error_traceback.txt\')\\n    with open(trace_path, \'w\', encoding=\'utf-8\') as handle:\\n        handle.write(raw_traceback)\\n        handle.flush()\\n        os.fsync(handle.fileno())\\n    debug = {\'path\': _mesh_debug_cae, \'saved\': False, \'official_success\': False}\\n    if _mesh_debug_cae is not None and \'mdb\' in globals() and not _mesh_closed and _mesh_phase not in (\'REOPEN_CAE\',\'CLOSE_REOPENED_CAE\',\'WRITE_COMPLETION\'):\\n        try:\\n            mdb.saveAs(pathName=_mesh_debug_cae)\\n            debug.update(saved=True, sha256=_mesh_file_sha(_mesh_debug_cae))\\n        except BaseException as debug_error:\\n            debug[\'error\'] = str(debug_error)\\n    failure = {\\n        \'schema\': _mesh_settings[\'failure_schema\'], \'status\': \'FAILED\',\\n        \'official_success\': False, \'failed_phase\': failed_phase,\\n        \'exception_type\': type(exc).__name__, \'exception_message\': str(exc),\\n        \'traceback\': raw_traceback,\\n        \'traceback_sha256\': hashlib.sha256(raw_traceback.encode(\'utf-8\')).hexdigest(),\\n        \'completed_phases\': list(_mesh_completed_phases),\\n        \'native_m9_completed\': \'M9\' in _mesh_completed_phases,\\n        \'official_cae_saved\': _mesh_saved, \'database_closed\': _mesh_closed,\\n        \'expected_exit_code\': 1, \'stop_after_module\': \'M9\', \'debug_cae\': debug,\\n        \'reopen_inventory\': globals().get(\'_mesh_inventory\'),\\n        \'expected_section_binding\': globals().get(\'_mesh_expected_sections\'),\\n        \'reopen_copy_observer\': globals().get(\'_mesh_reopen_observer\'),\\n    }\\n    failure.update(_mesh_bound_fields())\\n    _mesh_write_json(_mesh_settings[\'failure_receipt_path\'], failure)\\n"\n'
    '\n'
    'def _native_m9_section_binding(main):\n'
    '    tree = ast.parse(main)\n'
    "    mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']\n"
    '    if len(mains) != 1:\n'
    "        raise ValueError('Cannot bind unique native main for M9 section identities')\n"
    "    calls = [node for node in ast.walk(mains[0]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'm9_make_tet_and_assign']\n"
    '    if len(calls) != 1:\n'
    "        raise ValueError('Cannot bind unique native main M9 section invocation')\n"
    '    result = {}\n'
    "    for field in ('sec_mat', 'sec_inc'):\n"
    '        keywords = [item for item in calls[0].keywords if item.arg == field]\n'
    '        if len(keywords) != 1:\n'
    "            raise ValueError('Native M9 must explicitly bind ' + field)\n"
    '        value = ast.literal_eval(keywords[0].value)\n'
    '        if not isinstance(value, str) or not value:\n'
    "            raise ValueError('Native M9 section identity must be a nonempty string literal: ' + field)\n"
    '        result[field] = value\n'
    '    return result\n'
    '\n'
    'def _terminal_body(section_binding):\n'
    '    body = "\\n_fe_phase_end(\'BOOTSTRAP\')\\nmain()\\n_fe_phase_begin(\'SAVE_CAE\')\\nmdb.saveAs(pathName=_mesh_success_cae)\\n_mesh_saved=True\\n_fe_phase_end(\'SAVE_CAE\')\\n_fe_phase_begin(\'CLOSE_CAE\')\\nmdb.close()\\n_mesh_closed=True\\n_fe_phase_end(\'CLOSE_CAE\')\\n_fe_phase_begin(\'REOPEN_CAE\')\\n_mesh_saved_sha=_mesh_file_sha(_mesh_success_cae)\\n_mesh_saved_size=os.path.getsize(_mesh_success_cae)\\n_mesh_reopen_copy=_mesh_success_cae+\'.reopen_inspection.cae\'\\n_mesh_reopen_observer={\'schema\':\'pw-rve-m9-reopen-copy-observer-v1\',\'primary_path\':_mesh_success_cae,\'primary_before_sha256\':_mesh_saved_sha,\'primary_before_size_bytes\':_mesh_saved_size,\'copy_path\':_mesh_reopen_copy,\'opened\':False,\'closed\':False,\'explicit_copy_save_requested\':False,\'original_identity_guard_pass\':False,\'status\':\'CREATING_COPY\'}\\nwith open(_mesh_success_cae,\'rb\') as _mesh_copy_input:\\n    with open(_mesh_reopen_copy,\'xb\') as _mesh_copy_output:\\n        for _mesh_copy_chunk in iter(lambda:_mesh_copy_input.read(1048576),b\'\'):\\n            _mesh_copy_output.write(_mesh_copy_chunk)\\n        _mesh_copy_output.flush()\\n        os.fsync(_mesh_copy_output.fileno())\\n_mesh_reopen_observer[\'primary_before_open_sha256\']=_mesh_file_sha(_mesh_success_cae)\\n_mesh_reopen_observer[\'copy_before_open_sha256\']=_mesh_file_sha(_mesh_reopen_copy)\\n_mesh_reopen_observer[\'copy_before_open_size_bytes\']=os.path.getsize(_mesh_reopen_copy)\\nif _mesh_reopen_observer[\'primary_before_open_sha256\']!=_mesh_saved_sha or os.path.getsize(_mesh_success_cae)!=_mesh_saved_size:\\n    raise RuntimeError(\'Primary CAE changed while preparing reopen inspection copy\')\\nif _mesh_reopen_observer[\'copy_before_open_sha256\']!=_mesh_saved_sha or _mesh_reopen_observer[\'copy_before_open_size_bytes\']!=_mesh_saved_size:\\n    raise RuntimeError(\'Reopen inspection copy differs from saved CAE before open\')\\n_mesh_reopen_observer[\'status\']=\'COPY_VERIFIED_BEFORE_OPEN\'\\nmdb=openMdb(pathName=_mesh_reopen_copy)\\n_mesh_closed=False\\n_mesh_reopen_observer.update(opened=True,status=\'COPY_OPENED_FOR_INVENTORY\')\\nif \'RVE\' not in mdb.models or len(mdb.jobs)!=0:\\n    raise RuntimeError(\'Reopened M9 database must contain RVE and no solver jobs\')\\n_mesh_model=mdb.models[\'RVE\']\\n_mesh_inventory={\'parts\':{},\'instances\':{},\'materials\':{},\'sections\':{},\'jobs\':list(mdb.jobs.keys())}\\n_mesh_inventory[\'expected_section_binding\']=dict(_mesh_expected_sections)\\n_mesh_inventory[\'observed_section_keys\']=sorted(str(name) for name in _mesh_model.sections.keys())\\n_mesh_inventory[\'observed_section_materials\']={str(name):getattr(section,\'material\',None) for name,section in _mesh_model.sections.items()}\\n_mesh_inventory[\'observed_section_assignments\']={str(name):[assignment.sectionName for assignment in part.sectionAssignments] for name,part in _mesh_model.parts.items() if name in (\'P_MAT_TET\',\'P_INC_TET\')}\\nfor _mesh_part_name,_mesh_set_name,_mesh_instance_name,_mesh_section_name,_mesh_material_name in [(\'P_MAT_TET\',\'ESET_MAT\',\'I_MAT_TET\',\'SEC_MATRIX\',\'Al_matrix\'),(\'P_INC_TET\',\'ESET_INC\',\'I_INC_TET\',\'SEC_SIC\',\'SiC\')]:\\n    if _mesh_part_name not in _mesh_model.parts or _mesh_instance_name not in _mesh_model.rootAssembly.instances:\\n        raise RuntimeError(\'Reopened M9 tetrahedral part/instance missing: \'+_mesh_part_name)\\n    _mesh_part=_mesh_model.parts[_mesh_part_name]\\n    _mesh_instance=_mesh_model.rootAssembly.instances[_mesh_instance_name]\\n    _mesh_types=sorted(set(str(element.type) for element in _mesh_part.elements))\\n    if len(_mesh_part.nodes)<=0 or len(_mesh_part.elements)<=0 or _mesh_types!=[\'C3D4\'] or _mesh_set_name not in _mesh_part.sets:\\n        raise RuntimeError(\'Reopened M9 part lacks accepted tetrahedral mesh/set: \'+_mesh_part_name)\\n    if _mesh_section_name not in _mesh_model.sections or _mesh_model.sections[_mesh_section_name].material!=_mesh_material_name:\\n        raise RuntimeError(\'Reopened M9 material section mapping changed\')\\n    _mesh_assignments=[assignment.sectionName for assignment in _mesh_part.sectionAssignments]\\n    if _mesh_section_name not in _mesh_assignments:\\n        raise RuntimeError(\'Reopened M9 section assignment missing\')\\n    _mesh_inventory[\'parts\'][_mesh_part_name]={\'nodes\':len(_mesh_part.nodes),\'elements\':len(_mesh_part.elements),\'element_types\':_mesh_types,\'element_set\':_mesh_set_name,\'element_set_count\':len(_mesh_part.sets[_mesh_set_name].elements),\'section_assignments\':_mesh_assignments}\\n    _mesh_inventory[\'instances\'][_mesh_instance_name]={\'part_name\':_mesh_instance.partName,\'nodes\':len(_mesh_instance.nodes),\'elements\':len(_mesh_instance.elements)}\\n    _mesh_inventory[\'sections\'][_mesh_section_name]={\'material\':_mesh_material_name}\\nfor _mesh_mat_name,_mesh_expected_elastic in [(\'Al_matrix\',[[69000.0,0.33]]),(\'SiC\',[[450000.0,0.17]])]:\\n    if _mesh_mat_name not in _mesh_model.materials:\\n        raise RuntimeError(\'Reopened M9 fixed material missing\')\\n    _mesh_mat=_mesh_model.materials[_mesh_mat_name]\\n    _mesh_elastic=[[float(value) for value in row] for row in _mesh_mat.elastic.table]\\n    if _mesh_elastic!=_mesh_expected_elastic:\\n        raise RuntimeError(\'Reopened M9 immutable material elastic constants changed\')\\n    _mesh_inventory[\'materials\'][_mesh_mat_name]={\'elastic_table\':_mesh_elastic}\\n_mesh_al=_mesh_model.materials[\'Al_matrix\']\\n_mesh_plastic=[[float(value) for value in row] for row in _mesh_al.plastic.table]\\nif _mesh_plastic!=[[503.0,193.0,0.45,0.0,0.0,0.0]] or str(_mesh_al.plastic.hardening)!=\'JOHNSON_COOK\':\\n    raise RuntimeError(\'Reopened M9 original Al plastic declaration changed\')\\n_mesh_inventory[\'materials\'][\'Al_matrix\'][\'plastic\']={\'hardening\':str(_mesh_al.plastic.hardening),\'table\':_mesh_plastic}\\n_fe_phase_end(\'REOPEN_CAE\')\\n_fe_phase_begin(\'CLOSE_REOPENED_CAE\')\\nmdb.close()\\n_mesh_closed=True\\n_mesh_reopen_observer.update(closed=True,status=\'COPY_CLOSED_AFTER_INVENTORY\')\\n_mesh_reopen_observer[\'copy_after_close_sha256\']=_mesh_file_sha(_mesh_reopen_copy)\\n_mesh_reopen_observer[\'copy_after_close_size_bytes\']=os.path.getsize(_mesh_reopen_copy)\\n_mesh_reopen_observer[\'copy_bytes_changed_on_open_close\']=_mesh_reopen_observer[\'copy_after_close_sha256\']!=_mesh_reopen_observer[\'copy_before_open_sha256\'] or _mesh_reopen_observer[\'copy_after_close_size_bytes\']!=_mesh_reopen_observer[\'copy_before_open_size_bytes\']\\n_mesh_reopen_observer[\'primary_after_close_sha256\']=_mesh_file_sha(_mesh_success_cae)\\n_mesh_reopen_observer[\'primary_after_close_size_bytes\']=os.path.getsize(_mesh_success_cae)\\nif _mesh_reopen_observer[\'primary_after_close_sha256\']!=_mesh_saved_sha or _mesh_reopen_observer[\'primary_after_close_size_bytes\']!=_mesh_saved_size:\\n    raise RuntimeError(\'Read-only reopen inspection changed saved CAE bytes\')\\n_mesh_reopen_observer.update(original_identity_guard_pass=True,status=\'PRIMARY_PRESERVED_COPY_INSPECTION_CLOSED\')\\n_fe_phase_end(\'CLOSE_REOPENED_CAE\')\\n_fe_phase_begin(\'WRITE_COMPLETION\')\\nwith open(INPUT_JSON_PATH,\'r\',encoding=\'utf-8-sig\') as _mesh_input_handle:\\n    _mesh_input_box=json.load(_mesh_input_handle)[\'box\']\\nwith open(_mesh_settings[\'particles_work_json\'],\'r\',encoding=\'utf-8-sig\') as _mesh_work_handle:\\n    _mesh_actual_parameters=_validate_actual_mesh_parameters(json.load(_mesh_work_handle),_mesh_target_ratio,_mesh_input_box,_mesh_settings[\'node_match_tol\'])\\n_mesh_completion={\'schema\':_mesh_settings[\'completion_schema\'],\'status\':\'SUCCESS\',\'native_m9_completed\':True,\'official_cae_saved\':True,\'database_closed\':True,\'independent_cae_reopen_validation\':True,\'reopened_database_closed\':True,\'stop_after_module\':\'M9\',\'final_instances\':[\'I_MAT_TET\',\'I_INC_TET\'],\'completed_phases\':list(_mesh_completed_phases),\'official_cae_path\':_mesh_success_cae,\'official_cae_size_bytes\':os.path.getsize(_mesh_success_cae),\'official_cae_sha256\':_mesh_file_sha(_mesh_success_cae),\'particles_work_json_path\':_mesh_settings[\'particles_work_json\'],\'particles_work_json_sha256\':_mesh_file_sha(_mesh_settings[\'particles_work_json\']),\'actual_mesh_parameters\':_mesh_actual_parameters,\'reopen_inventory\':_mesh_inventory,\'reopen_copy_observer\':_mesh_reopen_observer}\\n_mesh_completion.update(_mesh_bound_fields())\\n_mesh_write_json(_mesh_receipt_path,_mesh_completion)\\n_fe_phase_end(\'WRITE_COMPLETION\')\\nprint(\'[MESH_ONLY_SUCCESS] native M9 saved, closed, independently reopened and closed:\',_mesh_success_cae,flush=True)\\n_mesh_rc=0\\n"\n'
    '    if body.count("\'SEC_MATRIX\'") != 1 or body.count("\'SEC_SIC\'") != 1:\n'
    "        raise ValueError('Ambiguous reopen section identity placeholders')\n"
    '    body = body.replace("\'SEC_MATRIX\'", repr(section_binding[\'sec_mat\'])).replace("\'SEC_SIC\'", repr(section_binding[\'sec_inc\']))\n'
    "    return '_mesh_expected_sections = ' + repr(section_binding) + '\\n' + body\n"
    '\n'
    'def _terminal_exception_tail():\n'
    '    return "\\nexcept BaseException as _mesh_exception:\\n    try:\\n        _mesh_record_failure(_mesh_exception)\\n    except BaseException:\\n        print(\'[MESH_FAILURE_RECEIPT_WRITE_FAILED]\', flush=True)\\n        traceback.print_exc()\\nfinally:\\n    if not _mesh_closed and \'mdb\' in globals():\\n        try:\\n            mdb.close()\\n        except BaseException:\\n            traceback.print_exc()\\nsys.exit(_mesh_rc)\\n"\n'
    '\n'
    'def _instrument_main(main):\n'
    '    tree = ast.parse(main)\n'
    '    function = tree.body[0]\n'
    '    body = function.body\n'
    '\n'
    '    def calls(statement, name):\n'
    '        return any((isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == name) for node in ast.walk(statement)))\n'
    '\n'
    '    def is_read_guard(statement):\n'
    "        return isinstance(statement, ast.If) and any((isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (node.func.attr == 'isfile') and any((isinstance(arg, ast.Name) and arg.id == 'PARTICLES_WORK_JSON' for arg in node.args)) for node in ast.walk(statement.test)))\n"
    '\n'
    '    def is_model_guard(statement):\n'
    "        return isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare) and isinstance(statement.test.left, ast.Name) and (statement.test.left.id == 'MODEL_NAME') and any((isinstance(op, ast.NotIn) for op in statement.test.ops))\n"
    "    starts = {'M1_PREPROCESS': 0}\n"
    '    reads = [i for i, node in enumerate(body) if is_read_guard(node)]\n'
    '    if not reads:\n'
    "        raise ValueError('Missing M3 input guard')\n"
    "    starts['M3'] = reads[0]\n"
    '    guards = [i for i, node in enumerate(body) if is_model_guard(node)]\n'
    '    if len(guards) != 3:\n'
    "        raise ValueError('Expected ordered native M2/M4/M5 model guards')\n"
    "    starts['M2'], starts['M4'], starts['M5'] = guards\n"
    "    inst_indices = [i for i, node in enumerate(body) if isinstance(node, ast.Assign) and any((isinstance(target, ast.Name) and target.id == 'inst' for target in node.targets)) and isinstance(node.value, ast.Subscript) and isinstance(node.value.slice, ast.Name) and (node.value.slice.id == 'MERGED_INSTANCE_NAME')]\n"
    '    if len(inst_indices) != 1 or inst_indices[0] < 2:\n'
    "        raise ValueError('Missing structural M6 assembly setup')\n"
    "    starts['M6'] = inst_indices[0] - 2\n"
    "    wanted = {'run_module7': 'M7', 'm8a_collect_registry_and_crop': 'M8A', 'm8p_make_periodic_shell': 'M8P', 'm8b_split_shells': 'M8B', 'm8c_flip_matrix_iface': 'M8C', 'm9_make_tet_and_assign': 'M9'}\n"
    '    for name, phase in wanted.items():\n'
    '        selected = [i for i, node in enumerate(body) if calls(node, name)]\n'
    '        if len(selected) != 1:\n'
    "            raise ValueError('Ambiguous native phase call: ' + phase)\n"
    '        starts[phase] = selected[0]\n'
    "    starts['M9'] = starts['M8C'] + 1\n"
    '    if [phase for phase, _ in sorted(starts.items(), key=lambda item: item[1])] != list(NATIVE_PHASES) or len(set(starts.values())) != len(NATIVE_PHASES):\n'
    "        raise ValueError('Structural native FE phase order differs')\n"
    '    result = []\n'
    '    previous = None\n'
    '    by_index = {index: phase for phase, index in starts.items()}\n'
    '    for i, statement in enumerate(body):\n'
    '        if i in by_index:\n'
    '            phase = by_index[i]\n'
    '            if previous is not None:\n'
    "                result += ast.parse('_fe_phase_end(%r)' % previous).body\n"
    "            result += ast.parse('_fe_phase_begin(%r)' % phase).body\n"
    '            previous = phase\n'
    '        result.append(statement)\n'
    '    result += ast.parse("_fe_phase_end(\'M9\')").body\n'
    '    function.body = result\n'
    "    return ast.unparse(ast.fix_missing_locations(tree)) + '\\n'\n"
    '\n'
    'def _native_ratio_override(src, ratio, helper_src):\n'
    '    if ratio is None:\n'
    '        return src\n'
    '    tree = ast.parse(src)\n'
    "    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'run_module6_v3')]\n"
    '    if len(calls) != 1:\n'
    "        raise ValueError('Cannot bind unique native M6 invocation')\n"
    "    keywords = [kw for kw in calls[0].keywords if kw.arg == 'mesh_target_ratio']\n"
    '    if len(keywords) != 1:\n'
    "        raise ValueError('Cannot bind unique mesh ratio keyword')\n"
    "    keywords[0].value = ast.Constant(float('%.12g' % float(ratio)))\n"
    "    return ast.unparse(ast.fix_missing_locations(tree)) + '\\n'\n"
    '\n'
    'def _script_mesh_ratio(src):\n'
    '    tree = ast.parse(src)\n'
    "    mains = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'main']\n"
    '    if len(mains) != 1:\n'
    "        raise ValueError('Cannot bind unique generated main')\n"
    "    calls = [node for node in ast.walk(mains[0]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'run_module6_v3')]\n"
    '    if len(calls) != 1:\n'
    "        raise ValueError('Cannot bind unique generated native M6 invocation')\n"
    "    keywords = [kw for kw in calls[0].keywords if kw.arg == 'mesh_target_ratio']\n"
    '    if len(keywords) != 1:\n'
    "        raise ValueError('Generated native M6 needs exactly one mesh_target_ratio keyword')\n"
    '    value = ast.literal_eval(keywords[0].value)\n'
    '    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value <= 0)):\n'
    "        raise ValueError('Generated native M6 mesh_target_ratio must be a finite positive literal')\n"
    '    return value\n'
    '\n'
    'def _verify_prepared_mesh_parameters_base(prepared):\n'
    "    _bound_sources(prepared['fe_root'])\n"
    "    script = Path(prepared['script_path'])\n"
    "    run_info = Path(prepared['run_info_path'])\n"
    "    work_dir = Path(prepared['cwd']).resolve()\n"
    '    if not script.resolve().is_relative_to(work_dir) or not run_info.resolve().is_relative_to(work_dir):\n'
    "        raise ValueError('Prepared mesh script/record outside attempt work directory')\n"
    "    if not script.name.endswith('_build_run.py'):\n"
    "        raise ValueError('Prepared mesh script name does not bind a build invocation')\n"
    "    if _sha(script) != prepared['script_sha256'] or _sha(run_info) != prepared['run_info_sha256']:\n"
    "        raise ValueError('Prepared mesh script or run_info integrity mismatch')\n"
    "    actual = _script_mesh_ratio(script.read_text(encoding='utf-8-sig'))\n"
    "    requested = prepared['mesh_target_ratio_call_literal']\n"
    "    effective = prepared['mesh_target_ratio']\n"
    '    if actual != requested:\n'
    "        raise ValueError('Generated M6 mesh_target_ratio does not equal requested literal')\n"
    "    native_defaults = inspect_native_mesh_defaults(prepared['fe_root'])\n"
    "    expected_effective = native_defaults['mesh_target_ratio_effective'] if requested is None else requested\n"
    '    if effective != expected_effective:\n'
    "        raise ValueError('Prepared effective mesh ratio differs from the requested/default value')\n"
    "    record = json.loads(run_info.read_text(encoding='utf-8-sig'))\n"
    "    fields = {'schema': 'pw-rve-m9-run-info-v1', 'mesh_target_ratio': effective, 'mesh_target_ratio_call_literal': requested, 'stop_after_module': 'M9', 'fresh_model': True, 'reuse_cae': False, 'reuse_mesh': False, 'source_model_sha256': MODEL_SHA256, 'source_helper_sha256': HELPER_SHA256, 'input_json_sha256': prepared['json_sha256'], 'script_sha256': prepared['script_sha256'], 'bootstrap_imports': ['abaqus', 'caeModules', 'part'], 'native_phase_order': list(NATIVE_PHASES), 'm9_section_binding': prepared['m9_section_binding']}\n"
    '    if any((key not in record or record[key] != value or (isinstance(value, bool) and type(record[key]) is not bool) for key, value in fields.items())):\n'
    "        raise ValueError('run_info does not bind the requested native mesh configuration')\n"
    "    for field, expected in (('input_json_path', prepared['json_path']), ('script_path', prepared['script_path'])):\n"
    "        if Path(record.get(field, '')).resolve() != Path(expected).resolve():\n"
    "            raise ValueError('run_info path mismatch: ' + field)\n"
    "    for field in ('particles_work_json', 'failure_receipt_path', 'module_events_path'):\n"
    '        path = Path(prepared[field]).resolve()\n'
    "        if not path.is_relative_to(work_dir) or Path(record.get(field, '')).resolve() != path:\n"
    "            raise ValueError('run_info output path mismatch: ' + field)\n"
    "    if _sha(prepared['json_path']) != prepared['json_sha256']:\n"
    "        raise ValueError('Prepared mesh input changed')\n"
    "    return {'status': 'PASS', 'mesh_target_ratio': effective, 'mesh_target_ratio_call_literal': actual, 'run_info_path': str(run_info), 'run_info_sha256': prepared['run_info_sha256'], 'script_sha256': prepared['script_sha256'], 'source_model_sha256': MODEL_SHA256, 'source_helper_sha256': HELPER_SHA256, 'read_only_verification': True}\n"
    '\n'
    'def prepare_mesh_only(*, fe_root, json_path, work_dir, success_cae, receipt_path, abaqus_cmd, debug_cae=None, mesh_target_ratio=None, particles_work_json=None, node_match_tol=None):\n'
    '    root, src, pipeline_src = _bound_sources(fe_root)\n'
    '    if mesh_target_ratio is not None:\n'
    '        if isinstance(mesh_target_ratio, bool):\n'
    "            raise ValueError('mesh_target_ratio must be a finite positive number')\n"
    '        mesh_target_ratio = float(mesh_target_ratio)\n'
    '        if not math.isfinite(mesh_target_ratio) or mesh_target_ratio <= 0:\n'
    "            raise ValueError('mesh_target_ratio must be a finite positive number')\n"
    '    json_path = Path(json_path).resolve()\n'
    '    if not json_path.is_file():\n'
    "        raise ValueError('Missing caller-supplied geometry JSON: %s' % json_path)\n"
    '    work_dir = Path(work_dir).resolve()\n'
    '    success_cae = Path(success_cae).resolve()\n'
    '    receipt_path = Path(receipt_path).resolve()\n'
    '    debug_cae = None if debug_cae is None else Path(debug_cae).resolve()\n'
    "    script_path = work_dir / 'geometry_build_run.py'\n"
    "    run_info_path = work_dir / 'run_info.json'\n"
    "    failure_receipt_path = work_dir / 'mesh_failure.json'\n"
    "    module_events_path = work_dir / 'module_events.jsonl'\n"
    "    particles_work_json = work_dir / 'particles_work.json' if particles_work_json is None else Path(particles_work_json).resolve()\n"
    "    outputs = (success_cae, receipt_path, script_path, run_info_path, particles_work_json, failure_receipt_path, module_events_path, work_dir / 'error_traceback.txt', debug_cae)\n"
    '    nonempty_outputs = [path for path in outputs if path is not None]\n'
    '    if len(nonempty_outputs) != len(set(nonempty_outputs)) or json_path in nonempty_outputs:\n'
    "        raise ValueError('FE output paths must be distinct and cannot overwrite the input JSON')\n"
    '    for path in outputs:\n'
    '        if path is None:\n'
    '            continue\n'
    '        if not path.is_relative_to(work_dir):\n'
    "            raise ValueError('FE attempt output outside work directory: %s' % path)\n"
    '        if path.exists():\n'
    "            raise ValueError('FE attempt output already exists: %s' % path)\n"
    "    if success_cae.suffix.lower() != '.cae':\n"
    "        raise ValueError('Official success path must end in .cae')\n"
    "    if debug_cae is not None and (debug_cae == success_cae or debug_cae.suffix.lower() != '.cae'):\n"
    "        raise ValueError('Debug CAE must have a distinct .cae destination')\n"
    '    work_dir.mkdir(parents=True, exist_ok=True)\n'
    '    success_cae.parent.mkdir(parents=True, exist_ok=True)\n'
    '    receipt_path.parent.mkdir(parents=True, exist_ok=True)\n'
    '    if debug_cae is not None:\n'
    '        debug_cae.parent.mkdir(parents=True, exist_ok=True)\n'
    '    particles_work_json.parent.mkdir(parents=True, exist_ok=True)\n'
    '    prefix, main = _native_segments(src)\n'
    '    m9_section_binding = _native_m9_section_binding(main)\n'
    '    prefix = _configure_json_paths(prefix, json_path, particles_work_json)\n'
    '    main = _native_ratio_override(main, mesh_target_ratio, pipeline_src)\n'
    "    input_box = json.loads(json_path.read_text(encoding='utf-8-sig'))['box']\n"
    '    effective_node_match_tol = 1e-06 * max(input_box) if node_match_tol is None else float(node_match_tol)\n'
    '    if isinstance(node_match_tol, bool) or not math.isfinite(effective_node_match_tol) or effective_node_match_tol <= 0:\n'
    "        raise ValueError('node_match_tol must be a finite positive coordinate length')\n"
    '    main = _native_node_match_override(main, node_match_tol)\n'
    '    main = _instrument_main(main)\n'
    '    native_mesh_defaults = inspect_native_mesh_defaults(root)\n'
    "    effective_ratio = native_mesh_defaults['mesh_target_ratio_effective'] if mesh_target_ratio is None else mesh_target_ratio\n"
    "    native_mesh_defaults.update({'override_applied': mesh_target_ratio is not None, 'mesh_target_ratio_effective': effective_ratio, 'mesh_target_ratio_rendered_call_literal': mesh_target_ratio})\n"
    '    instances = "\\n    # M9 output interface: only the two native tet parts need final instances.\\n    _fe_phase_begin(\'FINAL_INSTANCES\')\\n    a = mdb.models[\'RVE\'].rootAssembly\\n    if \'I_MAT_TET\' not in a.instances:\\n        a.Instance(name=\'I_MAT_TET\', part=p_mat_tet, dependent=ON)\\n    if \'I_INC_TET\' not in a.instances:\\n        a.Instance(name=\'I_INC_TET\', part=p_inc_tet, dependent=ON)\\n    a.regenerate()\\n    _fe_phase_end(\'FINAL_INSTANCES\')\\n\\n"\n'
    '    prelude = _runtime_prelude(success_cae, receipt_path, debug_cae, script_path, run_info_path, effective_ratio, failure_receipt_path, module_events_path, json_path=json_path, requested_ratio=mesh_target_ratio, particles_work_json=particles_work_json, node_match_tol=effective_node_match_tol)\n'
    '    body = "_fe_phase_begin(\'BOOTSTRAP\')\\n" + BOOTSTRAP_IMPORTS + prefix + main + instances + _terminal_body(m9_section_binding)\n'
    "    rendered = prelude + '\\ntry:\\n' + textwrap.indent(body, '    ') + _terminal_exception_tail()\n"
    '    rendered = _public_source(rendered)\n'
    '    tree = ast.parse(rendered)\n'
    '    for node in ast.walk(tree):\n'
    '        if isinstance(node, ast.Call):\n'
    "            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''\n"
    "            if name.startswith(('m10_', 'm11_', 'm12_')) or name in ('Job', 'submit', 'waitForCompletion', 'openOdb', 'Odb'):\n"
    "                raise ValueError('Forbidden post-M9 invocation in rendered script: ' + name)\n"
    "    with script_path.open('x', encoding='utf-8', newline='') as handle:\n"
    '        handle.write(rendered)\n'
    "    prepared = {'script_path': str(script_path), 'script_sha256': _sha(script_path), 'command': [str(abaqus_cmd), 'cae', 'noGUI=' + str(script_path)], 'cwd': str(work_dir), 'receipt_path': str(receipt_path), 'failure_receipt_path': str(failure_receipt_path), 'module_events_path': str(module_events_path), 'success_cae': str(success_cae), 'debug_cae': None if debug_cae is None else str(debug_cae), 'json_path': str(json_path), 'json_sha256': _sha(json_path), 'fe_root': str(root), 'particles_work_json': str(particles_work_json), 'native_mesh_defaults': native_mesh_defaults, 'mesh_target_ratio': effective_ratio, 'mesh_target_ratio_call_literal': mesh_target_ratio, 'run_info_path': str(run_info_path), 'stop_after_module': 'M9', 'dispatch_started': False, 'node_match_tol': effective_node_match_tol, 'node_match_tol_call_literal': node_match_tol, 'm9_section_binding': m9_section_binding}\n"
    "    record = {'schema': 'pw-rve-m9-run-info-v1', 'mesh_target_ratio': effective_ratio, 'mesh_target_ratio_call_literal': mesh_target_ratio, 'stop_after_module': 'M9', 'fresh_model': True, 'reuse_cae': False, 'reuse_mesh': False, 'source_model_sha256': MODEL_SHA256, 'source_helper_sha256': HELPER_SHA256, 'input_json_path': str(json_path), 'input_json_sha256': prepared['json_sha256'], 'script_path': str(script_path), 'script_sha256': prepared['script_sha256'], 'particles_work_json': str(particles_work_json), 'failure_receipt_path': str(failure_receipt_path), 'module_events_path': str(module_events_path), 'bootstrap_imports': ['abaqus', 'caeModules', 'part'], 'native_phase_order': list(NATIVE_PHASES), 'node_match_tol': effective_node_match_tol, 'node_match_tol_call_literal': node_match_tol, 'm9_section_binding': m9_section_binding}\n"
    "    with run_info_path.open('x', encoding='utf-8') as handle:\n"
    '        json.dump(record, handle, indent=2, sort_keys=True)\n'
    "    prepared['run_info_sha256'] = _sha(run_info_path)\n"
    "    prepared['parameter_verification'] = verify_prepared_mesh_parameters(prepared)\n"
    '    return prepared\n'
    '\n'
    'def _native_classifiers(fe_root, text):\n'
    '    _, _, src = _bound_sources(fe_root)\n'
    "    wanted = {'_classify_regen_required', '_classify_abaqus_lock_failure'}\n"
    '    nodes = [node for node in ast.parse(src).body if isinstance(node, ast.FunctionDef) and node.name in wanted]\n'
    '    if {node.name for node in nodes} != wanted:\n'
    "        raise ValueError('Missing native FE failure classifier')\n"
    "    namespace = {'Path': Path, 're': re, '_read_text_if_exists': lambda path: text if Path(path).name == 'abaqus_build.log' else '', '_read_abaqus_rpy_text': lambda *args, **kwargs: ''}\n"
    "    code = compile(ast.Module(body=nodes, type_ignores=[]), 'bound-native-classifiers', 'exec')\n"
    '    exec(code, namespace)\n'
    "    return (namespace['_classify_regen_required'](Path('unused')), namespace['_classify_abaqus_lock_failure'](Path('unused')))\n"
    '\n'
    'def classify_mesh_failure_text(fe_root, text):\n'
    '    text = str(text)\n'
    '    lower = text.lower()\n'
    "    license_failures = ('\\\\blicen[cs]e\\\\s+check[\\\\s-]?out\\\\s+(?:has\\\\s+)?(?:failed|failure|error|denied|unavailable)\\\\b', '\\\\b(?:failed|unable|cannot|could not)\\\\s+(?:to\\\\s+)?(?:check[\\\\s-]?out|obtain|acquire)[^\\\\r\\\\n]*\\\\blicen[cs]e\\\\b', '\\\\bno\\\\s+licen[cs]es?(?:\\\\s+(?:are|were))?\\\\s+(?:available|found)\\\\b', '\\\\blicen[cs]e(?:\\\\s+feature)?\\\\s+(?:is\\\\s+)?(?:unavailable|not available|denied)\\\\b', '\\\\blicens(?:e|ing)\\\\s+error\\\\b', '\\\\blicensed number of users[^\\\\r\\\\n]*(?:reached|exceeded|maximum)\\\\b', '\\\\bflex(?:net|lm)(?:\\\\s+licensing)?\\\\s+(?:error|failure|failed)\\\\b', '\\\\bflex(?:net|lm)[^\\\\r\\\\n]*[:=]\\\\s*-\\\\d+\\\\b', '\\\\b(?:cannot|unable to|could not)\\\\s+connect[^\\\\r\\\\n]*\\\\blicen[cs]e\\\\s+(?:server|manager)\\\\b', '\\\\blicen[cs]e\\\\s+(?:server|manager)[^\\\\r\\\\n]*(?:unavailable|not running|connection refused|unable to connect)\\\\b')\n"
    "    vetoes = (('python_environment', ('modulenotfounderror', 'importerror', 'syntaxerror', 'nameerror', 'dll load failed', 'fatal python error')), ('permissions', ('permissionerror', 'permission denied', 'access is denied')), ('process_crash', ('segmentation fault', 'access violation', 'kernel exited unexpectedly', 'unhandled exception')), ('missing_environment', ('is not recognized as an internal', 'no such file or directory', 'filenotfounderror', 'cannot find the file')))\n"
    '    (required, reason), lock_failure = _native_classifiers(fe_root, text)\n'
    "    if lock_failure or ('lock file' in lower and ('exists' in lower or 'cannot' in lower)):\n"
    "        return {'status': 'BLOCKED_abaqus_lock', 'reason': 'abaqus_lock', 'retryable': False}\n"
    '    if any((re.search(pattern, lower) for pattern in license_failures)):\n'
    "        return {'status': 'BLOCKED_license', 'reason': 'license', 'retryable': False}\n"
    '    for reason_veto, patterns in vetoes:\n'
    '        if any((pattern in lower for pattern in patterns)):\n'
    "            return {'status': 'BLOCKED_' + reason_veto, 'reason': reason_veto, 'retryable': False}\n"
    "    if required and (reason == 'main_plane_orphan_free_edge' or reason.startswith('m8p_')):\n"
    "        return {'status': 'RETRYABLE_FAILURE', 'reason': reason, 'retryable': True, 'classifier': 'native_FE_regeneration_trigger'}\n"
    "    return {'status': 'BLOCKED_unclassified_fe_failure', 'reason': reason or 'unclassified_fe_failure', 'retryable': False, 'classifier': 'native_FE_regeneration_trigger'}\n"
    '\n'
    'def assess_mesh_result(prepared, returncode, log_text=None):\n'
    "    work_dir = Path(prepared['cwd'])\n"
    '    chunks = [] if log_text is None else [str(log_text)]\n'
    "    for path in [work_dir / 'abaqus_build.log', work_dir / 'error_traceback.txt'] + sorted(work_dir.glob('abaqus.rpy*')):\n"
    '        if path.is_file():\n'
    "            chunks.append(path.read_text(encoding='utf-8', errors='replace'))\n"
    "    text = '\\n'.join(chunks)\n"
    "    if list(work_dir.rglob('*.odb')):\n"
    "        return {'status': 'BLOCKED_forbidden_odb_output', 'reason': 'forbidden_odb_output', 'retryable': False}\n"
    '    try:\n'
    '        verify_prepared_mesh_parameters(prepared)\n'
    '    except (OSError, ValueError, TypeError, KeyError) as exc:\n'
    "        return {'status': 'BLOCKED_mesh_parameter_binding', 'reason': 'mesh_parameter_binding', 'retryable': False, 'detail': str(exc)}\n"
    "    marker = Path(prepared['receipt_path'])\n"
    "    failure_marker = Path(prepared['failure_receipt_path'])\n"
    '    if marker.is_file() and failure_marker.is_file():\n'
    "        return {'status': 'BLOCKED_conflicting_native_receipts', 'reason': 'conflicting_native_receipts', 'retryable': False}\n"
    '    if failure_marker.is_file():\n'
    '        try:\n'
    "            failure = json.loads(failure_marker.read_text(encoding='utf-8-sig'))\n"
    '            _verify_failure_record(prepared, failure)\n'
    '        except (OSError, ValueError, TypeError, KeyError) as exc:\n'
    "            return {'status': 'BLOCKED_invalid_native_failure', 'reason': 'invalid_native_failure', 'retryable': False, 'detail': str(exc)}\n"
    "        classified = classify_mesh_failure_text(prepared['fe_root'], text + '\\n' + failure['traceback'])\n"
    "        if classified['reason'] == 'unclassified_fe_failure':\n"
    "            classified.update(status='BLOCKED_native_fe_failure', reason='native_fe_failure')\n"
    "        classified.update(failed_phase=failure['failed_phase'], exception_type=failure['exception_type'], exception_message=failure['exception_message'], traceback=failure['traceback'], failure_receipt=str(failure_marker), receipt=failure, outer_returncode=int(returncode), expected_script_exit_code=1)\n"
    '        return classified\n'
    '    if int(returncode) != 0:\n'
    "        return classify_mesh_failure_text(prepared['fe_root'], text)\n"
    '    if not marker.is_file():\n'
    "        failure = classify_mesh_failure_text(prepared['fe_root'], text)\n"
    "        if failure['retryable'] or failure['reason'] != 'unclassified_fe_failure':\n"
    '            return failure\n'
    "        return {'status': 'BLOCKED_missing_native_completion', 'reason': 'missing_native_completion', 'retryable': False}\n"
    '    try:\n'
    "        data = json.loads(marker.read_text(encoding='utf-8-sig'))\n"
    "        required = {'schema': RECEIPT_SCHEMA, 'status': 'SUCCESS', 'native_m9_completed': True, 'official_cae_saved': True, 'database_closed': True, 'independent_cae_reopen_validation': True, 'stop_after_module': 'M9', 'final_instances': ['I_MAT_TET', 'I_INC_TET'], 'input_json_sha256': prepared['json_sha256'], 'script_sha256': prepared['script_sha256'], 'run_info_sha256': prepared['run_info_sha256'], 'mesh_target_ratio': prepared['mesh_target_ratio'], 'mesh_target_ratio_call_literal': prepared['mesh_target_ratio_call_literal'], 'completed_phases': ['BOOTSTRAP', *NATIVE_PHASES, 'FINAL_INSTANCES', 'SAVE_CAE', 'CLOSE_CAE', 'REOPEN_CAE', 'CLOSE_REOPENED_CAE']}\n"
    '        if any((key not in data or data[key] != value or (isinstance(value, bool) and type(data[key]) is not bool) for key, value in required.items())):\n'
    "            raise ValueError('Completion fields do not bind the invoked native script')\n"
    "        for field, expected in (('input_json_path', prepared['json_path']), ('script_path', prepared['script_path']), ('run_info_path', prepared['run_info_path']), ('particles_work_json_path', prepared['particles_work_json']), ('official_cae_path', prepared['success_cae'])):\n"
    "            if Path(data.get(field, '')).resolve() != Path(expected).resolve():\n"
    "                raise ValueError('Completion path mismatch: ' + field)\n"
    "        success = Path(prepared['success_cae'])\n"
    '        if not success.is_file() or success.stat().st_size <= 0:\n'
    "            raise ValueError('Official CAE is absent or empty')\n"
    "        if success.stat().st_size != data.get('official_cae_size_bytes') or _sha(success) != data.get('official_cae_sha256'):\n"
    "            raise ValueError('Official CAE integrity mismatch')\n"
    "        if _sha(prepared['script_path']) != prepared['script_sha256'] or _sha(prepared['json_path']) != prepared['json_sha256']:\n"
    "            raise ValueError('Invoked script or input changed')\n"
    "        work_json = Path(prepared['particles_work_json'])\n"
    "        if _sha(work_json) != data.get('particles_work_json_sha256'):\n"
    "            raise ValueError('Native M6 persisted work JSON integrity mismatch')\n"
    "        actual_parameters = _validate_actual_mesh_parameters(json.loads(work_json.read_text(encoding='utf-8-sig')), prepared['mesh_target_ratio'], json.loads(Path(prepared['json_path']).read_text(encoding='utf-8-sig'))['box'], prepared['node_match_tol'])\n"
    "        if data.get('actual_mesh_parameters') != actual_parameters:\n"
    "            raise ValueError('Completion does not bind native M6 persisted mesh parameters')\n"
    '    except (OSError, ValueError, TypeError, KeyError) as exc:\n'
    "        return {'status': 'BLOCKED_invalid_native_completion', 'reason': 'invalid_native_completion', 'retryable': False, 'detail': str(exc)}\n"
    "    return {'status': 'SUCCESS', 'reason': 'native_m9_official_cae_saved_closed', 'retryable': False, 'receipt': data, 'official_success_cae': str(success)}\n"
    '\n'
    'def _verify_failure_record(prepared, record):\n'
    "    required = {'schema': FAILURE_SCHEMA, 'status': 'FAILED', 'official_success': False, 'stop_after_module': 'M9', 'expected_exit_code': 1, 'input_json_sha256': prepared['json_sha256'], 'script_sha256': prepared['script_sha256'], 'run_info_sha256': prepared['run_info_sha256'], 'mesh_target_ratio': prepared['mesh_target_ratio'], 'mesh_target_ratio_call_literal': prepared['mesh_target_ratio_call_literal']}\n"
    '    for key, value in required.items():\n'
    '        if key not in record or record[key] != value or (isinstance(value, bool) and type(record[key]) is not bool):\n'
    "            raise ValueError('Native failure record binding mismatch: ' + key)\n"
    "    if type(record.get('expected_exit_code')) is not int:\n"
    "        raise ValueError('Native failure exit code must be an integer')\n"
    "    for field, expected in (('input_json_path', prepared['json_path']), ('script_path', prepared['script_path']), ('run_info_path', prepared['run_info_path'])):\n"
    "        if Path(record.get(field, '')).resolve() != Path(expected).resolve():\n"
    "            raise ValueError('Native failure path mismatch: ' + field)\n"
    "    phase_order = ['BOOTSTRAP', *NATIVE_PHASES, 'FINAL_INSTANCES', 'SAVE_CAE', 'CLOSE_CAE', 'REOPEN_CAE', 'CLOSE_REOPENED_CAE', 'WRITE_COMPLETION']\n"
    "    completed = record.get('completed_phases')\n"
    '    if not isinstance(completed, list) or completed != phase_order[:len(completed)]:\n'
    "        raise ValueError('Native failure phase ledger differs')\n"
    "    if record.get('failed_phase') not in phase_order or type(record.get('native_m9_completed')) is not bool or record['native_m9_completed'] != ('M9' in completed):\n"
    "        raise ValueError('Native failure phase is missing or inconsistent')\n"
    '    allowed_failure_phases = phase_order[len(completed):len(completed) + 1] + completed[-1:]\n'
    "    if record['failed_phase'] not in allowed_failure_phases:\n"
    "        raise ValueError('Native failure phase does not follow completed phases')\n"
    "    for key in ('official_cae_saved', 'database_closed'):\n"
    '        if type(record.get(key)) is not bool:\n'
    "            raise ValueError('Native failure flag must be a boolean: ' + key)\n"
    "    for key in ('exception_type', 'exception_message', 'traceback'):\n"
    "        if not isinstance(record.get(key), str) or (key != 'exception_message' and (not record[key])):\n"
    "            raise ValueError('Native failure lacks original exception: ' + key)\n"
    "    if record.get('traceback_sha256') != hashlib.sha256(record['traceback'].encode('utf-8')).hexdigest():\n"
    "        raise ValueError('Native failure traceback integrity differs')\n"
    '\n'
    'def _public_source(src):\n'
    '    tree = ast.parse(src)\n'
    '    for node in ast.walk(tree):\n'
    '        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):\n'
    '            node.body = node.body[1:]\n'
    '            if not node.body and (not isinstance(node, ast.Module)):\n'
    '                node.body = [ast.Pass()]\n'
    "    return ast.unparse(ast.fix_missing_locations(tree)) + '\\n'\n"
    '\n'
    'def _native_node_match_override(main, node_match_tol):\n'
    '    if node_match_tol is None:\n'
    '        return main\n'
    '    if isinstance(node_match_tol, bool) or not math.isfinite(float(node_match_tol)) or float(node_match_tol) <= 0:\n'
    "        raise ValueError('node_match_tol must be finite positive')\n"
    '    tree = ast.parse(main)\n'
    "    mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']\n"
    '    if len(mains) != 1:\n'
    "        raise ValueError('Cannot bind unique native main')\n"
    "    sites = [i for i, node in enumerate(mains[0].body) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (node.value.func.id == 'preprocess')]\n"
    '    if len(sites) != 1:\n'
    "        raise ValueError('Cannot bind unique native preprocess result')\n"
    "    value = ast.Assign(targets=[ast.Subscript(value=ast.Name(id='out', ctx=ast.Load()), slice=ast.Constant('tol_pair'), ctx=ast.Store())], value=ast.Constant(float(node_match_tol)))\n"
    '    mains[0].body.insert(sites[0] + 1, value)\n'
    "    return ast.unparse(ast.fix_missing_locations(tree)) + '\\n'\n"
    '\n'
    'def _script_node_match_tolerance(src):\n'
    "    mains = [node for node in ast.walk(ast.parse(src)) if isinstance(node, ast.FunctionDef) and node.name == 'main']\n"
    '    if len(mains) != 1:\n'
    "        raise ValueError('Cannot bind rendered main')\n"
    '    values = []\n'
    '    for node in mains[0].body:\n'
    "        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript) and isinstance(node.targets[0].value, ast.Name) and (node.targets[0].value.id == 'out') and isinstance(node.targets[0].slice, ast.Constant) and (node.targets[0].slice.value == 'tol_pair'):\n"
    '            values.append(ast.literal_eval(node.value))\n'
    '    if len(values) > 1:\n'
    "        raise ValueError('Multiple node matching overrides')\n"
    '    return None if not values else values[0]\n'
    '\n'
    'def verify_prepared_mesh_parameters(prepared):\n'
    '    result = _verify_prepared_mesh_parameters_base(prepared)\n'
    "    requested = prepared['node_match_tol_call_literal']\n"
    "    actual = _script_node_match_tolerance(Path(prepared['script_path']).read_text(encoding='utf-8-sig'))\n"
    "    box = json.loads(Path(prepared['json_path']).read_text(encoding='utf-8-sig'))['box']\n"
    '    expected = 1e-06 * max(box) if requested is None else requested\n'
    "    record = json.loads(Path(prepared['run_info_path']).read_text(encoding='utf-8-sig'))\n"
    "    if actual != requested or prepared['node_match_tol'] != expected or record.get('node_match_tol') != expected or (record.get('node_match_tol_call_literal') != requested):\n"
    "        raise ValueError('Node correspondence tolerance binding mismatch')\n"
    "    result.update(node_match_tol=expected, node_match_tol_call_literal=requested, node_match_tol_unit='input_geometry_coordinate_length', native_field='tol_pair')\n"
    '    return result\n'
)
FE_SOLVE_SOURCE = (
    'from __future__ import annotations\n'
    '\n'
    'import ast\n'
    'import copy\n'
    'import hashlib\n'
    'import json\n'
    'import math\n'
    'from pathlib import Path\n'
    '\n'
    "MODEL_SHA256 = '96f56c421feebd899ecec5b20734a3a536d24748e7120e66d693cd9a785487ad'\n"
    'ROOT_RUN_FE_PATH = None\n'
    "STAGE2_ADAPTER_SHA256 = 'fed88680c15ca2d97d325a8563a0115eb8356dcc5e91002fe07bb54fde0f3938'\n"
    "STAGE2_HELPER_SHA256 = '04f236c1c7b41164415ac2b1006fe065630df940633a0ef1212190b27117a551'\n"
    '\n'
    'def digest(path):\n'
    '    return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n'
    '\n'
    'def bound_sources(fe_root):\n'
    '    root = Path(fe_root).resolve()\n'
    '    sources = {}\n'
    "    for name,wanted in (('602collaspse.py',MODEL_SHA256),('adapter_helpers.py',STAGE2_HELPER_SHA256)):\n"
    '        path = root/name\n'
    '        if not path.is_file() or digest(path)!=wanted:\n'
    "            raise ValueError('Stage 3 public source identity mismatch: '+str(path))\n"
    '        sources[name] = path\n'
    '    return sources\n'
    '\n'
    'def _case_from_args(args):\n'
    '    case_type = args.case_type.upper()\n'
    "    case = {'type': case_type, 'strain_pct': float(args.strain_pct), 'sign': float(args.sign)}\n"
    "    if case_type == 'AXIAL':\n"
    "        case['axial_dir'] = args.axial_dir.upper()\n"
    "    elif case_type == 'SHEAR':\n"
    "        case['shear_plane'] = args.shear_plane.upper()\n"
    '    else:\n'
    "        raise RuntimeError('Unsupported case type: %s' % case_type)\n"
    '    return case\n'
    '\n'
    'def _make_job_name(case, prefix):\n'
    "    sign = 'P' if float(case.get('sign', 1.0)) >= 0.0 else 'N'\n"
    "    sp = ('%.3f' % float(case['strain_pct'])).rstrip('0').rstrip('.').replace('.', 'p')\n"
    "    if case['type'].upper() == 'AXIAL':\n"
    "        return '%s_AXIAL_%s_%s_%s' % (prefix, case['axial_dir'].upper(), sign, sp)\n"
    "    return '%s_SHEAR_%s_%s_%s' % (prefix, case['shear_plane'].upper(), sign, sp)\n"
    '\n'
    'def validate_inputs(stage2_receipt,input_cae,particles_work_json,require_native):\n'
    '    if ROOT_RUN_FE_PATH is None or not Path(ROOT_RUN_FE_PATH).is_file():\n'
    "        raise ValueError('Unified run_fe.py path binding is absent')\n"
    '    driver_sha256 = digest(Path(ROOT_RUN_FE_PATH).resolve())\n'
    '    receipt_path = Path(stage2_receipt).resolve()\n'
    '    cae = Path(input_cae).resolve()\n'
    '    work = Path(particles_work_json).resolve()\n'
    "    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))\n"
    "    if not isinstance(receipt,dict) or receipt.get('schema')!='cbge-tafe-stage2-receipt-v1':\n"
    "        raise ValueError('Stage 3 requires the public Stage 2 receipt schema')\n"
    "    required_identity = {'driver_sha256':driver_sha256,'adapter_sha256':STAGE2_ADAPTER_SHA256,'public_fe_sha256':MODEL_SHA256,'helper_sha256':STAGE2_HELPER_SHA256,'original_fe_sha256':'26b575897547ef45aa9a7de893c5affea2c57efb12c13feb0d2041b6037a2ce7'}\n"
    '    if any(receipt.get(name)!=value for name,value in required_identity.items()):\n'
    "        raise ValueError('Stage 2 receipt does not bind the frozen public Stage 2 identities')\n"
    "    prepared = receipt.get('prepared_fe',{})\n"
    "    if Path(prepared.get('success_cae','')).resolve()!=cae or Path(prepared.get('particles_work_json','')).resolve()!=work:\n"
    "        raise ValueError('CAE and particles work paths must match the Stage 2 receipt')\n"
    "    result = {'stage2_receipt_path':str(receipt_path),'stage2_receipt_sha256':digest(receipt_path),'stage2_status':receipt.get('status'),'input_cae_path':str(cae),'particles_work_json_path':str(work),'input_cae_sha256':digest(cae) if cae.is_file() else None,'particles_work_json_sha256':digest(work) if work.is_file() else None,'native_input_gate_pass':False,'independent_CAE_reopen_validation':False,'gaps':[],'stage2_geometry_input_acceptance':receipt.get('geometry',{}).get('input_acceptance')}\n"
    "    if receipt.get('status')!='NATIVE_M9_CAE_SAVED_CLOSED' or receipt.get('scientific_success_claim') is not True or receipt.get('terminal') is not True or receipt.get('independent_cae_reopen_validation') is not True:\n"
    "        result['gaps'].append('A terminal native Stage 2 M9 saved/closed success receipt is absent')\n"
    '    if not cae.is_file() or cae.stat().st_size<=0:\n'
    "        result['gaps'].append('The nonempty native M9 CAE input is absent')\n"
    '    if not work.is_file():\n'
    "        result['gaps'].append('The native persisted particles_work.json is absent')\n"
    "    if not result['gaps']:\n"
    "        completion_path = Path(prepared.get('receipt_path','')).resolve()\n"
    "        completion = json.loads(completion_path.read_text(encoding='utf-8-sig'))\n"
    "        if completion!=receipt.get('result',{}).get('receipt'):\n"
    "            raise ValueError('Native completion file does not match the Stage 2 assessment')\n"
    "        fields = {'status':'SUCCESS','native_m9_completed':True,'official_cae_saved':True,'database_closed':True,'independent_cae_reopen_validation':True,'reopened_database_closed':True,'stop_after_module':'M9','final_instances':['I_MAT_TET','I_INC_TET']}\n"
    '        if any(completion.get(name)!=value for name,value in fields.items()):\n'
    "            raise ValueError('Native completion does not certify the existing M9 save/close interface')\n"
    "        if completion.get('official_cae_sha256')!=result['input_cae_sha256'] or completion.get('official_cae_size_bytes')!=cae.stat().st_size or completion.get('particles_work_json_sha256')!=result['particles_work_json_sha256']:\n"
    "            raise ValueError('The fixed CAE or work JSON changed after Stage 2 completion')\n"
    "        geometry = receipt['geometry']\n"
    "        accepted=geometry.get('input_acceptance',{})\n"
    "        if accepted.get('native_input_gate_pass') is not True or receipt.get('geometry_acceptance_certificate_unchanged') is not True or digest(accepted['path'])!=accepted['sha256']:\n"
    "            raise ValueError('Stage 2 native input acceptance certificate absent or changed')\n"
    "        geometry_path = Path(geometry['geometry_json_path'])\n"
    "        if digest(geometry_path)!=geometry['geometry_json_sha256'] or receipt.get('input_geometry_unchanged') is not True:\n"
    "            raise ValueError('Stage 2 protected geometry input identity changed')\n"
    "        data = json.loads(work.read_text(encoding='utf-8-sig'))\n"
    "        if not isinstance(data,dict) or not isinstance(data.get('box'),list) or len(data['box'])!=3 or any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0 for value in data['box']):\n"
    "            raise ValueError('Native work JSON box lacks its existing scientific interface')\n"
    "        result.update(native_input_gate_pass=True,independent_CAE_reopen_validation=True,native_completion_path=str(completion_path),native_completion_sha256=digest(completion_path),input_geometry_path=str(geometry_path.resolve()),input_geometry_sha256=geometry['geometry_json_sha256'],box=data['box'],geometry_acceptance_path=accepted['path'],geometry_acceptance_sha256=accepted['sha256'],stage2_actual_mesh_parameters=completion.get('actual_mesh_parameters'),stage2_reopen_inventory=completion.get('reopen_inventory'))\n"
    "    if require_native and not result['native_input_gate_pass']:\n"
    "        raise ValueError('Stage 3 native input gate is blocked: '+'; '.join(result['gaps']))\n"
    '    return result\n'
    '\n'
    'def scientific_kernel(model_path):\n'
    "    tree = ast.parse(Path(model_path).read_text(encoding='utf-8-sig'))\n"
    '    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef)]\n'
    '    definitions = {node.name:node for node in functions}\n'
    "    selected = {name for name in definitions if name.startswith(('_m10_','_m11_','m10_','m11_'))}\n"
    "    if not {'m10_apply_tie_and_pbc','m11_run_case'}<=selected:\n"
    "        raise ValueError('Frozen model lacks native M10/M11 functions')\n"
    '    while True:\n'
    '        referenced = {node.func.id for name in selected for node in ast.walk(definitions[name]) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in definitions}\n'
    '        if referenced<=selected:\n'
    '            break\n'
    '        selected |= referenced\n'
    "    forbidden = {'main','preprocess','run_module6_v3','run_module6_v2','run_module7','m8a_collect_registry_and_crop','m8p_make_periodic_shell','m8b_split_shells','m8c_flip_matrix_iface','m9_make_tet_and_assign'}\n"
    '    if selected & forbidden:\n'
    "        raise ValueError('Stage 3 M10/M11 closure would invoke a geometry or mesh stage')\n"
    '    nodes = []\n'
    '    for node in tree.body:\n'
    "        if isinstance(node,(ast.Import,ast.ImportFrom)) or isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id.startswith('_M10_') for target in node.targets) or isinstance(node,ast.FunctionDef) and node.name in selected:\n"
    '            nodes.append(copy.deepcopy(node))\n'
    "    return ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))+'\\n', [{'name':node.name,'source_line':node.lineno,'source_end_line':node.end_lineno} for node in nodes if isinstance(node,ast.FunctionDef)]\n"
    '\n'
    'def clean_source(text):\n'
    '    tree = ast.parse(text)\n'
    '    for node in ast.walk(tree):\n'
    '        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):\n'
    '            node.body = node.body[1:]\n'
    '            if not node.body and not isinstance(node,ast.Module):\n'
    '                node.body = [ast.Pass()]\n'
    "    return ast.unparse(ast.fix_missing_locations(tree))+'\\n'\n"
    '\n'
    'def prepare(fe_root,output,inputs,case,job_prefix,num_cpus,memory_percent,nout,step_name,inst_mat,inst_inc,set_mat,set_inc,abaqus_command):\n'
    '    sources = bound_sources(fe_root)\n'
    '    output = Path(output).resolve()\n'
    "    kernel,mapping = scientific_kernel(sources['602collaspse.py'])\n"
    '    job_name = _make_job_name(case,job_prefix)\n'
    "    record = {'schema':'cbge-tafe-stage3-run-info-v1','inputs':inputs,'case':case,'job_prefix':job_prefix,'job_name':job_name,'num_cpus':num_cpus,'memory_percent':memory_percent,'nout':nout,'step_name':step_name,'inst_mat':inst_mat,'inst_inc':inst_inc,'set_mat':set_mat,'set_inc':set_inc,'source_files':[{'path':str(path),'sha256':digest(path)} for path in sources.values()],'copied_input_cae':str(output/'input_model.cae'),'copied_work_json':str(output/'particles_work.json'),'solved_cae':str(output/'elastic_model.cae'),'odb_path':str(output/(job_name+'.odb')),'native_solve_receipt':str(output/'solve_completion.json'),'native_failure_receipt':str(output/'native_failure.json'),'scientific_kernel_function_mapping':mapping,'geometry_generation':False,'rebuild_or_remesh':False,'automatic_retry':False,'full_stiffness_tensor_claim':False}\n"
    '    settings = repr(record)\n'
    '    common = """\n'
    'import json\n'
    'import hashlib\n'
    'import os\n'
    'import sys\n'
    'import traceback\n'
    '\n'
    'def _stage3_sha(path):\n'
    "    with open(path,'rb') as handle:\n"
    '        return hashlib.sha256(handle.read()).hexdigest()\n'
    '\n'
    'def _stage3_persist(path,value):\n'
    "    temporary=path+'.tmp'\n"
    "    with open(temporary,'x',encoding='utf-8') as handle:\n"
    '        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False)\n'
    '        handle.flush()\n'
    '        os.fsync(handle.fileno())\n'
    '    os.replace(temporary,path)\n'
    '\n'
    'def _stage3_source_guard():\n'
    "    for source in _stage3_settings['source_files']:\n"
    "        if _stage3_sha(source['path'])!=source['sha256']:\n"
    "            raise RuntimeError('Stage 3 bound source changed: '+source['path'])\n"
    '\n'
    'def _stage3_input_guard():\n'
    "    for field,original_field in (('copied_input_cae','input_cae_sha256'),('copied_work_json','particles_work_json_sha256')):\n"
    "        if _stage3_sha(_stage3_settings[field])!=_stage3_settings['inputs'][original_field]:\n"
    "            raise RuntimeError('Copied Stage 3 fixed input changed: '+field)\n"
    '\n'
    "_stage3_phase='SOURCE_AND_INPUT_GUARDS'\n"
    '_stage3_rc=1\n'
    '\n'
    'def _stage3_boolean_name(value):\n'
    '    if value in (ON,True):\n'
    "        return 'ON'\n"
    '    if value in (OFF,False):\n'
    "        return 'OFF'\n"
    '    return str(value)\n'
    '\n'
    'def _stage3_material_check(model):\n'
    '    inventory={}\n'
    "    for name,wanted in [('Al_matrix',[[69000.0,0.33]]),('SiC',[[450000.0,0.17]])]:\n"
    '        if name not in model.materials:\n'
    "            raise RuntimeError('Original fixed material absent: '+name)\n"
    '        material=model.materials[name]\n'
    '        table=[[float(value) for value in row] for row in material.elastic.table]\n'
    '        if table!=wanted:\n'
    "            raise RuntimeError('Original fixed material elastic constants changed: '+name)\n"
    "        inventory[name]={'elastic_table':table}\n"
    "    plastic=model.materials['Al_matrix'].plastic\n"
    '    table=[[float(value) for value in row] for row in plastic.table]\n'
    "    if table!=[[503.0,193.0,0.45,0.0,0.0,0.0]] or str(plastic.hardening)!='JOHNSON_COOK':\n"
    "        raise RuntimeError('Original Al Johnson-Cook declaration changed')\n"
    "    inventory['Al_matrix']['plastic']={'hardening':str(plastic.hardening),'table':table}\n"
    '    return inventory\n'
    '"""\n'
    '    solve_tail = """\n'
    'from abaqus import openMdb\n'
    'try:\n'
    '    _stage3_source_guard()\n'
    '    _stage3_input_guard()\n'
    "    _stage3_phase='OPEN_EXISTING_CAE'\n"
    "    mdb=openMdb(pathName=_stage3_settings['copied_input_cae'])\n"
    "    if 'RVE' not in mdb.models:\n"
    "        raise RuntimeError('Stage 2 CAE lacks native RVE model')\n"
    "    _stage3_model=mdb.models['RVE']\n"
    "    if any(name not in _stage3_model.parts for name in ('P_MAT_TET','P_INC_TET')) or any(name not in _stage3_model.rootAssembly.instances for name in ('I_MAT_TET','I_INC_TET')):\n"
    "        raise RuntimeError('Stage 2 native M9 tetrahedral interface is missing')\n"
    '    _stage3_materials=_stage3_material_check(_stage3_model)\n'
    "    _stage3_phase='M10_M11_EXISTING_MESH_SOLVE'\n"
    "    _stage3_job=m11_run_case(model_name='RVE',particles_work_json=_stage3_settings['copied_work_json'],case=_stage3_settings['case'],step_name=_stage3_settings['step_name'],numCpus=_stage3_settings['num_cpus'],memoryPercent=_stage3_settings['memory_percent'],Nout=_stage3_settings['nout'],job_prefix=_stage3_settings['job_prefix'],wait=True)\n"
    "    if _stage3_job!=_stage3_settings['job_name']:\n"
    "        raise RuntimeError('Native job name differs from bound case')\n"
    "    if 'M10_TIE_IFACE_INC_TO_MAT' not in _stage3_model.constraints:\n"
    "        raise RuntimeError('Original ideal Tie constraint is absent after native M10')\n"
    "    _stage3_tie=_stage3_model.constraints['M10_TIE_IFACE_INC_TO_MAT']\n"
    "    _stage3_tie_fields={'positionToleranceMethod':str(getattr(_stage3_tie,'positionToleranceMethod','MISSING'))}\n"
    "    _stage3_tie_fields.update({name:_stage3_boolean_name(getattr(_stage3_tie,name,'MISSING')) for name in ('adjust','tieRotations','thickness')})\n"
    "    if _stage3_tie_fields!={'positionToleranceMethod':'COMPUTED','adjust':'OFF','tieRotations':'ON','thickness':'ON'}:\n"
    "        raise RuntimeError('Original ideal Tie settings changed or native attributes unavailable')\n"
    "    _stage3_equations=sorted(name for name in _stage3_model.constraints.keys() if name.startswith('M10_EQ_'))\n"
    "    if not _stage3_equations or _stage3_settings['step_name'] not in _stage3_model.steps:\n"
    "        raise RuntimeError('Original PBC equations/static step absent')\n"
    "    _stage3_step=_stage3_model.steps[_stage3_settings['step_name']]\n"
    "    if _stage3_boolean_name(getattr(_stage3_step,'nlgeom','MISSING'))!='OFF':\n"
    "        raise RuntimeError('Original small-deformation static step changed')\n"
    "    _stage3_inventory={'materials':_stage3_materials,'tie':{'name':'M10_TIE_IFACE_INC_TO_MAT','settings':_stage3_tie_fields},'pbc_equation_count':len(_stage3_equations),'pbc_equation_names':_stage3_equations,'step':{'name':_stage3_settings['step_name'],'native_type':type(_stage3_step).__name__,'nlgeom':_stage3_boolean_name(_stage3_step.nlgeom)},'case':_stage3_settings['case'],'job_status':str(mdb.jobs[_stage3_job].status),'no_geometry_generation_or_remesh':True}\n"
    "    _stage3_phase='SAVE_CLOSE_SOLVED_CAE'\n"
    "    mdb.saveAs(pathName=_stage3_settings['solved_cae'])\n"
    '    mdb.close()\n'
    "    _stage3_phase='SOLVE_RECEIPT'\n"
    "    _stage3_completion={'schema':'cbge-tafe-stage3-native-solve-v1','status':'SOLVE_COMPLETED','job_name':_stage3_job,'odb_path':_stage3_settings['odb_path'],'odb_sha256':_stage3_sha(_stage3_settings['odb_path']),'solved_cae_sha256':_stage3_sha(_stage3_settings['solved_cae']),'input_cae_sha256':_stage3_settings['inputs']['input_cae_sha256'],'particles_work_json_sha256':_stage3_settings['inputs']['particles_work_json_sha256'],'case':_stage3_settings['case'],'script_sha256':_stage3_sha(__file__),'run_info_sha256':_stage3_sha(os.path.join(os.path.dirname(__file__),'run_info.json')),'geometry_generation':False,'rebuild_or_remesh':False}\n"
    "    _stage3_completion['native_model_inventory']=_stage3_inventory\n"
    "    _stage3_persist(_stage3_settings['native_solve_receipt'],_stage3_completion)\n"
    '    _stage3_rc=0\n'
    'except BaseException as exc:\n'
    "    _stage3_persist(_stage3_settings['native_failure_receipt'],{'schema':'cbge-tafe-stage3-native-failure-v1','status':'FAILED','phase':_stage3_phase,'exception_type':type(exc).__name__,'exception_message':str(exc),'traceback':traceback.format_exc(),'scientific_success_claim':False})\n"
    '    traceback.print_exc()\n'
    'finally:\n'
    "    if 'mdb' in globals():\n"
    '        try:\n'
    '            mdb.close()\n'
    '        except BaseException:\n'
    '            pass\n'
    'sys.exit(_stage3_rc)\n'
    '"""\n'
    "    solve = clean_source('_stage3_settings = '+settings+'\\n'+common+kernel+solve_tail)\n"
    "    forbidden = {'main','preprocess','run_module6_v3','run_module6_v2','run_module7','m8a_collect_registry_and_crop','m8p_make_periodic_shell','m8b_split_shells','m8c_flip_matrix_iface','m9_make_tet_and_assign','generateMesh','seedPart','seedEdgeBySize','InstanceFromBooleanMerge'}\n"
    '    for text in (solve,):\n'
    '        for node in ast.walk(ast.parse(text)):\n'
    '            if isinstance(node,ast.Call):\n'
    "                name = node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''\n"
    '                if name in forbidden:\n'
    "                    raise ValueError('Forbidden generation/remesh call in Stage 3 renderer: '+name)\n"
    "    solve_path = output/'elastic_run.py'\n"
    '    for path,text in ((solve_path,solve),):\n'
    "        with path.open('x',encoding='utf-8') as handle:\n"
    '            handle.write(text)\n'
    "    record.update(solve_script=str(solve_path),solve_script_sha256=digest(solve_path),solve_command=[abaqus_command,'cae','noGUI='+str(solve_path)])\n"
    "    run_info = output/'run_info.json'\n"
    "    with run_info.open('x',encoding='utf-8') as handle:\n"
    '        json.dump(record,handle,indent=2,sort_keys=True,allow_nan=False)\n'
    '    record.update(run_info_path=str(run_info),run_info_sha256=digest(run_info))\n'
    '    return record\n'
    '\n'
    'def verify_prepared(prepared,fe_root):\n'
    '    bound_sources(fe_root)\n'
    "    for path_field,hash_field in (('solve_script','solve_script_sha256'),('run_info_path','run_info_sha256')):\n"
    '        if digest(prepared[path_field])!=prepared[hash_field]:\n'
    "            raise ValueError('Prepared Stage 3 invocation identity changed: '+path_field)\n"
    "    for field,path_field in (('input_cae_sha256','input_cae_path'),('particles_work_json_sha256','particles_work_json_path'),('stage2_receipt_sha256','stage2_receipt_path')):\n"
    "        if prepared['inputs'][field] is not None and digest(prepared['inputs'][path_field])!=prepared['inputs'][field]:\n"
    "            raise ValueError('Protected Stage 3 input identity changed: '+path_field)\n"
    "    for path_field,hash_field in (('input_geometry_path','input_geometry_sha256'),('geometry_acceptance_path','geometry_acceptance_sha256')):\n"
    "        if path_field in prepared['inputs'] and digest(prepared['inputs'][path_field])!=prepared['inputs'][hash_field]:\n"
    "            raise ValueError('Protected accepted geometry evidence changed: '+path_field)\n"
    "    return {'status':'PASS','read_only_verification':True,'solve_script_sha256':prepared['solve_script_sha256'],'run_info_sha256':prepared['run_info_sha256']}\n"
    '\n'
    'def assess_solve(prepared):\n'
    "    path = Path(prepared['native_solve_receipt'])\n"
    "    data = json.loads(path.read_text(encoding='utf-8-sig'))\n"
    "    fields = {'schema':'cbge-tafe-stage3-native-solve-v1','status':'SOLVE_COMPLETED','job_name':prepared['job_name'],'case':prepared['case'],'script_sha256':prepared['solve_script_sha256'],'run_info_sha256':prepared['run_info_sha256'],'input_cae_sha256':prepared['inputs']['input_cae_sha256'],'particles_work_json_sha256':prepared['inputs']['particles_work_json_sha256'],'geometry_generation':False,'rebuild_or_remesh':False}\n"
    '    if any(data.get(name)!=value for name,value in fields.items()):\n'
    "        raise ValueError('Native solve receipt identity/configuration mismatch')\n"
    "    if Path(data.get('odb_path','')).resolve()!=Path(prepared['odb_path']).resolve() or digest(prepared['odb_path'])!=data.get('odb_sha256') or digest(prepared['solved_cae'])!=data.get('solved_cae_sha256'):\n"
    "        raise ValueError('Native solve artifacts do not match their bound receipt')\n"
    "    return {'receipt':data,'receipt_sha256':digest(path)}\n"
    '\n'
)
FE_SOURCE_IDENTITIES = {'602collaspse.py': '96f56c421feebd899ecec5b20734a3a536d24748e7120e66d693cd9a785487ad', 'adapter_helpers.py': '04f236c1c7b41164415ac2b1006fe065630df940633a0ef1212190b27117a551', 'mesh_only_adapter.py': 'fed88680c15ca2d97d325a8563a0115eb8356dcc5e91002fe07bb54fde0f3938', 'elastic_adapter.py': '5d57eaf22f9997eece2d0a89e5ca8118f74e39fcc0b4a507dc8abd7754b1bad0'}
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite_positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Value must be finite and positive')
    return result
def tensile_strain(value):
    result = finite_positive(value)
    if result > 0.05:
        raise ValueError('Dimensionless tensile strain must satisfy 0 < strain <= 0.05')
    return result
def resolve_launcher(explicit=None, require_native=False):
    command = explicit or os.environ.get('CBGE_ABAQUS_COMMAND')
    if command is None:
        if require_native:
            raise ValueError('Set CBGE_ABAQUS_COMMAND or the advanced --abaqus-command option to the installed Abaqus launcher')
        return 'abaqus'
    resolved = shutil.which(command) or (str(Path(command).resolve()) if Path(command).is_file() else None)
    if require_native and resolved is None:
        raise ValueError('Configured Abaqus launcher is unavailable')
    return resolved or command
def _bundled_geometry_acceptance(geometry, wanted_hash):
    asset = (ROOT / BUNDLED_GEOMETRY_PATH).resolve()
    if geometry != asset and wanted_hash != BUNDLED_GEOMETRY_SHA256:
        return None
    if wanted_hash != BUNDLED_GEOMETRY_SHA256 or not asset.is_file() or asset.stat().st_size != BUNDLED_GEOMETRY_SIZE or (digest(asset) != BUNDLED_GEOMETRY_SHA256):
        raise ValueError('Bundled accepted geometry identity mismatch')
    certificate = {'schema': 'cbge-tafe-bundled-accepted-geometry-v1', 'geometry_json_sha256': wanted_hash, 'original_geometry_source_sha256': GEOMETRY_SOURCE_SHA256, 'provenance': 'APPROVED_ACCEPTED_GEOMETRY'}
    return {'declaration': False, 'source': 'BUNDLED_ACCEPTED_GEOMETRY_HASH', 'path': str(asset), 'sha256': BUNDLED_GEOMETRY_SHA256, 'geometry_json_sha256': wanted_hash, 'independent_acceptance_verified_by_stage2': False, 'native_input_gate_pass': True, 'certificate': certificate}
def discover_geometry_acceptance(input_json, require_native=True):
    geometry = Path(input_json).resolve()
    wanted_hash = digest(geometry)
    bundled = _bundled_geometry_acceptance(geometry, wanted_hash)
    if bundled is not None:
        return bundled
    certificate = geometry.parent / 'geometry_acceptance.json'
    if not certificate.is_file():
        raise ValueError('Adjacent geometry_acceptance.json is absent; this exact input requires bound independent acceptance evidence before FE')
    receipt = json.loads(certificate.read_text(encoding='utf-8-sig'))
    if receipt.get('schema') != 'cbge-tafe-geometry-acceptance-v2.2' or receipt.get('status') != 'GEOMETRY_ACCEPTED' or receipt.get('terminal_status') != 'GEOMETRY_ACCEPTED' or (receipt.get('accepted_count') != 1):
        raise ValueError('Geometry certificate is not a terminal accepted V2.2 realization')
    if receipt.get('geometry_json_sha256') != wanted_hash:
        raise ValueError('Geometry acceptance certificate does not bind the exact input JSON or all required audits')
    source = receipt.get('source_identity', {})
    if source.get('original_source_identity_sha256') != GEOMETRY_SOURCE_SHA256:
        raise ValueError('Geometry acceptance source identity mismatch')
    manifest_path = ROOT / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if source.get('identity_verified') is not True or source.get('identity_sha256') != manifest.get('identity_sha256') or source.get('manifest_sha256') != digest(manifest_path) or (source.get('source_files_verified') != manifest.get('member_count')):
        raise ValueError('Geometry acceptance source identity does not match the installed software')
    native = receipt.get('scientific_execution') is True and receipt.get('execution_kind') == 'NATIVE_SCIENTIFIC_GENERATION'
    if require_native and (not native):
        raise ValueError('Native FE refuses mock/synthetic geometry acceptance evidence')
    acceptance = receipt.get('acceptance', {})
    if digest(ROOT / 'public_cli/geometry_adapter.py') != GEOMETRY_ACCEPTANCE_ADAPTER_SHA256 or acceptance.get('adapter_sha256') != GEOMETRY_ACCEPTANCE_ADAPTER_SHA256:
        raise ValueError('Geometry acceptance adapter identity mismatch')
    if acceptance.get('metric') != 'native_orientation' or acceptance.get('authority') != 'NATIVE_FINAL_GEOMETRY_AUDITS' or acceptance.get('all_required_audits_passed') is not True:
        raise ValueError('Geometry acceptance lacks the bound native final audits')
    native_audits = acceptance.get('native_final_audits')
    if not isinstance(native_audits, dict) or native_audits.get('all_required_audits_passed') is not True or native_audits.get('physical_pw_passed') is not True or (native_audits.get('strict_periodic_audit_performed') is not True) or (native_audits.get('strict_periodic_passed') is not True):
        raise ValueError('Native physical/periodic acceptance audit is absent or failed')
    for role in ('geometry', 'orientation'):
        required = native_audits.get(role + '_required')
        passed = native_audits.get('geometry_passed' if role == 'geometry' else 'orientation_target_passed')
        if type(required) is not bool or (required and passed is not True):
            raise ValueError('Required native ' + role + ' audit absent or failed')
    data = json.loads(geometry.read_text(encoding='utf-8-sig'))
    summary = data.get('optimization_summary', {})
    if summary.get('final_audits') != native_audits:
        raise ValueError('Acceptance sidecar native final audits differ from the protected raw JSON')
    orientation = summary.get('final_orientation', {})
    native_orientation = {'target': orientation.get('orientation_target'), 'required': native_audits['orientation_required'], 'tolerance': orientation.get('tolerance'), 'max_abs_deviation': orientation.get('max_abs_deviation'), 'is_target_aligned': orientation.get('is_target_aligned')}
    if acceptance.get('native_orientation') != native_orientation:
        raise ValueError('Acceptance sidecar orientation values differ from protected raw JSON')
    return {'declaration': False, 'source': 'AUTO_DISCOVERED_HASH_BOUND_GEOMETRY_ACCEPTANCE', 'path': str(certificate), 'sha256': digest(certificate), 'geometry_json_sha256': wanted_hash, 'independent_acceptance_verified_by_stage2': native, 'native_input_gate_pass': native, 'certificate': receipt}

def _profile_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def runtime_config():
    value = copy.deepcopy(FE_RUNTIME)
    if value.get('schema') != 'cbge-fe-runtime-v2.2' or value.get('mesh_target_ratio_default') != 0.06:
        raise ValueError('Unsupported FE runtime configuration')
    for name in ('num_cpus', 'nout'):
        if type(value.get(name)) is not int or value[name] <= 0:
            raise ValueError('Invalid runtime integer: ' + name)
    if type(value.get('memory_percent')) is not int or not 1 <= value['memory_percent'] <= 100:
        raise ValueError('Invalid runtime memory percentage')
    value.update(profile_origin='EMBEDDED', profile_sha256=_profile_sha(FE_RUNTIME))
    return value

def material_config():
    return {'configuration': copy.deepcopy(FE_MATERIAL), 'sha256': _profile_sha(FE_MATERIAL), 'scope': MATERIAL_SCOPE, 'user_adjustable': False, 'profile_origin': 'EMBEDDED'}
def effective_fe_config(input_json=None, mesh_target_ratio=0.06, node_match_tol=None, solve_elastic=False, load_axis=None, strain=None):
    runtime = runtime_config()
    result = {'schema': 'cbge-fe-effective-config-v2.2', 'mesh_target_ratio': finite_positive(mesh_target_ratio), 'mesh_target_ratio_public_default': 0.06, 'native_mesh_ratio_default_unchanged': 0.04, 'mesh_target_len_formula': 'mesh_target_ratio * max(input_box)', 'node_match_tol_requested': node_match_tol, 'node_match_tol_default_formula': '1e-6 * max(input_box)', 'node_match_tol_unit': 'input coordinate length (um for the reference cases)', 'native_node_match_field': 'tol_pair', 'node_match_tol_role': 'Coordinate correspondence, separate from displacement residual and native COMPUTED tie position tolerance', 'solve_elastic': bool(solve_elastic), 'solver_default_enabled': False, 'material_model': material_config(), 'runtime': runtime, 'scientific_execution': False}
    if node_match_tol is not None:
        result['node_match_tol'] = finite_positive(node_match_tol)
    if input_json is not None:
        path = Path(input_json).resolve()
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        box = data.get('box')
        if not isinstance(box, list) or len(box) != 3 or any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) or (v <= 0) for v in box)):
            raise ValueError('Input JSON must expose a finite positive three-dimensional box')
        result.update(input_json=str(path), input_json_sha256=digest(path), box=box, mesh_target_len=result['mesh_target_ratio'] * max(box), node_match_tol=1e-06 * max(box) if node_match_tol is None else result['node_match_tol'])
    if load_axis is not None or strain is not None:
        if not solve_elastic:
            raise ValueError('Load axis and strain require explicit --solve-elastic')
        if load_axis not in ('x', 'y', 'z') or strain is None:
            raise ValueError('Explicit tensile load axis and dimensionless strain are required')
        strain = tensile_strain(strain)
        result['case'] = {'type': 'AXIAL', 'axial_dir': load_axis.upper(), 'sign': 1.0, 'strain_pct': 100.0 * strain, 'dimensionless_strain': strain}
    elif solve_elastic:
        raise ValueError('--solve-elastic requires explicit --load-axis and --strain')
    return result
def persist(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
def validate_geometry(path, status_path=None):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or data.get('schema') != 'pw85-ellipsoids-export' or data.get('periodic') is not True:
        raise ValueError('Stage 2 requires a periodic pw85-ellipsoids-export JSON')
    box = data.get('box')
    if not isinstance(box, list) or len(box) != 3 or any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) or (v <= 0) for v in box)):
        raise ValueError('Geometry box must contain three finite positive lengths')
    particles = data.get('particles')
    if not isinstance(particles, list) or not particles:
        raise ValueError('Geometry particles must be a nonempty list')
    ids = set()
    for particle in particles:
        if not isinstance(particle, dict) or type(particle.get('id')) is not int or particle['id'] in ids:
            raise ValueError('Each particle requires a distinct integer id')
        ids.add(particle['id'])
        a, b = (particle.get('a'), particle.get('b'))
        if any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) for v in (a, b))) or not a >= b > 0:
            raise ValueError('Particle axes must be finite and satisfy a >= b > 0')
        for name in ('c', 'p'):
            values = particle.get(name)
            if not isinstance(values, list) or len(values) != 3 or any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) for v in values)):
                raise ValueError('Particle %s must be a finite three-vector' % name)
        if sum((float(v) * float(v) for v in particle['p'])) == 0:
            raise ValueError('Particle orientation must be nonzero')
    summary = data.get('optimization_summary', {})
    native_reported = isinstance(summary, dict) and summary.get('success') is True and (summary.get('converged') is True) and (summary.get('generation_success') is True) and (summary.get('resource_status') == 'WITHIN_RESOURCE_BUDGET')
    result = {'geometry_json_path': str(path), 'geometry_json_sha256': digest(path), 'particle_count': len(particles), 'box': box, 'native_reported_success': native_reported, 'independent_geometry_audit_executed': False, 'native_summary': summary, 'native_summary_role': 'TRACE_ONLY_NOT_SCIENTIFIC_ACCEPTANCE'}
    if status_path is not None:
        status_path = Path(status_path).resolve()
        status = json.loads(status_path.read_text(encoding='utf-8-sig'))
        if not isinstance(status, dict) or status.get('geometry_json_sha256') != result['geometry_json_sha256']:
            raise ValueError('Geometry status receipt does not bind the exact input JSON hash')
        result['geometry_status'] = {'path': str(status_path), 'sha256': digest(status_path), 'schema': status.get('schema'), 'status': status.get('status'), 'role': 'SUPPLIED_HASH_BOUND_STATUS_NO_NEW_INDEPENDENT_AUDIT'}
    return result

def _materialize_fe_sources(output):
    root = output / 'runtime_sources'
    root.mkdir(exist_ok=False)
    for name, text in (('602collaspse.py', FE_MODEL_SOURCE), ('adapter_helpers.py', FE_HELPER_SOURCE), ('mesh_only_adapter.py', FE_MESH_SOURCE), ('elastic_adapter.py', FE_SOLVE_SOURCE)):
        raw = text.encode('utf-8')
        if hashlib.sha256(raw).hexdigest() != FE_SOURCE_IDENTITIES[name]:
            raise ValueError('Embedded FE source identity mismatch: ' + name)
        path = root / name
        with path.open('xb') as handle:
            handle.write(raw)
        if digest(path) != FE_SOURCE_IDENTITIES[name]:
            raise ValueError('Materialized FE source identity mismatch: ' + name)
    return root

def _load_adapter(root, filename, module_name):
    path = root / filename
    if digest(path) != FE_SOURCE_IDENTITIES[filename]:
        raise ValueError('FE adapter identity mismatch')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
def _execute_mesh(prepared, receipt, output, input_json, status_path, adapter):
    phase = 'VERIFY_DISPATCH_BINDINGS'
    receipt['native_dispatch_requested'] = False
    try:
        acceptance = receipt['geometry']['input_acceptance']
        if digest(input_json) != receipt['geometry']['geometry_json_sha256'] or digest(acceptance['path']) != acceptance['sha256'] or (status_path is not None and digest(status_path) != receipt['geometry']['geometry_status']['sha256']):
            raise ValueError('Protected geometry input changed before dispatch')
        adapter.verify_prepared_mesh_parameters(prepared)
        environment = dict(os.environ)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[name] = '1'
        phase = 'OPEN_NATIVE_LOGS'
        with (output / 'abaqus.stdout.log').open('x', encoding='utf-8') as stdout, (output / 'abaqus.stderr.log').open('x', encoding='utf-8') as stderr:
            phase = 'DISPATCH_NATIVE_FE'
            receipt.update(status='NATIVE_DISPATCH_REQUESTED', native_dispatch_requested=True, native_execution=None)
            persist(output / 'stage2_receipt.json', receipt)
            completed = subprocess.run(prepared['command'], cwd=prepared['cwd'], env=environment, stdout=stdout, stderr=stderr, shell=False)
        receipt.update(native_execution=True, native_returncode=completed.returncode)
        phase = 'ASSESS_NATIVE_RESULT'
        assessed = adapter.assess_mesh_result(prepared, completed.returncode)
        receipt.update(status='NATIVE_M9_CAE_SAVED_CLOSED' if assessed['status'] == 'SUCCESS' else 'FE_FAILED', result=assessed, independent_cae_reopen_validation=assessed['status'] == 'SUCCESS' and assessed.get('receipt', {}).get('independent_cae_reopen_validation') is True)
    except KeyboardInterrupt as exc:
        receipt.update(status='INTERRUPTED', error={'phase': phase, 'type': type(exc).__name__, 'message': str(exc)}, native_descendant_cleanup_independently_confirmed=False)
    except OSError as exc:
        receipt.update(status='DISPATCH_FAILED' if phase in ('OPEN_NATIVE_LOGS', 'DISPATCH_NATIVE_FE') else 'FE_FAILED', error={'phase': phase, 'type': type(exc).__name__, 'message': str(exc)}, native_execution=False if phase in ('OPEN_NATIVE_LOGS', 'DISPATCH_NATIVE_FE') else receipt['native_execution'])
    except Exception as exc:
        receipt.update(status='FE_FAILED', error={'phase': phase, 'type': type(exc).__name__, 'message': str(exc)})
    try:
        receipt['input_geometry_unchanged'] = digest(input_json) == receipt['geometry']['geometry_json_sha256']
        receipt['geometry_acceptance_certificate_unchanged'] = digest(receipt['geometry']['input_acceptance']['path']) == receipt['geometry']['input_acceptance']['sha256']
    except OSError:
        receipt['input_geometry_unchanged'] = False
        receipt['geometry_acceptance_certificate_unchanged'] = False
    receipt['terminal'] = True
    receipt['scientific_success_claim'] = receipt['status'] == 'NATIVE_M9_CAE_SAVED_CLOSED' and receipt['input_geometry_unchanged'] and receipt['geometry_acceptance_certificate_unchanged'] and receipt['independent_cae_reopen_validation']
    if receipt['status'] == 'NATIVE_M9_CAE_SAVED_CLOSED' and (not receipt['scientific_success_claim']):
        receipt['status'] = 'FE_FAILED_INPUT_OR_REOPEN_EVIDENCE'
    persist(output / 'stage2_receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'receipt': str(output / 'stage2_receipt.json'), 'cae': prepared['success_cae'] if receipt['scientific_success_claim'] else None}, indent=2))
    return 0 if receipt['scientific_success_claim'] else 2
def _build_fe(args):
    config = effective_fe_config(args.input_json, args.mesh_target_ratio, args.node_match_tol)
    if args.show_config:
        print(json.dumps(config, indent=2, allow_nan=False))
        return 0
    if args.input_json is None or args.output_dir is None:
        raise ValueError('--input-json and a fresh --output-dir are required for FE construction')
    output = args.output_dir.resolve()
    input_json = args.input_json.resolve()
    status_path = None if args.geometry_status is None else args.geometry_status.resolve()
    if output.exists():
        raise ValueError('Output directory must be fresh and must not exist')
    if output == ROOT or ROOT in output.parents or output in ROOT.parents:
        raise ValueError('Output directory must be isolated from the released source tree')
    if output == input_json or output in input_json.parents or (status_path is not None and (output == status_path or output in status_path.parents)):
        raise ValueError('Output directory must not contain the protected input files')
    if args.mesh_target_ratio is not None and (not math.isfinite(args.mesh_target_ratio) or args.mesh_target_ratio <= 0):
        raise ValueError('Explicit mesh-target-ratio must be finite and positive')
    geometry = validate_geometry(input_json, status_path)
    geometry['input_acceptance'] = discover_geometry_acceptance(input_json, require_native=not args.dry_run)
    command = resolve_launcher(args.abaqus_command, require_native=not args.dry_run)
    output.mkdir(parents=True, exist_ok=False)
    fe_root = _materialize_fe_sources(output)
    args._fe_root = fe_root
    adapter = _load_adapter(fe_root, 'mesh_only_adapter.py', 'cbge_fe_mesh_runtime')
    adapter._bound_sources(fe_root)
    receipt = {'schema': 'cbge-tafe-stage2-receipt-v1', 'status': 'PREPARING_NOT_EXECUTED', 'geometry': geometry, 'native_execution': False, 'solver_execution': False, 'geometry_generation': False, 'automatic_retry': False, 'scientific_success_claim': False, 'independent_cae_reopen_validation': False, 'driver_sha256': digest(Path(__file__)), 'adapter_sha256': digest(fe_root / 'mesh_only_adapter.py'), 'original_fe_sha256': adapter.ORIGINAL_MODEL_SHA256, 'public_fe_sha256': adapter.MODEL_SHA256, 'helper_sha256': adapter.HELPER_SHA256, 'stop_after_module': 'M9'}
    receipt['resolved_public_configuration'] = config
    receipt['public_interface_sha256'] = digest(Path(__file__))
    receipt['geometry_acceptance_certificate_unchanged'] = True
    try:
        prepared = adapter.prepare_mesh_only(fe_root=fe_root, json_path=input_json, work_dir=output, success_cae=output / 'official_success/final.cae', receipt_path=output / 'mesh_completion.json', abaqus_cmd=command, debug_cae=None, mesh_target_ratio=args.mesh_target_ratio, node_match_tol=args.node_match_tol)
        verification = adapter.verify_prepared_mesh_parameters(prepared)
        receipt.update(status='PREPARED_NOT_EXECUTED', prepared_fe=prepared, parameter_verification=verification)
    except BaseException as exc:
        receipt.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'PREPARATION_FAILED', terminal=True, error={'phase': 'PREPARATION', 'type': type(exc).__name__, 'message': str(exc)})
        persist(output / 'stage2_receipt.json', receipt)
        return 2
    if args.dry_run:
        receipt['status'] = 'DRY_RUN_STATIC_RENDER_PASS'
        receipt['terminal'] = True
        receipt['scientific_success_claim'] = False
        persist(output / 'stage2_receipt.json', receipt)
        print(json.dumps({'status': receipt['status'], 'receipt': str(output / 'stage2_receipt.json'), 'generated_script': prepared['script_path'], 'scientific_execution': False}, indent=2))
        return 0
    persist(output / 'stage2_receipt.json', receipt)
    return _execute_mesh(prepared, receipt, output, input_json, status_path, adapter)

def bind_original_case(args):
    if args.load_axis is None or args.strain is None:
        raise ValueError('Explicit load axis and dimensionless strain are required')
    args.case_type = 'AXIAL'
    args.axial_dir = args.load_axis.upper()
    args.shear_plane = 'XY'
    args.strain_pct = 100.0 * args.strain
    args.sign = 1.0
    for name in ('step_name', 'num_cpus', 'memory_percent', 'nout', 'job_prefix'):
        setattr(args, name, runtime_config()[name])

def _execute_tensile(prepared, receipt, output, adapter):
    phase = 'VERIFY_DISPATCH_BINDINGS'
    try:
        adapter.verify_prepared(prepared, output.parent / 'runtime_sources')
        for source_field, destination_field, hash_field in (('input_cae_path', 'copied_input_cae', 'input_cae_sha256'), ('particles_work_json_path', 'copied_work_json', 'particles_work_json_sha256')):
            with Path(prepared['inputs'][source_field]).open('rb') as source_handle, Path(prepared[destination_field]).open('xb') as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle)
            if digest(prepared[destination_field]) != prepared['inputs'][hash_field]:
                raise ValueError('Fixed tensile input copy identity mismatch')
        adapter.verify_prepared(prepared, output.parent / 'runtime_sources')
        environment = dict(os.environ)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[name] = '1'
        phase = 'SOLVE_DISPATCH'
        receipt.update(status='SOLVE_DISPATCH_REQUESTED', solver_execution=None)
        persist(output / 'solve_status.json', receipt)
        with (output / 'solve.stdout.log').open('x', encoding='utf-8') as stdout, (output / 'solve.stderr.log').open('x', encoding='utf-8') as stderr:
            completed = subprocess.run(prepared['solve_command'], cwd=output, env=environment, stdout=stdout, stderr=stderr, shell=False)
        receipt.update(native_execution=True, solver_execution=True, solve_returncode=completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError('Native tensile solve returned ' + str(completed.returncode))
        phase = 'SOLVE_ASSESSMENT'
        receipt['solve_assessment'] = adapter.assess_solve(prepared)
        receipt.update(status='TENSILE_SOLVE_COMPLETED', scientific_success_claim=True)
    except BaseException as exc:
        receipt.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'SOLVE_FAILED', scientific_success_claim=False, error={'phase': phase, 'type': type(exc).__name__, 'message': str(exc)})
    unchanged = True
    for path_field, hash_field in (('input_cae_path', 'input_cae_sha256'), ('particles_work_json_path', 'particles_work_json_sha256'), ('stage2_receipt_path', 'stage2_receipt_sha256'), ('input_geometry_path', 'input_geometry_sha256'), ('geometry_acceptance_path', 'geometry_acceptance_sha256')):
        if path_field in prepared['inputs']:
            try:
                unchanged = unchanged and digest(prepared['inputs'][path_field]) == prepared['inputs'][hash_field]
            except OSError:
                unchanged = False
    receipt['protected_inputs_unchanged'] = unchanged
    if not unchanged:
        receipt.update(status='FAILED_INPUT_IDENTITY_CHANGED', scientific_success_claim=False)
    receipt['terminal'] = True
    persist(output / 'solve_status.json', receipt)
    return 0 if receipt['scientific_success_claim'] else 2

def _run_tensile(args):
    output = args.output_dir.resolve() / 'elastic'
    stage2_path = args.output_dir.resolve() / 'stage2_receipt.json'
    stage2 = json.loads(stage2_path.read_text(encoding='utf-8-sig'))
    prepared_fe = stage2['prepared_fe']
    adapter = _load_adapter(args._fe_root, 'elastic_adapter.py', 'cbge_fe_solve_runtime')
    adapter.ROOT_RUN_FE_PATH = Path(__file__).resolve()
    inputs = adapter.validate_inputs(stage2_path, Path(prepared_fe['success_cae']), Path(prepared_fe['particles_work_json']), require_native=True)
    bind_original_case(args)
    case = adapter._case_from_args(args)
    output.mkdir(exist_ok=False)
    receipt = {'schema': 'cbge-tensile-solve-status-v1', 'status': 'PREPARING_NOT_EXECUTED', 'inputs': inputs, 'native_execution': False, 'solver_execution': False, 'scientific_success_claim': False, 'automatic_retry': False, 'geometry_generation': False, 'rebuild_or_remesh': False, 'automatic_result_extraction': False}
    try:
        prepared = adapter.prepare(args._fe_root, output, inputs, case, args.job_prefix, args.num_cpus, args.memory_percent, args.nout, args.step_name, 'I_MAT_TET', 'I_INC_TET', 'ESET_MAT', 'ESET_INC', resolve_launcher(args.abaqus_command, require_native=True))
        receipt['prepared'] = prepared
        adapter.verify_prepared(prepared, args._fe_root)
    except BaseException as exc:
        receipt.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'SOLVE_PREPARATION_FAILED', terminal=True, error={'type': type(exc).__name__, 'message': str(exc)})
        persist(output / 'solve_status.json', receipt)
        return 2
    return _execute_tensile(prepared, receipt, output, adapter)

def build_parser():
    parser = argparse.ArgumentParser(description='Build a CAE model from geometry JSON, with optional static tensile loading.')
    parser.add_argument('--input-json', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--mesh-target-ratio', type=finite_positive, default=0.06)
    parser.add_argument('--node-match-tol', type=finite_positive)
    parser.add_argument('--solve-elastic', action='store_true')
    parser.add_argument('--load-axis', choices=('x', 'y', 'z'))
    parser.add_argument('--strain', type=tensile_strain)
    parser.add_argument('--show-config', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--abaqus-command', help='Override the installed launcher; otherwise use CBGE_ABAQUS_COMMAND')
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    config = effective_fe_config(args.input_json, args.mesh_target_ratio, args.node_match_tol, args.solve_elastic, args.load_axis, args.strain)
    config.update(automatic_result_extraction=False, automatic_retry=False, geometry_generation=False)
    if args.show_config:
        print(json.dumps(config, indent=2, allow_nan=False))
        return 0
    args.geometry_status = None
    result = _build_fe(args)
    output = args.output_dir.resolve()
    status = 'FE_FAILED' if result else 'DRY_RUN_FE_RENDER_PASS' if args.dry_run else 'FE_CAE_SAVED'
    if result == 0 and args.solve_elastic and not args.dry_run:
        try:
            result = _run_tensile(args)
            status = 'TENSILE_SOLVE_FAILED' if result else 'TENSILE_SOLVE_COMPLETED'
        except BaseException as exc:
            result = 2
            status = 'TENSILE_SOLVE_BLOCKED'
            print(json.dumps({'status': status, 'error_type': type(exc).__name__, 'reason': str(exc)}, indent=2), file=sys.stderr)
    value = {'schema': 'cbge-fe-run-status-v1', 'status': status, 'automatic_retry': False, 'geometry_generation': False, 'automatic_result_extraction': False, 'solve_elastic_requested': args.solve_elastic, 'solver_execution': args.solve_elastic and not args.dry_run and status in ('TENSILE_SOLVE_COMPLETED', 'TENSILE_SOLVE_FAILED'), 'resolved_configuration': config, 'terminal': True}
    if result and status == 'FE_FAILED':
        value['next_action'] = 'Inspect the FE error and use a fresh output directory for the next request.'
    if args.dry_run and args.solve_elastic:
        value['tensile_input_status'] = 'NOT_EXECUTED_PENDING_NATIVE_CAE'
    if output.is_dir():
        persist(output / 'fe_status.json', value)
    print(json.dumps({'status': status, 'status_file': str(output / 'fe_status.json') if output.is_dir() else None, 'automatic_result_extraction': False, 'automatic_retry': False}, indent=2))
    return result

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'FE_BLOCKED', 'error_type': type(exc).__name__, 'reason': str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(3)
