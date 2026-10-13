from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
FeatureKind3D = Literal['face', 'edge', 'corner']

@dataclass(frozen=True)
class FaceFeature3D:
    name: str
    B: np.ndarray
    r: np.ndarray
    d_safe: float
    weight: float
    kind: FeatureKind3D = 'face'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'B', np.asarray(self.B, dtype=float))
        object.__setattr__(self, 'r', np.asarray(self.r, dtype=float))
        object.__setattr__(self, 'd_safe', float(self.d_safe))
        object.__setattr__(self, 'weight', float(self.weight))
        if self.B.shape != (1, 3):
            raise ValueError(f'FaceFeature3D {self.name}: expected B.shape == (1, 3), got {self.B.shape}.')
        if self.r.shape != (1,):
            raise ValueError(f'FaceFeature3D {self.name}: expected r.shape == (1,), got {self.r.shape}.')
        if self.d_safe <= 0.0:
            raise ValueError(f'FaceFeature3D {self.name}: d_safe must be > 0, got {self.d_safe}.')
        if self.weight < 0.0:
            raise ValueError(f'FaceFeature3D {self.name}: weight must be >= 0, got {self.weight}.')

@dataclass(frozen=True)
class EdgeFeature3D:
    name: str
    B: np.ndarray
    r: np.ndarray
    tangent: np.ndarray
    endpoint_A: np.ndarray
    length: float
    d_safe: float
    weight: float
    kind: FeatureKind3D = 'edge'

    def __post_init__(self) -> None:
        B_array = np.asarray(self.B, dtype=float)
        r_array = np.asarray(self.r, dtype=float)
        tangent_array = np.asarray(self.tangent, dtype=float)
        endpoint_array = np.asarray(self.endpoint_A, dtype=float)
        length_value = float(self.length)
        if B_array.shape != (2, 3):
            raise ValueError(f'EdgeFeature3D {self.name}: expected B.shape == (2, 3), got {B_array.shape}.')
        if r_array.shape != (2,):
            raise ValueError(f'EdgeFeature3D {self.name}: expected r.shape == (2,), got {r_array.shape}.')
        if tangent_array.shape != (3,):
            raise ValueError(f'EdgeFeature3D {self.name}: expected tangent.shape == (3,), got {tangent_array.shape}.')
        tangent_norm = float(np.linalg.norm(tangent_array))
        if tangent_norm == 0.0:
            raise ValueError(f'EdgeFeature3D {self.name}: tangent must be nonzero.')
        tangent_array = tangent_array / tangent_norm
        if endpoint_array.shape != (3,):
            raise ValueError(f'EdgeFeature3D {self.name}: expected endpoint_A.shape == (3,), got {endpoint_array.shape}.')
        if length_value <= 0.0:
            raise ValueError(f'EdgeFeature3D {self.name}: length must be > 0, got {length_value}.')
        d_safe_value = float(self.d_safe)
        weight_value = float(self.weight)
        if d_safe_value <= 0.0:
            raise ValueError(f'EdgeFeature3D {self.name}: d_safe must be > 0, got {d_safe_value}.')
        if weight_value < 0.0:
            raise ValueError(f'EdgeFeature3D {self.name}: weight must be >= 0, got {weight_value}.')
        object.__setattr__(self, 'B', B_array)
        object.__setattr__(self, 'r', r_array)
        object.__setattr__(self, 'tangent', tangent_array)
        object.__setattr__(self, 'endpoint_A', endpoint_array)
        object.__setattr__(self, 'length', length_value)
        object.__setattr__(self, 'd_safe', d_safe_value)
        object.__setattr__(self, 'weight', weight_value)

@dataclass(frozen=True)
class CornerFeature3D:
    name: str
    C: np.ndarray
    d_safe: float
    weight: float
    kind: FeatureKind3D = 'corner'

    @property
    def B(self) -> np.ndarray:
        return np.eye(3, dtype=float)

    @property
    def r(self) -> np.ndarray:
        return -np.asarray(self.C, dtype=float)

    def __post_init__(self) -> None:
        C_array = np.asarray(self.C, dtype=float)
        d_safe_value = float(self.d_safe)
        weight_value = float(self.weight)
        if C_array.shape != (3,):
            raise ValueError(f'CornerFeature3D {self.name}: expected C.shape == (3,), got {C_array.shape}.')
        if d_safe_value <= 0.0:
            raise ValueError(f'CornerFeature3D {self.name}: d_safe must be > 0, got {d_safe_value}.')
        if weight_value < 0.0:
            raise ValueError(f'CornerFeature3D {self.name}: weight must be >= 0, got {weight_value}.')
        object.__setattr__(self, 'C', C_array)
        object.__setattr__(self, 'd_safe', d_safe_value)
        object.__setattr__(self, 'weight', weight_value)

@dataclass(frozen=True)
class BoxGeometryFeatures3D:
    faces: list[FaceFeature3D]
    edges: list[EdgeFeature3D]
    corners: list[CornerFeature3D]

    def num_enabled_terms(self) -> int:
        return len(self.faces) + len(self.edges) + len(self.corners)

def _validate_box(box: np.ndarray) -> np.ndarray:
    box_array = np.asarray(box, dtype=float)
    if box_array.shape != (3,):
        raise ValueError(f'Expected box.shape == (3,), got {box_array.shape}.')
    if np.any(box_array <= 0.0):
        raise ValueError(f'Box lengths must be positive, got {box_array}.')
    return box_array

def _axis_unit(axis: int) -> np.ndarray:
    e = np.zeros(3, dtype=float)
    e[int(axis)] = 1.0
    return e

def _plane_row_and_offset(axis: int, value: float) -> tuple[np.ndarray, float]:
    return (_axis_unit(axis).reshape(1, 3), -float(value))

def build_box_geometry_features_3d(box: np.ndarray, *, d_face: float, d_edge: float, d_corner: float, w_face: float=1.0, w_edge: float=1.0, w_corner: float=1.0, include_face: bool=True, include_edge: bool=True, include_corner: bool=True) -> BoxGeometryFeatures3D:
    box_array = _validate_box(box)
    d_face_value = float(d_face)
    d_edge_value = float(d_edge)
    d_corner_value = float(d_corner)
    w_face_value = float(w_face)
    w_edge_value = float(w_edge)
    w_corner_value = float(w_corner)
    if d_face_value <= 0.0:
        raise ValueError(f'd_face must be > 0, got {d_face_value}.')
    if d_edge_value <= 0.0:
        raise ValueError(f'd_edge must be > 0, got {d_edge_value}.')
    if d_corner_value <= 0.0:
        raise ValueError(f'd_corner must be > 0, got {d_corner_value}.')
    if min(w_face_value, w_edge_value, w_corner_value) < 0.0:
        raise ValueError('Feature weights must be >= 0.')
    axis_names = ['x', 'y', 'z']
    faces: list[FaceFeature3D] = []
    if include_face:
        for axis in range(3):
            for side_name, value in (('0', 0.0), ('1', float(box_array[axis]))):
                B, r_scalar = _plane_row_and_offset(axis, value)
                faces.append(FaceFeature3D(name=f'{axis_names[axis]}{side_name}', B=B, r=np.array([r_scalar], dtype=float), d_safe=d_face_value, weight=w_face_value))
    edges: list[EdgeFeature3D] = []
    if include_edge:
        for fixed_axes in ((0, 1), (0, 2), (1, 2)):
            tangent_axis = ({0, 1, 2} - set(fixed_axes)).pop()
            tangent = _axis_unit(tangent_axis)
            length = float(box_array[tangent_axis])
            for side0 in (0, 1):
                for side1 in (0, 1):
                    fixed_values = [0.0, 0.0]
                    fixed_values[0] = float(box_array[fixed_axes[0]]) if side0 == 1 else 0.0
                    fixed_values[1] = float(box_array[fixed_axes[1]]) if side1 == 1 else 0.0
                    B0, r0 = _plane_row_and_offset(fixed_axes[0], fixed_values[0])
                    B1, r1 = _plane_row_and_offset(fixed_axes[1], fixed_values[1])
                    B = np.vstack([B0, B1])
                    r = np.array([r0, r1], dtype=float)
                    endpoint_A = np.zeros(3, dtype=float)
                    endpoint_A[fixed_axes[0]] = fixed_values[0]
                    endpoint_A[fixed_axes[1]] = fixed_values[1]
                    endpoint_A[tangent_axis] = 0.0
                    name = f'{axis_names[fixed_axes[0]]}{side0}{axis_names[fixed_axes[1]]}{side1}'
                    edges.append(EdgeFeature3D(name=name, B=B, r=r, tangent=tangent, endpoint_A=endpoint_A, length=length, d_safe=d_edge_value, weight=w_edge_value))
    corners: list[CornerFeature3D] = []
    if include_corner:
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    C = np.array([float(box_array[0]) if sx == 1 else 0.0, float(box_array[1]) if sy == 1 else 0.0, float(box_array[2]) if sz == 1 else 0.0], dtype=float)
                    corners.append(CornerFeature3D(name=f'x{sx}y{sy}z{sz}', C=C, d_safe=d_corner_value, weight=w_corner_value))
    return BoxGeometryFeatures3D(faces=faces, edges=edges, corners=corners)
