from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from src.utils.math_utils import normalize

@dataclass
class EllipsoidParticle3D:
    a: float
    b: float
    c: np.ndarray
    p: np.ndarray
    id: int | None = None

    def __post_init__(self) -> None:
        self.a = float(self.a)
        self.b = float(self.b)
        self.c = np.asarray(self.c, dtype=float)
        self.p = np.asarray(self.p, dtype=float)
        if not self.a >= self.b > 0.0:
            raise ValueError(f'Expected a >= b > 0, got a={self.a}, b={self.b}.')
        if self.c.shape != (3,):
            raise ValueError(f'Expected c.shape == (3,), got {self.c.shape}.')
        if self.p.shape != (3,):
            raise ValueError(f'Expected p.shape == (3,), got {self.p.shape}.')
        self.p = normalize(self.p)

    def clone(self) -> 'EllipsoidParticle3D':
        return EllipsoidParticle3D(a=self.a, b=self.b, c=self.c.copy(), p=self.p.copy(), id=self.id)

    def copy(self) -> 'EllipsoidParticle3D':
        return self.clone()

    def to_dict(self) -> dict[str, Any]:
        return {'a': self.a, 'b': self.b, 'c': self.c.copy(), 'p': self.p.copy(), 'id': self.id}
