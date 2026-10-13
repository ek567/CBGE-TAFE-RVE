from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PairContactResult3D:
    i: int
    j: int
    r_ij: np.ndarray
    F_ij: float
    lambda_star: float
    overlap: float
    psi_ij: float
    Q_star: np.ndarray | None = None
    s_ij: np.ndarray | None = None
    x0_ij: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.i = int(self.i)
        self.j = int(self.j)
        self.r_ij = np.asarray(self.r_ij, dtype=float)
        self.F_ij = float(self.F_ij)
        self.lambda_star = float(self.lambda_star)
        self.overlap = float(self.overlap)
        self.psi_ij = float(self.psi_ij)
        if self.r_ij.shape != (3,):
            raise ValueError(f'Expected r_ij.shape == (3,), got {self.r_ij.shape}.')
        if self.Q_star is not None:
            self.Q_star = np.asarray(self.Q_star, dtype=float)
            if self.Q_star.shape != (3, 3):
                raise ValueError(f'Expected Q_star.shape == (3, 3), got {self.Q_star.shape}.')
        if self.s_ij is not None:
            self.s_ij = np.asarray(self.s_ij, dtype=float)
            if self.s_ij.shape != (3,):
                raise ValueError(f'Expected s_ij.shape == (3,), got {self.s_ij.shape}.')
        if self.x0_ij is not None:
            self.x0_ij = np.asarray(self.x0_ij, dtype=float)
            if self.x0_ij.shape != (3,):
                raise ValueError(f'Expected x0_ij.shape == (3,), got {self.x0_ij.shape}.')
