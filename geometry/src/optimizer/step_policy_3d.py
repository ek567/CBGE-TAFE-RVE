from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class LineSearchConfig:
    enabled: bool = False
    alpha_init: float = 1.0
    rho: float = 0.5
    c1: float = 0.0001
    alpha_min: float = 1e-10
    max_trials: int = 25
    use_alpha_warm_start: bool = True
    alpha_growth: float = 1.2
    alpha_max: float = 1.0
    fallback_to_safe_step: bool = True
    safe_alpha: float = 0.001
    orientation_step_cap: float | None = None
    position_step_cap: float | None = None
    fast_energy_eval: bool = True
    early_abort: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.alpha_init = float(self.alpha_init)
        self.rho = float(self.rho)
        self.c1 = float(self.c1)
        self.alpha_min = float(self.alpha_min)
        self.max_trials = int(self.max_trials)
        self.use_alpha_warm_start = bool(self.use_alpha_warm_start)
        self.alpha_growth = float(self.alpha_growth)
        self.alpha_max = float(self.alpha_max)
        self.fallback_to_safe_step = bool(self.fallback_to_safe_step)
        self.safe_alpha = float(self.safe_alpha)
        self.orientation_step_cap = None if self.orientation_step_cap is None else float(self.orientation_step_cap)
        self.position_step_cap = None if self.position_step_cap is None else float(self.position_step_cap)
        self.fast_energy_eval = bool(self.fast_energy_eval)
        self.early_abort = bool(self.early_abort)
        if self.alpha_init <= 0.0:
            raise ValueError(f'alpha_init must be > 0, got {self.alpha_init}.')
        if not 0.0 < self.rho < 1.0:
            raise ValueError(f'rho must satisfy 0 < rho < 1, got {self.rho}.')
        if self.c1 <= 0.0:
            raise ValueError(f'c1 must be > 0, got {self.c1}.')
        if self.alpha_min <= 0.0:
            raise ValueError(f'alpha_min must be > 0, got {self.alpha_min}.')
        if self.max_trials <= 0:
            raise ValueError(f'max_trials must be > 0, got {self.max_trials}.')
        if self.alpha_growth <= 0.0:
            raise ValueError(f'alpha_growth must be > 0, got {self.alpha_growth}.')
        if self.alpha_max <= 0.0:
            raise ValueError(f'alpha_max must be > 0, got {self.alpha_max}.')
        if self.safe_alpha <= 0.0:
            raise ValueError(f'safe_alpha must be > 0, got {self.safe_alpha}.')
        if self.orientation_step_cap is not None and self.orientation_step_cap <= 0.0:
            raise ValueError(f'orientation_step_cap must be > 0 when provided, got {self.orientation_step_cap}.')
        if self.position_step_cap is not None and self.position_step_cap <= 0.0:
            raise ValueError(f'position_step_cap must be > 0 when provided, got {self.position_step_cap}.')

@dataclass
class LineSearchState:
    previous_alpha: float | None = None
    previous_success: bool = False

    def __post_init__(self) -> None:
        self.previous_alpha = None if self.previous_alpha is None else float(self.previous_alpha)
        self.previous_success = bool(self.previous_success)
        if self.previous_alpha is not None and self.previous_alpha <= 0.0:
            raise ValueError(f'previous_alpha must be > 0 when provided, got {self.previous_alpha}.')

def choose_initial_alpha(config: LineSearchConfig, state: LineSearchState | None) -> float:
    if not isinstance(config, LineSearchConfig):
        raise ValueError(f'config must be LineSearchConfig, got {type(config)!r}.')
    alpha = float(config.alpha_init)
    if config.use_alpha_warm_start and state is not None and state.previous_success and (state.previous_alpha is not None):
        alpha = min(float(state.previous_alpha) * float(config.alpha_growth), float(config.alpha_max))
    else:
        alpha = min(float(config.alpha_init), float(config.alpha_max))
    if alpha <= 0.0:
        raise RuntimeError(f'Computed non-positive initial alpha: {alpha}.')
    return float(alpha)

def cap_trial_step_sizes(direction_cs: list[np.ndarray], direction_ps: list[np.ndarray], alpha: float, position_step_cap: float | None, orientation_step_cap: float | None) -> float:
    if not isinstance(direction_cs, list):
        raise ValueError('direction_cs must be a list.')
    if not isinstance(direction_ps, list):
        raise ValueError('direction_ps must be a list.')
    if len(direction_cs) != len(direction_ps):
        raise ValueError(f'Expected len(direction_cs) == len(direction_ps), got {len(direction_cs)} != {len(direction_ps)}.')
    alpha_capped = float(alpha)
    if alpha_capped <= 0.0:
        raise ValueError(f'alpha must be > 0, got {alpha_capped}.')
    if position_step_cap is not None:
        position_cap_value = float(position_step_cap)
        if position_cap_value <= 0.0:
            raise ValueError(f'position_step_cap must be > 0, got {position_cap_value}.')
        max_norm_c = 0.0
        for idx, direction_c in enumerate(direction_cs):
            direction_c_array = np.asarray(direction_c, dtype=float)
            if direction_c_array.shape != (3,):
                raise ValueError(f'Expected direction_cs[{idx}].shape == (3,), got {direction_c_array.shape}.')
            max_norm_c = max(max_norm_c, float(np.linalg.norm(direction_c_array)))
        if max_norm_c > 0.0:
            alpha_capped = min(alpha_capped, position_cap_value / max_norm_c)
    if orientation_step_cap is not None:
        orientation_cap_value = float(orientation_step_cap)
        if orientation_cap_value <= 0.0:
            raise ValueError(f'orientation_step_cap must be > 0, got {orientation_cap_value}.')
        max_norm_p = 0.0
        for idx, direction_p in enumerate(direction_ps):
            direction_p_array = np.asarray(direction_p, dtype=float)
            if direction_p_array.shape != (3,):
                raise ValueError(f'Expected direction_ps[{idx}].shape == (3,), got {direction_p_array.shape}.')
            max_norm_p = max(max_norm_p, float(np.linalg.norm(direction_p_array)))
        if max_norm_p > 0.0:
            alpha_capped = min(alpha_capped, orientation_cap_value / max_norm_p)
    return float(alpha_capped)

def decide_fallback_alpha(config: LineSearchConfig) -> float | None:
    if not isinstance(config, LineSearchConfig):
        raise ValueError(f'config must be LineSearchConfig, got {type(config)!r}.')
    if config.fallback_to_safe_step:
        return float(config.safe_alpha)
    return None
