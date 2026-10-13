from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Iterator

def start_timer() -> float:
    return float(time.perf_counter())

def stop_timer(t_start: float) -> float:
    t0 = float(t_start)
    elapsed = float(time.perf_counter() - t0)
    if elapsed < 0.0:
        raise ValueError(f'Computed negative elapsed time: {elapsed}.')
    return elapsed

def add_elapsed_time(metrics: object | None, field_name: str, elapsed: float) -> None:
    if metrics is None:
        return
    elapsed_value = float(elapsed)
    if elapsed_value < 0.0:
        raise ValueError(f'elapsed must be >= 0, got {elapsed_value}.')
    if not hasattr(metrics, field_name):
        raise AttributeError(f'metrics object of type {type(metrics)!r} has no field {field_name!r}.')
    current_value = getattr(metrics, field_name)
    try:
        new_value = float(current_value) + elapsed_value
    except Exception as exc:
        raise TypeError(f'metrics.{field_name!r} must be numeric, got {type(current_value)!r}.') from exc
    setattr(metrics, field_name, new_value)

@contextmanager
def timed_field(metrics: object | None, field_name: str) -> Iterator[None]:
    if metrics is None:
        yield
        return
    t0 = start_timer()
    try:
        yield
    finally:
        elapsed = stop_timer(t0)
        add_elapsed_time(metrics=metrics, field_name=field_name, elapsed=elapsed)
