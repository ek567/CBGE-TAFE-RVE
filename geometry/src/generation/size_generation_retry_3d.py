from __future__ import annotations
import dataclasses
import re
from typing import Any
from src.generation.size_generator_3d import SizeGenerationConfig3D, SizeGenerationResult3D, generate_ellipsoid_sizes_from_truncnorm
SIZE_SEED_STRIDE = 1000003
VF_EXHAUSTION_MARKER = 'Size generation failed to meet VF tolerance'

class SizeGenerationRetryExhausted(RuntimeError):

    def __init__(self, retry_report: dict[str, Any], last_error: RuntimeError):
        self.size_generation_retry_report = retry_report
        super().__init__(f"Size generation seed retries exhausted after {retry_report['attempt_count']} attempts ({retry_report['max_seed_retries']} additional retries); requested_seed={retry_report['requested_seed']}, last_seed={retry_report['last_attempt_seed']}. {last_error}")

def _failure_values(message: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, pattern, convert in (('reported_last_vf', 'Last vf=([^,]+)', float), ('reported_target_vf', 'target=([^,]+)', float), ('reported_vf_abs_tol', 'abs_tol=([^,]+)', float), ('reported_last_N', 'last N=([^.,]+)', int)):
        match = re.search(pattern, message)
        if match:
            try:
                values[name] = convert(match.group(1))
            except (ValueError, OverflowError):
                values[name] = match.group(1)
    return values

def generate_sizes_with_seed_retries(config: SizeGenerationConfig3D, max_seed_retries: int=50) -> SizeGenerationResult3D:
    if type(max_seed_retries) is not int or max_seed_retries < 0:
        raise ValueError('max_seed_retries must be a nonnegative integer (additional attempts)')
    if max_seed_retries > 0 and (config.seed is None or not isinstance(config.seed, (int,))):
        raise ValueError('size seed retries require an explicit integer config.seed')
    requested_seed = config.seed
    report: dict[str, Any] = {'schema': 'pw-rve-size-generation-seed-retries-v1', 'requested_seed': requested_seed, 'effective_seed': None, 'max_seed_retries': max_seed_retries, 'max_attempts': max_seed_retries + 1, 'seed_stride': SIZE_SEED_STRIDE, 'max_adjust_iters_per_attempt': int(config.max_adjust_iters), 'attempt_count': 0, 'retries_used': 0, 'attempts': [], 'timing_scope': 'SIZE_SAMPLING_BEFORE_PARTICLE_INITIALIZATION_EXCLUDED_FROM_OPTIMIZATION_TIMER', 'admission_policy': 'UNCHANGED_PARENT_VF_TARGET_AND_ABSOLUTE_TOLERANCE'}
    for retry_index in range(max_seed_retries + 1):
        seed = requested_seed if retry_index == 0 else int(requested_seed) + retry_index * SIZE_SEED_STRIDE
        attempt_config = config if retry_index == 0 else dataclasses.replace(config, seed=seed)
        attempt: dict[str, Any] = {'retry_index': retry_index, 'seed': seed}
        report.update(attempt_count=retry_index + 1, retries_used=retry_index, last_attempt_seed=seed)
        try:
            result = generate_ellipsoid_sizes_from_truncnorm(attempt_config)
        except RuntimeError as exc:
            if VF_EXHAUSTION_MARKER not in str(exc):
                raise
            attempt.update(status='VF_ADMISSION_FAILED', exception_type=type(exc).__name__, exception=str(exc))
            attempt.update(_failure_values(str(exc)))
            report['attempts'].append(attempt)
            if retry_index == max_seed_retries:
                report['status'] = 'EXHAUSTED_NO_ACCEPTED_SIZES'
                if max_seed_retries == 0:
                    exc.size_generation_retry_report = report
                    raise
                raise SizeGenerationRetryExhausted(report, exc) from exc
            continue
        attempt.update(status='VF_ADMITTED', N=int(result.N), vf_sample=float(result.vf_sample), vf_error=float(result.vf_error))
        report['attempts'].append(attempt)
        report.update(status='SUCCESS', effective_seed=seed, aspect_ratio_seed=result.aspect_ratio_seed, N=int(result.N), vf_sample=float(result.vf_sample), vf_error=float(result.vf_error))
        result.report['size_generation_seed_retries'] = report
        return result
    raise AssertionError('bounded size retry loop must return or raise')
generate_ellipsoid_sizes_with_seed_retries = generate_sizes_with_seed_retries
