from __future__ import annotations
import hashlib
import json
import struct
TOTAL_REALIZATION_ITERATION_BUDGET = 7000
WITHIN_RESOURCE_BUDGET = 'WITHIN_RESOURCE_BUDGET'
RESOURCE_BUDGET_EXHAUSTED = 'RESOURCE_BUDGET_EXHAUSTED'

def committed_state_identity(particles) -> str:

    def binary64(value):
        return struct.pack('>d', float(value)).hex()
    state = [{'id': None if p.id is None else int(p.id), 'a': binary64(p.a), 'b': binary64(p.b), 'c': [binary64(v) for v in p.c], 'p': [binary64(v) for v in p.p]} for p in particles]
    encoded = json.dumps(state, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()

class RealizationIterationBudget:

    def __init__(self):
        self._completed_iterations = 0
        self._resource_status = WITHIN_RESOURCE_BUDGET
        self._iteration_admitted = False
        self._last_committed_state_sha256 = None
        self._native_stop_context = []

    @property
    def completed_iterations(self):
        return self._completed_iterations

    @property
    def resource_status(self):
        return self._resource_status

    @property
    def at_limit(self):
        return self._completed_iterations >= TOTAL_REALIZATION_ITERATION_BUDGET

    def bind_initial_state(self, particles):
        if self._last_committed_state_sha256 is None:
            self._last_committed_state_sha256 = committed_state_identity(particles)

    def admit_iteration(self):
        if self._iteration_admitted:
            raise RuntimeError('An optimizer iteration is already admitted.')
        if self.at_limit or self._resource_status == RESOURCE_BUDGET_EXHAUSTED:
            self.mark_exhausted('additional_optimizer_iteration_required')
            return False
        self._iteration_admitted = True
        return True

    def commit_iteration(self, particles):
        if not self._iteration_admitted or self.at_limit:
            raise RuntimeError('Completed iteration lacks an available admission.')
        self._completed_iterations += 1
        self._iteration_admitted = False
        self._last_committed_state_sha256 = committed_state_identity(particles)

    def record_native_stop(self, context):
        entry = dict(context)
        entry['G'] = self._completed_iterations
        if not self._native_stop_context or entry != self._native_stop_context[-1]:
            self._native_stop_context.append(entry)

    def mark_exhausted(self, native_stop_reason):
        if not self.at_limit or self._iteration_admitted:
            raise RuntimeError('Resource exhaustion requires the committed ceiling.')
        self.record_native_stop({'reason': str(native_stop_reason)})
        self._resource_status = RESOURCE_BUDGET_EXHAUSTED

    def snapshot(self):
        return {'cumulative_completed_iterations': self._completed_iterations, 'total_realization_iteration_budget': TOTAL_REALIZATION_ITERATION_BUDGET, 'resource_status': self._resource_status, 'last_committed_state_sha256': self._last_committed_state_sha256, 'native_stop_context': [dict(entry) for entry in self._native_stop_context]}

def apply_resource_result(result, budget):
    if result.stop_reason != RESOURCE_BUDGET_EXHAUSTED:
        budget.record_native_stop({'stop_reason': str(result.stop_reason), 'success': bool(result.success), 'converged': bool(result.converged), 'invocation_completed_iterations': int(result.steps_taken)})
    for name, value in budget.snapshot().items():
        setattr(result, name, value)
    result.terminal_state_sha256 = committed_state_identity(result.particles)
    if budget.resource_status == RESOURCE_BUDGET_EXHAUSTED:
        result.success = False
        result.converged = False
        result.stop_reason = RESOURCE_BUDGET_EXHAUSTED
        if result.metrics is not None:
            result.metrics.converged = False
    result.generation_success = bool(result.success)
    return result
