from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import uuid
MAX_ATTEMPTS = 21
SIZE_SEED_STRIDE = 1000003
DEFAULT_OUTER_SEED_STRIDE = 51 * SIZE_SEED_STRIDE
SCHEMA = 'pw-rve-geometry-acceptance-retry-v1'

class GeometryDiagnostic(RuntimeError):
    pass

class GeometryReceiptError(GeometryDiagnostic):

    def __init__(self, receipt, path, cause):
        self.receipt = receipt
        self.receipt_path = str(Path(path).resolve())
        super().__init__('RECEIPT_PERSISTENCE_FAILED: ' + str(cause))

def _persist_request(path, receipt, *, exclusive=False):
    try:
        _write_json(path, receipt, exclusive=exclusive)
    except (OSError, ValueError) as exc:
        receipt['terminal_status'] = 'GEOMETRY_FATAL_DIAGNOSTIC'
        receipt['failure_reason'] = 'RECEIPT_PERSISTENCE_FAILED: ' + str(exc)
        receipt['receipt_persisted'] = False
        raise GeometryReceiptError(receipt, path, exc) from exc

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def _publish_exclusive(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    published = False
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == 'nt':
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
        published = True
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                if not published:
                    raise

def verify_source_identity(root=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    manifest_path = root.parent / 'SOURCE_MANIFEST.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        rows = manifest['files']
        paths = [row['path'] for row in rows]
        if paths != sorted(set(paths)):
            raise GeometryDiagnostic('SOURCE_MANIFEST_PATH_ORDER_OR_DUPLICATE')
        for row in rows:
            name = row['path']
            member = PurePosixPath(name)
            if member.is_absolute() or '..' in member.parts or '\\' in name or (':' in name):
                raise GeometryDiagnostic('UNSAFE_SOURCE_MANIFEST_MEMBER')
            path = (root / name).resolve()
            if root not in path.parents:
                raise GeometryDiagnostic('SOURCE_MEMBER_ESCAPES_BOUND_ROOT')
            data = path.read_bytes()
            if len(data) != row['size'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise GeometryDiagnostic('SOURCE_IDENTITY_MISMATCH: ' + name)
        bound_python = {name for name in paths if name.endswith('.py')}
        actual_python = {path.relative_to(root).as_posix() for path in root.rglob('*.py')}
        if actual_python != bound_python:
            raise GeometryDiagnostic('SOURCE_PYTHON_INVENTORY_MISMATCH')
        canonical = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
        identity = hashlib.sha256(canonical).hexdigest()
        if identity != manifest['identity_sha256']:
            raise GeometryDiagnostic('SOURCE_CANONICAL_IDENTITY_MISMATCH')
        return {'schema': 'pw-rve-geometry-retry-source-identity-v1', 'identity_verified': True, 'baseline_id': manifest['baseline_id'], 'identity_sha256': identity, 'full_source_sha256': identity, 'source_files_verified': len(rows), 'source_root': str(root), 'manifest_path': str(manifest_path), 'manifest_sha256': _digest(manifest_path), 'original_source_identity_sha256': manifest.get('original_identity_sha256', manifest.get('original_source_identity_sha256')), 'scientific_execution': False}
    except GeometryDiagnostic:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise GeometryDiagnostic('SOURCE_IDENTITY_UNAVAILABLE: ' + str(exc)) from exc

def _boolean(value, label):
    if type(value) is not bool:
        raise GeometryDiagnostic('UNKNOWN_INDEPENDENT_ACCEPTANCE: ' + label + ' must be an exact bool')
    return value

def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        raise GeometryDiagnostic('UNKNOWN_INDEPENDENT_ACCEPTANCE: ' + label + ' must be finite')
    return float(value)

def evaluate_export(path, expected):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema') != 'pw85-ellipsoids-export' or not data.get('particles'):
            raise GeometryDiagnostic('UNKNOWN_INDEPENDENT_ACCEPTANCE: invalid or empty geometry export')
        summary = data['optimization_summary']
        audits = summary['final_audits']
        for key in ('physical_pw_passed', 'strict_periodic_audit_performed', 'strict_periodic_passed', 'geometry_required', 'geometry_audit_performed', 'orientation_required', 'orientation_target_passed', 'all_required_audits_passed'):
            _boolean(audits[key], key)
        if audits['geometry_required'] != expected['geometry_required'] or audits['orientation_required'] != expected['orientation_required']:
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_REQUIRED_AUDIT_POLICY_MISMATCH')
        if audits['strict_periodic_audit_performed'] != expected['periodic']:
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_PERIODIC_AUDIT_NOT_PERFORMED')
        if audits['geometry_required'] and (not audits['geometry_audit_performed']):
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_GEOMETRY_AUDIT_NOT_PERFORMED')
        physical_tol = _finite(audits['physical_acceptance_overlap_tol'], 'physical_acceptance_overlap_tol')
        if physical_tol != expected['physical_acceptance_overlap_tol']:
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_PHYSICAL_TOLERANCE_MISMATCH')
        orientation_tol = _finite(audits['orientation_tol'], 'orientation_tol')
        if orientation_tol != expected['orientation_tol']:
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_ORIENTATION_TOLERANCE_MISMATCH')
        if audits['physical_pw_passed'] != (_finite(summary['final_max_overlap_pw85_physical'], 'physical max overlap') <= physical_tol):
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_PHYSICAL_VERDICT_INCONSISTENT')
        for kind in ('face', 'edge', 'corner'):
            value = _finite(audits['geometry_violation_tolerances'][kind], kind + ' tolerance')
            if value != expected['geometry_violation_tolerances'][kind]:
                raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_GEOMETRY_TOLERANCE_MISMATCH: ' + kind)
        if audits['geometry_audit_performed']:
            violations = {kind: _finite(summary['final_geometry_physical']['max_' + kind + '_violation'], kind + ' violation') for kind in ('face', 'edge', 'corner')}
            geometry_ok = all((violations[kind] <= audits['geometry_violation_tolerances'][kind] for kind in ('face', 'edge', 'corner')))
            if _boolean(audits['geometry_passed'], 'geometry_passed') != geometry_ok:
                raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_GEOMETRY_VERDICT_INCONSISTENT')
        elif audits['geometry_passed'] is not None:
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_GEOMETRY_VERDICT_WITHOUT_AUDIT')
        orientation_ok = _boolean(summary['final_orientation']['is_target_aligned'], 'final orientation verdict')
        deviation = _finite(summary['final_orientation']['max_abs_deviation'], 'orientation max_abs_deviation')
        if deviation < 0 or orientation_ok != (deviation < orientation_tol):
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_ORIENTATION_NUMERIC_VERDICT_INCONSISTENT')
        if _finite(summary['final_orientation']['tolerance'], 'final orientation tolerance') != orientation_tol:
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_ORIENTATION_TOLERANCE_MISMATCH')
        if summary['final_orientation']['orientation_target'] != expected['orientation_target']:
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_ORIENTATION_TARGET_MISMATCH')
        if orientation_ok != audits['orientation_target_passed']:
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_ORIENTATION_VERDICT_INCONSISTENT')
        combined = audits['physical_pw_passed'] and audits['strict_periodic_passed'] and (not audits['geometry_required'] or audits['geometry_passed']) and (not audits['orientation_required'] or audits['orientation_target_passed'])
        if audits['all_required_audits_passed'] != bool(combined):
            raise GeometryDiagnostic('INDEPENDENT_ACCEPTANCE_COMBINED_VERDICT_INCONSISTENT')
        state = data['metadata']['input_state']
        for key in ('seed_init',):
            if type(state[key]) is not int or state[key] < 0:
                raise GeometryDiagnostic('UNKNOWN_ACTUAL_GENERATION_SEED: ' + key)
        if _boolean(state['size_distribution_mode'], 'size_distribution_mode'):
            for key in ('seed_size_requested', 'seed_size_effective'):
                if type(state[key]) is not int or state[key] < 0:
                    raise GeometryDiagnostic('UNKNOWN_ACTUAL_GENERATION_SEED: ' + key)
            report = state['size_report']
            trace = report['size_generation_seed_retries']
            rows = trace['attempts']
            retries = state['size_gen_seed_retries']
            if type(retries) is not int or retries < 0 or trace['status'] != 'SUCCESS' or (trace['requested_seed'] != state['seed_size_requested']) or (trace['effective_seed'] != state['seed_size_effective']) or (report['input']['seed'] != state['seed_size_effective']) or (trace['max_seed_retries'] != retries) or (trace['attempt_count'] != len(rows)) or (not 1 <= len(rows) <= retries + 1):
                raise GeometryDiagnostic('NATIVE_ACTUAL_SIZE_SEED_TRACE_MISMATCH')
            for index, row in enumerate(rows):
                if row['retry_index'] != index or row['seed'] != state['seed_size_requested'] + index * SIZE_SEED_STRIDE or row['status'] != ('VF_ADMITTED' if index == len(rows) - 1 else 'VF_ADMISSION_FAILED'):
                    raise GeometryDiagnostic('NATIVE_ACTUAL_SIZE_SEED_TRACE_MISMATCH')
            if rows[-1]['seed'] != state['seed_size_effective']:
                raise GeometryDiagnostic('NATIVE_ACTUAL_SIZE_SEED_TRACE_MISMATCH')
        elif state['seed_size_requested'] is not None or state['seed_size_effective'] is not None:
            raise GeometryDiagnostic('UNKNOWN_ACTUAL_GENERATION_SEED: non-size mode must retain native null size seeds')
        passed = audits['all_required_audits_passed']
        failed = [key for key in ('physical_pw_passed', 'strict_periodic_passed') if not audits[key]]
        if audits['geometry_required'] and (not audits['geometry_passed']):
            failed.append('geometry_passed')
        if audits['orientation_required'] and (not audits['orientation_target_passed']):
            failed.append('orientation_target_passed')
        return {'status': 'PASS' if passed else 'FAIL', 'performed': True, 'all_required_audits_passed': passed, 'verdicts': audits, 'failure_reason': None if passed else 'GEOMETRY_INDEPENDENT_ACCEPTANCE_FAILED: ' + ','.join(failed), 'input_state': state, 'optimization_summary': summary}
    except GeometryDiagnostic:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise GeometryDiagnostic('UNKNOWN_INDEPENDENT_ACCEPTANCE: ' + str(exc)) from exc

def plan_seed_pairs(initial_pairs, max_attempts=MAX_ATTEMPTS, inner_size_retries=50):
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_ATTEMPTS or type(inner_size_retries) is not int or (inner_size_retries < 0) or any((type(seed) is not int or seed < 0 for pair in initial_pairs for seed in pair)):
        raise GeometryDiagnostic('INVALID_SEED_PLAN_CONFIGURATION')
    span = max(51, int(inner_size_retries) + 1)
    stride = span * SIZE_SEED_STRIDE
    used_positions = {pair[0] for pair in initial_pairs}
    used_sizes = [pair[1] for pair in initial_pairs]
    plans = []

    def overlaps(candidate):
        return any(((candidate - used) % SIZE_SEED_STRIDE == 0 and abs((candidate - used) // SIZE_SEED_STRIDE) <= inner_size_retries for used in used_sizes))
    for position, size in initial_pairs:
        pairs = [(position, size)]
        next_position, next_size = (position, size)
        for _ in range(1, max_attempts):
            next_position += stride
            while next_position in used_positions:
                next_position += stride
            next_size += stride
            while overlaps(next_size):
                next_size += stride
            used_positions.add(next_position)
            used_sizes.append(next_size)
            pairs.append((next_position, next_size))
        plans.append(pairs)
    return plans

def _generation_status(summary=None, *, execution_status='RETURNED'):
    summary = summary or {}
    return {'execution_status': execution_status, 'native_success': summary.get('success'), 'native_converged': summary.get('converged'), 'generation_success': summary.get('generation_success'), 'resource_status': summary.get('resource_status'), 'stop_reason': summary.get('stop_reason'), 'steps_taken': summary.get('steps_taken'), 'iteration_cap_reached': summary.get('resource_status') == 'RESOURCE_BUDGET_EXHAUSTED' or 'MAX_ITER' in str(summary.get('stop_reason', '')).upper() or 'BUDGET' in str(summary.get('stop_reason', '')).upper()}

def run_request(*, request_dir, selected_path, receipt_path, seed_pairs, candidate_runner, acceptance_checker, requested_argv, source_identity, realization_index=None, scientific_execution=False, accepted_artifact_publisher=None):
    request_dir, selected_path, receipt_path = map(Path, (request_dir, selected_path, receipt_path))
    request_dir.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': SCHEMA, 'request_id': uuid.uuid4().hex, 'realization_index': realization_index, 'request_dir': str(request_dir.resolve()), 'requested_argv': list(requested_argv), 'source_identity': source_identity, 'attempt_count': 0, 'max_attempts': len(seed_pairs), 'accepted_count': 0, 'selected_json_path': None, 'selected_json_sha256': None, 'terminal_status': 'IN_PROGRESS', 'attempts': [], 'scientific_execution': False, 'execution_kind': 'NATIVE_SCIENTIFIC_GENERATION' if scientific_execution else 'SYNTHETIC_CONTROL_FLOW_TEST'}
    _persist_request(receipt_path, receipt, exclusive=True)
    for index, (position_seed, size_seed) in enumerate(seed_pairs, 1):
        attempt_dir = request_dir / ('a%02d' % index)
        attempt_dir.mkdir()
        candidate_path = attempt_dir / 'g.json'
        attempt = {'attempt_index': index, 'position_seed_requested': position_seed, 'position_seed': None, 'size_seed_requested': size_seed, 'size_seed': None, 'size_seed_effective': None, 'generation_status': _generation_status(execution_status='STARTING'), 'independent_acceptance': {'status': 'NOT_RUN', 'performed': False, 'all_required_audits_passed': None, 'verdicts': None}, 'failure_reason': None, 'geometry_json_sha256': None, 'artifact_write_status': 'NOT_PUBLISHED', 'infrastructure_failure_reason': None, 'artifact_paths': {'geometry_json': None, 'stdout': str((attempt_dir / 'out.txt').resolve()), 'stderr': str((attempt_dir / 'err.txt').resolve()), 'size_admission_failure_receipts': []}, 'elapsed_seconds': None}
        receipt['attempts'].append(attempt)
        receipt['attempt_count'] = index
        _persist_request(receipt_path, receipt)
        start = time.perf_counter()
        try:
            receipt['scientific_execution'] = bool(scientific_execution)
            candidate_runner(candidate_path, position_seed, size_seed, attempt)
            if not candidate_path.is_file():
                raise GeometryDiagnostic('NATIVE_GEOMETRY_JSON_NOT_PRODUCED')
            attempt['artifact_paths']['geometry_json'] = str(candidate_path.resolve())
            attempt['geometry_json_sha256'] = _digest(candidate_path)
            acceptance = acceptance_checker(candidate_path)
            if acceptance.get('status') not in ('PASS', 'FAIL'):
                raise GeometryDiagnostic('UNKNOWN_INDEPENDENT_ACCEPTANCE_STATUS')
            attempt['generation_status'] = _generation_status(acceptance['optimization_summary'])
            attempt['independent_acceptance'] = {key: acceptance[key] for key in ('status', 'performed', 'all_required_audits_passed', 'verdicts')}
            attempt['independent_failure_reason'] = acceptance['failure_reason']
            state = acceptance['input_state']
            if state['seed_init'] != position_seed or (state['size_distribution_mode'] and state['seed_size_requested'] != size_seed):
                raise GeometryDiagnostic('NATIVE_ACTUAL_SEED_REQUEST_MISMATCH')
            attempt['position_seed'] = state['seed_init']
            attempt['size_seed'] = attempt['size_seed_effective'] = state['seed_size_effective']
            attempt['size_generation_seed_trace'] = (state.get('size_report') or {}).get('size_generation_seed_retries')
            attempt['failure_reason'] = acceptance['failure_reason']
            if acceptance['status'] == 'PASS':
                if accepted_artifact_publisher is not None:
                    accepted_artifact_publisher(candidate_path, attempt)
                selected_path.parent.mkdir(parents=True, exist_ok=True)
                selected_bytes = candidate_path.read_bytes()
                selected_sha = hashlib.sha256(selected_bytes).hexdigest()
                if selected_sha != attempt['geometry_json_sha256']:
                    raise GeometryDiagnostic('GENERATED_JSON_CHANGED_BEFORE_ACCEPTED_PUBLICATION')
                _publish_exclusive(selected_path, selected_bytes)
                attempt['artifact_write_status'] = 'ACCEPTED_OUTPUT_PUBLISHED'
                receipt.update(terminal_status='GEOMETRY_ACCEPTED', accepted_count=1, selected_json_path=str(selected_path.resolve()), selected_json_sha256=selected_sha)
            else:
                receipt['terminal_status'] = 'GEOMETRY_INDEPENDENT_ACCEPTANCE_FAILED'
        except BaseException as exc:
            admission_report = getattr(exc, 'size_generation_retry_report', None)
            if admission_report is not None:
                attempt['size_generation_seed_trace'] = admission_report
                attempts = admission_report.get('attempts', [])
                if attempts:
                    attempt['size_seed'] = attempt['size_seed_effective'] = attempts[-1].get('seed')
                attempt['generation_status'] = _generation_status(execution_status='SIZE_ADMISSION_EXHAUSTED')
                attempt['independent_acceptance'] = {'status': 'FATAL', 'performed': False, 'all_required_audits_passed': None, 'verdicts': None}
                attempt['failure_reason'] = 'SIZE_GENERATION_ADMISSION_EXHAUSTED'
                receipt['terminal_status'] = 'GEOMETRY_FATAL_DIAGNOSTIC'
            else:
                if attempt['independent_acceptance']['status'] not in ('PASS', 'FAIL'):
                    attempt['generation_status'] = _generation_status(execution_status='FATAL')
                    attempt['independent_acceptance'] = {'status': 'FATAL', 'performed': False, 'all_required_audits_passed': None, 'verdicts': None}
                else:
                    attempt['artifact_write_status'] = 'FATAL_BEFORE_ACCEPTED_OUTPUT_PUBLICATION'
                    attempt['infrastructure_failure_reason'] = type(exc).__name__ + ': ' + str(exc)
                attempt['failure_reason'] = type(exc).__name__ + ': ' + str(exc)
                receipt['terminal_status'] = 'GEOMETRY_FATAL_DIAGNOSTIC'
            if candidate_path.is_file():
                attempt['artifact_paths']['geometry_json'] = str(candidate_path.resolve())
                attempt['geometry_json_sha256'] = _digest(candidate_path)
        finally:
            attempt['elapsed_seconds'] = time.perf_counter() - start
            attempt['artifact_paths']['size_admission_failure_receipts'] = [str(path.resolve()) for path in sorted(attempt_dir.glob('g.json.size_generation_failure.*.json'))]
            _persist_request(receipt_path, receipt)
        if receipt['terminal_status'] in ('GEOMETRY_ACCEPTED', 'GEOMETRY_FATAL_DIAGNOSTIC'):
            break
    if receipt['terminal_status'] == 'GEOMETRY_INDEPENDENT_ACCEPTANCE_FAILED' and len(seed_pairs) > 1:
        receipt['terminal_status'] = 'GEOMETRY_ACCEPTANCE_EXHAUSTED'
        _persist_request(receipt_path, receipt)
    print('[geometry acceptance receipt] ' + str(receipt_path.resolve()))
    print('[geometry acceptance] ' + receipt['terminal_status'] + ' attempts=' + str(receipt['attempt_count']))
    return receipt

def _replace_options(argv, replacements):
    out, index = ([], 0)
    while index < len(argv):
        token = argv[index]
        key = token.split('=', 1)[0]
        if key in replacements:
            index += 1 if '=' in token else 2
        else:
            out.append(token)
            index += 1
    for key, value in replacements.items():
        if value is not None:
            out.extend([key, str(value)])
    return out

def _fresh_request_dir(parent):
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        path = parent / ('.g' + uuid.uuid4().hex[:12])
        if not path.exists():
            return path

def _raw_option(argv, name):
    for index, token in enumerate(argv):
        if token.startswith(name + '='):
            return token.split('=', 1)[1]
        if token == name and index + 1 < len(argv):
            return argv[index + 1]
    return None

def _validate_finite_configuration(value, path='configuration'):
    if isinstance(value, float) and (not math.isfinite(value)):
        raise GeometryDiagnostic('NONFINITE_CONFIGURATION: ' + path)
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_configuration(child, path + '.' + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_configuration(child, path + '[' + str(index) + ']')

def _fatal_preflight(argv, exc, source_identity=None):
    requested_receipt = _raw_option(argv, '--geometry-receipt')
    raw_max = _raw_option(argv, '--geometry-max-attempts')
    try:
        maximum = MAX_ATTEMPTS if raw_max is None else int(raw_max)
        if not 1 <= maximum <= MAX_ATTEMPTS:
            maximum = None
    except ValueError:
        maximum = None
    receipt = {'schema': SCHEMA, 'request_id': uuid.uuid4().hex, 'request_dir': None, 'requested_argv': list(argv), 'source_identity': source_identity, 'attempt_count': 0, 'max_attempts': maximum, 'requested_max_attempts': raw_max, 'accepted_count': 0, 'selected_json_path': None, 'selected_json_sha256': None, 'terminal_status': 'GEOMETRY_FATAL_DIAGNOSTIC', 'failure_reason': type(exc).__name__ + ': ' + str(exc), 'attempts': [], 'scientific_execution': False, 'execution_kind': 'READ_ONLY_PREFLIGHT_DIAGNOSTIC', 'requested_receipt_path': requested_receipt, 'receipt_path': None, 'receipt_persisted': False}
    try:
        export = _raw_option(argv, '--export-json')
        parent = Path(export).resolve().parent if export else Path.cwd() / 'outputs'
        request_dir = _fresh_request_dir(parent)
        request_dir.mkdir()
        receipt['request_dir'] = str(request_dir)
        receipt_path = Path(requested_receipt or request_dir / 'receipt.json').resolve()
        if receipt_path.exists():
            receipt_path = request_dir / 'receipt.json'
        receipt['receipt_path'] = str(receipt_path)
        receipt['receipt_persisted'] = True
        _write_json(receipt_path, receipt, exclusive=True)
        print('[geometry acceptance receipt] ' + str(receipt_path), file=sys.stderr)
    except (OSError, ValueError) as write_error:
        receipt['receipt_persisted'] = False
        receipt['receipt_persistence_error'] = str(write_error)
        print('[geometry fatal receipt unavailable] ' + str(write_error), file=sys.stderr)
    print('[geometry diagnostic JSON] ' + json.dumps(receipt, ensure_ascii=False, allow_nan=False), file=sys.stderr)
    print('[geometry fatal] ' + receipt['failure_reason'], file=sys.stderr)
    return 2

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    source_identity = None
    try:
        source_identity = verify_source_identity()
        from src import main as native
        parser = native.build_arg_parser()
        parser.add_argument('--geometry-max-attempts', type=int, choices=range(1, MAX_ATTEMPTS + 1), default=MAX_ATTEMPTS, help='Independent-acceptance generation attempts per requested geometry (initial + at most 20 replacements).')
        parser.add_argument('--geometry-receipt', help='Exclusive explicit acceptance receipt path; one JSON for single mode or batch ledger.')
        args = parser.parse_args(argv)
        if args.baseline_identity:
            print(json.dumps(source_identity, ensure_ascii=False, indent=2))
            return 0
        if args.show_effective_config:
            config = native._effective_cli_configuration(args)
            config['geometry_acceptance_retry'] = {'max_attempts': args.geometry_max_attempts, 'default_max_attempts': MAX_ATTEMPTS, 'source_identity': source_identity, 'acceptance_authority': 'unmodified native _final_audit_verdicts', 'scientific_execution': False}
            print(json.dumps(config, ensure_ascii=False, indent=2))
            return 0
        count = int(args.realizations)
        if count < 1 or args.seed < 0 or (args.size_seed is not None and args.size_seed < 0):
            raise GeometryDiagnostic('INVALID_REALIZATION_COUNT_OR_SEED')
        if args.fixed_demo and (count > 1 or args.geometry_max_attempts > 1):
            raise GeometryDiagnostic('FIXED_DEMO_REQUIRES_SINGLE_ATTEMPT_SINGLE_REALIZATION')
        if count > 1 and (args.nonperiodic or (args.export_json and (not args.batch_export_json_dir))):
            raise GeometryDiagnostic('BATCH_REQUIRES_PERIODIC_AND_BATCH_EXPORT_JSON_DIR')
        if count > 1 and args.spatial_stats:
            raise GeometryDiagnostic('BATCH_AGGREGATE_SPATIAL_STATISTICS_NOT_SUPPORTED_BY_ACCEPTANCE_RETRY_ENTRY')
        config = native._effective_cli_configuration(args)
        _validate_finite_configuration(vars(args), 'parsed_args')
        _validate_finite_configuration(config, 'resolved_native_configuration')
        cfg = config['descent_config']
        expected = {'physical_acceptance_overlap_tol': cfg['overlap_tol'], 'orientation_tol': cfg['orientation_tol'], 'orientation_target': cfg['orientation_target'], 'periodic': not args.nonperiodic, 'geometry_required': bool(cfg['include_geometry_distortion'] and cfg['geometry_stop']), 'orientation_required': bool(cfg['include_orientation_regularization'] and cfg['orientation_stop']), 'geometry_violation_tolerances': {kind: cfg['geometry_' + kind + '_violation_tol'] for kind in ('face', 'edge', 'corner')}}
        base_seed = args.seed if args.base_seed is None else args.base_seed
        if base_seed < 0:
            raise GeometryDiagnostic('INVALID_BASE_SEED')
        initial_pairs = [(args.seed, args.seed if args.size_seed is None else args.size_seed)] if count == 1 else [(base_seed + 1000 + r, (base_seed if args.size_seed is None else args.size_seed) + r) for r in range(count)]
        plans = plan_seed_pairs(initial_pairs, args.geometry_max_attempts, args.size_gen_seed_retries)
        native_argv = _replace_options(argv, {'--geometry-max-attempts': None, '--geometry-receipt': None})
        batch_parent = Path(args.batch_export_json_dir or args.batch_outdir or 'outputs').resolve()
        request_receipts = []
        batch_receipt_path = Path(args.geometry_receipt).resolve() if count > 1 and args.geometry_receipt else None
        if batch_receipt_path is not None and batch_receipt_path.exists():
            raise GeometryDiagnostic('RECEIPT_PATH_ALREADY_EXISTS')
        explicit_artifact_targets = [(attribute, filename) for attribute, filename in (('metrics_json', 'metrics.json'), ('metrics_summary_json', 'summary.json'), ('metrics_csv', 'metrics.csv')) if getattr(args, attribute, None)]
        if count == 1:
            for attribute, _ in explicit_artifact_targets:
                if Path(getattr(args, attribute)).exists():
                    raise GeometryDiagnostic('ACCEPTED_METRICS_PATH_ALREADY_EXISTS: ' + attribute)
            if args.spatial_stats and args.spatial_stats_outdir and Path(args.spatial_stats_outdir).exists():
                raise GeometryDiagnostic('ACCEPTED_SPATIAL_STATISTICS_DIRECTORY_ALREADY_EXISTS')
        for r, pairs in enumerate(plans):
            selected = Path(args.export_json).resolve() if count == 1 and args.export_json else batch_parent / ('real_%03d.json' % r) if count > 1 else None
            parent = selected.parent if selected else Path.cwd() / 'outputs'
            request_dir = _fresh_request_dir(parent)
            selected = selected or request_dir / 'accepted.json'
            receipt_path = Path(args.geometry_receipt).resolve() if count == 1 and args.geometry_receipt else request_dir / 'receipt.json'
            if selected.exists() or receipt_path.exists():
                raise GeometryDiagnostic('SELECTED_JSON_OR_RECEIPT_PATH_ALREADY_EXISTS')

            def generate(candidate_path, position, size, attempt):
                replacements = {'--realizations': 1, '--seed': position, '--size-seed': size, '--export-json': candidate_path, '--batch-export-json-dir': None, '--batch-outdir': None, '--base-seed': None}
                for option, attribute, filename in (('--metrics-json', 'metrics_json', 'metrics.json'), ('--metrics-summary-json', 'metrics_summary_json', 'summary.json'), ('--metrics-csv', 'metrics_csv', 'metrics.csv')):
                    if getattr(args, attribute, None) is not None:
                        replacements[option] = candidate_path.parent / filename
                if args.spatial_stats:
                    replacements['--spatial-stats-outdir'] = candidate_path.parent / 'stats'
                call_argv = _replace_options(native_argv, replacements)
                attempt['native_argv'] = call_argv
                old_argv = sys.argv
                try:
                    sys.argv = [old_argv[0], *call_argv]
                    with Path(attempt['artifact_paths']['stdout']).open('x', encoding='utf-8') as out, Path(attempt['artifact_paths']['stderr']).open('x', encoding='utf-8') as err, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        native.main()
                finally:
                    sys.argv = old_argv

            def publish_artifacts(candidate_path, attempt):
                published = {}
                if count == 1:
                    for attribute, filename in explicit_artifact_targets:
                        artifact = candidate_path.parent / filename
                        if artifact.is_file():
                            target = Path(getattr(args, attribute)).resolve()
                            target.parent.mkdir(parents=True, exist_ok=True)
                            _publish_exclusive(target, artifact.read_bytes())
                            published[attribute] = str(target)
                    if args.spatial_stats and args.spatial_stats_outdir:
                        target = Path(args.spatial_stats_outdir).resolve()
                        shutil.copytree(candidate_path.parent / 'stats', target)
                        published['spatial_stats_outdir'] = str(target)
                attempt['artifact_paths']['accepted_auxiliary_artifacts'] = published
            receipt = run_request(request_dir=request_dir, selected_path=selected, receipt_path=receipt_path, seed_pairs=pairs, candidate_runner=generate, acceptance_checker=lambda path: evaluate_export(path, expected), requested_argv=argv, source_identity=source_identity, realization_index=r if count > 1 else None, scientific_execution=True, accepted_artifact_publisher=publish_artifacts)
            request_receipts.append({'realization_index': r, 'receipt_path': str(receipt_path.resolve()), **receipt})
            if receipt['terminal_status'] == 'GEOMETRY_FATAL_DIAGNOSTIC':
                break
        if count > 1:
            batch_receipt_path = batch_receipt_path or _fresh_request_dir(batch_parent) / 'batch_receipt.json'
            accepted = sum((item['accepted_count'] for item in request_receipts))
            batch = {'schema': 'pw-rve-geometry-acceptance-retry-batch-v1', 'request_id': uuid.uuid4().hex, 'requested_geometry_count': count, 'completed_request_count': len(request_receipts), 'max_attempts_per_geometry': args.geometry_max_attempts, 'accepted_count': accepted, 'attempt_count': sum((item['attempt_count'] for item in request_receipts)), 'terminal_status': 'GEOMETRY_BATCH_ACCEPTED' if accepted == count else 'GEOMETRY_FATAL_DIAGNOSTIC' if any((item['terminal_status'] == 'GEOMETRY_FATAL_DIAGNOSTIC' for item in request_receipts)) else 'GEOMETRY_ACCEPTANCE_EXHAUSTED', 'source_identity': source_identity, 'requested_argv': argv, 'requests': request_receipts}
            _persist_request(batch_receipt_path, batch, exclusive=True)
            print('[geometry batch receipt] ' + str(batch_receipt_path))
        if any((item['terminal_status'] == 'GEOMETRY_FATAL_DIAGNOSTIC' for item in request_receipts)):
            return 2
        return 0 if len(request_receipts) == count and all((item['accepted_count'] == 1 for item in request_receipts)) else 3
    except GeometryReceiptError as exc:
        print('[geometry fatal] ' + str(exc), file=sys.stderr)
        print('[geometry diagnostic JSON] ' + json.dumps(exc.receipt, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 2
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        return _fatal_preflight(argv, exc, source_identity)
    except Exception as exc:
        return _fatal_preflight(argv, exc, source_identity)
if __name__ == '__main__':
    raise SystemExit(main())
