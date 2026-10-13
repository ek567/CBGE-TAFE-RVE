import contextlib
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

GEOMETRY_SOURCE_SHA256 = 'ea8eec0848d9ec3b8a80c68aca8f4843828122dbeed53a62f137174583604d37'
ACCEPTANCE_SCHEMA = 'cbge-tafe-geometry-acceptance-v2.2'
ACCEPTANCE_AUTHORITY = 'NATIVE_FINAL_GEOMETRY_AUDITS'
ACCEPTANCE_FILENAME = 'geometry_acceptance.json'


def evaluate_public_export(path, expected, original_evaluator):
    return original_evaluator(path, expected)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _write_exclusive(path, value):
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(encoded)


def build_acceptance_sidecar(geometry_path, request_receipt, source_identity,
                             resolved_configuration, native_configuration):
    geometry_path = Path(geometry_path).resolve()
    if request_receipt['terminal_status'] != 'GEOMETRY_ACCEPTED' or request_receipt['accepted_count'] != 1:
        raise ValueError('ACCEPTANCE_SIDECAR_REQUIRES_ACCEPTED_REQUEST')
    geometry_sha = _digest(geometry_path)
    if geometry_sha != request_receipt['selected_json_sha256']:
        raise ValueError('ACCEPTED_GEOMETRY_HASH_MISMATCH')
    accepted = [attempt for attempt in request_receipt['attempts']
                if attempt['independent_acceptance']['status'] == 'PASS']
    if len(accepted) != 1:
        raise ValueError('ACCEPTED_ATTEMPT_COUNT_MISMATCH')
    attempt = accepted[0]
    native_audits = attempt['independent_acceptance']['verdicts']
    data = json.loads(geometry_path.read_text(encoding='utf-8-sig'))
    summary = data['optimization_summary']
    if (attempt['independent_acceptance']['performed'] is not True or
            attempt['independent_acceptance']['all_required_audits_passed'] is not True or
            native_audits['all_required_audits_passed'] is not True or
            native_audits != summary['final_audits']):
        raise ValueError('ACCEPTED_NATIVE_AUDIT_BINDING_MISMATCH')
    orientation = summary['final_orientation']
    native_orientation = {'target': orientation['orientation_target'],
                          'required': native_audits['orientation_required'],
                          'tolerance': orientation['tolerance'],
                          'max_abs_deviation': orientation['max_abs_deviation'],
                          'is_target_aligned': orientation['is_target_aligned']}
    return {'schema': ACCEPTANCE_SCHEMA, 'status': 'GEOMETRY_ACCEPTED',
            'terminal_status': 'GEOMETRY_ACCEPTED', 'accepted_count': 1,
            'geometry_json_path': str(geometry_path), 'geometry_json_sha256': geometry_sha,
            'source_identity': dict(source_identity), 'request_id': request_receipt['request_id'],
            'scientific_execution': request_receipt['scientific_execution'],
            'execution_kind': request_receipt['execution_kind'],
            'realization_index': request_receipt['realization_index'],
            'acceptance': {'authority': ACCEPTANCE_AUTHORITY, 'metric': 'native_orientation',
                           'adapter_sha256': _digest(__file__),
                           'all_required_audits_passed': True,
                           'native_final_audits': native_audits,
                           'native_orientation': native_orientation,
                           'native_success_is_separate': True,
                           'native_raw_json_preserved': True},
            'seed_lineage': {'position_seed_requested': attempt['position_seed_requested'],
                             'position_seed_actual': attempt['position_seed'],
                             'size_seed_requested': attempt['size_seed_requested'],
                             'size_seed_effective': attempt['size_seed_effective'],
                             'size_generation_seed_trace': attempt.get('size_generation_seed_trace'),
                             'accepted_attempt_index': attempt['attempt_index'],
                             'attempt_count': request_receipt['attempt_count'],
                             'max_attempts': request_receipt['max_attempts']},
            'native_generation_status': attempt['generation_status'],
            'resolved_configuration': resolved_configuration,
            'resolved_configuration_sha256': _canonical_sha(resolved_configuration),
            'native_configuration': native_configuration,
            'native_configuration_sha256': _canonical_sha(native_configuration)}


@contextlib.contextmanager
def _backend_context():
    geometry_root = Path(__file__).resolve().parents[1] / 'geometry'
    for name, module in tuple(sys.modules.items()):
        if name == 'src' or name.startswith('src.') or name == 'geometry_acceptance_retry':
            location = getattr(module, '__file__', None)
            if location is not None:
                resolved = Path(location).resolve()
                if geometry_root.resolve() not in resolved.parents:
                    raise RuntimeError('AMBIGUOUS_GEOMETRY_BACKEND_MODULE: ' + name)
    original_path = list(sys.path)
    original_cwd = Path.cwd()
    sys.path.insert(0, str(geometry_root))
    os.chdir(geometry_root)
    try:
        retry = importlib.import_module('geometry_acceptance_retry')
        source_identity = retry.verify_source_identity()
        if source_identity.get('original_source_identity_sha256') != GEOMETRY_SOURCE_SHA256:
            raise RuntimeError('GEOMETRY_SOURCE_IDENTITY_MISMATCH')
        native = importlib.import_module('src.main')
        yield retry, native, source_identity
    finally:
        sys.path[:] = original_path
        os.chdir(original_cwd)


def run_geometry(native_argv, output_dir, resolved_configuration):
    if not isinstance(native_argv, list) or any(not isinstance(value, str) for value in native_argv):
        raise ValueError('NATIVE_ARGV_MUST_BE_INTERNAL_STRING_LIST')
    protected_options = {'--export-json', '--geometry-receipt', '--geometry-max-attempts',
                         '--batch-outdir', '--batch-export-json-dir', '--base-seed'}
    if any(value.split('=', 1)[0] in protected_options for value in native_argv):
        raise ValueError('OUTPUT_OR_RETRY_CONTROL_IS_ADAPTER_OWNED')
    output = Path(output_dir).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    if output == repository_root or repository_root in output.parents or output in repository_root.parents:
        raise ValueError('GEOMETRY_OUTPUT_MUST_BE_OUTSIDE_PUBLIC_SOURCE_TREE')
    if output.exists():
        raise FileExistsError('GEOMETRY_OUTPUT_DIRECTORY_ALREADY_EXISTS')
    with _backend_context() as (retry, native, source_identity):
        args = native.build_arg_parser().parse_args(native_argv)
        if args.nonperiodic or args.fixed_demo or args.spatial_stats:
            raise ValueError('PUBLIC_GEOMETRY_REQUIRES_PERIODIC_SAMPLING_WITHOUT_DEMO_OR_SPATIAL_AGGREGATES')
        config = native._effective_cli_configuration(args)
        retry._validate_finite_configuration(vars(args), 'parsed_args')
        retry._validate_finite_configuration(config, 'resolved_native_configuration')
        count = int(args.realizations)
        if count < 1 or args.seed < 0 or (args.size_seed is not None and args.size_seed < 0):
            raise ValueError('INVALID_REALIZATION_COUNT_OR_SEED')
        descent = config['descent_config']
        expected = {'physical_acceptance_overlap_tol': descent['overlap_tol'],
                    'orientation_tol': descent['orientation_tol'],
                    'orientation_target': descent['orientation_target'], 'periodic': True,
                    'geometry_required': bool(descent['include_geometry_distortion'] and descent['geometry_stop']),
                    'orientation_required': bool(descent['include_orientation_regularization'] and descent['orientation_stop']),
                    'geometry_violation_tolerances': {kind: descent['geometry_' + kind + '_violation_tol']
                                                     for kind in ('face', 'edge', 'corner')}}
        initial_pairs = ([(args.seed, args.seed if args.size_seed is None else args.size_seed)] if count == 1 else
                         [(args.seed + 1000 + index, (args.seed if args.size_seed is None else args.size_seed) + index)
                          for index in range(count)])
        plans = retry.plan_seed_pairs(initial_pairs, retry.MAX_ATTEMPTS, args.size_gen_seed_retries)
        output.mkdir(parents=True, exist_ok=False)
        requests = []
        for index, pairs in enumerate(plans):
            realization_dir = output / ('realization_%03d' % (index + 1))
            realization_dir.mkdir()
            geometry_path = realization_dir / 'geometry.json'
            receipt_path = realization_dir / 'native_retry_receipt.json'
            def generate(candidate_path, position, size, attempt):
                call_argv = retry._replace_options(native_argv,
                    {'--realizations': 1, '--seed': position, '--size-seed': size,
                     '--export-json': candidate_path})
                attempt['native_argv'] = call_argv
                original_argv = sys.argv
                try:
                    sys.argv = [original_argv[0], *call_argv]
                    with Path(attempt['artifact_paths']['stdout']).open('x', encoding='utf-8') as stdout, \
                         Path(attempt['artifact_paths']['stderr']).open('x', encoding='utf-8') as stderr, \
                         contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        native.main()
                finally:
                    sys.argv = original_argv
            request = retry.run_request(request_dir=realization_dir / 'attempts', selected_path=geometry_path,
                receipt_path=receipt_path, seed_pairs=pairs, candidate_runner=generate,
                acceptance_checker=lambda path: evaluate_public_export(path, expected, retry.evaluate_export),
                requested_argv=native_argv, source_identity=source_identity, realization_index=index,
                scientific_execution=True)
            item = {'realization_index': index, 'native_retry_receipt_path': str(receipt_path),
                    'terminal_status': request['terminal_status'], 'accepted_count': request['accepted_count'],
                    'attempt_count': request['attempt_count'], 'geometry_json_path': request['selected_json_path'],
                    'geometry_json_sha256': request['selected_json_sha256'], 'acceptance_sidecar_path': None}
            if request['terminal_status'] == 'GEOMETRY_ACCEPTED':
                sidecar = build_acceptance_sidecar(geometry_path, request, source_identity, resolved_configuration, config)
                sidecar_path = realization_dir / ACCEPTANCE_FILENAME
                _write_exclusive(sidecar_path, sidecar)
                item['acceptance_sidecar_path'] = str(sidecar_path)
                item['acceptance_sidecar_sha256'] = _digest(sidecar_path)
            requests.append(item)
            if request['terminal_status'] == 'GEOMETRY_FATAL_DIAGNOSTIC':
                break
        accepted_count = sum(item['accepted_count'] for item in requests)
        status = (('GEOMETRY_ACCEPTED' if count == 1 else 'GEOMETRY_BATCH_ACCEPTED') if accepted_count == count else
                  'GEOMETRY_FATAL_DIAGNOSTIC' if any(item['terminal_status'] == 'GEOMETRY_FATAL_DIAGNOSTIC'
                                                   for item in requests) else 'GEOMETRY_ACCEPTANCE_EXHAUSTED')
        report = {'schema': 'cbge-tafe-public-geometry-run-v2.2', 'status': status,
                  'requested_geometry_count': count, 'completed_request_count': len(requests),
                  'accepted_count': accepted_count, 'attempt_count': sum(item['attempt_count'] for item in requests),
                  'source_identity': source_identity, 'requests': requests,
                  'resolved_configuration': resolved_configuration,
                  'resolved_configuration_sha256': _canonical_sha(resolved_configuration),
                  'native_configuration': config, 'native_configuration_sha256': _canonical_sha(config),
                  'scientific_execution': True, 'fe_execution': False,
                  'native_raw_geometry_json_preserved': True,
                  'orientation_acceptance_rule': 'native_orientation',
                  'acceptance_adapter_sha256': _digest(__file__)}
        report_path = output / 'geometry_run_receipt.json'
        _write_exclusive(report_path, report)
        report['receipt_path'] = str(report_path)
        return report
