import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIENTATION_MODES = ('off', '1d-x', '2d-xz', '3d-isotropic')
ORIENTATION_TENSORS = {
    '1d-x': [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    '2d-xz': [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
    '3d-isotropic': [[1.0 / 3.0, 0.0, 0.0], [0.0, 1.0 / 3.0, 0.0], [0.0, 0.0, 1.0 / 3.0]]
}


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('Expected a finite numeric value.') from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError('Expected a finite numeric value.')
    return result


def positive(value):
    result = finite(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('Expected a positive value.')
    return result


def nonnegative(value):
    result = finite(value)
    if result < 0:
        raise argparse.ArgumentTypeError('Expected a nonnegative value.')
    return result


def count(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('Expected an integer.') from exc
    if result < 1:
        raise argparse.ArgumentTypeError('Expected an integer >= 1.')
    return result


def seed_value(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('Expected an integer seed.') from exc
    if result < 0:
        raise argparse.ArgumentTypeError('Expected a nonnegative integer seed.')
    return result


def aspect(value):
    parts = value.split(':')
    try:
        numbers = [finite(part) for part in parts[1:]]
        if parts[0] == 'fixed' and len(numbers) == 1 and numbers[0] >= 1:
            return {'mode': 'fixed', 'ratio': numbers[0]}
        if parts[0] == 'uniform' and len(numbers) == 2 and 1 <= numbers[0] < numbers[1]:
            return {'mode': 'uniform', 'lower': numbers[0], 'upper': numbers[1]}
    except argparse.ArgumentTypeError:
        pass
    raise argparse.ArgumentTypeError('Expected fixed:R with R>=1, or uniform:LO:HI with 1<=LO<HI.')


def parser():
    p = argparse.ArgumentParser(description='Generate independently accepted periodic ellipsoidal geometry using the bound scientific engine.')
    p.add_argument('--profile', choices=('vf30-reference',), default='vf30-reference')
    p.add_argument('--box', nargs=3, type=positive, metavar=('LX', 'LY', 'LZ'))
    p.add_argument('--vf', type=positive)
    p.add_argument('--vf-tol', type=nonnegative)
    p.add_argument('--deq-truncnorm', nargs=4, type=positive, metavar=('MU', 'SIGMA', 'MIN', 'MAX'))
    p.add_argument('--aspect', type=aspect)
    p.add_argument('--min-gap-factor', type=nonnegative)
    p.add_argument('--geo', choices=('on', 'off'))
    widths = p.add_mutually_exclusive_group()
    widths.add_argument('--geo-width', type=positive)
    widths.add_argument('--geo-widths', nargs=3, type=positive, metavar=('FACE', 'EDGE', 'CORNER'))
    tolerances = p.add_mutually_exclusive_group()
    tolerances.add_argument('--geo-tol', type=nonnegative)
    tolerances.add_argument('--geo-tols', nargs=3, type=nonnegative, metavar=('FACE', 'EDGE', 'CORNER'))
    p.add_argument('--orientation', choices=ORIENTATION_MODES)
    p.add_argument('--orientation-tol', type=nonnegative, help='Orientation tolerance used by the native generation and final acceptance rules.')
    p.add_argument('--overlap-tol', type=nonnegative)
    p.add_argument('--realizations', type=count)
    p.add_argument('--seed', type=seed_value, help='Optional explicit reproducibility seed; no recommended seed value.')
    p.add_argument('--size-seed', type=seed_value, help='Optional independent size/shape seed; omission inherits the effective placement seed.')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--show-config', action='store_true', help='Resolve configuration without scientific imports, calculation or output creation.')
    return p


def resolve(argv=None):
    p = parser()
    args = p.parse_args(argv)
    profile_path = ROOT / 'config' / (args.profile + '.json')
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    defaults = profile['geometry']
    engine = profile['engine']
    selected = lambda name, default: default if getattr(args, name) is None else getattr(args, name)
    box = selected('box', defaults['box_um'])
    vf = selected('vf', defaults['vf'])
    if not 0 < vf < 1:
        p.error('--vf must lie strictly between 0 and 1.')
    diameters = selected('deq_truncnorm', defaults['deq_truncnorm_um'])
    if not diameters[2] < diameters[3]:
        p.error('--deq-truncnorm requires MIN < MAX.')
    shape = selected('aspect', aspect(defaults['aspect']))
    if args.geo_width is not None:
        widths = [args.geo_width] * 3
    else:
        widths = selected('geo_widths', defaults['geo_widths_um'])
    if args.geo_tol is not None:
        tolerances = [args.geo_tol] * 3
    else:
        tolerances = selected('geo_tols', defaults['geo_tols_um'])
    mode = selected('orientation', defaults['orientation'])
    factor = selected('min_gap_factor', defaults['min_gap_factor'])
    rho_max = shape['ratio'] if shape['mode'] == 'fixed' else shape['upper']
    minimum_semiminor_axis = (diameters[2] / 2.0) * rho_max ** (-1.0 / 3.0)
    delta = factor * minimum_semiminor_axis
    if not math.isfinite(delta):
        p.error('Resolved minimum gap must be finite.')
    result = {
        'schema': 'cbge-geometry-resolved-v2.2',
        'profile': {'name': args.profile, 'version': profile['version'], 'sha256': hashlib.sha256(profile_bytes).hexdigest()},
        'box_um': box, 'vf': vf, 'vf_tol': selected('vf_tol', defaults['vf_tol']),
        'deq_truncnorm_um': diameters, 'aspect': shape,
        'min_gap_factor': factor, 'minimum_gap_reference': defaults['min_gap_reference'],
        'theoretical_minimum_semiminor_axis_um': minimum_semiminor_axis, 'delta_min_um': delta,
        'geo': selected('geo', defaults['geo']), 'geo_widths_um': widths, 'geo_tols_um': tolerances,
        'orientation': mode, 'orientation_target_tensor': ORIENTATION_TENSORS.get(mode),
        'orientation_tol': selected('orientation_tol', defaults['orientation_tol']),
        'orientation_acceptance_rule': defaults['orientation_acceptance_rule'] if mode != 'off' else 'disabled',
        'overlap_tol': selected('overlap_tol', defaults['overlap_tol']),
        'realizations': selected('realizations', defaults['realizations']),
        'seed': engine['native_default_seed'] if args.seed is None else args.seed,
        'seed_selection': 'unchanged_native_default' if args.seed is None else 'explicit_user_argument',
        'size_seed': (engine['native_default_seed'] if args.seed is None else args.seed) if args.size_seed is None else args.size_seed,
        'size_seed_selection': 'inherit_effective_placement_seed' if args.size_seed is None else 'explicit_user_argument',
        'output_dir': None if args.output_dir is None else str(args.output_dir.resolve()),
        'engine': engine, 'scientific_execution': False,
        'original_geometry_source_sha256': profile['original_geometry_source_sha256']
    }
    if not args.show_config and args.output_dir is None:
        p.error('--output-dir is required for generation.')
    if not args.show_config and args.output_dir.exists():
        p.error('--output-dir must be fresh; existing output is preserved.')
    if not args.show_config and args.output_dir is not None:
        target = args.output_dir.resolve()
        if target == ROOT or ROOT in target.parents or target in ROOT.parents:
            p.error('Output must be outside the public source tree.')
    result['native_argv'] = translate(result)
    return args, result


def translate(config):
    argv = ['--box', *map(str, config['box_um']), '--vf-target', str(config['vf']),
            '--vf-abs-tol', str(config['vf_tol']), '--deq-mu', str(config['deq_truncnorm_um'][0]),
            '--deq-sigma', str(config['deq_truncnorm_um'][1]), '--deq-min', str(config['deq_truncnorm_um'][2]),
            '--deq-max', str(config['deq_truncnorm_um'][3]), '--aspect-ratio-mode', config['aspect']['mode'],
            '--delta-min', str(config['delta_min_um']), '--realizations', str(config['realizations']),
            '--steps', str(config['engine']['local_max_steps']), '--collect-metrics',
            '--pair-method', config['engine']['pair_method'], '--line-search', '--extent-filter',
            '--extent-sep-tol', str(config['engine']['extent_sep_tol']), '--overlap-tol', str(config['overlap_tol']),
            '--geometry-distortion' if config['geo'] == 'on' else '--no-geometry-distortion',
            '--orientation-regularization' if config['orientation'] != 'off' else '--no-orientation-regularization',
            '--orientation-target', config['orientation'] if config['orientation'] != 'off' else '3d-isotropic',
            '--orientation-tol', str(config['orientation_tol']),
            '--geo-d-face', str(config['geo_widths_um'][0]), '--geo-d-edge', str(config['geo_widths_um'][1]),
            '--geo-d-corner', str(config['geo_widths_um'][2]),
            '--geo-face-violation-tol', str(config['geo_tols_um'][0]), '--geo-edge-violation-tol', str(config['geo_tols_um'][1]),
            '--geo-corner-violation-tol', str(config['geo_tols_um'][2])]
    if config['aspect']['mode'] == 'fixed':
        argv.extend(['--aspect-ratio-k', str(config['aspect']['ratio'])])
    else:
        argv.extend(['--aspect-ratio-min', str(config['aspect']['lower']), '--aspect-ratio-max', str(config['aspect']['upper'])])
    if config['seed_selection'] == 'explicit_user_argument':
        argv.extend(['--seed', str(config['seed'])])
    if config['size_seed_selection'] == 'explicit_user_argument':
        argv.extend(['--size-seed', str(config['size_seed'])])
    return argv
