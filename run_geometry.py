import json
from public_cli.geometry_interface import resolve


def main(argv=None, dispatch=None):
    args, config = resolve(argv)
    if args.show_config:
        print(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if dispatch is None:
        from public_cli.geometry_adapter import run_geometry
        dispatch = run_geometry
    receipt = dispatch(config['native_argv'], args.output_dir, config)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if receipt.get('accepted_count') == config['realizations'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
