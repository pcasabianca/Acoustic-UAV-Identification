#!/usr/bin/env python3
"""CLI entrypoint to run all visualizations.

This module imports `6 - Visualization/scripts/viz_all.py` by path and calls its `run()`.
"""
from pathlib import Path
import argparse
import importlib.util
import sys


def _import_viz_all_module():
    base = Path(__file__).resolve().parent
    script_path = base / 'scripts' / 'viz_all.py'
    spec = importlib.util.spec_from_file_location('viz_all', str(script_path))
    module = importlib.util.module_from_spec(spec)
    # ensure repository root is on sys.path so `import config` in viz_all works
    repo_root = base.parent
    sys.path.insert(0, str(repo_root))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            # remove the entry we inserted
            sys.path.pop(0)
        except Exception:
            pass
    return module


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--perf-dir', default=None)
    p.add_argument('--results-dir', default=None)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--models', nargs='*', default=None)
    p.add_argument('--use-test-outputs', action='store_true', help='Write outputs into outputs_test/ instead of outputs/')
    p.add_argument('--snr-bins', default=None)
    p.add_argument('--strict', action='store_true')
    args = p.parse_args(argv)

    # determine if test outputs should be used (env var or flag)
    import os, sys
    use_test = args.use_test_outputs or (os.environ.get('VIS_TEST_OUTPUT', '') == '1')
    out_dir = args.out_dir
    if use_test:
        # ensure repo root is importable so we can import config
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        try:
            import config
            # place outputs_test next to configured VIZ_OUTPUT_DIR
            out_dir = str((Path(config.VIZ_OUTPUT_DIR).parent / 'outputs_test'))
        finally:
            try:
                sys.path.pop(0)
            except Exception:
                pass

    mod = _import_viz_all_module()
    res = mod.run(models=args.models, perf_dir=args.perf_dir, results_dir=args.results_dir, out_dir=out_dir,
                  snr_bins=args.snr_bins, strict=args.strict)

    # write manifest
    try:
        from datetime import datetime, timezone
        import json
        base_out = Path(out_dir) if out_dir else Path('6 - Visualization') / 'outputs'
        manifests_dir = base_out / 'manifests'
        manifests_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        manifest = {
            'timestamp': now.isoformat(),
            'params': {
                'models': args.models,
                'snr_bins': args.snr_bins,
                'strict': bool(args.strict),
            },
            'artifacts': res.get('artifacts') if isinstance(res, dict) else None,
        }
        mpath = manifests_dir / f"manifest_{now.strftime('%Y%m%d_%H%M%S')}.json"
        with open(mpath, 'w') as fh:
            json.dump(manifest, fh, indent=2)
        print('Wrote manifest:', str(mpath))
        # also copy manifest into the specific run folder (if viz_all returned one)
        try:
            run_out = res.get('out_dir') if isinstance(res, dict) else None
            if run_out:
                run_manifests = Path(run_out) / 'manifests'
                run_manifests.mkdir(parents=True, exist_ok=True)
                run_mpath = run_manifests / mpath.name
                with open(run_mpath, 'w') as fh2:
                    json.dump(manifest, fh2, indent=2)
                print('Also wrote run manifest:', str(run_mpath))
        except Exception:
            pass
    except Exception as e:
        print('Failed to write manifest:', e, file=sys.stderr)

    # exit code semantics: if strict and MVP missing => non-zero
    try:
        m = res.get('artifacts', {})
        has_perf_index = any('perf_index.csv' in str(p) for p in m.get('summary', []))
        has_perf_plot = len(m.get('perf_vs_snr', [])) > 0
        if args.strict and not (has_perf_index and has_perf_plot):
            print('Strict mode: MVP artifacts missing', file=sys.stderr)
            sys.exit(2)
    except Exception:
        pass

    print(res)


if __name__ == '__main__':
    main()
