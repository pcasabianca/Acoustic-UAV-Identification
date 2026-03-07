#!/usr/bin/env python3
"""Orchestrate TFLite conversion and validation for a list of models.

For each model:
  1. Convert the Keras model to TFLite via keras_to_tflite_converter.py
  2. Run tflite_conv_tester.py to produce a per-sample comparison JSON

Reads deployment_config.json to find Keras model filenames.
Reads config.py for TFLITE_DIR, TFLITE_QUANTIZATION, and TFLITE_COMPARISON_DIR.

Usage (called by run_full_pipeline.sh):
    python tflite_conv_orchestrator.py --models CNN RNN CRNN
    python tflite_conv_orchestrator.py --models CNN --split test --force

Exit code: 0 if all models succeeded, 1 if any failed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Import sibling modules by path so this works regardless of cwd
import importlib.util as _ilu


def _import_sibling(name: str):
    p = Path(__file__).parent / f'{name}.py'
    spec = _ilu.spec_from_file_location(name, str(p))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_converter = _import_sibling('keras_to_tflite_converter')
_tester    = _import_sibling('tflite_conv_tester')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_deployment_config() -> dict:
    cfg_path = Path(__file__).parent.parent / '8 - Deployment' / 'deployment_config.json'
    with open(cfg_path) as f:
        return json.load(f)


def _keras_path_for(model_name: str, deployment_cfg: dict) -> Path | None:
    """Return the .keras file path for model_name, or None if not declared."""
    available = deployment_cfg['models']['available_models']
    key = model_name  # names in deployment_config are mixed-case (CNN, RNN, etc.)
    if key not in available:
        return None
    filename = available[key].get('filename')
    if not filename:
        return None
    models_dir = Path(Path(__file__).parent.parent / '8 - Deployment' / deployment_cfg['models']['directory'])
    # Also accept absolute path from config
    keras_p = models_dir / filename
    # Fall back to config.MODELS_DIR
    if not keras_p.exists():
        keras_p = config.MODELS_DIR / filename
    return keras_p


# ---------------------------------------------------------------------------
# Per-model orchestration
# ---------------------------------------------------------------------------

def process_model(model_name: str, split: str, force: bool,
                  quantize: str, deployment_cfg: dict) -> dict:
    """Convert + test one model. Returns result dict."""
    model_upper = model_name.upper()
    result = {
        'model': model_upper,
        'convert_status': None,
        'test_status': None,
        'error': None,
        'elapsed_s': None,
    }
    t0 = time.time()

    # --- Locate Keras file ---
    keras_path = _keras_path_for(model_upper, deployment_cfg)
    if keras_path is None or not keras_path.exists():
        # Also try config.MODELS_DIR directly
        keras_path = config.MODELS_DIR / f'{model_name.lower()}_model.keras'
    if not keras_path.exists():
        result['convert_status'] = 'SKIPPED_NOT_FOUND'
        result['error'] = f'.keras file not found for {model_upper}'
        print(f'  [{model_upper}] SKIP — keras file not found')
        return result

    # --- TFLite output path ---
    tflite_path = config.TFLITE_DIR / f'{model_name.lower()}_model.tflite'

    # --- Step 1: Convert ---
    # Apply per-model quantization override from config when defined
    _overrides = getattr(config, 'TFLITE_QUANTIZATION_OVERRIDES', {})
    model_quantize = _overrides.get(model_upper, quantize)
    if model_quantize != quantize:
        print(f'  [{model_upper}] Quantization override: {quantize} → {model_quantize}')

    if tflite_path.exists() and not force:
        print(f'  [{model_upper}] SKIP conversion — {tflite_path.name} exists (--force to redo)')
        result['convert_status'] = 'SKIPPED_EXISTS'
    else:
        print(f'\n[{model_upper}] Converting ...')
        conv_res = _converter.convert(keras_path, tflite_path, quantize=model_quantize)
        result['convert_status'] = conv_res['status']
        if conv_res['status'] != 'OK':
            result['error'] = conv_res.get('error')
            result['elapsed_s'] = round(time.time() - t0, 1)
            return result

    # --- Step 2: Test ---
    comparison_path = (
        config.TFLITE_COMPARISON_DIR /
        f'{model_name.lower()}_{split}_tflite_comparison.json'
    )
    if comparison_path.exists() and not force:
        print(f'  [{model_upper}] SKIP testing — comparison JSON exists (--force to redo)')
        result['test_status'] = 'SKIPPED_EXISTS'
    else:
        print(f'\n[{model_upper}] Running TFLite vs Keras comparison ...')
        try:
            cmp_result = _tester.compare(model_upper, split)
            mean_delta = cmp_result['metadata']['mean_abs_delta']
            # Fidelity thresholds — independent of classification threshold
            # POOR: mean |Δprob| > 5% — the TFLite model diverges noticeably
            # WARN: mean |Δprob| > 1% — small but non-negligible drift
            if mean_delta > 0.05:
                result['test_status'] = f'POOR (mean_delta={mean_delta:.4f})'
            elif mean_delta > 0.01:
                result['test_status'] = f'WARN (mean_delta={mean_delta:.4f})'
            else:
                result['test_status'] = 'OK'
        except Exception as e:
            result['test_status'] = 'FAILED'
            result['error'] = str(e)
            print(f'  [{model_upper}] Test FAILED: {e}')

    result['elapsed_s'] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', required=True,
                    help='Model names to process (e.g. CNN RNN CONFORMER)')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                    help='Dataset split used for comparison (default: test)')
    ap.add_argument('--force', action='store_true',
                    help='Re-convert and re-test even if outputs exist')
    ap.add_argument('--quantize', default=None, choices=['none', 'float16'],
                    help='Quantization (default: from config.TFLITE_QUANTIZATION)')
    args = ap.parse_args()

    quantize = args.quantize or getattr(config, 'TFLITE_QUANTIZATION', 'float16')
    deployment_cfg = _load_deployment_config()

    print(f'\nTFLite Conversion Orchestrator')
    print(f'  models     : {args.models}')
    print(f'  split      : {args.split}')
    print(f'  quantize   : {quantize}')
    print(f'  force      : {args.force}')
    print(f'  output dir : {config.TFLITE_DIR}')
    print()

    results = []
    for model in args.models:
        res = process_model(model, args.split, args.force, quantize, deployment_cfg)
        results.append(res)

    # --- Summary ---
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    ok      = [r for r in results if r.get('test_status') in ('OK', 'SKIPPED_EXISTS')
               and r.get('convert_status') in ('OK', 'SKIPPED_EXISTS')]
    warned  = [r for r in results if (r.get('test_status') or '').startswith('WARN')
               and r.get('convert_status') in ('OK', 'SKIPPED_EXISTS')]
    poor    = [r for r in results if (r.get('test_status') or '').startswith('POOR')
               and r.get('convert_status') in ('OK', 'SKIPPED_EXISTS')]
    skipped = [r for r in results if 'SKIPPED' in (r.get('convert_status') or '')
               and r.get('test_status') in (None, 'SKIPPED_EXISTS')]
    failed  = [r for r in results if (r.get('convert_status') or '').startswith('FAILED')
               or (r.get('test_status') or '').startswith('FAILED')]

    for r in results:
        conv = r['convert_status'] or '?'
        test = r['test_status'] or 'not run'
        elapsed = f'{r["elapsed_s"]}s' if r.get('elapsed_s') else ''
        if r in ok:
            status = 'OK'
        elif r in warned:
            status = 'WARN'
        elif r in poor:
            status = 'POOR'
        elif r in failed:
            status = 'FAIL'
        else:
            status = 'SKIP'
        print(f'  {r["model"]:<20} convert={conv:<18} test={test:<30} [{status}] {elapsed}')

    print(f'\n  Total: {len(results)}  |  OK: {len(ok)}  |  Warn: {len(warned)}  |  Poor: {len(poor)}  |  Skipped: {len(skipped)}  |  Failed: {len(failed)}')

    if failed:
        print('\nFailed models:')
        for r in failed:
            print(f'  - {r["model"]}: {r.get("error", "unknown error")}')

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
