#!/usr/bin/env python3
"""Compare a Keras model against its TFLite equivalent on a dataset split.

Input:
  - Existing Keras perf JSON produced by Universal_Perf_Tester (contains
    per-sample keras_prob and true_label — no Keras re-inference needed).
  - The corresponding .tflite file (loaded via the TFLite Interpreter).
  - The MEL feature split (loaded via feature_loader to run TFLite inference).

Output: JSON written to config.TFLITE_COMPARISON_DIR/{model}_{split}_tflite_comparison.json
  {
    "metadata": {
      "model": "CNN",
      "split": "test",
      "quantization": "float16",
      "num_samples": 1234,
      "keras_accuracy": 0.952,
      "tflite_accuracy": 0.951,
      "agreement_rate": 0.998,
      "mean_abs_delta": 0.0014,
      "max_abs_delta": 0.031
    },
    "samples": [
      {
        "filename": "drone_500m_001.wav",
        "true_label": 1,
        "keras_prob": 0.872,
        "tflite_prob": 0.873,
        "keras_pred": 1,
        "tflite_pred": 1,
        "agreed": true
      }, ...
    ]
  }

Usage:
    python tflite_conv_tester.py --model CNN --split test
    python tflite_conv_tester.py --model CNN --split test --threshold 0.45
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from feature_loader import load_mel_features

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_keras_perf(model_name: str, split: str) -> dict:
    """Load the performance JSON saved by Universal_Perf_Tester.

    If the JSON does not exist, Universal_Perf_Tester is automatically invoked
    to generate it before loading, using the same split and the calibrated
    threshold from config.MODEL_THRESHOLDS when available.
    """
    path = config.PERFORMANCE_DIR / f'{model_name.lower()}_{split}.json'

    if not path.exists():
        print(f'  [INFO] Perf JSON not found for {model_name}/{split} — running Universal_Perf_Tester ...')
        _run_perf_tester(model_name, split)

    if not path.exists():
        raise FileNotFoundError(
            f'Keras perf JSON still missing after running Universal_Perf_Tester: {path}'
        )

    with open(path) as f:
        return json.load(f)


def _run_perf_tester(model_name: str, split: str) -> None:
    """Invoke Universal_Perf_Tester as a subprocess to generate the perf JSON.

    Uses the same Python interpreter as the current process and inherits the
    calibrated threshold from config.MODEL_THRESHOLDS when defined.
    """
    perf_tester = (
        Path(__file__).parent.parent
        / '3 - Single Model Performance Calculation'
        / 'Universal_Perf_Tester.py'
    )

    cmd = [sys.executable, str(perf_tester), '--model', model_name.upper(), '--split', split]

    # Pass calibrated threshold if available in config
    thresholds = getattr(config, 'MODEL_THRESHOLDS', {})
    threshold = thresholds.get(model_name.upper())
    if threshold is not None:
        cmd += ['--threshold', str(threshold)]

    print(f'  [INFO] Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f'Universal_Perf_Tester failed for {model_name}/{split} '
            f'(exit code {result.returncode})'
        )


def _index_keras_preds(perf: dict) -> dict[str, dict]:
    """Return {filename: {keras_prob, true_label, keras_pred}} from perf JSON."""
    index = {}
    for rec in perf.get('predictions', []):
        index[rec['filename']] = {
            'keras_prob': float(rec['prob_class1']),
            'true_label': int(rec['true_label']),
            'keras_pred': int(rec['pred_label']),
        }
    return index


def _run_tflite(tflite_path: Path, features: np.ndarray) -> np.ndarray:
    """Run TFLite inference on a batch of features. Returns prob_class1 array (N,).

    Handles both output shapes:
      - (1, 1): binary sigmoid — index [0, 0] is prob_class1
      - (1, 2): categorical softmax — index [0, 1] is prob_class1
    Also adapts input dims if the TFLite model expects a different number of
    dimensions than what was passed in (e.g. Conformer expects 3-D, no channel).
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # Determine expected input dtype (float32 or float16 for quantized models)
    inp_dtype = inp['dtype']
    # Expected input ndim (e.g. 4 for NHWC, 3 for NHW without channel)
    inp_ndim = len(inp['shape'])
    # Does this model produce 2-class softmax or binary sigmoid?
    out_is_categorical = (out['shape'][-1] == 2)

    results = []
    for i in range(len(features)):
        sample = features[i: i + 1]  # (1, H, W[, C])
        # Adapt channel dimension to match TFLite input
        if inp_ndim == 4 and sample.ndim == 3:
            sample = sample[..., np.newaxis]   # add channel
        elif inp_ndim == 3 and sample.ndim == 4:
            sample = sample[..., 0]            # drop channel
        sample = sample.astype(inp_dtype)
        interpreter.set_tensor(inp['index'], sample)
        interpreter.invoke()
        out_data = interpreter.get_tensor(out['index'])
        # For categorical (Dense(2, softmax)), index 1 = prob_class1
        # For binary     (Dense(1, sigmoid)), index 0 = prob_class1
        prob = float(out_data[0, 1] if out_is_categorical else out_data[0, 0])
        results.append(prob)
        if (i + 1) % 500 == 0 or (i + 1) == len(features):
            print(f'    TFLite inference: {i + 1}/{len(features)}', end='\r', flush=True)

    print()
    return np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare(model_name: str, split: str) -> dict:
    """Compare raw probability outputs between the Keras and TFLite models.

    The goal is purely numerical fidelity: does TFLite produce the same
    probability values as Keras?  Threshold / accuracy / agreement metrics
    are intentionally absent — those depend on the threshold chosen and are
    a separate concern.
    """
    model_upper = model_name.upper()
    quantization = getattr(config, 'TFLITE_QUANTIZATION', 'none')

    # --- Load Keras perf JSON (auto-generates it if missing) ---
    print(f'  Loading Keras perf JSON ({model_upper}/{split}) ...')
    keras_perf = _load_keras_perf(model_upper, split)
    keras_index = _index_keras_preds(keras_perf)

    # --- Locate TFLite file ---
    tflite_path = config.TFLITE_DIR / f'{model_name.lower()}_model.tflite'
    if not tflite_path.exists():
        raise FileNotFoundError(
            f'TFLite model not found: {tflite_path}\n'
            'Run the converter first (tflite_conv_orchestrator.py).'
        )

    # --- Load MEL features ---
    print(f'  Loading MEL features for split={split} ...')
    mels, labels, _mapping = load_mel_features(split)
    if len(mels) == 0:
        raise RuntimeError(f'No MEL features found for split={split}')

    # Adapt channel dim to match what the TFLite model actually expects.
    # Peek at the TFLite input shape to avoid unconditionally adding a channel
    # that models like Conformer (3-D input) do not expect.
    _peek = tf.lite.Interpreter(model_path=str(tflite_path))
    _peek.allocate_tensors()
    _inp_ndim = len(_peek.get_input_details()[0]['shape'])
    del _peek
    if _inp_ndim == 4 and mels.ndim == 3:
        mels = mels[..., np.newaxis]   # add channel dim
    elif _inp_ndim == 3 and mels.ndim == 4:
        mels = mels[..., 0]            # drop channel dim

    # --- Resolve filenames (same method as Universal_Perf_Tester) ---
    # The NPZ only stores mels/labels/mapping (not per-sample filenames).
    # Recover them by scanning the dataset directory in the same sorted order
    # used when the NPZ was built: class-0 WAVs sorted, then class-1 WAVs sorted.
    _dataset_dirs = {
        'train': config.DATASET_TRAIN_DIR,
        'val':   config.DATASET_VAL_DIR,
        'test':  config.DATASET_TEST_DIR,
    }
    _ds_dir = Path(_dataset_dirs[split])
    _files0 = sorted(p.name for p in (_ds_dir / '0').glob('*.wav')) if (_ds_dir / '0').exists() else []
    _files1 = sorted(p.name for p in (_ds_dir / '1').glob('*.wav')) if (_ds_dir / '1').exists() else []
    filenames = _files0 + _files1

    if len(filenames) != len(mels):
        raise RuntimeError(
            f'Dataset/features mismatch for split={split}: '
            f'{len(filenames)} WAV files vs {len(mels)} MEL features. '
            f'Re-extract features or re-build the dataset.'
        )

    # --- TFLite inference ---
    print(f'  Running TFLite inference ...')
    tflite_probs = _run_tflite(tflite_path, mels)

    # --- Build per-sample comparison (raw probs only, no threshold) ---
    samples = []
    for i, filename in enumerate(filenames):
        keras_rec = keras_index.get(filename)
        if keras_rec is None:
            continue  # filename present in features but absent from Keras perf JSON

        keras_prob  = float(keras_rec['keras_prob'])
        tflite_prob = float(tflite_probs[i])

        samples.append({
            'filename':    filename,
            'true_label':  int(labels[i]),
            'keras_prob':  round(keras_prob,  6),
            'tflite_prob': round(tflite_prob, 6),
            'delta':       round(abs(keras_prob - tflite_prob), 6),
        })

    n = len(samples)
    if n == 0:
        raise RuntimeError('No matching samples between features and Keras perf JSON.')

    deltas = np.array([s['delta'] for s in samples], dtype=np.float64)

    result = {
        'metadata': {
            'model':        model_upper,
            'split':        split,
            'quantization': quantization,
            'num_samples':  n,
            # Raw-output fidelity metrics — threshold-independent
            'mean_abs_delta': round(float(np.mean(deltas)),   6),
            'median_abs_delta': round(float(np.median(deltas)), 6),
            'p95_abs_delta':  round(float(np.percentile(deltas, 95)), 6),
            'p99_abs_delta':  round(float(np.percentile(deltas, 99)), 6),
            'max_abs_delta':  round(float(np.max(deltas)),    6),
        },
        'samples': samples,
    }

    # --- Save ---
    out_path = config.TFLITE_COMPARISON_DIR / f'{model_name.lower()}_{split}_tflite_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'  Saved comparison: {out_path}')
    _print_summary(result['metadata'])
    return result


def _print_summary(md: dict):
    """Print a threshold-independent output-fidelity summary."""
    print(f'\n  --- TFLite output fidelity ({md["model"]}/{md["split"]}) ---')
    print(f'    Samples          : {md["num_samples"]}')
    print(f'    Mean  |Δprob|    : {md["mean_abs_delta"]:.6f}')
    print(f'    Median |Δprob|   : {md["median_abs_delta"]:.6f}')
    print(f'    p95   |Δprob|    : {md["p95_abs_delta"]:.6f}')
    print(f'    p99   |Δprob|    : {md["p99_abs_delta"]:.6f}')
    print(f'    Max   |Δprob|    : {md["max_abs_delta"]:.6f}')
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True,
                    choices=['CNN', 'RNN', 'CRNN', 'ATTENTION_CRNN', 'EFFICIENTNET',
                             'MOBILENET', 'CONFORMER', 'TCN', 'AST', 'MULTIRES'])
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    args = ap.parse_args()

    try:
        compare(args.model, args.split)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
