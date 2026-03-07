#!/usr/bin/env python3
"""Convert a single Keras model (.keras) to TFLite (.tflite).

Two conversion attempts are made in order:
  1. tf.lite.TFLiteConverter.from_keras_model()   — direct, documented path
  2. model.export() → from_saved_model()          — Keras 3 + BatchNorm workaround

If standard TFLITE_BUILTINS fail, SELECT_TF_OPS is added automatically so that
ops like MultiHeadAttention are delegated back to TensorFlow kernels.

Quantization:
  --quantize float16   half-precision weights; ~2x smaller, no calibration needed
  --quantize none      full float32 (default)

Usage:
    python keras_to_tflite_converter.py --keras path/to/model.keras --out path/to/model.tflite
    python keras_to_tflite_converter.py --keras model.keras --out model.tflite --quantize float16
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

try:
    import tensorflow as tf
except ImportError:
    print('ERROR: TensorFlow is required.  pip install tensorflow')
    sys.exit(1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(keras_path: Path):
    """Load a Keras model, forcing float32 compute policy.

    Models trained with mixed_bfloat16 (e.g. AST, Conformer) store internal
    bfloat16 ops that survive into TFLite and cause CAST op version mismatches
    or large numerical deltas.  Loading under float32 policy removes those ops.
    """
    old_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy('float32')
    try:
        try:
            model = tf.keras.models.load_model(str(keras_path), compile=False, safe_mode=True)
        except TypeError:
            model = tf.keras.models.load_model(str(keras_path), compile=False)
    finally:
        tf.keras.mixed_precision.set_global_policy(old_policy)
    return model


def _make_converter_saved_model(tmp_dir: str, select_ops: bool, quantize: str):
    converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
    _apply_settings(converter, select_ops, quantize)
    return converter


def _apply_settings(converter, select_ops: bool, quantize: str):
    if quantize == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    if select_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        try:
            converter._experimental_lower_tensor_list_ops = False
        except Exception:
            pass


def _attempt(model, select_ops: bool, quantize: str) -> bytes | None:
    """One conversion attempt; returns tflite bytes or None on failure.

    stdout is suppressed during conversion to hide TF's verbose tracing output
    ("Saved artifact at ...", "Captures:", etc.).
    """
    _sink = io.StringIO()
    # Attempt 1: from_keras_model (direct)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        _apply_settings(converter, select_ops, quantize)
        with contextlib.redirect_stdout(_sink):
            return converter.convert()
    except Exception:
        pass

    # Attempt 2: SavedModel intermediate (needed for Keras 3 + BatchNorm)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(_sink):
                try:
                    model.export(tmp)          # Keras 3
                except AttributeError:
                    tf.saved_model.save(model, tmp)  # Keras 2 fallback
            converter = _make_converter_saved_model(tmp, select_ops, quantize)
            with contextlib.redirect_stdout(_sink):
                return converter.convert()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _ensure_float32_output(model):
    """Wrap model output with a float32 cast if it is not already float32.

    After loading with float32 global policy (see _load_model) this is mostly
    a no-op, but kept as a safety net.  Sequential models that haven't been
    called yet have no accessible .output — we skip silently in that case.
    """
    try:
        out = model.output
    except (AttributeError, RuntimeError):
        return model  # Unbuilt Sequential — already float32 from load policy
    if out.dtype != tf.float32:
        try:
            out = tf.keras.layers.Lambda(
                lambda x: tf.cast(x, tf.float32), name='cast_output_float32'
            )(out)
            model = tf.keras.Model(inputs=model.input, outputs=out)
        except Exception:
            pass  # If wrapping fails, proceed — conversion may still work
    return model


def convert(keras_path: Path, tflite_path: Path, quantize: str = 'none') -> dict:
    """Convert keras_path → tflite_path.  Returns result dict."""
    result = {
        'keras': str(keras_path),
        'tflite': str(tflite_path),
        'quantize': quantize,
        'ops': None,
        'size_kb': None,
        'status': None,
        'error': None,
    }

    print(f'  Loading {keras_path.name} ...', end=' ', flush=True)
    try:
        model = _load_model(keras_path)
        model = _ensure_float32_output(model)
    except Exception as e:
        result['status'] = 'FAILED_LOAD'
        result['error'] = str(e)
        print(f'FAILED: {e}')
        return result
    print('OK')

    for select_ops in (False, True):
        label = 'TFLITE_BUILTINS' + ('+SELECT_TF_OPS' if select_ops else '')
        print(f'  Converting ({label}) ...', end=' ', flush=True)
        tflite_bytes = _attempt(model, select_ops, quantize)
        if tflite_bytes is not None:
            tflite_path.write_bytes(tflite_bytes)
            result['ops'] = label
            result['size_kb'] = len(tflite_bytes) // 1024
            result['status'] = 'OK'
            print(f'OK  ({result["size_kb"]} KB)')
            return result
        print(f'failed')

    result['status'] = 'FAILED_CONVERT'
    result['error'] = 'All conversion attempts failed (standard + SELECT_TF_OPS)'
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keras', required=True, type=Path, help='Path to .keras file')
    ap.add_argument('--out', required=True, type=Path, help='Output .tflite path')
    ap.add_argument('--quantize', default='none', choices=['none', 'float16'],
                    help='Quantization mode (default: none)')
    ap.add_argument('--force', action='store_true', help='Overwrite existing .tflite')
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f'  SKIP — {args.out} already exists. Use --force to overwrite.')
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res = convert(args.keras, args.out, quantize=args.quantize)
    return 0 if res['status'] == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())
