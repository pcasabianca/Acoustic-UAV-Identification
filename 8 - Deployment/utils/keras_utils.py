"""Helpers for loading Keras/TensorFlow models and preparing inputs/outputs.

Provides safe-loading helpers and a `load_models_from_config` convenience
that mirrors the behavior expected by the rest of the codebase.
"""
from __future__ import annotations
import signal
import json
from pathlib import Path
from typing import Dict, Optional
import os

try:
    # Quiet TF INFO/WARNING messages about retracing when possible
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
except Exception:
    tf = None


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError('Model loading timed out')


def load_single_model(model_path: Path, safe_mode: bool = True, timeout: int = 30):
    """Load a single Keras model with optional timeout; returns None on failure."""
    if tf is None:
        raise RuntimeError('TensorFlow not available')

    try:
        if timeout > 0 and hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        # Attempt to pass through `safe_mode` when available (newer Keras API).
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=safe_mode)
        except TypeError:
            # Older TF/Keras may not accept `safe_mode` kwarg; fall back.
            model = tf.keras.models.load_model(str(model_path), compile=False)

        # Ensure Keras has built its internal predict function (warm start)
        try:
            if hasattr(model, 'make_predict_function'):
                try:
                    model.make_predict_function()
                except Exception:
                    pass
        except Exception:
            pass

        if timeout > 0 and hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        return model
    except Exception:
        try:
            if timeout > 0 and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        except Exception:
            pass
        return None


def load_models_from_config(config: Dict, logger=None, timeout: int = 30) -> Dict[str, Dict]:
    """Load enabled models from configuration and attach thresholds if available."""
    models_dir = Path(config['models']['directory'])
    enabled = config['detection']['enabled_models']
    available = config['models']['available_models']

    # Attempt to load thresholds
    thresholds = {}
    try:
        thresh_file = config['detection'].get('thresholds_file')
        if thresh_file:
            with open(thresh_file, 'r') as f:
                data = json.load(f)
            for k, v in data.get('models', {}).items():
                thresholds[k] = v.get('threshold', 0.5)
    except Exception:
        thresholds = {}

    loaded = {}
    for name in enabled:
        if name not in available:
            if logger:
                logger.warning(f'Model {name} not declared in config')
            continue
        info = available[name]
        path = models_dir / info['filename']
        if not path.exists():
            if logger:
                logger.warning(f'Model file missing: {path}')
            continue
        # If the model declares `safe_mode=False` it requires unsafe
        # deserialization (e.g. Python `lambda` in a `Lambda` layer). Calling
        # into Keras to allow unsafe deserialization has caused hard crashes
        # (segfaults) in some environments. To avoid bringing down the
        # process, do not attempt unsafe loads automatically here — instead
        # skip the model and warn the user with remediation steps.
        if not info.get('safe_mode', True):
            # Models saved with Keras 2 Lambda layers cannot be deserialized on
            # Keras 3 without a segfault — even with enable_unsafe_deserialization().
            # The only fix is to re-save the model from the training environment
            # using proper subclassed layers instead of Lambda layers.
            msg = (
                f"Model {name} declares safe_mode=False and requires unsafe deserialization. "
                "Skipping load: unsafe deserialization causes segfaults on Keras 3 when the model "
                "contains Keras-2-style Lambda layers. Re-save the model without Lambda layers "
                "from the training environment (Keras 3 native layers or subclassed layers)."
            )
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            continue

        model = load_single_model(path, safe_mode=True, timeout=timeout)
        if model is None:
            if logger:
                logger.error(f'Failed to load model {name}')
            continue
        calibrated_key = info.get('calibrated_key', name)
        loaded[name] = {
            'model': model,
            'threshold': thresholds.get(calibrated_key, 0.5),
            'description': info.get('description', ''),
            'architecture': info.get('architecture')
        }
        # store input shape for downstream adapt
        try:
            loaded[name]['input_shape'] = getattr(model, 'input_shape', None)
        except Exception:
            loaded[name]['input_shape'] = None

        # Simple warm-up predict with a zero tensor matching the model's
        # reported input shape (batch=1). This uses plain Keras API and helps
        # avoid repeated retracing on first inference calls.
        try:
            import numpy as _np
            ish = loaded[name].get('input_shape')
            if ish is None:
                X = _np.zeros((1, 64, 173, 1), dtype=_np.float32)
            else:
                try:
                    ish_t = tuple(ish)
                except Exception:
                    ish_t = tuple(list(ish)) if hasattr(ish, '__iter__') else None
                if ish_t is None:
                    X = _np.zeros((1, 64, 173, 1), dtype=_np.float32)
                else:
                    dims = list(ish_t)
                    if dims and dims[0] is None:
                        dims = dims[1:]
                    defaults = [64, 173, 1]
                    final = []
                    for i, d in enumerate(dims):
                        if d is None:
                            final.append(defaults[i] if i < len(defaults) else 1)
                        else:
                            final.append(int(d))
                    X_shape = tuple([1] + final)
                    X = _np.zeros(X_shape, dtype=_np.float32)
            try:
                model.predict(X, verbose=0)
            except Exception:
                # ignore warm-up failures; prediction can still be attempted later
                pass
        except Exception:
            pass

        # Use simple Keras prediction call by default. Do not create tf.function
        # wrappers here — leave prediction calls to callers who may choose how
        # to invoke the model (we'll use plain `model.predict` elsewhere).
        loaded[name]['predict_fn'] = None

    return loaded


if __name__ == '__main__':
    import argparse
    import time
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='deployment_config.json')
    parser.add_argument('--runs', type=int, default=3, help='number of predict runs per model')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    models = load_models_from_config(cfg)
    print('Loaded models:', list(models.keys()))

    # For each model, build a dummy input matching reported input_shape and run predict
    for name, info in models.items():
        model = info.get('model')
        ish = info.get('input_shape') or getattr(model, 'input_shape', None)
        print(f'Running smoke predict for model: {name}, input_shape={ish}')
        try:
            # Build dummy input
            if ish is None:
                # fallback to (1,64,173,1)
                X = np.zeros((1, 64, 173, 1), dtype=np.float32)
            else:
                # Normalize ish to tuple and replace None dims
                try:
                    ish_t = tuple(ish)
                except Exception:
                    ish_t = tuple(list(ish)) if hasattr(ish, '__iter__') else None
                if ish_t is None:
                    X = np.zeros((1, 64, 173, 1), dtype=np.float32)
                else:
                    # Determine shape including batch
                    # If ish includes batch (None as first), drop it and fill defaults
                    dims = list(ish_t)
                    if dims[0] is None:
                        dims = dims[1:]
                    # Fill None with defaults
                    defaults = [64, 173, 1]
                    final = []
                    for i, d in enumerate(dims):
                        if d is None:
                            final.append(defaults[i] if i < len(defaults) else 1)
                        else:
                            final.append(int(d))
                    # Prepend batch dim
                    X_shape = tuple([1] + final)
                    X = np.zeros(X_shape, dtype=np.float32)

            # Run several predicts and time them
            for i in range(args.runs):
                t0 = time.time()
                try:
                    out = model.predict(X, verbose=0)
                except Exception as e:
                    print(f'  predict failed on run {i}: {e}')
                    out = None
                t1 = time.time()
                print(f'  run {i+1}/{args.runs} took {(t1-t0)*1000:.1f} ms, out_shape={None if out is None else getattr(out, "shape", None)}')
        except Exception as e:
            print(f'  Unable to run smoke predict for {name}: {e}')
