"""Prediction helpers: load thresholds and run models on features.

This mirrors parts of `core.predictor.Predictor` but is a lightweight
utility to drive models from orchestrators.
"""
from __future__ import annotations
import time
from typing import Dict, Any
import numpy as np
import argparse
import json
from .keras_utils import load_single_model
from .features_utils import FeatureExtractor

def load_thresholds_from_config(config: Dict) -> Dict[str, float]:
    try:
        thresh_file = config['detection'].get('thresholds_file')
        if not thresh_file:
            return {}
        import json
        with open(thresh_file, 'r') as f:
            data = json.load(f)
        res = {k: v.get('threshold', 0.5) for k, v in data.get('models', {}).items()}
        return res
    except Exception:
        return {}


def run_models(models: Dict[str, Dict], feature_extractor, features: np.ndarray, logger=None) -> Dict[str, float]:
    """Run all models on `features`. Returns {model_name: probability} and attaches timing to model dicts."""
    predictions: Dict[str, float] = {}
    timings: Dict[str, int] = {}
    total_ms = 0.0

    for name, info in models.items():
        try:
            model = info['model']
            # Adapt features to model input. If a feature_extractor with
            # `adapt_to_model` is provided, use it; otherwise try to reshape
            # features into a batched tensor expected by common architectures.
            if feature_extractor is not None and hasattr(feature_extractor, 'adapt_to_model'):
                X = feature_extractor.adapt_to_model(features, model)
            else:
                # Default: treat `features` as 2D MEL (H, W)
                Hf, Wf = features.shape[:2]
                ish = info.get('input_shape')
                # ish might be e.g. (None, H, W, C) or (H, W, C) or (None, N)
                target_H = None
                target_W = None
                target_C = 1
                if ish is not None:
                    try:
                        # normalize to tuple
                        ish_t = tuple([int(x) for x in ish if x is not None])
                    except Exception:
                        ish_t = None
                    if ish_t:
                        if len(ish_t) == 3:
                            # (H, W, C)
                            target_H, target_W, target_C = ish_t
                        elif len(ish_t) == 2:
                            # (H, W) or (None, N)
                            target_H, target_W = ish_t
                        elif len(ish_t) == 1:
                            # flattened vector
                            target_H = None

                if target_H is not None and target_W is not None:
                    # pad/crop to (target_H, target_W)
                    try:
                        feat2 = pad_crop_2d(features, target_H, target_W)
                        X = np.expand_dims(feat2, 0)
                        if target_C and target_C > 1:
                            X = np.expand_dims(X, -1)
                        else:
                            X = np.expand_dims(X, -1)
                    except Exception:
                        X = np.expand_dims(features, 0)
                else:
                    # Fallback: flatten or batch
                    try:
                        flat = features.flatten()
                        expected_len = None
                        if ish is not None and isinstance(ish, (list, tuple)):
                            # try to pick last dimension as expected length
                            try:
                                expected_len = int(ish[-1])
                            except Exception:
                                expected_len = None
                        if expected_len and expected_len > 0:
                            vec = np.zeros((expected_len,), dtype=flat.dtype)
                            vec[:min(expected_len, flat.size)] = flat[:min(expected_len, flat.size)]
                            X = np.expand_dims(vec, 0)
                        else:
                            X = np.expand_dims(flat, 0)
                    except Exception:
                        X = np.expand_dims(features, 0)
            t0 = time.time()
            # Use simple Keras predict API. Ensure input dtype is float32.
            try:
                X = X.astype('float32', copy=False)
            except Exception:
                pass
            pred = model.predict(X, verbose=0)
            elapsed = (time.time() - t0) * 1000.0
            timings[name] = int(elapsed)
            total_ms += elapsed

            if pred.shape[1] == 2:
                prob = float(pred[0][1])
            elif pred.shape[1] == 1:
                prob = float(pred[0][0])
            else:
                prob = 0.0
            predictions[name] = prob
        except Exception as e:
            if logger:
                logger.error(f'Prediction failed for {name}: {e}')
            predictions[name] = 0.0
            timings[name] = 0

    # attach timings to returned structure
    return {'predictions': predictions, 'timings_ms': timings, 'total_inference_ms': int(total_ms)}


if __name__ == '__main__':
    # Quick smoke test: load models via keras_utils and run run_models once with a
    # dummy MEL feature to observe runtime behavior and any warnings/exceptions.

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='deployment_config.json')
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    models = keras_utils.load_models_from_config(cfg)
    print('Loaded models for predictor smoke:', list(models.keys()))

    fe = FeatureExtractor(cfg)
    # Create a default MEL-like array (n_mels x width) — use common default 44x173
    mel = np.zeros((44, 173), dtype=np.float32)

    for i in range(args.runs):
        t0 = time.time()
        try:
            res = run_models(models, fe, mel, logger=print)
            print(f'Run {i+1} result:', res)
        except Exception as e:
            print(f'Run {i+1} failed: {e}')
        t1 = time.time()
        print(f'  elapsed {(t1-t0)*1000:.1f} ms')
