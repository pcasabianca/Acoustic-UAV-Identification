#!/usr/bin/env python3
"""Diagnose average drone probabilities per category for each model to check thresholds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Ensure visualization module path is available
sys.path.insert(0, str(Path(__file__).parent.parent / '6 - Visualization'))
import json
import numpy as np
import performance_by_distance as pbd

# Use project's venv when running

def main():
    print('Loading augmentation config...')
    aug = pbd.load_augmentation_config()
    cats = pbd.get_categories_from_config(aug)
    cats = pbd.detect_categories_from_files(cats)
    cat_files = pbd.get_category_files(cats)
    precomputed = pbd.load_precomputed_features()
    if precomputed is None or 'test_index' not in precomputed and 'features' not in precomputed:
        print('No precomputed features available. Aborting (precomputed-only mode).')
        return
    models = pbd.load_models()

    thresholds = {
        'CNN': 0.85,
        'RNN': 0.95,
        'CRNN': 0.90,
        'Attention-CRNN': 0.95
    }

    for model_name, model in models.items():
        print('\n=== Model:', model_name, '===')
        thr = thresholds.get(model_name, 0.5)
        for cat_name, info in cat_files.items():
            files = info['files'][:100]
            probs = []
            for f in files:
                try:
                    feat = pbd.get_precomputed_features_for_file(f, precomputed)
                    if model_name in ['CNN', 'CRNN', 'Attention-CRNN']:
                        x = feat[np.newaxis, ..., np.newaxis]
                    else:
                        x = feat[np.newaxis, ...]
                    pred = model.predict(x, verbose=0)[0]
                    drone_prob = float(pred[1])
                    probs.append(drone_prob)
                except Exception as e:
                    continue
            if not probs:
                continue
            probs = np.array(probs)
            mean = probs.mean()
            pct_above = (probs >= thr).mean() * 100
            print(f"{info['display_name']:25s}: mean_prob={mean:.3f}, pct_above_thr={pct_above:.1f}% (thr={thr})")

if __name__ == '__main__':
    main()
