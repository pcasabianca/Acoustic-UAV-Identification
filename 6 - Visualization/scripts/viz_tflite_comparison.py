"""Plugin: TFLite vs Keras per-sample probability scatter plots.

For each model that has a comparison JSON in config.TFLITE_COMPARISON_DIR,
produces one scatter plot:
  - x-axis : Keras output probability
  - y-axis : TFLite output probability
  - color  : true label (0 = non-drone, 1 = drone)
  - diagonal reference line at y = x

One PNG per model, written to <out_dir>/tflite/.
"""
import json
import logging
import sys
from pathlib import Path

from tools.viz_core import VizResult

logger = logging.getLogger(__name__)

VIZ_NAME = 'tflite_comparison'
VIZ_ORDER = 95   # runs near the end, after standard performance plots


def run(ctx, args=None):
    vr = VizResult()

    # Resolve comparison directory from config (available via ctx.repo_root)
    sys.path.insert(0, str(ctx.repo_root))
    try:
        import config
        comparison_dir = Path(config.TFLITE_COMPARISON_DIR)
    except Exception as e:
        vr.skipped = True
        vr.errors.append(f'Cannot import config: {e}')
        return vr
    finally:
        sys.path.pop(0)

    if not comparison_dir.exists():
        vr.skipped = True
        vr.errors.append(f'TFLITE_COMPARISON_DIR not found: {comparison_dir}')
        return vr

    # Discover comparison JSONs for the requested models
    # Filename pattern: {model}_{split}_tflite_comparison.json
    json_files = sorted(comparison_dir.glob('*_tflite_comparison.json'))
    if not json_files:
        vr.skipped = True
        vr.errors.append('No tflite comparison JSON files found.')
        return vr

    # Filter to models requested by the viz run (ctx.models is uppercase)
    models_upper = {m.upper() for m in ctx.models} if ctx.models else None

    out_dir = Path(ctx.out_dir) / 'tflite'
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError as e:
        vr.skipped = True
        vr.errors.append(f'matplotlib not available: {e}')
        return vr

    artifacts = []

    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning('Failed to read %s: %s', json_path.name, e)
            continue

        md = data.get('metadata', {})
        model_name = md.get('model', json_path.stem)
        split = md.get('split', 'test')

        if models_upper and model_name.upper() not in models_upper:
            continue

        samples = data.get('samples', [])
        if not samples:
            logger.warning('No samples in %s', json_path.name)
            continue

        keras_probs  = np.array([s['keras_prob']  for s in samples], dtype=np.float32)
        tflite_probs = np.array([s['tflite_prob'] for s in samples], dtype=np.float32)
        true_labels  = np.array([s['true_label']  for s in samples], dtype=np.int32)

        # Separate by class
        mask_drone     = true_labels == 1
        mask_non_drone = true_labels == 0

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(
            keras_probs[mask_non_drone], tflite_probs[mask_non_drone],
            s=6, alpha=0.35, color='steelblue', label='Non-drone (0)', rasterized=True
        )
        ax.scatter(
            keras_probs[mask_drone], tflite_probs[mask_drone],
            s=6, alpha=0.35, color='tomato', label='Drone (1)', rasterized=True
        )

        # Reference diagonal y = x
        ax.plot([0, 1], [0, 1], color='black', linewidth=0.8, linestyle='--', alpha=0.6)

        # Annotate with key metrics
        agreement = md.get('agreement_rate', 0.0)
        mean_delta = md.get('mean_abs_delta', 0.0)
        keras_acc  = md.get('keras_accuracy', 0.0)
        tflite_acc = md.get('tflite_accuracy', 0.0)
        quantize   = md.get('quantization', 'unknown')
        n          = md.get('num_samples', len(samples))

        ax.set_xlabel('Keras probability', fontsize=11)
        ax.set_ylabel('TFLite probability', fontsize=11)
        ax.set_title(
            f'{model_name} — Keras vs TFLite ({split}, {quantize})\n'
            f'n={n}  agree={agreement:.3f}  mean|delta|={mean_delta:.5f}\n'
            f'Keras acc={keras_acc:.3f}  TFLite acc={tflite_acc:.3f}',
            fontsize=9
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.legend(fontsize=9, markerscale=2)
        ax.grid(True, linestyle=':', alpha=0.4)

        out_path = out_dir / f'tflite__{model_name.lower()}__{split}.png'
        fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
        plt.close(fig)
        artifacts.append(out_path)
        logger.info('Saved %s', out_path.name)

    vr.artifacts = artifacts
    vr.summary = {'produced': len(artifacts)}
    return vr
