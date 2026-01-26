#!/usr/bin/env python3
"""Plugin: training curves (real or placeholders).

Exposes `VIZ_NAME` and `run(ctx)` so `viz_all` can discover it.
"""
from pathlib import Path
import logging
from tools.viz_core import VizResult
from tools.io.training_loader import load_history_csv
from tools.plotting.training import plot_learning_curves
import config

logger = logging.getLogger(__name__)

VIZ_NAME = 'training'
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

VIZ_ORDER = int(getattr(config, 'MEL_TIME_FRAMES', 90))


def discover_history(results_dir: str, model: str) -> tuple:
    p = Path(results_dir)
    model_base = model.lower()
    candidates = list(p.rglob(f'*{model_base}*history*.csv')) + list(p.rglob(f'*{model_base}*history.csv'))
    if candidates:
        history = str(candidates[0])
    else:
        raise FileNotFoundError(f'History CSV not found for {model} in {results_dir}')
    meta_candidates = list(p.rglob(f'*{model_base}*training_metadata*.json'))
    meta = str(meta_candidates[0]) if meta_candidates else None
    return history, meta


def run(ctx, args=None):
    out_dir = Path(ctx.out_dir) / 'training'
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for model in ctx.models:
        try:
            history, meta = discover_history(ctx.results_dir, model)
        except Exception as e:
            logger.warning('History not found for model %s: %s', model, e)
            try:
                import matplotlib.pyplot as plt
                out_path = out_dir / f'training__{model}.png'
                fig = plt.figure(figsize=(6, 3))
                fig.text(0.5, 0.5, f'History CSV not found for {model}', ha='center', va='center')
                fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
                plt.close(fig)
                artifacts.append(str(out_path))
            except Exception as e2:
                logger.warning('Failed to write placeholder for %s: %s', model, e2)
            continue
        try:
            df = load_history_csv(history)
            out = plot_learning_curves(df, model, str(out_dir))
            artifacts.append(str(out))
            # include JSON summary if produced
            summary_json = Path(out).with_suffix('.json')
            if summary_json.exists():
                artifacts.append(str(summary_json))
        except Exception as e:
            logger.warning('Failed plotting history for %s: %s', model, e)
    vr = VizResult()
    vr.artifacts = [Path(a) for a in artifacts]
    vr.summary = {'produced': len(artifacts)}
    return vr


if __name__ == '__main__':
    # allow running standalone for quick checks
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default=config.RESULTS_DIR)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()
    # simple invocation using discovered args
    class C: pass
    c = C()
    c.out_dir = args.out_dir or (Path(config.VIZ_OUTPUT_DIR) / 'debug')
    c.results_dir = args.results_dir
    c.models = args.models
    run(c)
