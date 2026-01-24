"""Plugin: global metrics barplot per split (multi-model).
"""
from tools.viz_core import VizResult
from pathlib import Path
import numpy as np

VIZ_NAME = 'global_metrics'
VIZ_ORDER = 20


def run(ctx, args=None):
    out_dir = Path(ctx.out_dir) / 'benchmark'
    out_dir.mkdir(parents=True, exist_ok=True)
    perfs = ctx.perfs
    models = ctx.models
    splits = ctx.splits
    artifacts = []
    try:
        import matplotlib.pyplot as plt
        for split in splits:
            metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'specificity': []}
            labels = []
            for m in models:
                # find perf
                p = next((pp for pp in perfs if pp.get('metadata', {}).get('model') == m and pp.get('metadata', {}).get('split') == split), None)
                if not p:
                    continue
                gm = p.get('global_metrics', {})
                labels.append(m)
                for k in metrics.keys():
                    metrics[k].append(gm.get(k) or 0.0)

            if not labels:
                continue
            x = np.arange(len(labels))
            width = 0.15
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, (k, vals) in enumerate(metrics.items()):
                ax.bar(x + i * width, vals, width, label=k)
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend()
            ax.set_title(f'Global metrics - {split}')
            outp = out_dir / f'global_metrics__{split}.png'
            fig.savefig(str(outp), dpi=200, bbox_inches='tight')
            plt.close(fig)
            artifacts.append(str(outp))
    except Exception as e:
        vr = VizResult()
        vr.skipped = True
        vr.errors.append(str(e))
        return vr

    vr = VizResult()
    vr.artifacts = [Path(a) for a in artifacts]
    vr.summary = {'produced': len(artifacts)}
    return vr
