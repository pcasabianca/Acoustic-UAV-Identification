"""Plugin: confusion matrix grid per split.
"""
from tools.viz_core import VizResult
from pathlib import Path
import numpy as np

VIZ_NAME = 'confusions'
VIZ_ORDER = 25


def _confusion_from_preds(preds):
    tp = sum(1 for p in preds if p.get('true_label') == 1 and p.get('pred_label') == 1)
    tn = sum(1 for p in preds if p.get('true_label') == 0 and p.get('pred_label') == 0)
    fp = sum(1 for p in preds if p.get('true_label') == 0 and p.get('pred_label') == 1)
    fn = sum(1 for p in preds if p.get('true_label') == 1 and p.get('pred_label') == 0)
    return np.array([[tn, fp], [fn, tp]])


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
            figs = []
            n = len(models)
            cols = min(4, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            axes = axes.reshape(rows, cols) if rows > 1 else axes.reshape(1, cols)
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= n:
                        axes[r, c].axis('off')
                        idx += 1
                        continue
                    m = models[idx]
                    p = next((pp for pp in perfs if pp.get('metadata', {}).get('model') == m and pp.get('metadata', {}).get('split') == split), None)
                    if not p:
                        axes[r, c].set_title(m + ' (missing)')
                        axes[r, c].axis('off')
                        idx += 1
                        continue
                    cm = _confusion_from_preds(p.get('predictions', []))
                    im = axes[r, c].imshow(cm, cmap='Blues')
                    axes[r, c].set_title(m)
                    axes[r, c].set_xlabel('pred')
                    axes[r, c].set_ylabel('true')
                    for (i, j), val in np.ndenumerate(cm):
                        axes[r, c].text(j, i, int(val), ha='center', va='center')
                    idx += 1
            outp = out_dir / f'confusions__{split}.png'
            fig.tight_layout()
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
