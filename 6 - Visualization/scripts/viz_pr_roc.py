"""Plugin: PR and ROC curves overlaying models per split.
"""
from tools.viz_core import VizResult
from pathlib import Path
import numpy as np

VIZ_NAME = 'pr_roc'
VIZ_ORDER = 30


def _pr_curve(points):
    # points: list of (score, true_label)
    pts = sorted(points, key=lambda x: -x[0])
    tp = 0
    fp = 0
    fn = sum(1 for _, y in pts if y == 1)
    tn = sum(1 for _, y in pts if y == 0)
    precisions = []
    recalls = []
    for s, y in pts:
        if y == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
    return recalls, precisions


def _roc_curve(points):
    pts = sorted(points, key=lambda x: -x[0])
    P = sum(1 for _, y in pts if y == 1)
    N = sum(1 for _, y in pts if y == 0)
    tpr_list = []
    fpr_list = []
    tp = 0
    fp = 0
    for s, y in pts:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return fpr_list, tpr_list


def run(ctx, args=None):
    out_dir = Path(ctx.out_dir) / 'pr_roc'
    out_dir.mkdir(parents=True, exist_ok=True)
    perfs = ctx.perfs
    models = ctx.models
    splits = ctx.splits
    artifacts = []
    try:
        import matplotlib.pyplot as plt
        for split in splits:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for m in models:
                p = next((pp for pp in perfs if pp.get('metadata', {}).get('model') == m and pp.get('metadata', {}).get('split') == split), None)
                if not p:
                    continue
                points = [(float(pr.get('prob_class1') or 0.0), pr.get('true_label')) for pr in p.get('predictions', [])]
                if not points:
                    continue
                rec, prec = _pr_curve(points)
                fpr, tpr = _roc_curve(points)
                axes[0].plot(rec, prec, label=m)
                axes[1].plot(fpr, tpr, label=m)
            axes[0].set_title('PR Curve')
            axes[0].set_xlabel('recall')
            axes[0].set_ylabel('precision')
            axes[1].set_title('ROC Curve')
            axes[1].set_xlabel('fpr')
            axes[1].set_ylabel('tpr')
            # place legend at bottom center to avoid occlusion
            ncols = min(4, max(1, len(models)))
            axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=ncols, fontsize='small')
            axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=ncols, fontsize='small')
            outp = out_dir / f'pr_roc__{split}.png'
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
