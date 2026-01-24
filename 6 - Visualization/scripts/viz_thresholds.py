"""Plugin: Visualize threshold sweep and calibrated thresholds per model.
"""
from tools.viz_core import VizResult
from pathlib import Path
import json

VIZ_NAME = 'thresholds'
VIZ_ORDER = 35


def run(ctx, args=None):
    out_dir = Path(ctx.out_dir) / 'thresholds'
    out_dir.mkdir(parents=True, exist_ok=True)
    perfs = ctx.perfs
    models = ctx.models
    artifacts = []
    try:
        import matplotlib.pyplot as plt
        from tools.io.thresholds_loader import load_calibrated_thresholds, get_threshold
        # try to find a calibration file in ctx if provided
        calib = None
        try:
            calib = load_calibrated_thresholds(str(ctx.thresholds_path)) if getattr(ctx, 'thresholds_path', None) else None
        except Exception:
            calib = None

        for m in models:
            model_up = m.upper()
            # find a representative perf (prefer val then test)
            chosen = None
            for s in ('val', 'test', 'train'):
                for p in perfs:
                    md = p.get('metadata', {})
                    if md.get('model') == model_up and md.get('split') == s:
                        chosen = p
                        break
                if chosen:
                    break
            if not chosen:
                # fallback: any perf for model
                for p in perfs:
                    if p.get('metadata', {}).get('model') == model_up:
                        chosen = p
                        break
            if not chosen:
                # nothing to do for this model
                continue

            preds = chosen.get('predictions', [])
            probs = []
            y_true = []
            for pr in preds:
                try:
                    probs.append(float(pr.get('prob_class1', 0.0)))
                    y_true.append(int(pr.get('true_label', 0)))
                except Exception:
                    pass

            if not probs:
                continue

            import numpy as _np
            thresholds = _np.linspace(0.0, 1.0, 101)
            f1s = []
            prec_drone = []
            prec_ambient = []
            recall = []
            specificity = []
            for t in thresholds:
                preds_label = [1 if p >= t else 0 for p in probs]
                tp = sum(1 for p, yt in zip(preds_label, y_true) if p == 1 and yt == 1)
                fp = sum(1 for p, yt in zip(preds_label, y_true) if p == 1 and yt == 0)
                tn = sum(1 for p, yt in zip(preds_label, y_true) if p == 0 and yt == 0)
                fn = sum(1 for p, yt in zip(preds_label, y_true) if p == 0 and yt == 1)
                prec_d = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
                prec_a = float(tn) / (tn + fn) if (tn + fn) > 0 else 0.0
                rec = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
                f1 = (2 * prec_d * rec) / (prec_d + rec) if (prec_d + rec) > 0 else 0.0
                f1s.append(f1)
                prec_drone.append(prec_d)
                prec_ambient.append(prec_a)
                recall.append(rec)
                specificity.append(spec)

            # select optimal threshold by max f1
            opt_idx = int(_np.nanargmax(_np.array(f1s)))
            opt_t = float(thresholds[opt_idx])

            # plot 2x2 grid similar to previous example
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            # F1
            ax = axes[0, 0]
            ax.plot(thresholds, f1s, marker='o', markersize=3, color='C0')
            ax.axvline(opt_t, color='r', linestyle='--', label=f'Optimal: {opt_t:.3f}')
            ax.set_title(f'{model_up} - Threshold Analysis')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('F1-Score')
            ax.legend()

            # Precisions
            ax = axes[0, 1]
            ax.plot(thresholds, prec_drone, marker='o', markersize=3, color='C0', label='Precision Drone (PPV)')
            ax.plot(thresholds, prec_ambient, marker='s', markersize=3, color='C1', label='Precision Ambient (NPV)')
            ax.axvline(opt_t, color='r', linestyle='--')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Precision')
            ax.legend()

            # Recall/Specificity
            ax = axes[1, 0]
            ax.plot(thresholds, recall, marker='o', markersize=3, color='C2', label='Recall (Sensitivity)')
            ax.plot(thresholds, specificity, marker='s', markersize=3, color='C3', label='Specificity')
            ax.axvline(opt_t, color='r', linestyle='--')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Rate')
            ax.legend()

            # Precision-Recall scatter (Pareto)
            ax = axes[1, 1]
            ax.scatter(recall, prec_drone, c=thresholds, cmap='viridis', s=40)
            ax.scatter([recall[opt_idx]], [prec_drone[opt_idx]], color='r', marker='*', s=120, label='Optimal')
            ax.set_xlabel('Recall (Sensitivity)')
            ax.set_ylabel('Precision Drone (PPV)')
            ax.legend()

            fig.tight_layout()
            outp = out_dir / f'thresholds__{model_up}.png'
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
