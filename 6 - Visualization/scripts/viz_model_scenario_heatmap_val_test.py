"""Plugin: model x scenario heatmap restricted to val+test splits.

Same behavior as `viz_model_scenario_heatmap` but only uses perfs whose
metadata.split is in ('val','test').
"""
from tools.viz_core import VizResult
from pathlib import Path
import numpy as np

VIZ_NAME = 'model_scenario_heatmap_val_test'
VIZ_ORDER = 23


def _get_scenario(pr):
    md = pr.get('metadata') or {}
    return md.get('scenario') or md.get('noise_type') or pr.get('scenario') or 'unknown'


def run(ctx, args=None):
    out_dir = Path(ctx.out_dir) / 'heatmaps'
    out_dir.mkdir(parents=True, exist_ok=True)
    perfs_all = ctx.perfs
    models = ctx.models
    artifacts = []
    try:
        import matplotlib.pyplot as plt

        # filter perfs to val+test splits
        allowed = set(['val', 'test'])
        perfs = [p for p in perfs_all if (p.get('metadata', {}).get('split') in allowed)]

        # collect global scenario set across perfs
        scenario_set = set()
        for p in perfs:
            for pr in p.get('predictions', []):
                scenario_set.add(_get_scenario(pr))
        scenarios = sorted(list(scenario_set))
        if not scenarios:
            vr = VizResult()
            vr.skipped = True
            vr.summary = {'reason': 'no scenarios found in predictions for val+test'}
            return vr

        matrix = np.full((len(models), len(scenarios)), np.nan)
        counts = np.zeros((len(models), len(scenarios)), dtype=int)

        for i, m in enumerate(models):
            mup = m.upper()
            preds = []
            for p in perfs:
                if p.get('metadata', {}).get('model') != mup:
                    continue
                preds.extend(p.get('predictions', []))
            if not preds:
                continue
            for j, sc in enumerate(scenarios):
                sel = [pr for pr in preds if _get_scenario(pr) == sc]
                if not sel:
                    continue
                tp = sum(1 for pr in sel if pr.get('true_label') == 1 and pr.get('pred_label') == 1)
                tn = sum(1 for pr in sel if pr.get('true_label') == 0 and pr.get('pred_label') == 0)
                fp = sum(1 for pr in sel if pr.get('true_label') == 0 and pr.get('pred_label') == 1)
                fn = sum(1 for pr in sel if pr.get('true_label') == 1 and pr.get('pred_label') == 0)
                total = tp + tn + fp + fn
                if total == 0:
                    continue
                acc = float(tp + tn) / float(total)
                matrix[i, j] = acc
                counts[i, j] = total

        # order scenarios by avg_snr similar to original
        avg_snr = {}
        for sc in scenarios:
            vals = []
            for p in perfs:
                for pr in p.get('predictions', []):
                    if _get_scenario(pr) == sc:
                        md = pr.get('metadata') or {}
                        rd = md.get('raw_debug') or md
                        sv = rd.get('resulting_snr_db') if isinstance(rd, dict) else md.get('snr_db')
                        if sv is None:
                            sv = pr.get('snr_db')
                        try:
                            if sv is not None:
                                vals.append(float(sv))
                        except Exception:
                            pass
            avg_snr[sc] = float(np.mean(vals)) if vals else float('nan')

        order = sorted(range(len(scenarios)), key=lambda i: (-(avg_snr[scenarios[i]] if not np.isnan(avg_snr[scenarios[i]]) else -1e9), scenarios[i]))
        scenarios_ord = [scenarios[i] for i in order]
        matrix_ord = matrix[:, order]
        counts_ord = counts[:, order]

        def _detect_drone_flag_for_list(prs):
            for pr in prs:
                md = pr.get('metadata') or {}
                for k in ('drone', 'has_drone', 'drone_present', 'contains_drone', 'is_drone'):
                    if k in md:
                        try:
                            return 1 if int(bool(md.get(k))) else 0
                        except Exception:
                            pass
                rd = md.get('raw_debug') if isinstance(md, dict) else None
                if isinstance(rd, dict):
                    for k in ('drone', 'has_drone'):
                        if k in rd:
                            try:
                                return 1 if int(bool(rd.get(k))) else 0
                            except Exception:
                                pass
            if prs:
                name = _get_scenario(prs[0])
                if isinstance(name, str) and 'drone' in name.lower():
                    return 1
            return 0

        drone_flags = []
        for sc in scenarios_ord:
            preds_for_sc = [p for p in perfs for p in p.get('predictions', []) if _get_scenario(p) == sc]
            drone_flags.append(_detect_drone_flag_for_list(preds_for_sc))

        fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(models))))
        cmap = plt.get_cmap('RdYlGn')
        vmin = np.nanmin(matrix_ord)
        vmax = np.nanmax(matrix_ord)
        if np.isnan(vmin) or np.isnan(vmax):
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(matrix_ord, aspect='auto', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)

        xticks = []
        for sc in scenarios_ord:
            sn = avg_snr.get(sc)
            if np.isnan(sn):
                xticks.append(sc)
            else:
                xticks.append(f"{sc}\n({sn:.0f} dB)")

        ax.set_xticks(range(len(scenarios_ord)))
        ax.set_xticklabels(xticks, rotation=45, ha='right')
        ax.set_xlabel('Scenario (avg SNR)')
        ax.set_title('Accuracy Heatmap by Scenario (val+test)')

        for i in range(matrix_ord.shape[0]):
            for j in range(matrix_ord.shape[1]):
                v = matrix_ord[i, j]
                if np.isnan(v):
                    continue
                txt = f'{v:.3f}\n({counts_ord[i,j]})'
                ax.text(j, i, txt, ha='center', va='center', fontsize='8')

        boundary = None
        for i in range(1, len(drone_flags)):
            if drone_flags[i-1] != drone_flags[i]:
                boundary = i
                break
        if boundary is not None:
            ax.axvline(boundary - 0.5, color='black', linewidth=2, zorder=5)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Accuracy')
        outp = out_dir / f'model_scenario_heatmap__val_test.png'
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
