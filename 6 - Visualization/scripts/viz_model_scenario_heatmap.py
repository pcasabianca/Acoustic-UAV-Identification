"""Plugin: model x scenario heatmap of a metric (default: accuracy).

Rows: models, Columns: scenarios, Value: accuracy (tp+tn / total)
"""
from tools.viz_core import VizResult
from pathlib import Path
import numpy as np

VIZ_NAME = 'model_scenario_heatmap'
VIZ_ORDER = 22


def _get_scenario(pr):
    md = pr.get('metadata') or {}
    return md.get('scenario') or md.get('noise_type') or pr.get('scenario') or 'unknown'


def run(ctx, args=None):
    out_dir = Path(ctx.out_dir) / 'heatmaps'
    out_dir.mkdir(parents=True, exist_ok=True)
    perfs = ctx.perfs
    models = ctx.models
    splits = ctx.splits
    artifacts = []
    try:
        import matplotlib.pyplot as plt

        # collect global scenario set across perfs
        scenario_set = set()
        for p in perfs:
            for pr in p.get('predictions', []):
                scenario_set.add(_get_scenario(pr))
        scenarios = sorted(list(scenario_set))
        if not scenarios:
            vr = VizResult()
            vr.skipped = True
            vr.summary = {'reason': 'no scenarios found in predictions'}
            return vr

        # build matrix: rows=models, cols=scenarios
        matrix = np.full((len(models), len(scenarios)), np.nan)
        counts = np.zeros((len(models), len(scenarios)), dtype=int)

        for i, m in enumerate(models):
            mup = m.upper()
            # collect all preds for this model across splits
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

        # order scenarios by average SNR for readability
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

        # sort scenarios by avg_snr descending (higher SNR first)
        order = sorted(range(len(scenarios)), key=lambda i: (-(avg_snr[scenarios[i]] if not np.isnan(avg_snr[scenarios[i]]) else -1e9), scenarios[i]))
        scenarios_ord = [scenarios[i] for i in order]
        matrix_ord = matrix[:, order]
        counts_ord = counts[:, order]

        # detect drone flag per scenario (1 if any example indicates drone present)
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
            # fallback name check
            if prs:
                name = _get_scenario(prs[0])
                if isinstance(name, str) and 'drone' in name.lower():
                    return 1
            return 0

        drone_flags = []
        for sc in scenarios_ord:
            preds_for_sc = [p for p in perfs for p in p.get('predictions', []) if _get_scenario(p) == sc]
            drone_flags.append(_detect_drone_flag_for_list(preds_for_sc))

            # use shared utility to create multiple heatmaps (all, val, test, val_test)
            from tools.heatmap_utils import generate_model_scenario_heatmap

            plots = [
                ('all', None, 'Accuracy Heatmap by Scenario (all splits)'),
                ('val', ['val'], 'Accuracy Heatmap by Scenario (val)'),
                ('test', ['test'], 'Accuracy Heatmap by Scenario (test)'),
                ('val_test', ['val', 'test'], 'Accuracy Heatmap by Scenario (val+test)'),
            ]

            any_ok = False
            for suffix, splits_arg, title in plots:
                outp = out_dir / f'model_scenario_heatmap__{suffix}.png'
                try:
                    generate_model_scenario_heatmap(perfs, models, outp, title=title, splits=splits_arg, show_snr_only_for_drone=True)
                    artifacts.append(str(outp))
                    any_ok = True
                except Exception:
                    # continue to next plot; we don't fail the whole plugin for one missing split
                    continue
            if not any_ok:
                vr = VizResult()
                vr.skipped = True
                vr.summary = {'reason': 'no heatmaps produced for any split'}
                return vr

    except Exception as e:
        vr = VizResult()
        vr.skipped = True
        vr.errors.append(str(e))
        return vr

    vr = VizResult()
    vr.artifacts = [Path(a) for a in artifacts]
    vr.summary = {'produced': len(artifacts)}
    return vr
