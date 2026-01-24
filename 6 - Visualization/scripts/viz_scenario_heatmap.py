"""Plugin: scenario x SNR heatmap of a chosen metric (default: recall).

Produces one heatmap per split showing metric values for each scenario (row)
and SNR bin (column). Falls back gracefully if scenario metadata missing.
"""
from tools.viz_core import VizResult
from pathlib import Path
import numpy as np

VIZ_NAME = 'scenario_heatmap'
VIZ_ORDER = 20


def _get_scenario(pr):
    md = pr.get('metadata') or {}
    return md.get('scenario') or md.get('noise_type') or pr.get('scenario') or 'unknown'


def run(ctx, args=None):
    out_dir = Path(ctx.out_dir) / 'heatmaps'
    out_dir.mkdir(parents=True, exist_ok=True)
    perfs = ctx.perfs
    splits = ctx.splits
    artifacts = []
    split_summaries = {}
    try:
        import matplotlib.pyplot as plt

        for split in splits:
            # gather preds across all models for this split (only positives needed for recall)
            all_preds = []
            for p in perfs:
                md = p.get('metadata', {})
                if md.get('split') != split:
                    continue
                for pr in p.get('predictions', []):
                    if pr.get('true_label') == 1:
                        all_preds.append(pr)

            if not all_preds:
                continue

            # collect scenarios and snr values
            scenarios = {}
            snr_vals = []
            for pr in all_preds:
                sc = _get_scenario(pr)
                scenarios.setdefault(sc, []).append(pr)
                md = pr.get('metadata') or {}
                rd = md.get('raw_debug') or md
                sv = rd.get('resulting_snr_db') if isinstance(rd, dict) else md.get('snr_db')
                if sv is None:
                    sv = pr.get('snr_db')
                try:
                    if sv is not None:
                        snr_vals.append(float(sv))
                except Exception:
                    pass

            if not snr_vals:
                continue

            nbins = 20
            lo = min(snr_vals)
            hi = max(snr_vals)
            if lo == hi:
                bins = np.array([lo - 0.5, hi + 0.5])
            else:
                bins = np.linspace(lo, hi, nbins + 1)

            scenario_list = sorted(scenarios.keys())
            matrix = np.full((len(scenario_list), nbins), np.nan)

            for i, sc in enumerate(scenario_list):
                preds = scenarios[sc]
                # compute snr array for preds
                snrs = []
                for pr in preds:
                    md = pr.get('metadata') or {}
                    rd = md.get('raw_debug') or md
                    sv = rd.get('resulting_snr_db') if isinstance(rd, dict) else md.get('snr_db')
                    if sv is None:
                        sv = pr.get('snr_db')
                    try:
                        snrs.append(float(sv))
                    except Exception:
                        snrs.append(np.nan)

                for bidx in range(nbins):
                    lo_b = bins[bidx]
                    hi_b = bins[bidx + 1]
                    sel = [pr for pr, s in zip(preds, snrs) if (not np.isnan(s)) and (s >= lo_b and s < hi_b)]
                    if not sel:
                        continue
                    tp = sum(1 for p in sel if p.get('pred_label') == 1 and p.get('true_label') == 1)
                    fn = sum(1 for p in sel if p.get('pred_label') == 0 and p.get('true_label') == 1)
                    if tp + fn == 0:
                        val = np.nan
                    else:
                        val = float(tp) / float(tp + fn)
                    matrix[i, bidx] = val

            # order scenarios by avg snr (descending)
            avg_snr = []
            for sc in scenario_list:
                vals = []
                for pr in scenarios[sc]:
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
                avg_snr.append((sc, float(np.mean(vals)) if vals else float('nan')))

            order = [s for s, _ in sorted(avg_snr, key=lambda x: (-(x[1] if not np.isnan(x[1]) else -1e9), x[0]))]
            scenario_order = order
            # build ordered matrix
            mat_ord = np.full((len(scenario_list), nbins), np.nan)
            for i, sc in enumerate(scenario_order):
                orig_i = scenario_list.index(sc)
                mat_ord[i, :] = matrix[orig_i, :]

            fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(scenario_list))))
            cmap = plt.get_cmap('viridis')
            vmin = np.nanmin(mat_ord)
            vmax = np.nanmax(mat_ord)
            if np.isnan(vmin) or np.isnan(vmax):
                vmin, vmax = 0.0, 1.0
            im = ax.imshow(mat_ord, aspect='auto', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_yticks(range(len(scenario_order)))
            ax.set_yticklabels(scenario_order)
            centers = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)]
            ax.set_xticks(range(len(centers)))
            ax.set_xticklabels([f'{c:.0f}' for c in centers], rotation=45)
            ax.set_xlabel('SNR (dB)')
            ax.set_title(f'{split} - recall heatmap by scenario vs SNR')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('recall')
            outp = out_dir / f'heatmap__{split}.png'
            fig.tight_layout()
            fig.savefig(str(outp), dpi=200, bbox_inches='tight')
            plt.close(fig)
            artifacts.append(str(outp))

            # record summary info for this split
            try:
                split_summaries[split] = {'n_scenarios': len(scenario_list)}
            except Exception:
                pass

    except Exception as e:
        vr = VizResult()
        vr.skipped = True
        vr.errors.append(str(e))
        return vr

    vr = VizResult()
    vr.artifacts = [Path(a) for a in artifacts]
    vr.summary = {'produced': len(artifacts), 'split_summaries': split_summaries}
    return vr
