"""Plugin: dense perf vs SNR curves overlaying models.

Exposes VIZ_NAME, VIZ_ORDER, add_args, run(ctx, args) -> VizResult
"""
from tools.viz_core import VizResult
from pathlib import Path
from collections import defaultdict
from typing import List
import math

VIZ_NAME = 'perf_vs_snr'
VIZ_ORDER = 10


def _quantile_bins(values: List[float], n_bins: int = 30) -> List[float]:
    vals = sorted([v for v in values if v is not None and (not math.isnan(v))])
    if len(vals) < 2:
        return []
    bins = []
    L = len(vals)
    for i in range(n_bins + 1):
        idx = int(i * (L - 1) / n_bins)
        bins.append(vals[idx])
    # ensure monotonic
    bins = sorted(list(dict.fromkeys(bins)))
    if len(bins) < 2:
        return []
    return bins


def run(ctx, args=None):
    out_dir = Path(ctx.out_dir) / 'perf_vs_snr'
    out_dir.mkdir(parents=True, exist_ok=True)
    perfs = ctx.perfs
    models = ctx.models
    splits = ctx.splits
    # determine which splits to plot (prefer test and val and train if present)
    present_splits = [s for s in ('train', 'val', 'test') if s in splits]
    if not present_splits:
        return VizResult()

    # gather all snr values across predictions for binning
    snr_vals = []
    for p in perfs:
        for pr in p.get('predictions', []):
            if pr.get('true_label') == 1:
                md = pr.get('metadata', {}) or {}
                rd = md.get('raw_debug') or md
                sv = rd.get('resulting_snr_db') if isinstance(rd, dict) else md.get('snr_db')
                if sv is None:
                    sv = pr.get('snr_db')
                try:
                    if sv is not None:
                        snr_vals.append(float(sv))
                except Exception:
                    pass

    bins = _quantile_bins(snr_vals, n_bins=30)
    if not bins:
        return VizResult()

    try:
        from tools.plotting.style import set_style
        import matplotlib.pyplot as plt
        import numpy as _np
        plt = set_style()

        n_sub = len(present_splits)
        fig, axes = plt.subplots(1, n_sub, figsize=(5 * n_sub, 5), squeeze=False)
        axes = axes[0]

        from tools.perf.snr import bin_by_snr, metric_over_bins

        legend_lines = []
        legend_labels = []

        for ax_idx, split in enumerate(present_splits):
            ax = axes[ax_idx]
            centers = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)]
            for m in models:
                # gather all perf entries for this model+split
                model_perfs = [p for p in perfs if p.get('metadata', {}).get('model') == m and p.get('metadata', {}).get('split') == split]
                series_list = []
                for perf in model_perfs:
                    preds = [pr for pr in perf.get('predictions', []) if pr.get('true_label') == 1]
                    binned = bin_by_snr(preds, bins)
                    stats = metric_over_bins(binned, metric='recall')
                    metric_series = [stats[i]['metric'] for i in sorted(stats.keys())]
                    series_list.append(metric_series)

                if not series_list:
                    continue
                arr = _np.array(series_list, dtype=float)
                if arr.ndim == 1 or arr.shape[0] == 1:
                    median = arr.mean(axis=0)
                    p10 = median
                    p90 = median
                else:
                    median = _np.nanmedian(arr, axis=0)
                    p10 = _np.nanpercentile(arr, 10, axis=0)
                    p90 = _np.nanpercentile(arr, 90, axis=0)

                # smoothing median with small window
                try:
                    if median.size >= 5:
                        w = max(3, int(median.size // 10))
                        if w % 2 == 0:
                            w += 1
                        pad = (w - 1) // 2
                        median = _np.convolve(_np.pad(median, pad, mode='edge'), _np.ones(w) / w, mode='valid')
                        p10 = _np.convolve(_np.pad(p10, pad, mode='edge'), _np.ones(w) / w, mode='valid')
                        p90 = _np.convolve(_np.pad(p90, pad, mode='edge'), _np.ones(w) / w, mode='valid')
                except Exception:
                    pass

                line, = ax.plot(centers, median, linewidth=1.5, label=m)
                ax.fill_between(centers, p10, p90, alpha=0.15, color=line.get_color())
                if ax_idx == 0:
                    legend_lines.append(line)
                    legend_labels.append(m)

            ax.set_title(f'{split} - recall vs SNR')
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel('recall')

        # single legend at bottom for all subplots
        if legend_lines:
            fig.legend(legend_lines, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=min(6, max(1, len(legend_lines))), fontsize='small')

        out_path = out_dir / f'perf_vs_snr__combined__recall.png'
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
        plt.close(fig)
        artifacts = [str(out_path)]
    except Exception as e:
        vr = VizResult()
        vr.skipped = True
        vr.errors.append(str(e))
        return vr

    vr = VizResult()
    vr.artifacts = [Path(a) for a in artifacts]
    vr.summary = {'produced': len(artifacts)}
    return vr
#!/usr/bin/env python3
import argparse
from tools.io.perf_index import scan_perf_dir, query, closest_threshold, best_for_model_split
from tools.io.thresholds_loader import load_calibrated_thresholds, get_threshold
from tools.perf.snr import filter_drone_only, bin_by_snr, metric_over_bins, parse_bins_from_str
from tools.plotting.snr import plot_perf_vs_snr
import config
import logging

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--perf-dir', default=config.PERFORMANCE_DIR)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--split', default='val')
    p.add_argument('--metric', default='recall')
    p.add_argument('--snr-bins', default='-30,-20,-10,0,10,20,30')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--use-calibrated-thresholds', action='store_true')
    p.add_argument('--rethreshold', action='store_true', help='Recompute pred_label using calibrated threshold')
    args = p.parse_args()

    perfs = scan_perf_dir(args.perf_dir, recursive=True, strict=False)
    calib = None
    try:
        calib = load_calibrated_thresholds(config.CALIBRATION_FILE_PATH)
    except Exception:
        calib = None

    bins = parse_bins_from_str(args.snr_bins)

    for model in args.models:
        model_up = model.upper()
        chosen = None
        if args.use_calibrated_thresholds and calib:
            t = get_threshold(calib, model_up)
            if t is not None:
                chosen = closest_threshold(perfs, model_up, args.split, t)
        if chosen is None:
            # fallback to best by metric
            chosen = best_for_model_split(perfs, model_up, split=args.split, metric=args.metric)
        if not chosen:
            logger.warning('No perf file found for model %s split %s', model_up, args.split)
            continue

        perf = chosen
        preds = filter_drone_only(perf) if True else perf.get('predictions', [])
        if args.rethreshold and calib:
            t = get_threshold(calib, model_up)
            if t is not None:
                for p in preds:
                    p['pred_label'] = 1 if float(p.get('prob_class1', 0.0)) >= float(t) else 0

        binned = bin_by_snr(preds, bins)
        stats = metric_over_bins(binned, metric=args.metric)
        metric_series = [stats[i]['metric'] for i in sorted(stats.keys())]
        out = plot_perf_vs_snr(metric_series, bins, model_up, args.out_dir, args.metric)
        print('Saved:', out)


if __name__ == '__main__':
    main()
