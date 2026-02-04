"""Shared utilities to generate model x scenario heatmaps.

Provides a single function `generate_model_scenario_heatmap` which encapsulates
the logic previously duplicated across multiple viz plugins.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def _get_scenario(pr):
    md = pr.get('metadata') or {}
    return md.get('scenario') or md.get('noise_type') or pr.get('scenario') or 'unknown'


def generate_model_scenario_heatmap(perfs: List[dict], models: List[str], out_path: Path,
                                    title: str = 'Accuracy Heatmap', splits: Optional[List[str]] = None,
                                    show_snr_only_for_drone: bool = True) -> Tuple[Path, dict]:
    """Generate and save a model x scenario heatmap.

    perfs: list of perf dicts (each containing 'metadata' and 'predictions')
    models: list of model names (rows)
    out_path: full path where PNG will be written
    title: figure title
    splits: optional list of split names to include (e.g., ['val','test'])
    show_snr_only_for_drone: when True, SNR labels are shown only for scenarios
        detected as containing drone sounds.

    Returns: (out_path, info_dict)
    """
    import matplotlib.pyplot as plt

    # optionally filter perfs by split
    if splits is not None:
        perfs = [p for p in perfs if (p.get('metadata', {}).get('split') in set(splits))]

    # collect scenario set
    scenario_set = set()
    for p in perfs:
        for pr in p.get('predictions', []):
            scenario_set.add(_get_scenario(pr))
    scenarios = sorted(list(scenario_set))
    if not scenarios:
        raise ValueError('no scenarios found in provided perfs')

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

    # compute avg_snr per scenario
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

    # ordering
    order = sorted(range(len(scenarios)), key=lambda i: (-(avg_snr[scenarios[i]] if not np.isnan(avg_snr[scenarios[i]]) else -1e9), scenarios[i]))
    scenarios_ord = [scenarios[i] for i in order]
    matrix_ord = matrix[:, order]
    counts_ord = counts[:, order]

    # drone detection
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

    # plot
    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(models))))
    cmap = plt.get_cmap('RdYlGn')
    vmin = np.nanmin(matrix_ord)
    vmax = np.nanmax(matrix_ord)
    if np.isnan(vmin) or np.isnan(vmax):
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(matrix_ord, aspect='auto', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    # annotate scenario with avg SNR when appropriate
    xticks = []
    for j, sc in enumerate(scenarios_ord):
        sn = avg_snr.get(sc)
        if show_snr_only_for_drone and drone_flags[j] == 0:
            xticks.append(sc)
        else:
            if np.isnan(sn):
                xticks.append(sc)
            else:
                xticks.append(f"{sc}\n({sn:.0f} dB)")

    ax.set_xticks(range(len(scenarios_ord)))
    ax.set_xticklabels(xticks, rotation=45, ha='right')
    ax.set_xlabel('Scenario (avg SNR shown only for drone sounds)' if show_snr_only_for_drone else 'Scenario (avg SNR)')
    ax.set_title(title)

    for i in range(matrix_ord.shape[0]):
        for j in range(matrix_ord.shape[1]):
            v = matrix_ord[i, j]
            if np.isnan(v):
                continue
            txt = f'{v:.3f}\n({counts_ord[i,j]})'
            ax.text(j, i, txt, ha='center', va='center', fontsize='8')

    # separator between drone and no-drone
    boundary = None
    for i in range(1, len(drone_flags)):
        if drone_flags[i-1] != drone_flags[i]:
            boundary = i
            break
    if boundary is not None:
        ax.axvline(boundary - 0.5, color='black', linewidth=2, zorder=5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Accuracy')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close(fig)

    info = {'scenarios': scenarios_ord, 'drone_flags': drone_flags, 'avg_snr': avg_snr}
    return out_path, info
