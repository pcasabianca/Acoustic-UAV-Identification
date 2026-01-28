from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

import config
from tools.viz_core import VizResult

VIZ_NAME = 'dataset_scenario_counts'
VIZ_ORDER = 10


def _load_build_config(path: Path):
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def _compute_counts(cfg):
    total = int(cfg.get('output', {}).get('total_samples', 0))
    splits = cfg.get('output', {}).get('split_ratio', [0.8, 0.1, 0.1])
    scenarios = cfg.get('scenarios', [])

    names = [s.get('name', f'scn_{i}') for i, s in enumerate(scenarios)]
    labels = [s.get('label', 0) for s in scenarios]
    weights = [float(s.get('weight', 0.0)) for s in scenarios]

    totals = np.array(weights) * float(total)
    per_split = np.vstack([totals * splits[0], totals * splits[1], totals * splits[2]]).T

    # compute component counts per sample for each scenario
    drone_components_per_sample = []
    non_drone_components_per_sample = []
    for s in scenarios:
        comps = s.get('components', [])
        dcount = 0
        ndcount = 0
        for c in comps:
            role = (c.get('role') or '').lower()
            cnt = int(c.get('count', 1))
            if role == 'drone':
                dcount += cnt
            else:
                ndcount += cnt
        drone_components_per_sample.append(float(dcount))
        non_drone_components_per_sample.append(float(ndcount))

    # Use per-sample component counts (not multiplied by total samples)
    # This shows, for one example of the scenario, how many drone/non-drone
    # components are present on average in that scenario.
    drone = np.array(drone_components_per_sample)
    ambient = np.array(non_drone_components_per_sample)

    return {
        'names': names,
        'per_split': per_split,
        'drone': drone,
        'ambient': ambient,
        'labels': labels,
    }


def _plot_counts(counts, out_dir: Path):
    names = counts['names']
    per_split = counts['per_split']
    drone = counts['drone']
    ambient = counts['ambient']
    labels = counts['labels']

    n = len(names)
    x = np.arange(n)
    width = 0.25

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, max(6, 0.6 * n)), sharex=True,
                                         gridspec_kw={'height_ratios': [2, 1]})

    # Top: grouped bars per split
    # Base hue per scenario: blue for drone scenarios, red for noise scenarios
    base_blue = (0.2, 0.4, 0.8)
    base_red = (0.85, 0.3, 0.2)
    base_colors = [base_blue if lab == 1 else base_red for lab in labels]
    split_alphas = (0.95, 0.75, 0.55)

    max_val = per_split.max() if per_split.size else 1.0
    top_offset = max(6.0, 0.03 * max_val)
    for i in range(3):
        bar_colors = [(*base_colors[j], split_alphas[i]) for j in range(n)]
        bars = ax_top.bar(x + (i - 1) * width, per_split[:, i], width, label=('train' if i==0 else 'val' if i==1 else 'test'), color=bar_colors)
        # annotate top bars with counts (rounded ints) placed slightly above bars
        for b, val in zip(bars, per_split[:, i]):
            h = float(val)
            txt = f"{int(round(h))}"
            ax_top.text(b.get_x() + b.get_width()/2, h + top_offset, txt, ha='center', va='bottom', rotation=90, fontsize=7)
    ax_top.set_ylabel('Number of samples')
    ax_top.set_title('Samples per scenario by split')
    ax_top.legend()
    ax_top.grid(axis='y', linestyle='--', alpha=0.4)

    # Bottom: per-scenario component composition
    drone_bars = []
    ambient_bars = []
    for j in range(n):
        base = base_colors[j]
        # use same hue per scenario; differentiate ambient/drone by alpha
        ambient_color = (*base, 0.6)
        drone_color = (*base, 0.95)
        b1 = ax_bot.bar(x[j], -ambient[j], width * 1.0, color=ambient_color)
        b2 = ax_bot.bar(x[j], drone[j], width * 1.0, color=drone_color)
        ambient_bars.append(b1)
        drone_bars.append(b2)

    # annotate bottom bars: place annotations slightly outside bars
    max_comp = max(drone.max() if drone.size else 0.0, ambient.max() if ambient.size else 0.0)
    comp_offset = max(1.0, 0.06 * max_comp)
    for j in range(n):
        h = float(drone[j])
        txt = f"{h:.1f}" if (h != int(h)) else f"{int(h)}"
        ax_bot.text(x[j], h + comp_offset, txt, ha='center', va='bottom', rotation=90, fontsize=7)
        h2 = float(ambient[j])
        txt2 = f"{h2:.1f}" if (h2 != int(h2)) else f"{int(h2)}"
        ax_bot.text(x[j], -h2 - comp_offset, txt2, ha='center', va='top', rotation=90, fontsize=7)
    ax_bot.set_ylabel('Total samples (drone above / ambient below)')
    ax_bot.axhline(0, color='black', linewidth=0.8)
    ax_bot.grid(axis='y', linestyle='--', alpha=0.4)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(names, rotation=60, ha='right')

    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / 'scenario_counts__all.png'
    fig.tight_layout()
    fig.savefig(str(outp), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return outp


def run(ctx, args=None):
    vr = VizResult()
    try:
        cfg_path = Path(config.AUGMENTATION_CONFIG)
        if not cfg_path.exists():
            vr.skipped = True
            vr.summary = {'reason': f'build_config not found: {cfg_path}'}
            return vr

        cfg = _load_build_config(cfg_path)
        counts = _compute_counts(cfg)
        # attempt to order scenarios the same way as the heatmap: by avg SNR
        scenarios = counts['names']
        # compute avg_snr per scenario from ctx.perfs if available
        avg_snr = {s: float('nan') for s in scenarios}
        perfs = getattr(ctx, 'perfs', []) or []
        def _get_scenario_from_pred(pr):
            md = pr.get('metadata') or {}
            return md.get('scenario') or md.get('noise_type') or pr.get('scenario') or 'unknown'

        for s in scenarios:
            vals = []
            for p in perfs:
                for pr in p.get('predictions', []):
                    sc = _get_scenario_from_pred(pr)
                    if sc == s:
                        md = pr.get('metadata') or {}
                        rd = md.get('raw_debug') or md
                        sv = None
                        if isinstance(rd, dict):
                            sv = rd.get('resulting_snr_db') or rd.get('snr_db')
                        if sv is None:
                            sv = pr.get('snr_db')
                        try:
                            if sv is not None:
                                vals.append(float(sv))
                        except Exception:
                            pass
            if vals:
                avg_snr[s] = float(np.mean(vals))

        # order scenarios by avg_snr descending, fallback to original order
        order = sorted(range(len(scenarios)), key=lambda i: (-(avg_snr[scenarios[i]] if not np.isnan(avg_snr[scenarios[i]]) else -1e9), scenarios[i]))
        # reorder counts according to order
        names_ord = [counts['names'][i] for i in order]
        per_split_ord = counts['per_split'][order, :]
        drone_ord = counts['drone'][order]
        ambient_ord = counts['ambient'][order]
        labels_ord = [counts['labels'][i] for i in order]

        # build a new counts dict for plotting
        counts_ord = {
            'names': names_ord,
            'per_split': per_split_ord,
            'drone': drone_ord,
            'ambient': ambient_ord,
            'labels': labels_ord,
        }

        out_dir = Path(ctx.out_dir) / 'summary'
        outp = _plot_counts(counts_ord, out_dir)
        vr.artifacts = [outp]
        vr.summary = {'produced': 1}
        return vr
    except Exception as e:
        vr.skipped = True
        vr.errors.append(str(e))
        return vr
