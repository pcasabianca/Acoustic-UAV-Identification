from .style import set_style
from .common import resolve_output_dir, savefig
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List


def plot_perf_vs_snr(metric_series: List[float], bins: List[float], model: str, out_dir: str, metric_name: str):
    plt = set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    # bins center
    centers = [(bins[i] + bins[i+1]) / 2.0 for i in range(len(bins)-1)]
    ax.plot(centers, metric_series, marker='o')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{model} - {metric_name} vs SNR')
    out = resolve_output_dir(out_dir)
    out_path = out / f'perf_vs_snr__{model}__{metric_name}.png'
    savefig(fig, out_path)
    return str(out_path)
