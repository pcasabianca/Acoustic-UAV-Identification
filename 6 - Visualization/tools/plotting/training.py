"""
Improved training curves plotting: smoothing, PR/F1 handling, overfit gap annotations,
learning-rate in log scale, and a small JSON summary exported next to the PNG.
"""
from .style import set_style
from .common import resolve_output_dir, savefig
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import json


def _smooth_series(series, window=None):
    arr = np.array(series, dtype=float)
    n = arr.size
    if n < 3:
        return arr
    if window is None:
        window = max(3, int(n // 10))
    if window % 2 == 0:
        window += 1
    pad = (window - 1) // 2
    padded = np.pad(arr, pad, mode='edge')
    kernel = np.ones(window) / float(window)
    smooth = np.convolve(padded, kernel, mode='valid')
    return smooth


def _find_col(df, name):
    if name in df.columns:
        return name
    for c in df.columns:
        if name in c:
            return c
    return None


def plot_learning_curves(df, model_name: str, out_dir: Optional[str] = None):
    plt = set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fontsize_title = 14
    fontsize_ticks = 10

    # epochs
    if 'epoch' in df.columns:
        epochs = df['epoch'].values
    else:
        epochs = np.arange(len(df)) + 1

    # Loss
    ax = axes[0, 0]
    plotted_loss = False
    loss_present = _find_col(df, 'loss') is not None
    val_loss_present = _find_col(df, 'val_loss') is not None
    if loss_present:
        col = _find_col(df, 'loss')
        raw = df[col].values.astype(float)
        # plot raw in background with 50% transparency
        ax.plot(epochs, raw, color='C0', alpha=0.5, linewidth=0.8)
        series = _smooth_series(raw)
        ax.plot(epochs, series, label='train_loss', linewidth=1.5)
        plotted_loss = True
    if val_loss_present:
        col = _find_col(df, 'val_loss')
        raw = df[col].values.astype(float)
        ax.plot(epochs, raw, color='C1', alpha=0.5, linewidth=0.8)
        series = _smooth_series(raw)
        ax.plot(epochs, series, label='val_loss', linewidth=1.5)
        plotted_loss = True
    # also plot the overfit gap on the loss subplot and mark the epoch of maximum gap
    if loss_present and val_loss_present:
        try:
            raw_gap = df[_find_col(df, 'val_loss')].astype(float).values - df[_find_col(df, 'loss')].astype(float).values
            sg = _smooth_series(raw_gap)
            ax2 = ax.twinx()
            ax2.plot(epochs, sg, color='C4', linewidth=1.0, label='val_loss - train_loss')
            ax2.axhline(0.0, color='k', linewidth=0.6, alpha=0.5)
            try:
                idx = int(np.nanargmax(sg))
                ax2.scatter([epochs[idx]], [sg[idx]], color='C3', s=40, zorder=5)
                ax2.annotate(f'max gap {sg[idx]:.4f} (ep {int(epochs[idx])})', xy=(epochs[idx], sg[idx]), xytext=(10, 10), textcoords='offset points', fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8))
            except Exception:
                pass
            ax2.set_ylabel('Gap', fontsize=9)
            ax2.tick_params(axis='y', labelsize=fontsize_ticks)
            ax2.legend(loc='upper right', fontsize='small')
        except Exception:
            pass
    ax.set_title('Loss', fontsize=fontsize_title)
    ax.grid(True, alpha=0.25)
    if plotted_loss:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
    else:
        ax.text(0.5, 0.5, 'No loss columns found', ha='center', va='center')

    # Accuracy
    ax = axes[0, 1]
    plotted_acc = False
    if _find_col(df, 'accuracy'):
        col = _find_col(df, 'accuracy')
        raw = df[col].values.astype(float)
        ax.plot(epochs, raw, color='C0', alpha=0.5, linewidth=0.8)
        series = _smooth_series(raw)
        ax.plot(epochs, series, label='train_accuracy', linewidth=1.5)
        plotted_acc = True
    if _find_col(df, 'val_accuracy'):
        col = _find_col(df, 'val_accuracy')
        raw = df[col].values.astype(float)
        ax.plot(epochs, raw, color='C1', alpha=0.5, linewidth=0.8)
        series = _smooth_series(raw)
        ax.plot(epochs, series, label='val_accuracy', linewidth=1.5)
        plotted_acc = True
    ax.set_title('Accuracy', fontsize=fontsize_title)
    ax.grid(True, alpha=0.25)
    if plotted_acc:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
    else:
        ax.text(0.5, 0.5, 'No accuracy columns found', ha='center', va='center')

    # Confusion Matrix Evolution
    ax = axes[1, 0]
    cm_cols = {}
    for metric in ('val_tp', 'val_tn', 'val_fp', 'val_fn'):
        col = _find_col(df, metric)
        if col:
            cm_cols[metric] = col
    
    if len(cm_cols) == 4:
        # Plot all 4 confusion matrix components
        colors = {'val_tp': 'C2', 'val_tn': 'C0', 'val_fp': 'C3', 'val_fn': 'C1'}
        labels = {'val_tp': 'True Pos', 'val_tn': 'True Neg', 'val_fp': 'False Pos', 'val_fn': 'False Neg'}
        
        for metric, col in cm_cols.items():
            raw = df[col].astype(float).values
            ax.plot(epochs, raw, color=colors[metric], alpha=0.5, linewidth=0.8)
            series = _smooth_series(raw)
            ax.plot(epochs, series, label=labels[metric], color=colors[metric], linewidth=1.25)
        
        ax.set_title('Confusion Matrix Evolution', fontsize=fontsize_title)
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
    else:
        ax.set_title('Confusion Matrix Evolution', fontsize=fontsize_title)
        ax.text(0.5, 0.5, 'TP/TN/FP/FN not recorded in history', ha='center', va='center')

    # Derived Metrics (Specificity, NPV, Balanced Accuracy)
    ax = axes[1, 1]
    derived_cols = {}
    for metric in ('val_specificity', 'val_npv', 'val_balanced_accuracy'):
        col = _find_col(df, metric)
        if col:
            derived_cols[metric] = col
    
    if derived_cols:
        colors = {'val_specificity': 'C5', 'val_npv': 'C6', 'val_balanced_accuracy': 'C4'}
        labels = {'val_specificity': 'Specificity', 'val_npv': 'NPV', 'val_balanced_accuracy': 'Balanced Acc'}
        
        for metric, col in derived_cols.items():
            raw = df[col].astype(float).values
            ax.plot(epochs, raw, color=colors.get(metric, 'C0'), alpha=0.5, linewidth=0.8)
            series = _smooth_series(raw)
            ax.plot(epochs, series, label=labels.get(metric, metric), 
                   color=colors.get(metric, 'C0'), linewidth=1.25)
        
        # Also add Precision/Recall/F1 if available for comparison
        for metric in ('val_precision', 'val_recall', 'val_f1'):
            col = _find_col(df, metric)
            if col:
                raw = df[col].astype(float).values
                color = 'C0' if 'prec' in metric else ('C1' if 'rec' in metric else 'C2')
                series = _smooth_series(raw)
                label = metric.replace('val_', '').replace('_', ' ').title()
                ax.plot(epochs, series, label=label, color=color, 
                       linewidth=1.0, linestyle='--', alpha=0.7)
        
        ax.set_ylim(0.0, 1.0)
        ax.set_title('Derived Metrics', fontsize=fontsize_title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
    else:
        ax.set_title('Derived Metrics', fontsize=fontsize_title)
        ax.text(0.5, 0.5, 'Derived metrics not recorded in history', ha='center', va='center')

    # ticks and layout
    for axr in axes.ravel():
        for label in axr.get_xticklabels():
            label.set_fontsize(fontsize_ticks)
        for label in axr.get_yticklabels():
            label.set_fontsize(fontsize_ticks)

    fig.suptitle(f'Training summary: {model_name}', fontsize=16)
    # avoid expensive tight_layout; use a conservative subplots_adjust
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.88, hspace=0.35, wspace=0.3)

    out = resolve_output_dir(out_dir)
    out_path = out / f'training__{model_name}.png'
    # summary json
    summary = {}
    # best epoch heuristics
    best_metric = None
    if _find_col(df, 'val_f1'):
        col = _find_col(df, 'val_f1')
        idx = int(np.nanargmax(df[col].astype(float).values))
        best_metric = ('val_f1', float(df[col].astype(float).values[idx]), int(df['epoch'].values[idx]) if 'epoch' in df.columns else idx + 1)
    elif _find_col(df, 'val_accuracy'):
        col = _find_col(df, 'val_accuracy')
        idx = int(np.nanargmax(df[col].astype(float).values))
        best_metric = ('val_accuracy', float(df[col].astype(float).values[idx]), int(df['epoch'].values[idx]) if 'epoch' in df.columns else idx + 1)
    elif _find_col(df, 'val_loss') and _find_col(df, 'loss'):
        col = _find_col(df, 'val_loss')
        idx = int(np.nanargmin(df[col].astype(float).values))
        best_metric = ('val_loss', float(df[col].astype(float).values[idx]), int(df['epoch'].values[idx]) if 'epoch' in df.columns else idx + 1)
    if best_metric:
        summary['best_metric'] = {'name': best_metric[0], 'value': best_metric[1], 'epoch': best_metric[2]}
    
    # final values - include all important metrics
    final = {}
    metric_names = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 
                   'val_precision', 'val_recall', 'val_f1',
                   'val_specificity', 'val_npv', 'val_balanced_accuracy',
                   'val_tp', 'val_tn', 'val_fp', 'val_fn']
    for k in metric_names:
        col = _find_col(df, k)
        if col:
            try:
                final[k] = float(df[col].astype(float).values[-1])
            except Exception:
                final[k] = None
    summary['final'] = final
    
    summary_path = out / f'training__{model_name}.json'
    try:
        with open(summary_path, 'w') as fh:
            json.dump(summary, fh, indent=2)
    except Exception:
        pass

    savefig(fig, out_path)
    return str(out_path)
