"""
Performance Comparison (Test Set)
Charge les résultats de test précalculés et génère des visualisations.
Ne sélectionne pas ou n'impose pas de thresholds.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

plot_utils.set_style()
OUTPUT_DIR = plot_utils.get_output_dir(__file__)

def plot_global_metrics_comparison(results, output_dir):
    """Comparaison des métriques globales - meilleurs résultats uniquement."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Global Performance Comparison (Test Set)', fontsize=16, fontweight='bold')
    
    # Grouper par modèle et split
    model_split_data = defaultdict(lambda: defaultdict(dict))
    
    for (model, split, _), data in results.items():
        metrics = data.get('global_metrics', {})
        model_split_data[model][split] = metrics
    
    models = sorted(model_split_data.keys())
    splits = sorted({split for model_data in model_split_data.values() for split in model_data.keys()})
    
    metrics_to_plot = [
        ('accuracy', 'Accuracy', axes[0, 0]),
        ('f1_score', 'F1-Score', axes[0, 1]),
        ('precision', 'Precision', axes[1, 0]),
        ('recall', 'Recall', axes[1, 1])
    ]
    
    colors = {'train': '#2ecc71', 'val': '#3498db', 'test': '#e74c3c'}
    width = 0.25
    
    for metric_key, metric_name, ax in metrics_to_plot:
        x = np.arange(len(models))
        
        for i, split in enumerate(splits):
            values = [model_split_data[model].get(split, {}).get(metric_key, 0) for model in models]
            offset = (i - len(splits)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=split.capitalize(), 
                         color=colors.get(split, '#95a5a6'), alpha=0.8, edgecolor='black')
            
            # Ajouter les valeurs sur les barres
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    output_path = output_dir / 'best_global_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(results, output_dir):
    """Matrices de confusion - test set uniquement."""
    test_results = {k: v for k, v in results.items() if k[1] == 'test'}
    
    if not test_results:
        print("  [SKIP] No test results found")
        return
    
    n_models = len(test_results)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 8*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]
    
    fig.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')
    
    for idx, ((model, split, _), data) in enumerate(sorted(test_results.items())):
        metrics = data.get('global_metrics', {})
        
        tp = metrics.get('tp', 0)
        tn = metrics.get('tn', 0)
        fp = metrics.get('fp', 0)
        fn = metrics.get('fn', 0)
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_title(f'{model}\nAcc={metrics.get("accuracy", 0):.3f}, F1={metrics.get("f1_score", 0):.3f}', 
                    fontweight='bold')
        ax.set_xticklabels(['No Drone', 'Drone'])
        ax.set_yticklabels(['No Drone', 'Drone'])
    
    # Masquer axes inutilisés
    for idx in range(len(test_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'best_confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_class_performance(results, output_dir):
    """Performance par classe (drone vs ambient) - test set."""
    test_results = {k: v for k, v in results.items() if k[1] == 'test'}
    
    if not test_results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Class Performance (Test Set)', fontsize=16, fontweight='bold')
    
    models = sorted([k[0] for k in test_results.keys()])
    
    # Préparer les données
    drone_acc = []
    ambient_acc = []
    drone_f1 = []
    ambient_f1 = []
    
    for model in models:
        key = [k for k in test_results.keys() if k[0] == model][0]
        data = test_results[key]
        class_metrics = data.get('class_metrics', {})

        # Require explicit class metrics naming. Accept either ('drone','ambient')
        # or ('class_1','class_0') but do NOT fallback to arbitrary keys.
        if 'drone' in class_metrics and 'ambient' in class_metrics:
            drone_metrics = class_metrics.get('drone', {})
            ambient_metrics = class_metrics.get('ambient', {})
        elif 'class_1' in class_metrics and 'class_0' in class_metrics:
            # Map class_1 -> drone, class_0 -> ambient (based on training conventions)
            drone_metrics = class_metrics.get('class_1', {})
            ambient_metrics = class_metrics.get('class_0', {})
        else:
            raise RuntimeError(f"class_metrics missing required keys ('drone'/'ambient' or 'class_1'/'class_0') for model {model}")

        drone_acc.append(drone_metrics.get('accuracy', 0))
        ambient_acc.append(ambient_metrics.get('accuracy', 0))
        drone_f1.append(drone_metrics.get('f1_score', 0))
        ambient_f1.append(ambient_metrics.get('f1_score', 0))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Subplot 1: Accuracy
    axes[0].bar(x - width/2, drone_acc, width, label='Drone', color='#3498db', alpha=0.8, edgecolor='black')
    axes[0].bar(x + width/2, ambient_acc, width, label='Ambient', color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Model', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('Accuracy by Class', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1.05)
    
    # Subplot 2: F1-Score
    axes[1].bar(x - width/2, drone_f1, width, label='Drone', color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[1].bar(x + width/2, ambient_f1, width, label='Ambient', color='#f39c12', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Model', fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontweight='bold')
    axes[1].set_title('F1-Score by Class', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    output_path = output_dir / 'best_class_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_subcategory_heatmap(results, output_dir):
    """Heatmap de performance par sous-catégorie - test set."""
    test_results = {k: v for k, v in results.items() if k[1] == 'test'}
    
    if not test_results:
        return
    
    # Require explicit `scenario` in each prediction's metadata.
    scenario_set = set()
    missing = []
    for data in test_results.values():
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            # scenario may be stored at top-level metadata['scenario'] or under raw_meta.scenario
            sc = None
            if isinstance(meta, dict):
                sc = (meta.get('scenario') or (meta.get('raw_meta') or {}).get('scenario'))
            if not sc:
                # prefer path for examples if available
                fname = (meta.get('path') or (meta.get('raw_meta') or {}).get('filename') or p.get('filename') or '<no-fname>')
                missing.append(fname)
            else:
                scenario_set.add(sc)

    if missing:
        examples = list(dict.fromkeys(missing))[:12]
        raise RuntimeError(
            "Missing 'scenario' in prediction metadata.raw_meta for some predictions. "
            "Ensure augmentation JSONL and performance JSONs include raw_meta.scenario. "
            "Examples: " + ", ".join(examples)
        )

    # Build matrix aggregated by scenario
    all_subcats = sorted(list(scenario_set))
    
    # If scenarios look like distances (e.g. '850m'), try to sort numerically descending.
    import re
    def extract_distance(s):
        match = re.search(r'(\d+)m', s)
        return int(match.group(1)) if match else None

    # Sort scenarios: numeric distances first (desc), then alphabetical
    distances = [s for s in all_subcats if extract_distance(s) is not None]
    others = [s for s in all_subcats if extract_distance(s) is None]
    distances_sorted = sorted(distances, key=lambda x: extract_distance(x), reverse=True)
    all_subcats = distances_sorted + sorted(others)
    
    models = sorted([k[0] for k in test_results.keys()])
    
    # Build accuracy matrix. If using scenarios, aggregate per-scenario from predictions.
    accuracy_matrix = np.full((len(models), len(all_subcats)), np.nan)

    for i, model in enumerate(models):
        key = [k for k in test_results.keys() if k[0] == model][0]
        data = test_results[key]

        # Aggregate by scenario from predictions using canonical top-level metadata first,
        # then fall back to raw_meta (but prefer canonical as produced by the tester/worker).
        counts = {s: {'correct': 0, 'total': 0} for s in all_subcats}
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            raw_meta = meta.get('raw_meta') or {}
            sc = meta.get('scenario') or (raw_meta.get('scenario') if isinstance(raw_meta, dict) else None)
            if sc in counts:
                counts[sc]['total'] += 1
                if p.get('true_label') == p.get('pred_label'):
                    counts[sc]['correct'] += 1

        for j, subcat in enumerate(all_subcats):
            cnt = counts.get(subcat, {})
            if cnt.get('total', 0) > 0:
                accuracy_matrix[i, j] = cnt['correct'] / cnt['total']
            else:
                accuracy_matrix[i, j] = np.nan
    
    # Plot
    # Also compute mean resulting SNR per scenario (deduplicated by filename)
    # and attach it to x tick labels.
    scenario_snrs = {s: [] for s in all_subcats}
    seen_files = set()
    missing_snr = []
    for data in test_results.values():
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            raw_meta = (meta.get('raw_meta') or {}) if isinstance(meta, dict) else {}
            raw_debug = (meta.get('raw_debug') or {}) if isinstance(meta, dict) else {}
            # determine filename/path used for deduplication
            fname = raw_meta.get('filename') or meta.get('path') or raw_meta.get('path') or p.get('filename')
            if not fname or fname in seen_files:
                continue
            seen_files.add(fname)
            # scenario may appear at metadata['scenario'] or raw_meta['scenario']
            sc = meta.get('scenario') or raw_meta.get('scenario')
            if sc not in scenario_snrs:
                continue
            # Require resulting_snr_db to be present in raw_debug
            if not isinstance(raw_debug, dict) or 'resulting_snr_db' not in raw_debug:
                missing_snr.append(fname)
                continue
            try:
                snr_val = float(raw_debug.get('resulting_snr_db'))
            except Exception:
                missing_snr.append(fname)
                continue
            scenario_snrs[sc].append(snr_val)

    if missing_snr:
        examples = list(dict.fromkeys(missing_snr))[:12]
        raise RuntimeError(
            "Missing or invalid 'resulting_snr_db' in some prediction raw_debug entries. Examples: " + ", ".join(examples)
        )

    # Compute mean SNR per scenario
    scenario_snr_mean = {s: (np.mean(scenario_snrs[s]) if len(scenario_snrs[s])>0 else float('nan')) for s in all_subcats}

    fig, ax = plt.subplots(figsize=(max(14, len(all_subcats) * 0.8), max(8, len(models) * 0.6)))
    
    # Calculer vmin/vmax dynamiquement basé sur les données réelles (ignore NaN)
    valid_data = accuracy_matrix[~np.isnan(accuracy_matrix)]
    vmin_dynamic = valid_data.min() if len(valid_data) > 0 else 0
    vmax_dynamic = valid_data.max() if len(valid_data) > 0 else 1
    
    # Determine scenario class (majority vote) strictly from per-file labels in test_results
    scenario_class_counts = {s: {'class_0': 0, 'class_1': 0} for s in all_subcats}
    seen = set()
    for data in test_results.values():
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            raw_meta = (meta.get('raw_meta') or {}) if isinstance(meta, dict) else {}
            fname = raw_meta.get('filename') or meta.get('path') or raw_meta.get('path') or p.get('filename')
            if not fname or fname in seen:
                continue
            seen.add(fname)
            sc = meta.get('scenario') or raw_meta.get('scenario')
            if sc not in scenario_class_counts:
                continue
            # Use only the explicit numeric label field (do not rely on class_name)
            cls = raw_meta.get('label') if 'label' in raw_meta else meta.get('label')
            if cls is None:
                raise RuntimeError(f"Missing class label for file {fname}")
            try:
                ival = int(cls)
            except Exception:
                raise RuntimeError(f"Unrecognized class label '{cls}' for file {fname}")
            if ival == 0:
                scenario_class_counts[sc]['class_0'] += 1
            else:
                scenario_class_counts[sc]['class_1'] += 1

    # Decide majority class per scenario and group
    class0 = []
    class1 = []
    ties = []
    for s in all_subcats:
        c0 = scenario_class_counts[s]['class_0']
        c1 = scenario_class_counts[s]['class_1']
        if c0 > c1:
            class0.append(s)
        elif c1 > c0:
            class1.append(s)
        else:
            ties.append((s, c0, c1))
    if ties:
        examples = ', '.join([f"{t[0]}({t[1]} vs {t[2]})" for t in ties[:12]])
        raise RuntimeError(f"Tie or no majority class for scenarios: {examples}")

    # Note: do not warn about single-class scenarios here; keep heatmap columns visible.

    # Keep original ordering reference for indexing later
    orig_all_subcats = list(all_subcats)

    # Sort each group by mean SNR ascending (lower SNR -> harder)
    class0_sorted = sorted(class0, key=lambda x: scenario_snr_mean.get(x, float('nan')))
    class1_sorted = sorted(class1, key=lambda x: scenario_snr_mean.get(x, float('nan')))
    ordered_all_subcats = class0_sorted + class1_sorted

    # Keep single-class scenarios visible without special marking.

    def _snr_label(s):
        v = scenario_snr_mean.get(s, float('nan'))
        # Detect NaN
        if v != v:
            snr_part = 'n/a'
        else:
            snr_part = f"{v:.1f} dB"
        return f"{s}\n{snr_part}"

    # Build xtick labels that include mean SNR per scenario in the new order
    xtick_labels = [_snr_label(s) for s in ordered_all_subcats]

    # Reorder accuracy_matrix columns to match ordered_all_subcats
    col_idx = [orig_all_subcats.index(s) for s in ordered_all_subcats]
    accuracy_matrix_reordered = accuracy_matrix[:, col_idx]

    sns.heatmap(accuracy_matrix_reordered, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
               xticklabels=xtick_labels, yticklabels=models, cbar_kws={'label': 'Accuracy'},
               vmin=vmin_dynamic, vmax=vmax_dynamic, linewidths=0.5, linecolor='gray')
    
    ax.set_title('Accuracy Heatmap by Subcategory (Test Set)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Subcategory', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / 'best_subcategory_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_subcategory_heatmap_all(results_all, output_dir):
    """Heatmap aggregated across all splits and all performance JSONs.
    This builds per-model accuracy per scenario using all available predictions,
    deduplicating by filename per model to avoid double-counting.
    """
    # Collect scenario set from all results
    scenario_set = set()
    for data in results_all.values():
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            sc = meta.get('scenario') or (meta.get('raw_meta') or {}).get('scenario')
            if sc:
                scenario_set.add(sc)

    if not scenario_set:
        print("  ⚠ No scenarios found in all-results; skipping all-splits heatmap")
        return

    all_subcats = sorted(list(scenario_set))
    import re
    def extract_distance(s):
        match = re.search(r'(\d+)m', s)
        return int(match.group(1)) if match else None
    distances = [s for s in all_subcats if extract_distance(s) is not None]
    others = [s for s in all_subcats if extract_distance(s) is None]
    distances_sorted = sorted(distances, key=lambda x: extract_distance(x), reverse=True)
    all_subcats = distances_sorted + sorted(others)

    models = sorted(list({k[0] for k in results_all.keys()}))

    accuracy_matrix = np.full((len(models), len(all_subcats)), np.nan)

    for i, model in enumerate(models):
        # aggregate across all entries for this model
        counts = {s: {'correct': 0, 'total': 0} for s in all_subcats}
        seen_files = set()
        for key, data in results_all.items():
            if key[0] != model:
                continue
            for p in data.get('predictions', []):
                meta = p.get('metadata', {}) or {}
                raw_meta = meta.get('raw_meta') or {}
                fname = meta.get('path') or raw_meta.get('filename') or p.get('filename')
                if not fname or fname in seen_files:
                    continue
                seen_files.add(fname)
                sc = meta.get('scenario') or raw_meta.get('scenario')
                if sc in counts:
                    counts[sc]['total'] += 1
                    if p.get('true_label') == p.get('pred_label'):
                        counts[sc]['correct'] += 1

        for j, subcat in enumerate(all_subcats):
            cnt = counts.get(subcat, {})
            if cnt.get('total', 0) > 0:
                accuracy_matrix[i, j] = cnt['correct'] / cnt['total']
            else:
                accuracy_matrix[i, j] = np.nan

    # compute mean SNR per scenario across all_results
    scenario_snrs = {s: [] for s in all_subcats}
    seen = set()
    missing_snr = []
    for data in results_all.values():
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            raw_meta = meta.get('raw_meta') or {}
            raw_debug = meta.get('raw_debug') or {}
            fname = raw_meta.get('filename') or meta.get('path') or raw_meta.get('path') or p.get('filename')
            if not fname or fname in seen:
                continue
            seen.add(fname)
            sc = meta.get('scenario') or raw_meta.get('scenario')
            if sc not in scenario_snrs:
                continue
            if not isinstance(raw_debug, dict) or 'resulting_snr_db' not in raw_debug:
                missing_snr.append(fname)
                continue
            try:
                snr_val = float(raw_debug.get('resulting_snr_db'))
            except Exception:
                missing_snr.append(fname)
                continue
            scenario_snrs[sc].append(snr_val)

    if missing_snr:
        # do not fail the entire pipeline for missing SNR in all-splits heatmap; warn and proceed
        print(f"  ⚠ Missing resulting_snr_db for {len(missing_snr)} files; continuing without those values")

    scenario_snr_mean = {s: (np.mean(scenario_snrs[s]) if len(scenario_snrs[s])>0 else float('nan')) for s in all_subcats}

    fig, ax = plt.subplots(figsize=(max(14, len(all_subcats) * 0.8), max(8, len(models) * 0.6)))
    valid_data = accuracy_matrix[~np.isnan(accuracy_matrix)]
    vmin_dynamic = valid_data.min() if len(valid_data) > 0 else 0
    vmax_dynamic = valid_data.max() if len(valid_data) > 0 else 1

    # decide majority class per scenario across all_results
    scenario_class_counts = {s: {'class_0': 0, 'class_1': 0} for s in all_subcats}
    seen_files = set()
    for data in results_all.values():
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            raw_meta = meta.get('raw_meta') or {}
            fname = raw_meta.get('filename') or meta.get('path') or raw_meta.get('path') or p.get('filename')
            if not fname or fname in seen_files:
                continue
            seen_files.add(fname)
            sc = meta.get('scenario') or raw_meta.get('scenario')
            if sc not in scenario_class_counts:
                continue
            cls = raw_meta.get('label') if 'label' in raw_meta else meta.get('label')
            if cls is None:
                continue
            try:
                ival = int(cls)
            except Exception:
                continue
            if ival == 0:
                scenario_class_counts[sc]['class_0'] += 1
            else:
                scenario_class_counts[sc]['class_1'] += 1

    # decide ordering as before
    class0 = []
    class1 = []
    for s in all_subcats:
        c0 = scenario_class_counts[s]['class_0']
        c1 = scenario_class_counts[s]['class_1']
        if c0 > c1:
            class0.append(s)
        else:
            class1.append(s)

    class0_sorted = sorted(class0, key=lambda x: scenario_snr_mean.get(x, float('nan')))
    class1_sorted = sorted(class1, key=lambda x: scenario_snr_mean.get(x, float('nan')))
    ordered_all_subcats = class0_sorted + class1_sorted

    def _snr_label(s):
        v = scenario_snr_mean.get(s, float('nan'))
        if v != v:
            snr_part = 'n/a'
        else:
            snr_part = f"{v:.1f} dB"
        return f"{s}\n{snr_part}"

    xtick_labels = [_snr_label(s) for s in ordered_all_subcats]
    orig_all_subcats = list(all_subcats)
    col_idx = [orig_all_subcats.index(s) for s in ordered_all_subcats]
    accuracy_matrix_reordered = accuracy_matrix[:, col_idx]

    sns.heatmap(accuracy_matrix_reordered, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
               xticklabels=xtick_labels, yticklabels=models, cbar_kws={'label': 'Accuracy'},
               vmin=vmin_dynamic, vmax=vmax_dynamic, linewidths=0.5, linecolor='gray')
    ax.set_title('Accuracy Heatmap by Subcategory (All Splits)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Subcategory', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_path = output_dir / 'best_subcategory_heatmap_all_splits.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

def load_results(folder):
    """
    Charge les résultats de performance depuis le dossier spécifié.
    """
    perf_dir = Path(folder)
    if not perf_dir.exists():
        print(f"[ERROR] Performance directory not found: {perf_dir}")
        return {}

    results = {}
    for json_path in sorted(perf_dir.glob("*_test.json")):
        try:
            with open(json_path, 'r') as f:
                full_data = json.load(f)

            # Minimal validation
            for tk in ('metadata', 'global_metrics', 'class_metrics', 'predictions'):
                if tk not in full_data:
                    print(f"  ✗ {json_path.name} missing key: {tk}")
                    raise RuntimeError(f"Invalid performance file: {json_path.name}")

            meta = full_data.get('metadata', {})
            model = meta.get('model')
            split = meta.get('split')
            if model is None or split is None:
                print(f"  ✗ {json_path.name} missing metadata.model or metadata.split")
                continue

            file_threshold = meta.get('threshold')
            if file_threshold is None:
                print(f"  ✗ {json_path.name} missing metadata.threshold — produced JSONs must include this field")
                raise RuntimeError(f"Missing metadata.threshold in {json_path.name}")

            key_tuple = (model, split, float(file_threshold))
            results[key_tuple] = full_data
            print(f"  ✓ Loaded: {model} | {split} | t={float(file_threshold):.3f}")

        except Exception as e:
            print(f"  ✗ Error loading {json_path.name}: {e}")

    if not results:
        print("[WARN] No valid performance JSONs loaded from PERFORMANCE_DIR")
    return results


def load_all_results(folder):
    """
    Load all performance JSONs (all splits) from `folder` and return a dict keyed by (model, split, threshold).
    """
    perf_dir = Path(folder)
    if not perf_dir.exists():
        print(f"[ERROR] Performance directory not found: {perf_dir}")
        return {}

    results = {}
    for json_path in sorted(perf_dir.glob("*.json")):
        try:
            with open(json_path, 'r') as f:
                full_data = json.load(f)

            # Minimal validation
            for tk in ('metadata', 'global_metrics', 'class_metrics', 'predictions'):
                if tk not in full_data:
                    print(f"  ✗ {json_path.name} missing key: {tk}")
                    raise RuntimeError(f"Invalid performance file: {json_path.name}")

            meta = full_data.get('metadata', {})
            model = meta.get('model')
            split = meta.get('split')
            if model is None or split is None:
                print(f"  ✗ {json_path.name} missing metadata.model or metadata.split")
                continue

            file_threshold = meta.get('threshold')
            if file_threshold is None:
                print(f"  ✗ {json_path.name} missing metadata.threshold — produced JSONs must include this field")
                raise RuntimeError(f"Missing metadata.threshold in {json_path.name}")

            key_tuple = (model, split, float(file_threshold))
            results[key_tuple] = full_data
            print(f"  ✓ Loaded: {model} | {split} | t={float(file_threshold):.3f}")

        except Exception as e:
            print(f"  ✗ Error loading {json_path.name}: {e}")

    if not results:
        print("[WARN] No valid performance JSONs loaded from PERFORMANCE_DIR")
    return results
    

def main():

    print("\n" + "=" * 80)
    print("\n[1/5]Loading results...")
    
    print(f"{'Loading performance results from:':<40} {config.PERFORMANCE_DIR}")
    results = load_results(config.PERFORMANCE_DIR)
    
    print("\n[2/5] Plotting global metrics comparison...")
    plot_global_metrics_comparison(results, OUTPUT_DIR)
    
    print("\n[3/5] Plotting confusion matrices (test set)...")
    plot_confusion_matrices(results, OUTPUT_DIR)
    
    print("\n[4/5] Plotting class performance...")
    plot_class_performance(results, OUTPUT_DIR)
    
    print("\n[5/5] Plotting subcategory heatmap...")
    plot_subcategory_heatmap(results, OUTPUT_DIR)
    # Also generate heatmap aggregated across all splits
    try:
        all_results = load_all_results(config.PERFORMANCE_DIR)
        if all_results:
            print("\n[6/5] Plotting subcategory heatmap (all splits)...")
            plot_subcategory_heatmap_all(all_results, OUTPUT_DIR)
    except Exception:
        pass
    
    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
