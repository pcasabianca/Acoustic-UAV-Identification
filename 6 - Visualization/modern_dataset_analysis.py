"""
Modern Dataset Analysis (strict)
Analyse la composition et les statistiques du dataset à partir de config.py.

Features:
- Utilise config.DATASET_TEST_DIR, DATASET_TRAIN_DIR, DATASET_VAL_DIR
- Statistiques par split (train/val/test)
- Distribution par catégorie (distances, ambient types, originals)
- Visualisations de la composition du dataset
- Pas de fallback, configuration centralisée
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
import re

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from tools import plot_utils

# Apply plotting style
plot_utils.set_style()

# Output directory
OUTPUT_DIR = plot_utils.get_output_dir(__file__)


def analyze_split(split_dir, split_name):
    """
    Analyse un split (train/val/test) et retourne les statistiques.
    """
    stats = {
        'split': split_name,
        'total': 0,
        'class_0': 0,
        'class_1': 0,
        'subcategories': defaultdict(int),
        'files_by_class': {'0': [], '1': []}
    }
    
    # For strict analysis we read canonical performance JSONs and aggregate from their
    # per-file metadata. No filename heuristics or directory fallbacks are used.
    perf_dir = Path(config.PERFORMANCE_DIR)
    if not perf_dir.exists():
        raise RuntimeError(f"Performance directory not found: {perf_dir}")

    # Aggregate counts per split using the raw_meta.split value (must be present).
    # Use deduplication by filename so that multiple performance JSONs (one per model)
    # do not multiply counts. We count each source file only once per split.
    counts_by_split = defaultdict(lambda: {'total': 0, 'class_0': 0, 'class_1': 0, 'subcategories': defaultdict(int)})
    seen_files = defaultdict(set)

    missing_meta_examples = []

    # Build canonical dataset index (filename -> canonical metadata) from dataset metadata.jsonl files
    dataset_index = {}
    for sd, name in [(config.DATASET_TRAIN_DIR, 'train'), (config.DATASET_VAL_DIR, 'val'), (config.DATASET_TEST_DIR, 'test')]:
        md_path = Path(sd) / 'metadata.jsonl'
        if not md_path.exists():
            continue
        try:
            with open(md_path, 'r') as mf:
                for line in mf:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    entry_path = entry.get('path') or entry.get('filename')
                    if not entry_path:
                        continue
                    b = Path(entry_path).name
                    # store canonical metadata and the split it belongs to
                    dataset_index[b] = {'entry': entry, 'split': name}
        except Exception:
            continue

    for j in perf_dir.glob('*.json'):
        try:
            with open(j, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        for p in data.get('predictions', []):
            # Prediction-level metadata may be incomplete; use canonical dataset metadata.jsonl as source-of-truth
            meta = p.get('metadata') or {}
            if not isinstance(meta, dict):
                missing_meta_examples.append({'file': j.name, 'prediction': p.get('filename', '<no-fname>')})
                continue

            fname = meta.get('path') or p.get('filename')
            if not fname:
                missing_meta_examples.append({'file': j.name, 'prediction': p.get('filename', '<no-fname>')})
                continue
            bname = Path(fname).name

            # Look up canonical entry from dataset metadata
            canonical = dataset_index.get(bname)
            if not canonical:
                missing_meta_examples.append({'file': j.name, 'prediction': bname})
                continue
            canonical_entry = canonical.get('entry', {})
            canonical_split = canonical.get('split')

            # Only count if this file belongs to the split being analyzed
            if canonical_split != split_name:
                continue

            # Deduplicate by filename within this split
            if bname in seen_files[split_name]:
                continue
            seen_files[split_name].add(bname)

            counts_by_split[split_name]['total'] += 1

            # Determine label and scenario from canonical entry (preferred) or prediction metadata
            label_val = meta.get('label') if 'label' in meta else canonical_entry.get('label')
            scenario = meta.get('scenario') if 'scenario' in meta else canonical_entry.get('scenario')

            # Validate required fields
            if label_val is None or not scenario:
                missing_meta_examples.append({'file': j.name, 'prediction': bname})
                continue

            try:
                ival = int(label_val)
            except Exception:
                missing_meta_examples.append({'file': j.name, 'prediction': bname})
                continue
            if ival == 0:
                counts_by_split[split_name]['class_0'] += 1
            else:
                counts_by_split[split_name]['class_1'] += 1

            counts_by_split[split_name]['subcategories'][scenario] += 1

    if missing_meta_examples:
        msg = 'Missing or incomplete per-file metadata in performance JSONs.\nShow examples: '
        msg += ', '.join([f"{e.get('file')}@{e.get('prediction')}" for e in missing_meta_examples[:10]])
        raise RuntimeError(msg)

    # Convert counts_by_split into the stats structure expected by plotting code
    # Use canonical split names from keys (no fallback mapping)
    stats_list = []
    for split_key, v in counts_by_split.items():
        s = {
            'split': split_key,
            'total': v['total'],
            'class_0': v['class_0'],
            'class_1': v['class_1'],
            'subcategories': v['subcategories'],
            'files_by_class': {'0': [], '1': []}
        }
        stats_list.append(s)

    # Return aggregated stats for the requested split_name if present
    for s in stats_list:
        if s['split'] == split_name or s['split'].endswith(split_name):
            return s
    # If no exact match, return an empty stats (caller will handle)
    return {'split': split_name, 'total': 0, 'class_0': 0, 'class_1': 0, 'subcategories': defaultdict(int), 'files_by_class': {'0': [], '1': []}}
    
    return stats


def plot_split_distribution(all_stats, output_dir):
    """
    Plot la distribution des classes par split.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    splits = [s['split'] for s in all_stats]
    class_0_counts = [s['class_0'] for s in all_stats]
    class_1_counts = [s['class_1'] for s in all_stats]
    
    x = np.arange(len(splits))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, class_0_counts, width, label='Class 0 (Non-Drone)', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, class_1_counts, width, label='Class 1 (Drone)',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Split', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    ax.set_title('Dataset Distribution by Split', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'dataset_split_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_subcategory_distribution(all_stats, output_dir):
    """
    Plot la distribution des sous-catégories pour le split de test.
    """
    # Rebuild subcategory counts directly from performance JSONs for the TEST split
    # to ensure we use the exact same `scenario` extraction and deduplication
    # logic as the heatmap.
    # To exactly match the heatmap ordering and scenario discovery we load the
    # same best-results set used by the heatmap and derive scenarios from it.
    try:
        from . import performance_comparison_best as pcb
    except Exception:
        try:
            import performance_comparison_best as pcb
        except Exception:
            print("  ⚠ Could not import performance_comparison_best; skipping subcategory plots")
            return

    # Use load_results (no "best" selection) to obtain all performance JSONs
    results = pcb.load_results(config.PERFORMANCE_DIR)
    if not results:
        print("  ⚠ No best results available; skipping subcategory plots")
        return

    # Collect canonical scenario set from ALL splits (train/val/test), using the
    # same strict rules as the heatmap: fail-fast on any missing `scenario`.
    scenario_set = set()
    missing_scenarios = []
    for data in results.values():
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            if not isinstance(meta, dict):
                missing_scenarios.append(p.get('filename', '<no-fname>'))
                continue
            sc = meta.get('scenario')
            if sc:
                scenario_set.add(sc)
            else:
                missing_scenarios.append(p.get('filename', '<no-fname>'))

    if missing_scenarios:
        examples = list(dict.fromkeys(missing_scenarios))[:12]
        raise RuntimeError(
            "Missing 'scenario' in prediction metadata.raw_meta for some predictions. "
            "Ensure augmentation JSONL and performance JSONs include raw_meta.scenario. "
            "Examples: " + ", ".join(examples)
        )

    # Create ordered subcategories exactly like the heatmap
    all_subcats = sorted(list(scenario_set))
    def extract_distance(cat_name):
        match = re.search(r'(\d+)m', cat_name)
        return int(match.group(1)) if match else None
    distances = [s for s in all_subcats if extract_distance(s) is not None]
    others = [s for s in all_subcats if extract_distance(s) is None]
    distances_sorted = sorted(distances, key=lambda x: extract_distance(x), reverse=True)
    ordered_subcats = distances_sorted + sorted(others)

    # Rebuild counts by iterating all predictions (all splits) and deduplicating filenames globally
    seen = set()
    subcat_counts = defaultdict(int)
    for data in results.values():
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            if not isinstance(meta, dict):
                continue
            fname = meta.get('path') or p.get('filename')
            if not fname or fname in seen:
                continue
            seen.add(fname)
            sc = meta.get('scenario')
            if not sc:
                # Should not happen due to earlier validation
                raise RuntimeError(f"Missing scenario for file {fname} in best_results")
            subcat_counts[sc] += 1

    # Build ordered list of scenarios using the exact same sorting as the heatmap
    all_subcats = sorted(list(subcat_counts.keys()))

    def extract_distance(cat_name):
        match = re.search(r'(\d+)m', cat_name)
        return int(match.group(1)) if match else None

    distances = [s for s in all_subcats if extract_distance(s) is not None]
    others = [s for s in all_subcats if extract_distance(s) is None]
    distances_sorted = sorted(distances, key=lambda x: extract_distance(x), reverse=True)
    ordered_subcats = distances_sorted + sorted(others)

    # Determine whether each scenario is 'drone' or 'ambient' strictly from
    # the per-file class labels (no hardcoded name checks). We aggregate class
    # labels across all predictions (deduplicated by filename) and decide per
    # scenario by majority vote. Fail-fast on ties or missing labels.
    scenario_class_counts = {s: {'class_0': 0, 'class_1': 0} for s in ordered_subcats}
    seen_files = set()

    for data in results.values():
        for p in data.get('predictions', []):
            meta = p.get('metadata', {}) or {}
            if not isinstance(meta, dict):
                continue
            fname = meta.get('path') or p.get('filename')
            if not fname or fname in seen_files:
                continue
            seen_files.add(fname)
            sc = meta.get('scenario')
            if sc not in scenario_class_counts:
                # scenario outside the canonical set (shouldn't happen)
                continue
            # Determine class from metadata (canonical: numeric label at top-level)
            cls = meta.get('label') if 'label' in meta else meta.get('class_name')
            if cls is None:
                raise RuntimeError(f"Missing class label for file {fname}")
            try:
                if int(cls) == 0:
                    scenario_class_counts[sc]['class_0'] += 1
                else:
                    scenario_class_counts[sc]['class_1'] += 1
            except Exception:
                raise RuntimeError(f"Unrecognized class label '{cls}' for file {fname}")

    # Now split scenarios by majority class (strict)
    drone_cats = {}
    ambient_cats = {}
    ties = []
    for s in ordered_subcats:
        c0 = scenario_class_counts[s]['class_0']
        c1 = scenario_class_counts[s]['class_1']
        if c1 > c0:
            drone_cats[s] = subcat_counts[s]
        elif c0 > c1:
            ambient_cats[s] = subcat_counts[s]
        else:
            ties.append((s, c0, c1))

    if ties:
        examples = ', '.join([f"{t[0]}({t[1]} vs {t[2]})" for t in ties[:10]])
        raise RuntimeError(f"Tie or no class majority for scenarios: {examples}. Fix labels before plotting.")
    
    # Plot drones
    if drone_cats:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Trier par distance using the same pattern as the heatmap: look for '<number>m'
        def extract_dist(cat_name):
            if 'orig' in cat_name:
                return 0
            match = re.search(r'(\d+)m', cat_name)
            return int(match.group(1)) if match else None

        # Sort distances descending, others after
        distances = [s for s in drone_cats.keys() if extract_dist(s) is not None]
        others = [s for s in drone_cats.keys() if extract_dist(s) is None]
        distances_sorted = sorted(distances, key=lambda x: extract_dist(x), reverse=True)
        ordered = distances_sorted + sorted(others)
        sorted_cats = [(k, drone_cats[k]) for k in ordered]
        labels = [cat.replace('_', ' ').title() for cat, _ in sorted_cats]
        values = [count for _, count in sorted_cats]
        
        bars = ax.bar(range(len(labels)), values, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Drone Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
        ax.set_title('Drone Distribution by Distance (All Splits)', fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = output_dir / 'dataset_drone_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    # Plot ambients
    if ambient_cats:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Keep alphabetical order for ambients but normalize label formatting
        sorted_cats = sorted(ambient_cats.items())
        labels = [cat.replace('_', ' ').title() for cat, _ in sorted_cats]
        values = [count for _, count in sorted_cats]
        
        bars = ax.bar(range(len(labels)), values, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Ambient Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
        ax.set_title('Ambient Distribution by Type (All Splits)', fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = output_dir / 'dataset_ambient_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def generate_summary_report(all_stats, output_dir):
    """
    Génère un rapport texte avec toutes les statistiques.
    """
    report_path = output_dir / 'dataset_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATASET ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Stats globales
        total_samples = sum(s['total'] for s in all_stats)
        total_class_0 = sum(s['class_0'] for s in all_stats)
        total_class_1 = sum(s['class_1'] for s in all_stats)
        
        f.write(f"GLOBAL STATISTICS:\n")
        f.write(f"  Total Samples: {total_samples}\n")
        f.write(f"  Class 0 (Non-Drone): {total_class_0} ({total_class_0/total_samples*100:.1f}%)\n")
        f.write(f"  Class 1 (Drone): {total_class_1} ({total_class_1/total_samples*100:.1f}%)\n")
        f.write(f"  Class Balance: {min(total_class_0, total_class_1) / max(total_class_0, total_class_1):.2f}\n\n")
        
        # Stats par split
        for stats in all_stats:
            f.write(f"\n{'='*80}\n")
            f.write(f"{stats['split'].upper()} SPLIT\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"  Total: {stats['total']} samples\n")
            f.write(f"  Class 0: {stats['class_0']} ({stats['class_0']/stats['total']*100:.1f}%)\n")
            f.write(f"  Class 1: {stats['class_1']} ({stats['class_1']/stats['total']*100:.1f}%)\n\n")
            
            if stats['subcategories']:
                f.write(f"  Subcategories:\n")
                for cat, count in sorted(stats['subcategories'].items(), key=lambda x: -x[1]):
                    f.write(f"    {cat:25s}: {count:4d} samples\n")
    
    print(f"  ✓ Saved: {report_path}")


def main():
    """Point d'entrée principal."""
    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80 + "\n")
    
    # Analyser tous les splits
    print("[1/4] Analyzing splits...")
    all_stats = []
    
    splits = [
        ('train', config.DATASET_TRAIN_DIR),
        ('val', config.DATASET_VAL_DIR),
        ('test', config.DATASET_TEST_DIR)
    ]
    
    for split_name, split_dir in splits:
        print(f"  Analyzing {split_name}...")
        stats = analyze_split(split_dir, split_name)
        all_stats.append(stats)
        print(f"    Total: {stats['total']}, Class 0: {stats['class_0']}, Class 1: {stats['class_1']}")
    
    # Générer les visualisations
    print("\n[2/4] Generating split distribution plot...")
    plot_split_distribution(all_stats, OUTPUT_DIR)
    
    print("\n[3/4] Generating subcategory distribution plots...")
    plot_subcategory_distribution(all_stats, OUTPUT_DIR)
    
    print("\n[4/4] Generating summary report...")
    generate_summary_report(all_stats, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print(f"✓ Dataset analysis complete! Results in: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
