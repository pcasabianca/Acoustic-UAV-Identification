#!/usr/bin/env python3
"""
Best Results Selector
Analyse tous les fichiers JSON de performance et sélectionne le meilleur threshold
pour chaque modèle/split en fonction du F1-Score.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_and_select_best_results():
    """
    Charge tous les résultats de performance et retourne les meilleurs.
    
    Returns:
        dict: {(model, split): data_dict} - Un seul résultat par (modèle, split)
    """
    perf_dir = Path(config.PERFORMANCE_DIR)
    
    if not perf_dir.exists():
        print(f"[ERROR] Performance directory not found: {perf_dir}")
        return {}
    
    # Structure: {(model, split): {threshold: data}}
    all_results = defaultdict(dict)
    
    # Charger tous les JSONs
    for json_file in sorted(perf_dir.glob("*.json")):
        # Skip summary files
        if 'summary' in json_file.name or 'calibration' in json_file.name or 'optimization' in json_file.name:
            continue
        
        # Parse filename: {model}_{split}.json (canonical format)
        name_parts = json_file.stem.split('_')
        if len(name_parts) < 2:
            continue
        
        # Extract model and split
        # Support multi-word models: attention_crnn_test.json → model=ATTENTION_CRNN, split=test
        if name_parts[-1] in ['train', 'val', 'test']:
            split = name_parts[-1]
            model_token = '_'.join(name_parts[:-1])
            model = model_token.upper()
        else:
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            threshold = metadata.get('threshold', 0.5)
            
            # Store this result
            all_results[(model, split)][threshold] = {
                'file': json_file,
                'data': data,
                'threshold': threshold
            }
            
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
    
    # Sélectionner le meilleur threshold pour chaque (model, split) basé sur F1
    best_results = {}
    
    for (model, split), threshold_results in all_results.items():
        if not threshold_results:
            continue
        
        # Trouver le threshold avec le meilleur F1
        best_entry = max(
            threshold_results.items(),
            key=lambda x: x[1]['data'].get('global_metrics', {}).get('f1_score', 0)
        )
        
        best_threshold, best_data = best_entry
        best_results[(model, split)] = best_data['data']
        
        f1 = best_data['data'].get('global_metrics', {}).get('f1_score', 0)
        print(f"  ✓ {model:<15} {split:<8} threshold={best_threshold:.3f} F1={f1:.4f}")
    
    return best_results


def save_best_results_summary(best_results, output_file):
    """
    Sauvegarde un résumé des meilleurs résultats.
    """
    summary = {
        'selection_criteria': 'Best F1-Score per (model, split)',
        'total_configurations': len(best_results),
        'results': {}
    }
    
    for (model, split), data in best_results.items():
        key = f"{model}_{split}"
        
        metrics = data.get('global_metrics', {})
        metadata = data.get('metadata', {})
        
        summary['results'][key] = {
            'model': model,
            'split': split,
            'threshold': metadata.get('threshold', 0.5),
            'metrics': {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'specificity': metrics.get('specificity', 0)
            },
            'confusion_matrix': {
                'tp': metrics.get('tp', 0),
                'tn': metrics.get('tn', 0),
                'fp': metrics.get('fp', 0),
                'fn': metrics.get('fn', 0)
            }
        }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved: {output_file}")


def main():
    print("=" * 80)
    print("  BEST RESULTS SELECTOR")
    print("=" * 80)
    """Deprecated select_best_results stub.

    This module used to select the best threshold per model/split, producing
    `best_results_summary.json`. That behavior is deprecated because it re-derived
    thresholds and caused inconsistencies. The visualizer now trusts each
    performance JSON's `metadata.threshold` (fallback 0.5).

    This file is intentionally a no-op to avoid accidental execution.
    """

    from pathlib import Path

    def main():
        print("[SKIP] select_best_results is deprecated and intentionally disabled.")
        return 0

    if __name__ == '__main__':
        main()
    print("-" * 80)
    
    for (model, split), data in sorted(best_results.items()):
        metrics = data.get('global_metrics', {})
        metadata = data.get('metadata', {})
        
        threshold = metadata.get('threshold', 0.5)
        f1 = metrics.get('f1_score', 0)
        acc = metrics.get('accuracy', 0)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        
        print(f"{model:<15} {split:<8} {threshold:<10.3f} {f1:<8.4f} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f}")
    
    print("=" * 80)
    
    return best_results


if __name__ == '__main__':
    main()
