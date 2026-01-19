#!/usr/bin/env python3
"""
Model Performance Comparison (Style de l'ancien plot)
Crée des visualisations synthétiques comparant les performances des modèles.
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_best_results():
    """
    Charge le résumé des meilleurs résultats (créé par select_best_results.py).
    """
    summary_file = config.PERFORMANCE_DIR / "best_results_summary.json"
    
    if not summary_file.exists():
        print(f"[WARNING] Best results summary not found: {summary_file}")
        print("[INFO] Ensure canonical performance JSONs exist in config.PERFORMANCE_DIR or run the visualizer pipeline.")
        return None
    
    with open(summary_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', {})
    # Validation stricte: chaque entrée doit contenir les métriques attendues
    required_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    missing = []
    for key, entry in results.items():
        metrics = entry.get('metrics', {})
        for m in required_metrics:
            if m not in metrics:
                missing.append(f"{key}: missing metrics.{m}")

    if missing:
        print("[ERROR] best_results_summary.json is missing required metric keys:")
        for m in missing:
            print(f" - {m}")
        raise RuntimeError('best_results_summary.json missing required metrics')

    return results


def load_distance_performance():
    """
    Charge les performances par distance (si disponible).
    Retourne: dict[model] = dict[distance] = metrics
    """
    perf_by_distance = defaultdict(lambda: defaultdict(dict))
    
    perf_dir = config.PERFORMANCE_DIR
    distance_files = list(perf_dir.glob("*_distance_*.json"))
    
    for dist_file in distance_files:
        try:
            with open(dist_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            model = metadata.get('model', 'UNKNOWN')
            distance = metadata.get('distance', 'UNKNOWN')
            
            perf_by_distance[model][distance] = data.get('metrics', {})
            
        except Exception as e:
            pass  # Ignorer les erreurs
    
    return perf_by_distance


def plot_model_comparison(best_results, output_file):
    """
    Crée un graphique à 4 subplots comparant les modèles.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Extraire les modèles uniques et leurs métriques
    models_data = defaultdict(lambda: {'accuracy': [], 'f1': [], 'precision': [], 'recall': []})
    
    for key, result in best_results.items():
        model = result['model']
        metrics = result['metrics']
        
        models_data[model]['accuracy'].append(metrics.get('accuracy', 0))
        models_data[model]['f1'].append(metrics.get('f1_score', 0))
        models_data[model]['precision'].append(metrics.get('precision', 0))
        models_data[model]['recall'].append(metrics.get('recall', 0))
    
    # Calculer les moyennes
    model_names = sorted(models_data.keys())
    avg_accuracy = [np.mean(models_data[m]['accuracy']) for m in model_names]
    avg_f1 = [np.mean(models_data[m]['f1']) for m in model_names]
    avg_precision = [np.mean(models_data[m]['precision']) for m in model_names]
    avg_recall = [np.mean(models_data[m]['recall']) for m in model_names]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Subplot 1: Accuracy Comparison
    x_pos = np.arange(len(model_names))
    axes[0, 0].bar(x_pos, avg_accuracy, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(avg_accuracy):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Subplot 2: F1-Score Comparison
    axes[0, 1].bar(x_pos, avg_f1, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('F1-Score', fontsize=12)
    axes[0, 1].set_title('Model F1-Score Comparison', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(avg_f1):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Subplot 3: Precision vs Recall
    axes[1, 0].bar(x_pos - 0.2, avg_precision, width=0.4, label='Precision', 
                   color='#2ca02c', alpha=0.7, edgecolor='black')
    axes[1, 0].bar(x_pos + 0.2, avg_recall, width=0.4, label='Recall', 
                   color='#d62728', alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Precision vs Recall', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Overall Performance Radar (simplifié en barres groupées)
    metrics_names = ['Accuracy', 'F1', 'Precision', 'Recall']

    # Create a readable grouped bar chart for overall metrics.
    # Compute a dynamic width so all models fit without overlap.
    n_models = len(model_names)
    group_width = 0.8
    if n_models > 0:
        width = group_width / max(n_models, 1)
    else:
        width = 0.2

    all_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Center groups on each metric tick
    x_base = np.arange(len(metrics_names))
    for i, model in enumerate(model_names):
        values = [
            np.mean(models_data[model]['accuracy']),
            np.mean(models_data[model]['f1']),
            np.mean(models_data[model]['precision']),
            np.mean(models_data[model]['recall'])
        ]
        # center offset
        offset = (i - (n_models - 1) / 2) * width
        x_positions = x_base + offset
        color = all_colors[i % len(all_colors)]
        axes[1, 1].bar(x_positions, values, width=width * 0.95, label=model,
                      color=color, alpha=0.9, edgecolor='black')

    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Overall Performance Metrics', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x_base)
    axes[1, 1].set_xticklabels(metrics_names, rotation=0, ha='center')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Place legend outside the plot to avoid occlusion
    axes[1, 1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_performance_by_distance(perf_by_distance, output_file):
    """
    Crée un graphique montrant les performances en fonction de la distance.
    """
    if not perf_by_distance:
        print("[INFO] No distance-specific performance data available")
        return
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance by Distance', fontsize=16, fontweight='bold')
    
    colors = {
        'CNN': '#1f77b4',
        'RNN': '#ff7f0e',
        'CRNN': '#2ca02c',
        'ATTENTION_CRNN': '#d62728'
    }
    
    for model, distances in perf_by_distance.items():
        # Trier par distance
        sorted_distances = sorted(distances.items(), key=lambda x: float(x[0].replace('m', '')))
        
        dist_labels = [d[0] for d in sorted_distances]
        accuracies = [d[1].get('accuracy', 0) for d in sorted_distances]
        f1_scores = [d[1].get('f1_score', 0) for d in sorted_distances]
        
        color = colors.get(model, '#333333')
        
        # Subplot 1: Accuracy by Distance
        axes[0].plot(dist_labels, accuracies, marker='o', label=model, 
                    color=color, linewidth=2, markersize=6)
        
        # Subplot 2: F1-Score by Distance
        axes[1].plot(dist_labels, f1_scores, marker='s', label=model, 
                    color=color, linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Distance', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy by Distance', fontsize=13, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    axes[1].set_xlabel('Distance', fontsize=12)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('F1-Score by Distance', fontsize=13, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    print("=" * 80)
    print("  MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    
    # Charger les meilleurs résultats
    print("[INFO] Loading best results...")
    best_results = load_best_results()
    
    if not best_results:
        print("[ERROR] No best results found! Run select_best_results.py first.")
        return
    
    print(f"[INFO] Found {len(best_results)} best configurations")
    
    # Créer le répertoire de sortie
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Graphique de comparaison générale
    print("[INFO] Creating model comparison plot...")
    output_file1 = output_dir / "model_performance_comparison.png"
    plot_model_comparison(best_results, output_file1)
    
    # Graphique par distance (si disponible)
    print("[INFO] Loading distance-specific performance...")
    perf_by_distance = load_distance_performance()
    
    if perf_by_distance:
        print("[INFO] Creating performance by distance plot...")
        output_file2 = output_dir / "performance_by_distance.png"
        plot_performance_by_distance(perf_by_distance, output_file2)
    
    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
