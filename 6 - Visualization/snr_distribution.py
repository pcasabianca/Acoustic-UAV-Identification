#!/usr/bin/env python3
"""
SNR Distribution Visualization
Analyse la distribution du SNR dans les données augmentées.
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_augmentation_metadata():
    """
    Charge les métadonnées d'augmentation pour extraire les valeurs de SNR.
    """
    # Récupère le SNR à partir des fichiers de performance générés (predictions[].metadata.raw_debug.resulting_snr_db)
    perf_dir = Path(config.PERFORMANCE_DIR)
    if not perf_dir.exists():
        print(f"[ERROR] Performance directory not found: {perf_dir}")
        return None

    snr_entries = {}
    # Parcourir tous les JSON de performance
    for j in perf_dir.glob('*.json'):
        try:
            with open(j, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        preds = data.get('predictions', [])
        for p in preds:
            meta = p.get('metadata', {})
            raw_debug = meta.get('raw_debug', {})
            snr = raw_debug.get('resulting_snr_db')
            # Category by subcategory if present, else class_name
            cat = p.get('subcategory') or meta.get('class_name') or meta.get('raw_meta', {}).get('class_name', 'unknown')
            if snr is None:
                continue
            snr_entries.setdefault(cat, []).append(float(snr))

    return snr_entries


def extract_snr_distribution(metadata):
    """
    Extrait les valeurs de SNR par catégorie d'augmentation.
    Retourne: dict[category] = list[snr_values]
    """
    snr_by_category = defaultdict(list)
    
    for filename, aug_info in metadata.items():
        if not isinstance(aug_info, dict):
            continue
        
        # Extraire le type d'augmentation et SNR
        aug_type = aug_info.get('augmentation_type', 'unknown')
        
        # Rechercher SNR dans les paramètres
        params = aug_info.get('parameters', {})
        
        if 'snr_db' in params:
            snr = params['snr_db']
            snr_by_category[aug_type].append(snr)
        elif 'noise_level' in params:
            # Convertir noise_level en SNR approximatif
            noise_level = params['noise_level']
            snr = -20 * np.log10(noise_level) if noise_level > 0 else 40
            snr_by_category[aug_type].append(snr)
    
    return snr_by_category


def plot_snr_distribution(snr_by_category, output_file):
    """
    Crée un graphique montrant la distribution du SNR.
    """
    if not snr_by_category:
        print("[ERROR] No SNR data available!")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('SNR Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Préparer les données
    categories = sorted(snr_by_category.keys())
    data_for_box = [snr_by_category[cat] for cat in categories]
    
    # Calculer les statistiques globales
    all_snr = []
    for snr_list in snr_by_category.values():
        all_snr.extend(snr_list)
    
    global_mean = np.mean(all_snr)
    global_median = np.median(all_snr)
    global_std = np.std(all_snr)
    
    # Subplot 1: Box plots par catégorie
    bp = axes[0].boxplot(data_for_box, labels=categories, patch_artist=True, 
                         notch=True, showmeans=True)
    
    # Colorier les boîtes
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].set_ylabel('SNR (dB)', fontsize=12)
    axes[0].set_title('SNR Distribution by Augmentation Category', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Ligne de moyenne globale
    axes[0].axhline(global_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Global Mean: {global_mean:.1f} dB')
    axes[0].legend(loc='best')
    
    # Subplot 2: Histogramme global
    axes[1].hist(all_snr, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(global_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {global_mean:.1f} dB')
    axes[1].axvline(global_median, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {global_median:.1f} dB')
    
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Overall SNR Distribution', fontsize=13, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Ajouter les statistiques en texte
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {global_mean:.2f} dB\n'
    stats_text += f'Median: {global_median:.2f} dB\n'
    stats_text += f'Std Dev: {global_std:.2f} dB\n'
    stats_text += f'Min: {min(all_snr):.2f} dB\n'
    stats_text += f'Max: {max(all_snr):.2f} dB\n'
    stats_text += f'Total Samples: {len(all_snr)}'
    
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def print_snr_statistics(snr_by_category):
    """
    Affiche les statistiques de SNR par catégorie.
    """
    print("\n" + "=" * 80)
    print("  SNR STATISTICS BY CATEGORY")
    print("=" * 80)
    print(f"{'Category':<20} {'Count':<8} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    for category in sorted(snr_by_category.keys()):
        snr_values = snr_by_category[category]
        
        count = len(snr_values)
        mean = np.mean(snr_values)
        median = np.median(snr_values)
        std = np.std(snr_values)
        min_val = min(snr_values)
        max_val = max(snr_values)
        
        print(f"{category:<20} {count:<8} {mean:<10.2f} {median:<10.2f} {std:<10.2f} {min_val:<8.2f} {max_val:<8.2f}")
    
    # Statistiques globales
    all_snr = []
    for snr_list in snr_by_category.values():
        all_snr.extend(snr_list)
    
    print("-" * 80)
    print(f"{'GLOBAL':<20} {len(all_snr):<8} {np.mean(all_snr):<10.2f} {np.median(all_snr):<10.2f} "
          f"{np.std(all_snr):<10.2f} {min(all_snr):<8.2f} {max(all_snr):<8.2f}")
    print("=" * 80)


def main():
    print("=" * 80)
    print("  SNR DISTRIBUTION VISUALIZATION")
    print("=" * 80)
    print()
    
    # Charger les SNR depuis les fichiers de performance
    print("[INFO] Loading SNR values from performance JSONs...")
    snr_by_category = load_augmentation_metadata()

    if not snr_by_category:
        print("[ERROR] No SNR data found in performance JSONs!")
        print("[INFO] Ensure performance JSONs include predictions[].metadata.raw_debug.resulting_snr_db")
        return
    
    print(f"[INFO] Found SNR data for {len(snr_by_category)} categories")
    
    # Afficher les statistiques
    print_snr_statistics(snr_by_category)
    
    # Créer la visualisation
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "snr_distribution.png"
    
    print("\n[INFO] Creating SNR distribution plot...")
    plot_snr_distribution(snr_by_category, output_file)
    
    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
