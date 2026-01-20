#!/usr/bin/env python3
"""
HTML Report Generator
Génère un rapport HTML interactif avec tous les résultats et visualisations.
Utilise des chemins relatifs pour les images (plus léger et plus rapide que base64).
"""
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def copy_image_to_output(image_path, images_dir, output_name=None):
    """
    Copie une image vers le dossier images/ du rapport.
    Retourne le chemin relatif pour le HTML.
    """
    try:
        if not image_path.exists():
            return None
        
        dest_name = output_name or image_path.name
        dest_path = images_dir / dest_name
        shutil.copy2(image_path, dest_path)
        
        return f"./images/{dest_name}"
    except Exception as e:
        print(f"  ✗ Error copying image {image_path}: {e}")
        return None


def load_best_results_summary():
    """
    Charge le résumé des meilleurs résultats.
    """
    summary_file = config.PERFORMANCE_DIR / "best_results_summary.json"
    
    if not summary_file.exists():
        return None

    with open(summary_file, 'r') as f:
        return json.load(f)


def generate_html_report(output_file):
    """
    Génère le rapport HTML complet.
    """
    # Charger les données
    best_results = load_best_results_summary()
    
    if not best_results:
        print("[ERROR] No best results found! Ensure canonical performance JSONs exist in config.PERFORMANCE_DIR and rerun visualizations.")
        return

    # Validation stricte: chaque entrée du résumé doit correspondre à un fichier de performance contenant les clés minimales
    perf_dir = config.PERFORMANCE_DIR
    missing = []
    for key, entry in best_results.get('results', {}).items():
        model = entry.get('model')
        split = entry.get('split')
        if not model or not split:
            missing.append(f"summary entry {key} missing model/split")
            continue
        pf = perf_dir / f"{model.lower()}_{split}.json"
        if not pf.exists():
            missing.append(f"missing perf file: {pf}")
            continue
        try:
            with open(pf, 'r') as f:
                data = json.load(f)
        except Exception as e:
            missing.append(f"invalid json {pf}: {e}")
            continue

        for tk in ('metadata', 'global_metrics', 'class_metrics', 'predictions'):
            if tk not in data:
                missing.append(f"{pf}: missing top-level key '{tk}'")

    if missing:
        print("[ERROR] Validation failed for HTML report inputs:")
        for m in missing:
            print(f" - {m}")
        raise RuntimeError('generate_html_report: missing or invalid performance inputs')
    
    # Préparer les dossiers de sortie
    output_dir = Path(output_file).parent
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Charger et copier les images
    viz_dir = Path(__file__).parent / "outputs"
    
    images = {
        'threshold_calibration': viz_dir / "threshold_calibration_comparison.png",
        'model_comparison': viz_dir / "model_performance_comparison.png",
        'best_global_metrics': viz_dir / "best_global_metrics_comparison.png",
        'best_confusion_matrices': viz_dir / "best_confusion_matrices.png",
        'snr_distribution': viz_dir / "snr_distribution.png"
    }
    
    # Copier les images et obtenir chemins relatifs
    image_paths = {}
    for name, path in images.items():
        if path.exists():
            rel_path = copy_image_to_output(path, images_dir)
            if rel_path:
                image_paths[name] = rel_path
                print(f"  ✓ Copied: {name}")
    
    # Générer le HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UAV Acoustic Identification - Performance Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        nav {{
            background: #f8f9fa;
            padding: 15px 40px;
            border-bottom: 2px solid #e0e0e0;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        
        nav ul {{
            list-style: none;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }}
        
        nav a {{
            color: #667eea;
            text-decoration: none;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }}
        
        nav a:hover {{
            background: #667eea;
            color: white;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        section {{
            margin-bottom: 60px;
        }}
        
        h2 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        h3 {{
            color: #764ba2;
            font-size: 1.5em;
            margin: 30px 0 15px 0;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-card h4 {{
            font-size: 1.1em;
            margin-bottom: 10px;
            opacity: 0.9;
        }}
        
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        tbody tr:hover {{
            background: #f5f5f5;
        }}
        
        .best-row {{
            background: #d4edda !important;
            font-weight: bold;
        }}
        
        .image-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}
        
        .highlight {{
            background: #fff3cd;
            padding: 20px;
            border-left: 5px solid #ffc107;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .badge-success {{
            background: #28a745;
            color: white;
        }}
        
        .badge-info {{
            background: #17a2b8;
            color: white;
        }}
        
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }}
        
        .progress-fill {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 UAV Acoustic Identification</h1>
            <p>Performance Analysis Report</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <nav>
            <ul>
                <li><a href="#overview">Overview</a></li>
                <li><a href="#results">Best Results</a></li>
                <li><a href="#threshold">Threshold Calibration</a></li>
                <li><a href="#comparison">Model Comparison</a></li>
                <li><a href="#distance">Performance by Distance</a></li>
                <li><a href="#snr">SNR Analysis</a></li>
            </ul>
        </nav>
        
        <div class="content">
            <section id="overview">
                <h2>📊 Overview</h2>
                <p>This report presents the performance analysis of multiple deep learning models for UAV acoustic identification. The models were evaluated using various metrics and threshold configurations.</p>
                
                <div class="highlight">
                    <strong>Selection Criteria:</strong> {best_results.get('selection_criteria', 'Best F1-Score per (model, split)')}
                    <br>
                    <strong>Total Configurations:</strong> {best_results.get('total_configurations', 0)}
                </div>
            </section>
            
            <section id="results">
                <h2>🏆 Best Results Summary</h2>
                {generate_results_table(best_results.get('results', {}))}
                {generate_metrics_cards(best_results.get('results', {}))}
            </section>
"""
    
    # Ajouter les sections de visualisation
    if 'threshold_calibration' in image_paths:
        html_content += f"""
            <section id="threshold">
                <h2>📈 Threshold Calibration Analysis</h2>
                <p>This analysis shows how different classification thresholds affect model performance across various metrics.</p>
                <div class="image-container">
                    <img src="{image_paths['threshold_calibration']}" alt="Threshold Calibration">
                </div>
            </section>
"""
    
    if 'model_comparison' in image_paths:
        html_content += f"""
            <section id="comparison">
                <h2>🔍 Model Performance Comparison</h2>
                <p>Comparative analysis of all models across different performance metrics.</p>
                <div class="image-container">
                    <img src="{image_paths['model_comparison']}" alt="Model Comparison">
                </div>
            </section>
"""
    
    if 'best_global_metrics' in image_paths:
        html_content += f"""
            <section id="metrics">
                <h2>📊 Global Metrics Comparison</h2>
                <p>Comparison of accuracy, precision, recall, and F1-score across all models.</p>
                <div class="image-container">
                    <img src="{image_paths['best_global_metrics']}" alt="Global Metrics">
                </div>
            </section>
"""
    
    if 'best_confusion_matrices' in image_paths:
        html_content += f"""
            <section id="confusion">
                <h2>🎯 Confusion Matrices</h2>
                <p>Detailed confusion matrices for each model showing classification performance.</p>
                <div class="image-container">
                    <img src="{image_paths['best_confusion_matrices']}" alt="Confusion Matrices">
                </div>
            </section>
"""
    
    if 'snr_distribution' in image_paths:
        html_content += f"""
            <section id="snr">
                <h2>🔊 SNR Distribution Analysis</h2>
                <p>Distribution of Signal-to-Noise Ratio (SNR) across augmented dataset categories.</p>
                <div class="image-container">
                    <img src="{image_paths['snr_distribution']}" alt="SNR Distribution">
                </div>
            </section>
"""
    
    html_content += """
        </div>
        
        <footer>
            <p>&copy; 2024 UAV Acoustic Identification Project</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Generated with ❤️ by the Performance Analysis Pipeline</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Sauvegarder le fichier HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ HTML report saved: {output_file}")


def generate_results_table(results):
    """
    Génère le tableau HTML des résultats.
    """
    if not results:
        return "<p>No results available.</p>"
    
    # Trouver le meilleur F1-score global
    best_f1 = max(r['metrics']['f1_score'] for r in results.values())
    
    html = """
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Split</th>
                <th>Threshold</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Specificity</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for key, result in sorted(results.items()):
        model = result['model']
        split = result['split']
        threshold = result['threshold']
        metrics = result['metrics']
        
        # Marquer la meilleure ligne
        is_best = abs(metrics['f1_score'] - best_f1) < 0.0001
        row_class = 'best-row' if is_best else ''
        
        html += f"""
            <tr class="{row_class}">
                <td><strong>{model}</strong> {'<span class="badge badge-success">BEST</span>' if is_best else ''}</td>
                <td>{split}</td>
                <td>{threshold:.3f}</td>
                <td>{metrics['accuracy']:.4f}</td>
                <td>{metrics['precision']:.4f}</td>
                <td>{metrics['recall']:.4f}</td>
                <td><strong>{metrics['f1_score']:.4f}</strong></td>
                <td>{metrics['specificity']:.4f}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    
    return html


def generate_metrics_cards(results):
    """
    Génère les cartes de métriques.
    """
    if not results:
        return ""
    
    # Calculer les moyennes
    all_accuracies = [r['metrics']['accuracy'] for r in results.values()]
    all_f1 = [r['metrics']['f1_score'] for r in results.values()]
    all_precision = [r['metrics']['precision'] for r in results.values()]
    all_recall = [r['metrics']['recall'] for r in results.values()]
    
    avg_acc = sum(all_accuracies) / len(all_accuracies)
    avg_f1 = sum(all_f1) / len(all_f1)
    avg_prec = sum(all_precision) / len(all_precision)
    avg_rec = sum(all_recall) / len(all_recall)
    
    html = """
    <h3>📊 Average Performance Metrics</h3>
    <div class="metrics-grid">
    """
    
    metrics_data = [
        ("Average Accuracy", f"{avg_acc:.3f}", avg_acc),
        ("Average F1-Score", f"{avg_f1:.3f}", avg_f1),
        ("Average Precision", f"{avg_prec:.3f}", avg_prec),
        ("Average Recall", f"{avg_rec:.3f}", avg_rec)
    ]
    
    for title, value, score in metrics_data:
        html += f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <div class="value">{value}</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {score * 100}%"></div>
            </div>
        </div>
        """
    
    html += """
    </div>
    """
    
    return html


def main():
    print("=" * 80)
    print("  HTML REPORT GENERATOR")
    print("=" * 80)
    print()
    
    print("[INFO] Loading visualization images...")
    
    # Créer le rapport HTML
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "performance_report.html"
    
    print("[INFO] Generating HTML report...")
    generate_html_report(output_file)
    
    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)
    print(f"\n  Open: {output_file}")


if __name__ == '__main__':
    main()
