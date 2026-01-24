#!/usr/bin/env python3
"""
Threshold Calibration - Multi-Criteria Hierarchical Optimization

Optimise les thresholds de classification en utilisant une approche hiérarchique:

Tier 1 (HARD CONSTRAINTS - must satisfy):
    - Recall > min_recall (défaut: 0.90)
    - Precision_drone > min_precision_drone (défaut: 0.70)
    - NPV_ambient > min_precision_ambient (défaut: 0.85)

Tier 2 (OPTIMIZATION TARGET):
    - Maximize: balanced_precision = min(precision_drone, NPV_ambient)
    - Force équilibre entre les deux classes

Tier 3 (TIE-BREAKER):
    - Si égalité balanced_precision → maximize F1-score
    - Si égalité F1 → maximize recall
    - Si égalité recall → prefer lower threshold (plus permissif)

Relaxation progressive si aucun threshold ne satisfait les contraintes.

Usage:
    # Calibrer un modèle
    python calibrate_thresholds.py --model CNN
    
    # Calibrer tous les modèles
    python calibrate_thresholds.py --all-models
    
    # Avec contraintes personnalisées
    python calibrate_thresholds.py --model RNN --min-recall 0.95 --min-precision-drone 0.80
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def calculate_metrics(y_true, y_pred_proba, threshold):
    """
    Calcule toutes les métriques pour un threshold donné.
    
    Returns:
        dict avec métriques complètes
    """
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Métriques principales
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision_drone = tp / (tp + fp) if (tp + fp) > 0 else 0  # PPV (Positive Predictive Value)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity / True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    
    # NPV (Negative Predictive Value) = precision pour la classe ambient (0)
    precision_ambient = tn / (tn + fn) if (tn + fn) > 0 else 0  # NPV
    
    # F1-score
    f1 = 2 * (precision_drone * recall) / (precision_drone + recall) if (precision_drone + recall) > 0 else 0
    
    # Balanced precision (notre critère d'optimisation Tier 2)
    balanced_precision = min(precision_drone, precision_ambient)
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision_drone': float(precision_drone),  # PPV
        'precision_ambient': float(precision_ambient),  # NPV
        # Per-class accuracies: drone accuracy approximated by recall, ambient accuracy by specificity
        'accuracy_drone': float(recall),
        'accuracy_ambient': float(specificity),
        'recall': float(recall),  # Sensitivity
        'specificity': float(specificity),
        'f1_score': float(f1),
        'balanced_precision': float(balanced_precision),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def optimize_threshold_hierarchical(y_true, y_pred_proba, 
                                    min_recall=0.90,
                                    min_precision_drone=0.70,
                                    min_precision_ambient=0.85,
                                    threshold_range=(0.05, 0.95),
                                    threshold_step=0.01,
                                    verbose=True,
                                    max_accuracy_gap=None):
    """
    Optimise threshold avec critères hiérarchiques et relaxation progressive.
    
    Returns:
        tuple: (optimal_threshold, optimal_metrics, all_tested_metrics, relaxation_info)
    """
    if verbose:
        print("\n" + "="*80)
        print("  THRESHOLD OPTIMIZATION - Hierarchical Multi-Criteria")
        print("="*80)
        print(f"\nConstraints (Tier 1):")
        print(f"  • Min Recall (Sensitivity):         {min_recall:.2%}")
        print(f"  • Min Precision Drone (PPV):        {min_precision_drone:.2%}")
        print(f"  • Min Precision Ambient (NPV):      {min_precision_ambient:.2%}")
        if max_accuracy_gap is not None:
            print(f"  • Max per-class accuracy gap:       {max_accuracy_gap:.2%} (abs(accuracy_drone - accuracy_ambient))")
        print(f"\nOptimization Target (Tier 2):")
        print(f"  • Maximize: balanced_precision = min(PPV, NPV)")
        print(f"\nTie-breakers (Tier 3):")
        print(f"  • F1-score → Recall → Lower threshold")
        print("="*80)
    
    # Tester tous les thresholds dans la plage
    all_results = []
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    
    if verbose:
        print(f"\nTesting {len(thresholds)} thresholds from {threshold_range[0]:.2f} to {threshold_range[1]:.2f}...")
    
    for thresh in thresholds:
        metrics = calculate_metrics(y_true, y_pred_proba, thresh)
        all_results.append(metrics)
    
    # Tier 1: Filtrer par contraintes DURES
    def satisfies_hard_constraints(r, min_recall_, min_prec_drone_, min_prec_amb_, max_acc_gap_):
        if not (r['recall'] >= min_recall_ and
                r['precision_drone'] >= min_prec_drone_ and
                r['precision_ambient'] >= min_prec_amb_):
            return False
        if max_acc_gap_ is not None:
            gap = abs(r.get('accuracy_drone', 0.0) - r.get('accuracy_ambient', 0.0))
            if gap > max_acc_gap_:
                return False
        return True

    candidates = [r for r in all_results if satisfies_hard_constraints(r, min_recall, min_precision_drone, min_precision_ambient, max_accuracy_gap)]
    
    relaxation_applied = None
    
    if not candidates:
        if verbose:
            print("\n⚠️  No threshold satisfies all hard constraints!")
            print("    Applying progressive constraint relaxation...")
        
        # Relaxation progressive (3 niveaux)
        relaxation_levels = [
            {'name': 'Level 1: Relax ambient precision', 
             'min_precision_ambient': min_precision_ambient * 0.95},  # -5%
            
            {'name': 'Level 2: Relax drone precision',
             'min_precision_ambient': min_precision_ambient * 0.90,  # -10%
             'min_precision_drone': min_precision_drone * 0.95},  # -5%
            
            {'name': 'Level 3: Relax recall (last resort)',
             'min_precision_ambient': min_precision_ambient * 0.85,  # -15%
             'min_precision_drone': min_precision_drone * 0.90,  # -10%
             'min_recall': min_recall * 0.95}  # -5%
        ]
        
        for i, level in enumerate(relaxation_levels):
            relaxed_ambient = level.get('min_precision_ambient', min_precision_ambient)
            relaxed_drone = level.get('min_precision_drone', min_precision_drone)
            relaxed_recall = level.get('min_recall', min_recall)
            
            candidates = [
                r for r in all_results
                if satisfies_hard_constraints(r, relaxed_recall, relaxed_drone, relaxed_ambient, max_accuracy_gap)
            ]
            
            if candidates:
                relaxation_applied = {
                    'level': i + 1,
                    'name': level['name'],
                    'relaxed_constraints': {
                        'min_recall': relaxed_recall,
                        'min_precision_drone': relaxed_drone,
                        'min_precision_ambient': relaxed_ambient
                    }
                }
                if verbose:
                    print(f"\n✓ {level['name']} successful")
                    print(f"  Found {len(candidates)} candidate(s)")
                break
        
        if not candidates:
            # Dernière option: prendre le meilleur F1 sans contraintes
            if verbose:
                print("\n⚠️  All relaxation levels failed!")
                print("    Falling back to best F1-score (no constraints)")
            candidates = all_results
            relaxation_applied = {
                'level': 4,
                'name': 'Fallback: Best F1 (no constraints)',
                'relaxed_constraints': None
            }
    
    # Tier 2: Trier par balanced_precision (desc)
    candidates.sort(key=lambda x: x['balanced_precision'], reverse=True)
    
    # Tier 3: Départager les égalités
    best_balanced = candidates[0]['balanced_precision']
    ties = [c for c in candidates if abs(c['balanced_precision'] - best_balanced) < 1e-6]
    
    if len(ties) > 1:
        # Tie-breaker: F1 → recall → lower threshold
        optimal = max(ties, key=lambda x: (x['f1_score'], x['recall'], -x['threshold']))
    else:
        optimal = candidates[0]
    
    if verbose:
        print(f"\n{'='*80}")
        print("  OPTIMAL THRESHOLD FOUND")
        print("="*80)
        print(f"\nThreshold: {optimal['threshold']:.4f}")
        print(f"\nMetrics:")
        print(f"  Accuracy:                {optimal['accuracy']:.4f}")
        print(f"  Precision Drone (PPV):   {optimal['precision_drone']:.4f}")
        print(f"  Precision Ambient (NPV): {optimal['precision_ambient']:.4f}")
        print(f"  Recall (Sensitivity):    {optimal['recall']:.4f}")
        print(f"  Specificity:             {optimal['specificity']:.4f}")
        print(f"  F1-Score:                {optimal['f1_score']:.4f}")
        print(f"  Balanced Precision:      {optimal['balanced_precision']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP={optimal['tp']}, TN={optimal['tn']}, FP={optimal['fp']}, FN={optimal['fn']}")
        
        if relaxation_applied:
            print(f"\n⚠️  Constraints Relaxation Applied:")
            print(f"  Level: {relaxation_applied['level']}")
            print(f"  Name: {relaxation_applied['name']}")
    
    return optimal['threshold'], optimal, all_results, relaxation_applied


def load_validation_predictions(model_name):
    """
    Charge les prédictions de validation pour un modèle.
    Utilise les features NPZ si disponibles.
    """
    import tensorflow as tf
    # Allow loading models that include Python `Lambda` layers when running
    # in this trusted local environment. Keras blocks unsafe deserialization
    # by default; enable it here so calibrations/tests succeed on saved
    # artifacts that contain lambdas.
    try:
        import keras
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass
    
    # Charger le modèle
    model_paths = {
        'CNN': config.CNN_MODEL_PATH,
        'RNN': config.RNN_MODEL_PATH,
        'CRNN': config.CRNN_MODEL_PATH,
        'ATTENTION_CRNN': config.ATTENTION_CRNN_MODEL_PATH,
        'EFFICIENTNET': config.EFFICIENTNET_MODEL_PATH,
        'MOBILENET': config.MOBILENET_MODEL_PATH,
        'CONFORMER': config.CONFORMER_MODEL_PATH,
        'TCN': config.TCN_MODEL_PATH
    }
    
    model_path = model_paths.get(model_name.upper())
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    
    # Charger les features de validation
    val_features_path = config.MEL_VAL_NPZ_PATH
    
    if not val_features_path.exists():
        raise FileNotFoundError(f"Validation features not found: {val_features_path}")
    
    print(f"Loading validation features: {val_features_path}")
    data = np.load(val_features_path, allow_pickle=True)
    
    # Extract features and labels (handle different NPZ formats)
    if 'features' in data:
        X_val = data['features']
        y_val = data['labels']
    elif 'mels' in data:
        X_val = data['mels']
        y_val = data['labels']
    else:
        raise KeyError(f"Could not find features/mels in NPZ. Available keys: {list(data.keys())}")
    
    print(f"Validation samples: {len(y_val)}")
    print(f"Feature shape: {X_val.shape}")
    
    # Add channel dimension for Conv2D models (CNN/CRNN/Attention/EfficientNet/MobileNet need 4D)
    # RNN/TCN/Conformer use 1D processing and expect 3D
    models_needing_3d = ['RNN', 'TCN', 'CONFORMER']
    if X_val.ndim == 3 and model_name.upper() not in models_needing_3d:
        X_val = X_val[..., np.newaxis]
        print(f"Added channel dimension: {X_val.shape}")
    
    # Predict
    print("Generating predictions...")
    predictions = model.predict(X_val, batch_size=32, verbose=0)
    
    # Extract probability for class 1
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        y_pred_proba = predictions[:, 1]
    else:
        y_pred_proba = predictions.reshape(-1)
    
    return y_val, y_pred_proba


def save_calibration_results(calibration_data, output_path):
    """Sauvegarde les résultats de calibration en JSON.
    
    Si le fichier existe déjà, fusionne les nouveaux modèles avec les existants.
    Cela permet d'appeler le script plusieurs fois (--model CNN, --model RNN, etc.)
    sans écraser les résultats précédents.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Charger le fichier existant s'il existe
    existing_data = None
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not read existing calibration file: {e}")
            print("   Creating new calibration file...")
    
    # Fusionner les données
    if existing_data is not None:
        # Garder la structure existante et fusionner les modèles
        existing_data['models'].update(calibration_data['models'])
        # Mettre à jour la date
        existing_data['calibration_date'] = calibration_data['calibration_date']
        # Utiliser les nouvelles contraintes si présentes
        if 'constraints' in calibration_data:
            existing_data['constraints'] = calibration_data['constraints']
        final_data = existing_data
    else:
        final_data = calibration_data
    
    # Sauvegarder atomiquement (tmp + rename)
    tmp_path = output_path.with_suffix('.json.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    # Renommer atomiquement
    tmp_path.replace(output_path)
    
    print(f"\n✓ Calibration saved: {output_path}")
    print(f"   Models in file: {', '.join(final_data['models'].keys())}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate classification thresholds')
    parser.add_argument('--model', type=str, choices=['CNN', 'RNN', 'CRNN', 'ATTENTION_CRNN', 'EFFICIENTNET', 'MOBILENET', 'CONFORMER', 'TCN'],
                        help='Model to calibrate')
    parser.add_argument('--all-models', action='store_true',
                        help='Calibrate all models')
    parser.add_argument('--min-recall', type=float, default=0.90,
                        help='Minimum recall constraint (default: 0.90)')
    parser.add_argument('--min-precision-drone', type=float, default=0.70,
                        help='Minimum drone precision constraint (default: 0.70)')
    parser.add_argument('--min-precision-ambient', type=float, default=0.85,
                        help='Minimum ambient precision constraint (default: 0.85)')
    parser.add_argument('--max-accuracy-gap', type=float, default=None,
                        help='Maximum allowed absolute gap between per-class accuracies (drone vs ambient). e.g. 0.25')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (default: results/calibrated_thresholds.json)')
    
    args = parser.parse_args()
    
    if not args.model and not args.all_models:
        parser.error("Either --model or --all-models must be specified")
    
    # Déterminer les modèles à calibrer
    if args.all_models:
        models = ['CNN', 'RNN', 'CRNN', 'ATTENTION_CRNN', 'EFFICIENTNET', 'MOBILENET', 'CONFORMER', 'TCN']
    else:
        models = [args.model]
    
    # Output path
    output_path = Path(args.output) if args.output else config.CALIBRATION_FILE_PATH
    
    # Structure de données pour tout stocker
    calibration_data = {
        'version': '1.0',
        'calibration_date': datetime.now().isoformat(),
        'calibration_mode': 'balanced_precision',
        'constraints': {
            'min_recall': args.min_recall,
            'min_precision_drone': args.min_precision_drone,
            'min_precision_ambient': args.min_precision_ambient
        },
        'models': {}
    }
    # Record the max_accuracy_gap constraint explicitly if provided
    if args.max_accuracy_gap is not None:
        calibration_data['constraints']['max_accuracy_gap'] = args.max_accuracy_gap
    
    # Calibrer chaque modèle
    for model_name in models:
        print(f"\n{'#'*80}")
        print(f"  CALIBRATING: {model_name}")
        print(f"{'#'*80}")
        
        try:
            # Charger prédictions validation
            y_val, y_pred_proba = load_validation_predictions(model_name)
            
            # Optimiser threshold
            optimal_threshold, optimal_metrics, all_results, relaxation_info = optimize_threshold_hierarchical(
                y_val, y_pred_proba,
                min_recall=args.min_recall,
                min_precision_drone=args.min_precision_drone,
                min_precision_ambient=args.min_precision_ambient,
                verbose=True,
                max_accuracy_gap=args.max_accuracy_gap
            )
            
            # Vérifier contraintes
            constraints_met = {
                f'min_recall_{args.min_recall:.2f}': optimal_metrics['recall'] >= args.min_recall,
                f'min_precision_drone_{args.min_precision_drone:.2f}': optimal_metrics['precision_drone'] >= args.min_precision_drone,
                f'min_precision_ambient_{args.min_precision_ambient:.2f}': optimal_metrics['precision_ambient'] >= args.min_precision_ambient
            }
            
            # Stocker résultats
            calibration_data['models'][model_name] = {
                'threshold': optimal_threshold,
                'validation_samples': int(len(y_val)),
                'metrics_at_threshold': optimal_metrics,
                'constraints_met': constraints_met,
                'relaxation_applied': relaxation_info,
                'optimization_summary': {
                    'num_thresholds_tested': len(all_results),
                    'threshold_range': [float(all_results[0]['threshold']), float(all_results[-1]['threshold'])],
                    'best_f1_threshold': float(max(all_results, key=lambda x: x['f1_score'])['threshold']),
                    'best_balanced_threshold': optimal_threshold
                },
                'all_tested_thresholds': all_results  # Pour visualisation
            }
            
            print(f"\n✓ {model_name} calibration complete")
            
        except Exception as e:
            print(f"\n✗ {model_name} calibration failed: {e}")
            import traceback
            traceback.print_exc()
            calibration_data['models'][model_name] = {
                'error': str(e),
                'threshold': 0.5  # Fallback
            }
    
    # Sauvegarder
    save_calibration_results(calibration_data, output_path)
    
    # Résumé
    print(f"\n{'='*80}")
    print("  CALIBRATION SUMMARY")
    print("="*80)
    for model_name, data in calibration_data['models'].items():
        if 'error' in data:
            print(f"\n{model_name}: ✗ FAILED - {data['error']}")
        else:
            print(f"\n{model_name}:")
            print(f"  Threshold: {data['threshold']:.4f}")
            print(f"  F1-Score:  {data['metrics_at_threshold']['f1_score']:.4f}")
            print(f"  Balanced:  {data['metrics_at_threshold']['balanced_precision']:.4f}")
            if data.get('relaxation_applied'):
                print(f"  ⚠️  Relaxation: {data['relaxation_applied']['name']}")
    
    print(f"\n{'='*80}")
    print(f"Calibration file: {output_path}")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
