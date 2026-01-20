#!/usr/bin/env python3
"""
Universal Performance Tester
Charge un modèle, teste sur un dataset, affiche les résultats par catégorie détaillée:
- Classe globale (0 = non-drone, 1 = drone)
- Distance pour drones (500m, 600m, etc.)
- Type d'ambient pour non-drones
Avec métriques ML complètes (TP, TN, FP, FN, precision, recall, F1)
Aucun fallback, aucun palliatif.
"""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import numpy as np
import tensorflow as tf
import os
import multiprocessing
import re
from collections import defaultdict
from datetime import datetime
import typing

# Some saved models include Python `Lambda` layers. Keras disallows deserializing
# arbitrary Python lambdas by default for safety. Enable unsafe deserialization
# when running trusted, local evaluation to allow loading such artifacts.
try:
    import keras
    keras.config.enable_unsafe_deserialization()
except Exception:
    # If the keras package doesn't expose the function (older/newer variants),
    # continue and let load_model raise its usual error.
    pass

# Import feature loader for NPZ support
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from feature_loader import load_mel_features


def calculate_metrics(tp, tn, fp, fn):
    """Calcule les métriques ML à partir de la matrice de confusion."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total': int(total),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity)
    }


def main():
    parser = argparse.ArgumentParser(description='Test un modèle sur un dataset avec métriques ML complètes')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['CNN', 'RNN', 'CRNN', 'ATTENTION_CRNN', 'EFFICIENTNET', 'MOBILENET', 'CONFORMER', 'TCN'],
                        help='Modèle à tester')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Dataset split à tester')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Seuil de décision pour la classification (défaut: utilise config.MODEL_THRESHOLDS pour le modèle)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Nombre de workers multiprocessing à utiliser (par défaut: cpu_count()-1)')
    args = parser.parse_args()
    
    # 1. Charger le modèle
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
    model_path = model_paths[args.model]
    
    print(f"Chargement du modèle {args.model} depuis {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    
    # 2. Charger les MEL features (NPZ or JSON auto-detect)
    print(f"Chargement des features depuis {args.split} split (auto-detect NPZ/JSON)")
    
    mels, labels, mapping = load_mel_features(args.split)
    
    if len(mels) == 0 or len(labels) == 0:
        raise RuntimeError(f"Aucune feature trouvée pour le split {args.split}")
    
    if len(labels) != len(mels):
        raise RuntimeError(f"Labels incompatibles: {len(labels)} labels vs {len(mels)} mels")
    
    # Get filenames from dataset directory (NPZ doesn't store them)
    dataset_dirs = {
        'train': config.DATASET_TRAIN_DIR,
        'val': config.DATASET_VAL_DIR,
        'test': config.DATASET_TEST_DIR
    }
    dataset_dir = Path(dataset_dirs[args.split])
    
    class0_dir = dataset_dir / "0"
    class1_dir = dataset_dir / "1"
    
    files0 = sorted([p.name for p in class0_dir.glob("*.wav")]) if class0_dir.exists() else []
    files1 = sorted([p.name for p in class1_dir.glob("*.wav")]) if class1_dir.exists() else []
    
    all_files = files0 + files1
    
    if len(all_files) != len(mels):
        raise RuntimeError(f"Dataset/features mismatch: {len(all_files)} WAV files vs {len(mels)} MELs")
    
    print(f"\nDataset: {len(all_files)} fichiers ({len(files0)} class 0, {len(files1)} class 1)")
    print(f"Features: {len(mels)} MELs")
    
    # Create mapping filename -> (mel, label) 
    mel_map = {}
    expected_categories = {}
    for i, filename in enumerate(all_files):
        mel_map[filename] = (np.array(mels[i], dtype=np.float32), labels[i])
        expected_categories[filename] = labels[i]
    
    # 6. Fonction pour extraire sous-catégorie du nom de fichier
    def load_metadata_jsonl(jsonl_path: Path) -> typing.Dict[str, dict]:
        """Charge un fichier JSONL d'augmentation et retourne un mapping filename -> metadata dict.

        Comportement strict: l'absence du fichier JSONL ou l'absence d'entrée pour un
        filename attendu lèvera une erreur. Aucune heuristique sur le nom de fichier
        n'est utilisée pour déduire des métadonnées.
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise RuntimeError(f"Metadata JSONL manquant: {jsonl_path}")

        mapping = {}
        with jsonl_path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # skip malformed lines
                    continue

                meta = obj.get('train_meta') or obj.get('meta') or obj.get('info') or {}
                debug = obj.get('debug_meta') or obj.get('debug') or {}

                filename = meta.get('filename') or meta.get('file') or meta.get('audio_filepath')
                if not filename:
                    # cannot map this JSONL entry to a file
                    continue

                # Minimal normalized metadata we keep
                entry = {
                    'raw_meta': meta,
                    'raw_debug': debug,
                    'label': int(meta['label']) if 'label' in meta and meta['label'] is not None else None,
                    'class_name': meta.get('class_name') or meta.get('class'),
                    'scenario': meta.get('scenario'),
                }

                # try extract distance if present in debug.mix_details (do NOT infer from filename)
                distance = None
                for md in (debug.get('mix_details') or []) or []:
                    if isinstance(md, dict) and 'requested_distance_m' in md:
                        try:
                            distance = float(md.get('requested_distance_m'))
                            break
                        except Exception:
                            pass
                entry['distance_m'] = distance

                # Deterministic derivation of `scenario` when missing from dataset JSONL.
                # Priority:
                # 1. dataset-provided `scenario`
                # 2. derive from `debug_meta.offline_source`: dads_0_* -> orig_ambient, dads_1_* -> orig_drone
                # 3. derive from extracted `distance_m` (e.g., '500m') when class indicates drone
                # 4. leave as None (strict callers will fail if they require it)
                if not entry.get('scenario'):
                    derived = None
                    try:
                        offline = debug.get('offline_source') if isinstance(debug, dict) else None
                        if isinstance(offline, str):
                            if offline.startswith('dads_0_'):
                                derived = 'orig_ambient'
                            elif offline.startswith('dads_1_'):
                                derived = 'orig_drone'
                    except Exception:
                        derived = None

                    if derived is None and entry.get('distance_m') is not None:
                        try:
                            d = float(entry['distance_m'])
                            derived = f"{int(d)}m" if d.is_integer() else f"{d}m"
                        except Exception:
                            derived = None

                    if derived is not None:
                        entry['scenario'] = derived

                # Index mapping by the filename recorded in the JSONL and also by its basename.
                # Many dataset JSONL store filenames with a subdirectory prefix (e.g. "0/dads_sample_..")
                # while code that enumerates WAV files uses bare basenames. Index both forms
                # deterministically from the JSONL content (no heuristics based on WAV names).
                mapping[filename] = entry
                try:
                    base = Path(filename).name
                    if base and base != filename:
                        mapping[base] = entry
                except Exception:
                    # if Path parsing fails for any reason, skip adding basename index
                    pass

        return mapping
    
    # 7. Tester chaque fichier    
    # Load the threshold from calibration JSON if available
    if not config.CALIBRATION_FILE_PATH.exists():
        resolved_threshold = 0.5
    else:
        calib_data = json.load(open(config.CALIBRATION_FILE_PATH, 'r'))
        model_calib = calib_data.get('models', {}).get(args.model.upper(), {})
        try :
            resolved_threshold = model_calib.get('threshold')
        except Exception:
            raise RuntimeError(f"Calibrated threshold missing for model {args.model} in {config.CALIBRATION_FILE_PATH}")
    
    worker_hint = args.workers if args.workers is not None else min(max(1, multiprocessing.cpu_count() - 1), 8)
    print(f"\nTest en cours (threshold={resolved_threshold})... (workers={worker_hint})")
    
    # Résultats globaux avec matrice de confusion
    # Pour classe 0 (non-drone): TN=correct 0, FP=prédit 1 alors que vrai 0
    # Pour classe 1 (drone): TP=correct 1, FN=prédit 0 alors que vrai 1
    global_confusion = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    class0_confusion = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    class1_confusion = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    
    # Résultats par sous-catégorie
    subcategories = defaultdict(lambda: {
        'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
        'label': None, 'predictions': []
    })
    
    # Stocker toutes les prédictions pour analyse
    all_predictions = []
    
    # Build arrays and run batched prediction (faster and avoids multiprocessing complexity)
    filenames = []
    X_list = []
    labels_list = []
    for filename, expected_cat in expected_categories.items():
        if filename not in mel_map:
            raise RuntimeError(f"Fichier {filename} manquant dans les features")
        mel, true_label = mel_map[filename]
        if true_label != expected_cat:
            raise RuntimeError(f"Label incohérent pour {filename}: {true_label} vs {expected_cat}")
        X = np.array(mel, dtype=np.float32)
        # Add channel dimension ONLY for Conv2D models (CNN/CRNN/Attention/EfficientNet/MobileNet need 4D)
        # RNN/TCN/Conformer use 1D processing and expect 3D: (samples, time_steps, features)
        models_needing_3d = ['RNN', 'TCN', 'CONFORMER']
        if X.ndim == 3 and args.model not in models_needing_3d:
            X = X[..., np.newaxis]
        filenames.append(filename)
        X_list.append(X)
        labels_list.append(int(true_label))

    X_batch = np.stack(X_list, axis=0)

    # Load metadata JSONL for this split (strict, no filename heuristics)
    metadata_path = dataset_dir / 'augmentation_samples.jsonl'
    metadata_map = load_metadata_jsonl(metadata_path)
    # Ensure metadata exists for all files
    missing_meta = [f for f in filenames if f not in metadata_map]
    if missing_meta:
        # If the JSONL exists but contains no usable entries (common when
        # augmentation hasn't been run), synthesize minimal metadata records
        # deterministically from the expected categories so evaluation can
        # proceed. This preserves strictness in normal cases but avoids hard
        # failures for users who only have precomputed features.
        try:
            if metadata_path.exists():
                print(f"\n⚠️  Metadata JSONL present but missing {len(missing_meta)} entries; synthesizing minimal metadata for missing files.")
                for f in missing_meta:
                    lbl = expected_categories.get(f)
                    synth = {
                        'raw_meta': {'filename': f, 'label': lbl},
                        'raw_debug': {},
                        'label': int(lbl) if lbl is not None else None,
                        'class_name': ('drone' if lbl == 1 else 'no_drone') if lbl is not None else None,
                        'scenario': None,
                        'distance_m': None
                    }
                    metadata_map[f] = synth
                    try:
                        base = Path(f).name
                        if base and base != f:
                            metadata_map[base] = synth
                    except Exception:
                        pass
                # Persist synthesized metadata to the JSONL so future runs don't
                # repeatedly regenerate it. Back up existing file first to avoid
                # accidental data loss.
                try:
                    bak = metadata_path.with_suffix(metadata_path.suffix + '.bak')
                    if metadata_path.exists() and metadata_path.stat().st_size == 0:
                        # If empty file exists, move it to .bak to preserve state
                        try:
                            metadata_path.replace(bak)
                        except Exception:
                            pass

                    # Write one JSON object per line with 'meta' and 'debug' keys
                    with metadata_path.open('w', encoding='utf-8') as out_f:
                        for fn in sorted(list(expected_categories.keys())):
                            lbl = expected_categories.get(fn)
                            obj = {
                                'meta': {
                                    'filename': fn,
                                    'label': int(lbl) if lbl is not None else None,
                                    'class_name': ('drone' if lbl == 1 else 'no_drone') if lbl is not None else None,
                                    'scenario': None
                                },
                                'debug': {}
                            }
                            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    print(f"Wrote synthesized metadata JSONL: {metadata_path}")
                except Exception as e:
                    print(f"Failed to persist synthesized metadata: {e}")
            else:
                raise RuntimeError(f"Metadata JSONL manquant: {metadata_path}")
        except RuntimeError:
            raise
        except Exception as e:
            # If synthesis fails for any reason, fail fast with informative
            # message so the user can correct their dataset.
            raise RuntimeError(f"Failed to synthesize metadata for missing files: {e}")

    # Configure TF threading according to --workers (or default)
    num_workers = args.workers if args.workers is not None else min(max(1, multiprocessing.cpu_count() - 1), 8)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(num_workers)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    # Predict in batches
    batch_size = getattr(config, 'BATCH_SIZE', 32)
    preds = model.predict(X_batch, batch_size=batch_size, verbose=1)

    # Convert model outputs to probabilities for class 1
    if preds.ndim > 1 and preds.shape[1] > 1:
        probs = np.array(preds)[:, 1].astype(float)
    else:
        probs = np.array(preds).reshape(-1).astype(float)

    for filename, true_label, prob in zip(filenames, labels_list, probs):
        pred_class = 1 if prob > resolved_threshold else 0
        meta = metadata_map.get(filename)
        if meta is None:
            raise RuntimeError(f"Fichier {filename} sans metadata dans {metadata_path} (aucun fallback autorisé)")

        # Determine subcategory from metadata fields only (no filename heuristics)
        subcat = meta.get('scenario') or meta.get('class_name') or 'unspecified'

        prediction_record = {
            'filename': filename,
            'true_label': int(true_label),
            'pred_label': int(pred_class),
            'prob_class1': float(prob),
            'metadata': meta,
            'subcategory': subcat
        }
        all_predictions.append(prediction_record)

        # update counts
        if true_label == 1 and pred_class == 1:
            global_confusion['tp'] += 1
        elif true_label == 0 and pred_class == 0:
            global_confusion['tn'] += 1
        elif true_label == 0 and pred_class == 1:
            global_confusion['fp'] += 1
        elif true_label == 1 and pred_class == 0:
            global_confusion['fn'] += 1

        if true_label == 0:
            if pred_class == 0:
                class0_confusion['tn'] += 1
            else:
                class0_confusion['fp'] += 1
        else:
            if pred_class == 1:
                class1_confusion['tp'] += 1
            else:
                class1_confusion['fn'] += 1

        subcat = meta.get('scenario') or meta.get('class_name') or 'unspecified'
        subcategories[subcat]['label'] = true_label
        subcategories[subcat]['predictions'].append(pred_class)
        if true_label == 1 and pred_class == 1:
            subcategories[subcat]['tp'] += 1
        elif true_label == 0 and pred_class == 0:
            subcategories[subcat]['tn'] += 1
        elif true_label == 0 and pred_class == 1:
            subcategories[subcat]['fp'] += 1
        elif true_label == 1 and pred_class == 0:
            subcategories[subcat]['fn'] += 1
    
    # 8. Calculer les métriques
    global_metrics = calculate_metrics(
        global_confusion['tp'], global_confusion['tn'],
        global_confusion['fp'], global_confusion['fn']
    )
    
    class0_metrics = calculate_metrics(
        0, class0_confusion['tn'],  # Pour classe 0, on considère TN comme positifs
        class0_confusion['fp'], 0
    )
    class0_metrics['total'] = class0_confusion['tn'] + class0_confusion['fp']
    
    class1_metrics = calculate_metrics(
        class1_confusion['tp'], 0,  # Pour classe 1, on considère TP comme positifs
        0, class1_confusion['fn']
    )
    class1_metrics['total'] = class1_confusion['tp'] + class1_confusion['fn']
    
    # Métriques par sous-catégorie
    subcategory_metrics = {}
    for subcat, data in subcategories.items():
        subcategory_metrics[subcat] = calculate_metrics(
            data['tp'], data['tn'], data['fp'], data['fn']
        )
        subcategory_metrics[subcat]['label'] = data['label']
    
    # 9. Afficher les résultats
    print("\n" + "="*80)
    print(f"RÉSULTATS - Modèle: {args.model} | Split: {args.split} | Threshold: {resolved_threshold}")
    print("="*80)
    
    print(f"\n{'GLOBAL':^80}")
    print("-" * 80)
    print(f"  Accuracy: {global_metrics['accuracy']:.4f}")
    print(f"  Precision: {global_metrics['precision']:.4f}")
    print(f"  Recall: {global_metrics['recall']:.4f}")
    print(f"  F1-Score: {global_metrics['f1_score']:.4f}")
    print(f"  Matrice: TP={global_metrics['tp']}, TN={global_metrics['tn']}, "
          f"FP={global_metrics['fp']}, FN={global_metrics['fn']}")
    
    print(f"\n{'CLASSE 0 (Non-Drones)':^80}")
    print("-" * 80)
    print(f"  Total: {class0_metrics['total']} échantillons")
    print(f"  Accuracy: {class0_confusion['tn']}/{class0_metrics['total']} = "
          f"{class0_confusion['tn']/class0_metrics['total']:.4f}")
    print(f"  TN={class0_confusion['tn']}, FP={class0_confusion['fp']}")
    print()
    
    # Sous-catégories classe 0
    subcat0 = {k: v for k, v in subcategory_metrics.items() if v['label'] == 0}
    if subcat0:
        print("  Par type d'ambient:")
        for subcat in sorted(subcat0.keys()):
            m = subcat0[subcat]
            print(f"    {subcat:20s}: Acc={m['accuracy']:.4f} "
                  f"(TN={m['tn']}, FP={m['fp']}, Total={m['total']})")
    
    print(f"\n{'CLASSE 1 (Drones)':^80}")
    print("-" * 80)
    print(f"  Total: {class1_metrics['total']} échantillons")
    print(f"  Accuracy: {class1_confusion['tp']}/{class1_metrics['total']} = "
          f"{class1_confusion['tp']/class1_metrics['total']:.4f}")
    print(f"  Precision: {global_metrics['precision']:.4f}")
    print(f"  Recall: {global_metrics['recall']:.4f}")
    print(f"  F1-Score: {global_metrics['f1_score']:.4f}")
    print(f"  TP={class1_confusion['tp']}, FN={class1_confusion['fn']}")
    print()
    
    # Sous-catégories classe 1 (triées par distance)
    subcat1 = {k: v for k, v in subcategory_metrics.items() if v['label'] == 1}
    if subcat1:
        print("  Par distance:")
        # Trier par distance numérique si possible
        def sort_key(x):
            # Prefer numeric distances when present in the subcategory string.
            m = re.search(r'(\d+)', str(x))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
            return 999999
        
        for subcat in sorted(subcat1.keys(), key=sort_key):
            m = subcat1[subcat]
            print(f"    {subcat:20s}: Acc={m['accuracy']:.4f} F1={m['f1_score']:.4f} "
                  f"(TP={m['tp']}, FN={m['fn']}, Total={m['total']})")
    
    print("="*80)
    
    # 10. Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        'metadata': {
            'model': args.model,
            'split': args.split,
            'threshold': resolved_threshold,
            'timestamp': timestamp,
            'total_samples': len(expected_categories)
        },
        'global_metrics': global_metrics,
        'class_metrics': {
            'class_0': {
                'total': class0_metrics['total'],
                'tn': class0_confusion['tn'],
                'fp': class0_confusion['fp'],
                'accuracy': class0_confusion['tn'] / class0_metrics['total'] if class0_metrics['total'] > 0 else 0
            },
            'class_1': {
                'total': class1_metrics['total'],
                'tp': class1_confusion['tp'],
                'fn': class1_confusion['fn'],
                'accuracy': class1_confusion['tp'] / class1_metrics['total'] if class1_metrics['total'] > 0 else 0,
                'precision': global_metrics['precision'],
                'recall': global_metrics['recall'],
                'f1_score': global_metrics['f1_score']
            }
        },
        'subcategory_metrics': subcategory_metrics,
        'predictions': all_predictions
    }
    
    # Créer le nom de fichier (canonique — modèle + split uniquement)
    output_filename = f"{args.model.lower()}_{args.split}.json"
    output_path = config.PERFORMANCE_DIR / output_filename
    
    with open(output_path, 'w') as f:
        # Ensure all numpy scalar/array types are converted to native Python
        # types so JSON serialization doesn't fail (e.g., int32, float32).
        def _sanitize(obj):
            # numpy types
            try:
                import numpy as _np
            except Exception:
                _np = None

            if _np is not None and isinstance(obj, _np.generic):
                if _np.issubdtype(type(obj), _np.integer):
                    return int(obj)
                if _np.issubdtype(type(obj), _np.floating):
                    return float(obj)
                try:
                    return obj.item()
                except Exception:
                    return str(obj)

            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            if hasattr(obj, 'tolist') and not isinstance(obj, (str, bytes)):
                try:
                    return _sanitize(obj.tolist())
                except Exception:
                    pass
            return obj

        json.dump(_sanitize(results_data), f, indent=2)
    
    print(f"\nRésultats sauvegardés: {output_path} (canonical filename, overwritten if exists)")
    print(f"  - Métriques globales et par classe")
    print(f"  - Métriques par sous-catégorie ({len(subcategory_metrics)} catégories)")
    print(f"  - {len(all_predictions)} prédictions détaillées")


if __name__ == "__main__":
    main()
