"""
Performance by Distance Analysis - Dynamic Configuration Version
Analyzes REAL model performance as a function of simulated drone distance (SNR categories).

This script:
1. Dynamically loads augmentation categories from augment_config_v2.json
2. Tests trained models on each category separately
3. Generates real performance metrics by distance/SNR
4. Adapts to any future configuration changes

Features:
- Automatic category detection from config
- Real accuracy by SNR category
- Performance degradation analysis
- Per-model robustness comparison
- Confusion matrices by distance
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
import librosa
import math
import warnings
import re
import argparse

# Suppress warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal of length=.*')

# Import project config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Import ML libraries
try:
    from tensorflow import keras
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
except ImportError:
    print("[ERROR] TensorFlow/scikit-learn not installed")
    sys.exit(1)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Paths
AUGMENT_CONFIG_PATH = Path(__file__).parent.parent / "0 - DADS dataset extraction" / "augment_config_v3.json"
DATASET_COMBINED_PATH = Path(__file__).parent.parent / "0 - DADS dataset extraction" / "dataset_combined"
FEATURES_JSON_PATH = Path(__file__).parent.parent / "0 - DADS dataset extraction" / "extracted_features" / "mel_pitch_shift_9.0.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"
TEST_INDEX_PATH = Path(__file__).parent.parent / "0 - DADS dataset extraction" / "extracted_features" / "mel_test_index.json"


def load_augmentation_config():
    """Load augmentation configuration to get dynamic categories."""
    if not AUGMENT_CONFIG_PATH.exists():
        print(f"[ERROR] Config not found: {AUGMENT_CONFIG_PATH}")
        return None
    
    try:
        with open(AUGMENT_CONFIG_PATH, 'r') as f:
            config_data = json.load(f)
        
        print(f"[OK] Loaded augmentation config v{config_data.get('version', 'unknown')}")
        return config_data
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return None


def get_categories_from_config(aug_config):
    """
    Extract category definitions from augmentation config.
    
    Returns dict with category info:
    {
        'drone_500m': {
            'name': 'drone_500m',
            'display_name': '500m (-32dB)',
            'snr_db': -32,
            'proportion': 0.40,
            'description': '...',
            'label': 1
        },
        ...
    }
    """
    if not aug_config:
        return {}
    
    categories = {}
    
    # Extract drone augmentation categories
    if aug_config.get('drone_augmentation', {}).get('enabled'):
        for cat in aug_config['drone_augmentation']['categories']:
            name = cat['name']
            snr = cat['snr_db']
            
            # Parse distance from name (e.g., 'drone_500m' -> '500m')
            distance = name.replace('drone_', '').replace('_', ' ')
            
            categories[name] = {
                'name': name,
                'display_name': f"{distance} ({snr:+d}dB)".replace('+-', '-'),
                'snr_db': snr,
                'proportion': cat.get('proportion', 0),
                'description': cat.get('description', ''),
                'label': cat.get('label', 1),
                'file_pattern': f"aug_{name}_"
            }
    
    # Extract no-drone augmentation categories (if any)
    if aug_config.get('no_drone_augmentation', {}).get('enabled'):
        for cat in aug_config['no_drone_augmentation']['categories']:
            name = cat['name']
            snr = cat.get('snr_db', 0)
            
            categories[name] = {
                'name': name,
                'display_name': cat.get('display_name', name),
                'snr_db': snr,
                'proportion': cat.get('proportion', 0),
                'description': cat.get('description', ''),
                'label': cat.get('label', 0),
                'file_pattern': f"aug_{name}_"
            }
    
    # Add original clean samples category
    categories['original_clean'] = {
        'name': 'original_clean',
        'display_name': 'Original (Clean)',
        'snr_db': float('inf'),  # Perfect quality
        'proportion': 0,
        'description': 'Original unaugmented samples',
        'label': None,  # Will contain both classes
        'file_pattern': 'orig_'
    }
    
    return categories


def detect_categories_from_files(categories):
    """Detect additional categories present in the dataset files that are not declared in the config.

    This scans `DATASET_COMBINED_PATH` (both class folders) for file name patterns like
    `aug_drone_600m_...wav` and registers a category `drone_600m` when found.
    For detected drone distance categories we set a synthetic `snr_db` value equal to
    negative distance (e.g. 600m -> -600) so ordering remains consistent (more negative = harder).
    """
    detected = set()

    if not DATASET_COMBINED_PATH.exists():
        return categories

    pattern = re.compile(r"aug_([a-z0-9_]+)_")

    for class_dir in ['0', '1']:
        class_path = DATASET_COMBINED_PATH / class_dir
        if not class_path.exists():
            continue
        for f in class_path.glob('*.wav'):
            m = pattern.search(f.name)
            if m:
                detected.add(m.group(1))

    # Add detected categories that are missing in config
    for name in sorted(detected):
        if name in categories:
            continue

        # Build a display name
        display_name = name.replace('drone_', '').replace('_', ' ')

        # Attempt to extract distance in meters (drone_600m -> 600)
        distance_m = None
        m = re.search(r'drone_(\d+)m', name)
        if m:
            distance_m = int(m.group(1))
            # We DO NOT assume distance == dB. Keep snr_db unknown and store distance_m
            snr_db = float('nan')
            # Try to find snr_db in any available augment_config files (v2, v3, custom)
            try:
                cfg_dir = Path(__file__).parent.parent / "0 - DADS dataset extraction"
                for cfg_file in sorted(cfg_dir.glob('augment_config*.json')):
                    try:
                        with open(cfg_file, 'r') as cf:
                            cfg = json.load(cf)
                        # Look under drone_augmentation.categories or drone_augmentation.categories list
                        drone_section = cfg.get('drone_augmentation', {})
                        cats = drone_section.get('categories', [])
                        for cat in cats:
                            if cat.get('name') == name:
                                snr_db = cat.get('snr_db', float('nan'))
                                break
                        if np.isfinite(snr_db) or (not np.isnan(snr_db) and snr_db is not None):
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        else:
            snr_db = float('nan')

        if distance_m is not None:
            if np.isfinite(snr_db):
                display = f"{distance_m}m ({int(snr_db):+d}dB)"
            else:
                display = f"{distance_m}m (distance only)"

        categories[name] = {
            'name': name,
            'display_name': display,
            'snr_db': snr_db,
            'proportion': 0,
            'description': 'Detected from dataset files (not present in config)',
            'label': 1 if name.startswith('drone_') else 0,
            'file_pattern': f"aug_{name}_",
            'distance_m': distance_m
        }

    return categories


def get_category_files(categories):
    """
    Get all audio files for each category from dataset_combined.
    
    Returns:
    {
        'category_name': {
            'files': [Path objects],
            'display_name': 'Display Name',
            'snr_db': -32,
            'expected_label': 0 or 1 or None
        }
    }
    """
    if not DATASET_COMBINED_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_COMBINED_PATH}")
        return {}
    
    category_files = {}
    
    for cat_name, cat_info in categories.items():
        pattern = cat_info['file_pattern']
        files = []
        
        # Search in both class folders (0 and 1)
        for class_dir in ['0', '1']:
            class_path = DATASET_COMBINED_PATH / class_dir
            if class_path.exists():
                # Find files matching pattern
                matching_files = list(class_path.glob(f"{pattern}*.wav"))
                files.extend(matching_files)
        
        if files:
            category_files[cat_name] = {
                'files': sorted(files),
                'display_name': cat_info['display_name'],
                'snr_db': cat_info['snr_db'],
                'expected_label': cat_info['label']
            }
    
    return category_files


def load_precomputed_features():
    """Load pre-computed MEL features from JSON file."""
    # Prefer the per-file test index (one MEL per audio file) when available.
    if TEST_INDEX_PATH.exists():
        try:
            print(f"[INFO] Loading test-indexed features from {TEST_INDEX_PATH}...")
            with open(TEST_INDEX_PATH, 'r') as f:
                data = json.load(f)

            # Expect keys: 'mapping' (optional), 'names', 'mel', 'labels' (optional)
            print(f"[OK] Loaded test index with {len(data.get('names', []))} entries")
            return {'test_index': data}
        except Exception as e:
            print(f"[WARNING] Failed to load test index {TEST_INDEX_PATH}: {e}")

    # If we reach here, no test-index present. If PRECOMPUTED_ONLY is enforced, abort.
    if getattr(config, 'PRECOMPUTED_ONLY', False):
        print(f"[ERROR] PRECOMPUTED_ONLY mode enabled but test index not found at {TEST_INDEX_PATH}")
        return None

    # Fallback to the large segmented features JSON (multiple segments per file)
    if not FEATURES_JSON_PATH.exists():
        print(f"[ERROR] Features JSON not found: {FEATURES_JSON_PATH}")
        return None

    try:
        print(f"[INFO] Loading pre-computed features from {FEATURES_JSON_PATH}...")
        with open(FEATURES_JSON_PATH, 'r') as f:
            data = json.load(f)

        mel_features = np.array(data.get('mel', []))  # Shape: (N_segments, 44, 90)
        labels = np.array(data.get('labels', []))

        print(f"[OK] Loaded {len(mel_features)} pre-computed MEL feature segments")
        if len(mel_features) > 0:
            print(f"[INFO] Feature shape: {mel_features[0].shape}")

        return {'features': mel_features, 'labels': labels, 'mapping': data.get('mapping')}
    except Exception as e:
        print(f"[ERROR] Failed to load features: {e}")
        return None


def load_models():
    """Load trained models."""
    models = {}
    
    model_paths = {
        'CNN': config.CNN_MODEL_PATH,
        'RNN': config.RNN_MODEL_PATH,
        'CRNN': config.CRNN_MODEL_PATH,
        'Attention-CRNN': config.ATTENTION_CRNN_MODEL_PATH
    }
    
    for name, path in model_paths.items():
        if path.exists():
            try:
                # Try to load with fallback custom_objects for custom loss function if present
                custom_objects = None
                try:
                    from loss_functions import get_loss_function
                    fallback_loss = get_loss_function('recall_focused', fn_penalty=50.0)
                    custom_objects = {'loss_fn': fallback_loss}
                except Exception:
                    custom_objects = None

                if custom_objects:
                    models[name] = keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                else:
                    models[name] = keras.models.load_model(path, compile=False)
                print(f"[OK] Loaded {name} model")
            except Exception as e:
                print(f"[WARNING] Failed to load {name}: {e}")
        else:
            print(f"[WARNING] Model not found: {path}")
    
    return models


def get_precomputed_features_for_file(file_path, precomputed_data, num_segments=10):
    """Get MEL features using EXACT same parameters as training.
    
    CRITICAL: Training uses 4 seconds with shape (44, 90).
    Test MUST match this exactly or we get catastrophic failure.
    
    Args:
        file_path: Path to audio file
        precomputed_data: Not used (kept for compatibility)
        num_segments: Ignored - we use single 4-second window like training
    
    Returns:
        Array of shape (44, 90) - SINGLE MEL spectrogram matching training
    """
    n_mels = 44  # MATCH training (not 90!)
    n_fft = 2048
    hop_length = 512
    
    # ONLY use precomputed per-file MELs. Do NOT compute on-the-fly here.
    if not precomputed_data or 'test_index' not in precomputed_data:
        raise RuntimeError(
            "Precomputed test index not available. Rebuild `mel_test_index.json` and retry."
        )

    test_index = precomputed_data['test_index']
    names = test_index.get('names', [])
    if not names:
        raise RuntimeError("Precomputed test index contains no names; rebuild `mel_test_index.json`.")

    fname = file_path.name
    try:
        idx = names.index(fname)
    except ValueError:
        raise RuntimeError(f"File '{fname}' not found in precomputed test index. Rebuild the index to include this file.")

    mel_db = np.array(test_index['mel'][idx]).astype(float)

    # Enforce exact trainer shape: (n_mels, 90)
    expected_shape = (n_mels, 90)
    if mel_db.shape != expected_shape:
        raise RuntimeError(
            f"Precomputed MEL for '{fname}' has shape {mel_db.shape}; expected {expected_shape}.\n"
            "Do not modify precomputed MELs in-place — rebuild `mel_test_index.json` using the trainer's extraction settings."
        )

    return mel_db


def evaluate_model_on_category(model, model_name, files, precomputed_data, feature_type='mel', thresholds=None):
    """
    Evaluate a model on files from a specific category.
    
    Args:
        model: Trained model
        model_name: Name of model (CNN, RNN, CRNN)
        files: List of Path objects to audio files
        precomputed_data: Pre-computed features dict (not used, kept for compatibility)
        feature_type: 'mel' or 'mfcc'
    
    Returns:
        Dictionary with accuracy, precision, recall, f1 metrics
    """
    print(f"  Evaluating {len(files)} files... ", end='', flush=True)
    
    predictions = []
    labels = []
    file_names = []
    
    for file_path in files:
        try:
            # Get true label from parent directory name
            true_label = int(file_path.parent.name)
            
            # Get features - shape (44, 90) matching training exactly
            if feature_type == 'mel':
                features = get_precomputed_features_for_file(file_path, precomputed_data)
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            
            # Reshape for model input - add batch and channel dimensions
            # Training shape: (batch, 44, 90, 1)
            # Test must match: (1, 44, 90, 1)
            if model_name in ['CNN', 'CRNN', 'Attention-CRNN']:
                features = features[np.newaxis, ..., np.newaxis]  # (44, 90) -> (1, 44, 90, 1)
            else:  # RNN
                features = features[np.newaxis, ...]  # (44, 90) -> (1, 44, 90)
            
            # Predict - single prediction, no averaging
            pred = model.predict(features, verbose=0)

            # Extract prediction from batch dimension
            pred = pred[0]  # (1, 2) -> (2) or (1)

            # If thresholds provided, apply per-model threshold on the positive-class score
            pred_label = None
            if thresholds and model_name in thresholds:
                # Extract positive-class score
                p = np.array(pred)
                if p.size == 1:
                    pos = float(p[0])
                else:
                    pos = float(p[1])
                thr = thresholds[model_name].get('best_threshold', 0.5)
                pred_label = int(pos >= thr)
                predicted_index = 1 if pred_label == 1 else 0
            else:
                # Use hard argmax-style prediction (legacy behavior)
                try:
                    predicted_index = int(np.argmax(pred))
                except Exception:
                    predicted_index = int(np.array(pred) >= 0.5)
                pred_label = int(predicted_index)

            # Debug first few predictions (show raw scores and chosen index)
            if len(predictions) < 3:
                try:
                    raw_scores_display = np.array(pred)
                except Exception:
                    raw_scores_display = pred
                print(f"\n  [DEBUG] File: {file_path.name}, True label: {true_label}")
                print(f"  [DEBUG] Features shape: {features.shape}, raw_scores: {raw_scores_display}, predicted_index: {predicted_index}, pred_label: {pred_label}")
            
            predictions.append(pred_label)
            labels.append(true_label)
            file_names.append(file_path.name)
            
        except Exception as e:
            print(f"\n[WARNING] Error processing {file_path.name}: {e}")
            continue
    
    if not predictions:
        print("No valid predictions")
        return None
    
    # Debug: Show prediction distribution
    pred_array = np.array(predictions)
    label_array = np.array(labels)
    print(f"\n  [DEBUG] Predictions: {np.sum(pred_array == 0)} no-drones, {np.sum(pred_array == 1)} drones")
    print(f"  [DEBUG] True labels: {np.sum(label_array == 0)} no-drones, {np.sum(label_array == 1)} drones")
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Handle cases where category has only one class
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        # Single class - use simple accuracy
        precision = accuracy
        recall = accuracy
        f1 = accuracy
    else:
        # Binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, 
                                                                    average='binary', 
                                                                    zero_division=0)
    
    print(f"  Accuracy: {accuracy*100:.1f}%, Recall: {recall*100:.1f}%, Precision: {precision*100:.1f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'labels': labels,
        'file_names': file_names
    }


def test_models_by_distance(categories, category_files, force_retest=False, thresholds=None):
    """Test all models on each distance category."""
    print("[1/3] Loading pre-computed features...")
    precomputed_data = load_precomputed_features()
    
    if precomputed_data is None:
        print("[ERROR] Failed to load pre-computed features")
        return None
    
    print("\n[2/3] Loading models...")
    models = load_models()
    
    if not models:
        print("[ERROR] No models loaded")
        return None
    
    print(f"\n[3/3] Testing on {len(category_files)} categories...")
    
    # Test each model on each category
    results = defaultdict(dict)
    
    for model_name, model in models.items():
        print(f"\nTesting {model_name}:")
        print("-" * 60)
        
        # All models use mel spectrograms in this project
        feature_type = 'mel'
        
        # If a per-model by-category cached file exists and re-test not forced, load it
        model_key = model_name.lower().replace('-', '_')
        by_category_path = config.PREDICTIONS_DIR / f"{model_key}_by_category.json"
        if by_category_path.exists() and not force_retest:
            try:
                with open(by_category_path, 'r') as f:
                    loaded = json.load(f)
                # loaded should be a mapping cat_name -> results per category
                results[model_name] = loaded
                print(f"[SKIP] Loaded cached per-category results for {model_name}: {by_category_path}")
                continue
            except Exception as e:
                print(f"[WARNING] Failed to load cached results {by_category_path}: {e}")

        # Sort categories in SNR-proportional order for testing: finite snr_db first (ascending),
        # then detected distances (farther first), then any remaining categories.
        def _cat_sort(item):
            cat_name, cat_data = item
            snr = cat_data.get('snr_db', float('nan'))
            dist = cat_data.get('distance_m', None)
            if np.isfinite(snr):
                return (0, float(snr))
            elif dist is not None:
                return (1, -int(dist))
            else:
                return (2, 0)

        sorted_cats = sorted(list(category_files.items()), key=_cat_sort)

        for cat_name, cat_data in sorted_cats:
            files = cat_data['files']
            display_name = cat_data['display_name']

            print(f"{display_name:25s} ({len(files):4d} files): ", end='')

            # Limit to 100 files for speed
            result = evaluate_model_on_category(model, model_name, files[:100], precomputed_data, feature_type, thresholds=thresholds)
            
            if result:
                results[model_name][cat_name] = {
                    **result,
                    'display_name': display_name,
                    'snr_db': cat_data['snr_db']
                }

        # After testing (or loading cached results), produce canonical outputs for this model
        try:
            # Build aggregated lists in the same order as categories were iterated
            agg_names = []
            agg_preds = []
            agg_labels = []

            # Use the same SNR-proportional ordering for aggregation to keep canonical outputs consistent
            ordered_cat_names = [cn for cn, _ in sorted_cats]
            for cat_name in ordered_cat_names:
                if cat_name in results[model_name]:
                    entry = results[model_name][cat_name]
                    # Expect entry to contain 'file_names', 'predictions', 'labels'
                    fns = entry.get('file_names') or []
                    ps = entry.get('predictions') or []
                    ls = entry.get('labels') or []
                    agg_names.extend(fns)
                    agg_preds.extend(ps)
                    agg_labels.extend(ls)

            # If we have no aggregated predictions, skip writing
            if agg_names:
                model_key = model_name.lower().replace('-', '_')
                # Prediction file (names, mapping, predictions, labels)
                pred_out = {
                    'mapping': ['0', '1'],
                    'names': agg_names,
                    'predictions': agg_preds,
                    'labels': agg_labels
                }

                pred_path = getattr(config, f"{model_key.upper()}_PREDICTIONS_PATH", None)
                if pred_path is None:
                    pred_path = config.PREDICTIONS_DIR / f"{model_key}_predictions.json"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                with open(pred_path, 'w') as f:
                    json.dump(pred_out, f, indent=4)
                print(f"[OK] Saved predictions: {pred_path}")

                # Scores file (newer flat format with confusion_matrix dict)
                TP = int(sum(1 for p, l in zip(agg_preds, agg_labels) if p == 1 and l == 1))
                FN = int(sum(1 for p, l in zip(agg_preds, agg_labels) if p == 0 and l == 1))
                TN = int(sum(1 for p, l in zip(agg_preds, agg_labels) if p == 0 and l == 0))
                FP = int(sum(1 for p, l in zip(agg_preds, agg_labels) if p == 1 and l == 0))

                total = TP + TN + FP + FN
                accuracy = float((TP + TN) / total) if total > 0 else 0.0
                precision = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
                recall = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
                f1 = float((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)

                scores_out = {
                    'confusion_matrix': {'TP': TP, 'FN': FN, 'TN': TN, 'FP': FP},
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }

                scores_path = getattr(config, f"{model_key.upper()}_SCORES_PATH", None)
                if scores_path is None:
                    scores_path = config.PREDICTIONS_DIR / f"{model_key}_scores.json"
                with open(scores_path, 'w') as f:
                    json.dump(scores_out, f, indent=4)
                print(f"[OK] Saved scores: {scores_path}")

                # Accuracy file (visualizer format)
                # Build confusion matrix in visualizer orientation [[TN, FP],[FN, TP]]
                confusion_matrix = [[TN, FP], [FN, TP]]

                # classification_report similar to convert_results_for_viz
                classification_report = {
                    "0": {
                        "precision": (TN / (TN + FN)) if (TN + FN) > 0 else 0,
                        "recall": (TN / (TN + FP)) if (TN + FP) > 0 else 0,
                        "f1-score": 0,
                        "support": int(TN + FP)
                    },
                    "1": {
                        "precision": precision,
                        "recall": recall,
                        "f1-score": f1,
                        "support": int(TP + FN)
                    },
                    "accuracy": accuracy,
                    "macro avg": {
                        "precision": 0,
                        "recall": 0,
                        "f1-score": 0,
                        "support": int(total)
                    },
                    "weighted avg": {
                        "precision": precision,
                        "recall": recall,
                        "f1-score": f1,
                        "support": int(total)
                    }
                }

                # Fill class 0 f1 and macro avg
                if classification_report["0"]["precision"] > 0 and classification_report["0"]["recall"] > 0:
                    classification_report["0"]["f1-score"] = (
                        2 * classification_report["0"]["precision"] * classification_report["0"]["recall"] /
                        (classification_report["0"]["precision"] + classification_report["0"]["recall"])
                    )

                classification_report["macro avg"]["precision"] = (
                    classification_report["0"]["precision"] + classification_report["1"]["precision"]
                ) / 2
                classification_report["macro avg"]["recall"] = (
                    classification_report["0"]["recall"] + classification_report["1"]["recall"]
                ) / 2
                classification_report["macro avg"]["f1-score"] = (
                    classification_report["0"]["f1-score"] + classification_report["1"]["f1-score"]
                ) / 2

                accuracy_out = {
                    "model_name": model_name,
                    "test_accuracy": accuracy,
                    "test_loss": None,
                    "confusion_matrix": confusion_matrix,
                    "classification_report": classification_report,
                    "metrics": {
                        "TP": TP,
                        "FN": FN,
                        "TN": TN,
                        "FP": FP,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    }
                }

                acc_path = getattr(config, f"{model_key.upper()}_ACC_PATH", None)
                if acc_path is None:
                    acc_path = config.RESULTS_DIR / f"{model_key}_accuracy.json"
                acc_path.parent.mkdir(parents=True, exist_ok=True)
                with open(acc_path, 'w') as f:
                    json.dump(accuracy_out, f, indent=4)
                print(f"[OK] Saved accuracy: {acc_path}")

                # Write by-category cache so subsequent runs can skip heavy inference
                by_category_path = config.PREDICTIONS_DIR / f"{model_key}_by_category.json"
                with open(by_category_path, 'w') as f:
                    json.dump(results[model_name], f, indent=4)
                print(f"[OK] Saved per-category cache: {by_category_path}")

        except Exception as e:
            print(f"[WARNING] Failed to write canonical outputs for {model_name}: {e}")
    
    return results


def generate_performance_table(results, categories):
    """Generate detailed performance table by distance."""
    if not results:
        return None
    
    # Sort categories in a robust way:
    # 1) categories with finite snr_db sorted by snr_db (worst/most-negative first)
    # 2) categories detected from filenames with a distance_m are placed after, sorted by distance (farther first)
    def _sort_key(item):
        cat_info = item[1]
        snr = cat_info.get('snr_db', float('nan'))
        dist = cat_info.get('distance_m', None)
        if np.isfinite(snr):
            return (0, float(snr))
        elif dist is not None:
            # place detected distances after configured SNRs, order by distance descending
            return (1, -int(dist))
        else:
            return (2, 0)

    sorted_cats = sorted([(cat_name, cat_info) for cat_name, cat_info in categories.items()], key=_sort_key)
    
    table_data = []
    
    for cat_name, cat_info in sorted_cats:
        # Check if this category has results
        has_results = any(cat_name in model_results for model_results in results.values())
        if not has_results:
            continue
        
        row = {'Category': cat_info['display_name']}
        
        for model_name, model_results in results.items():
            if cat_name in model_results:
                res = model_results[cat_name]
                row[f'{model_name} Acc (%)'] = f"{res['accuracy']*100:.2f}"
                row[f'{model_name} F1 (%)'] = f"{res['f1']*100:.2f}"
        
        if len(row) > 1:  # Has data beyond category name
            table_data.append(row)
    
    if not table_data:
        return None
    
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "performance_by_distance.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved: {csv_path}")
    
    # Print formatted table
    print("\n" + "=" * 100)
    print("REAL PERFORMANCE BY DISTANCE/SNR")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    return df


def plot_performance_by_distance(results, categories):
    """Plot real model performance by distance category."""
    if not results:
        print("[WARNING] No results to plot")
        return None
    
    # Sort categories by SNR (worst to best)
    sorted_cats = sorted(
        [(cat_name, cat_info) for cat_name, cat_info in categories.items()],
        key=lambda x: x[1]['snr_db']
    )
    
    # Filter to categories with results
    cat_order = [
        (cat_name, cat_info) 
        for cat_name, cat_info in sorted_cats 
        if any(cat_name in model_results for model_results in results.values())
    ]
    
    if not cat_order:
        print("[WARNING] No categories with results")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract display names and SNRs
    display_names = [cat_info['display_name'] for _, cat_info in cat_order]
    cat_names = [cat_name for cat_name, _ in cat_order]
    snr_values = [cat_info['snr_db'] for _, cat_info in cat_order]
    
    # Plot 1: Accuracy by category
    ax1 = axes[0, 0]
    x = np.arange(len(cat_names))
    width = 0.25
    
    colors = {'CNN': 'steelblue', 'RNN': 'coral', 'CRNN': 'lightgreen'}
    
    for i, (model_name, model_results) in enumerate(results.items()):
        accuracies = [
            model_results[cat_name]['accuracy'] * 100 if cat_name in model_results else 0
            for cat_name in cat_names
        ]
        ax1.bar(x + i*width, accuracies, width, label=model_name, color=colors.get(model_name, 'gray'))
    
    ax1.set_xlabel('Distance Category')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy by Drone Distance', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Plot 2: F1-Score by category
    ax2 = axes[0, 1]
    
    for i, (model_name, model_results) in enumerate(results.items()):
        f1_scores = [
            model_results[cat_name]['f1'] * 100 if cat_name in model_results else 0
            for cat_name in cat_names
        ]
        ax2.bar(x + i*width, f1_scores, width, label=model_name, color=colors.get(model_name, 'gray'))
    
    ax2.set_xlabel('Distance Category')
    ax2.set_ylabel('F1-Score (%)')
    ax2.set_title('Model F1-Score by Drone Distance', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(display_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Performance vs SNR (line plot)
    ax3 = axes[1, 0]
    
    # Filter out infinite SNR (original clean)
    finite_snr_indices = [i for i, snr in enumerate(snr_values) if np.isfinite(snr)]
    
    if finite_snr_indices:
        finite_snrs = [snr_values[i] for i in finite_snr_indices]
        finite_cats = [cat_names[i] for i in finite_snr_indices]
        
        for model_name, model_results in results.items():
            accuracies = [
                model_results[cat_name]['accuracy'] * 100 if cat_name in model_results else 0
                for cat_name in finite_cats
            ]
            ax3.plot(finite_snrs, accuracies, marker='o', linewidth=2, 
                    label=model_name, color=colors.get(model_name, 'gray'))
    
    ax3.set_xlabel('SNR (dB)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Performance vs. Drone Distance (SNR)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # Plot 4: Performance degradation
    ax4 = axes[1, 1]
    
    if finite_snr_indices and len(finite_snr_indices) > 1:
        # Calculate degradation from best to worst SNR
        for model_name, model_results in results.items():
            accuracies = [
                model_results[cat_name]['accuracy'] * 100 if cat_name in model_results else 0
                for cat_name in finite_cats
            ]
            
            # Degradation relative to best performance
            best_acc = max(accuracies) if accuracies else 0
            degradations = [best_acc - acc for acc in accuracies]
            
            ax4.plot(finite_snrs, degradations, marker='s', linewidth=2,
                    label=model_name, color=colors.get(model_name, 'gray'))
    
    ax4.set_xlabel('SNR (dB)')
    ax4.set_ylabel('Performance Degradation (%)')
    ax4.set_title('Model Robustness to Noise (Lower is Better)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "real_performance_by_distance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {output_path}")
    
    plt.close()
    return output_path


def plot_difficulty_spectrum(categories, category_files):
    """Plot the difficulty spectrum showing SNR levels and sample counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort categories by SNR
    sorted_cats = sorted(
        [(cat_name, cat_info) for cat_name, cat_info in categories.items()],
        key=lambda x: x[1]['snr_db']
    )
    
    # Filter to categories with files
    filtered_cats = [
        (cat_name, cat_info) 
        for cat_name, cat_info in sorted_cats 
        if cat_name in category_files
    ]
    
    if not filtered_cats:
        print("[WARNING] No categories to plot")
        return
    
    display_names = [cat_info['display_name'] for _, cat_info in filtered_cats]
    cat_names = [cat_name for cat_name, _ in filtered_cats]
    snr_values = [cat_info['snr_db'] if np.isfinite(cat_info['snr_db']) else 0 for _, cat_info in filtered_cats]
    file_counts = [len(category_files[cat_name]['files']) for cat_name in cat_names]
    
    # Plot 1: SNR levels
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cat_names)))
    
    bars1 = ax1.bar(display_names, snr_values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('Difficulty Spectrum: SNR by Category', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='SNR = 0dB')
    ax1.legend()
    
    # Add value labels on bars
    for bar, snr in zip(bars1, snr_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{snr:+.0f}dB' if np.isfinite(snr) else 'Clean',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Sample distribution
    bars2 = ax2.bar(display_names, file_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Dataset Distribution by Category', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars2, file_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "difficulty_spectrum.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    
    plt.close()


def main(force_retest=None):
    """Main execution function."""
    print("=" * 80)
    print("REAL PERFORMANCE BY DISTANCE ANALYSIS")
    print("=" * 80)
    print("\nTesting trained models on each SNR category...")
    print("This will take several minutes...\n")
    
    # Determine force_retest: prefer explicit parameter, else parse CLI
    if force_retest is None:
        parser = argparse.ArgumentParser(description='Performance by distance analysis')
        parser.add_argument('--force-retest', action='store_true', help='Force re-running model inference even if cached results exist')
        args = parser.parse_args()
        force_retest = args.force_retest
    force_retest = True
    # Load configuration
    print("[STEP 1/5] Loading augmentation configuration...")
    aug_config = load_augmentation_config()
    if not aug_config:
        print("[ERROR] Cannot proceed without configuration")
        return
    
    # Extract categories from config
    print("\n[STEP 2/5] Extracting category definitions...")
    categories = get_categories_from_config(aug_config)

    # Detect additional categories present in dataset files (fallback)
    categories = detect_categories_from_files(categories)

    print(f"[OK] Found {len(categories)} categories (config + detected):")
    for cat_name, cat_info in categories.items():
        print(f"  - {cat_info['display_name']}: {cat_info.get('description', 'N/A')[:60]}")
    
    # Get files for each category
    print(f"\n[STEP 3/5] Scanning dataset files...")
    category_files = get_category_files(categories)
    print(f"[OK] Found files for {len(category_files)} categories:")
    for cat_name, cat_data in category_files.items():
        print(f"  - {cat_data['display_name']}: {len(cat_data['files'])} files")
    
    if not category_files:
        print("[ERROR] No category files found!")
        return
    
    # Test models
    print(f"\n[STEP 4/5] Testing models by distance category...")

    # Load optional calibrated thresholds (auto-run calibration if enabled and not present)
    thresholds_path = Path(__file__).parent / 'outputs' / 'model_thresholds.json'
    thresholds = None
    if thresholds_path.exists():
        try:
            with open(thresholds_path, 'r') as f:
                thresholds = json.load(f)
            print(f"[OK] Loaded model thresholds from: {thresholds_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load thresholds: {e}")
            thresholds = None
    else:
        if getattr(config, 'AUTO_CALIBRATE_THRESHOLDS', False):
            print("[INFO] No thresholds file found. Running automatic threshold calibration...")
            try:
                # Import and run the calibration script
                from . import threshold_calibration as tc
                tc.main()
                with open(thresholds_path, 'r') as f:
                    thresholds = json.load(f)
                print(f"[OK] Calibration complete, loaded thresholds from: {thresholds_path}")
            except Exception as e:
                print(f"[WARNING] Auto-calibration failed: {e}")
                thresholds = None

    results = test_models_by_distance(categories, category_files, force_retest=force_retest, thresholds=thresholds)
    
    if not results:
        print("[ERROR] No results generated")
        return
    
    # Generate outputs
    print(f"\n[STEP 5/5] Generating performance analysis...")
    
    print("\n[5a/5] Generating performance table...")
    generate_performance_table(results, categories)
    
    print("\n[5b/5] Plotting performance by distance...")
    plot_performance_by_distance(results, categories)
    
    print("\n[5c/5] Plotting difficulty spectrum...")
    plot_difficulty_spectrum(categories, category_files)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Real performance by distance analysis complete!")
    print("=" * 80)
    print(f"\nOutputs saved in: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
