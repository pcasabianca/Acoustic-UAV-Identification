"""
Project Configuration File
Centralized path configuration for all scripts in the project.
Works on both Linux and Windows.
"""

import os
from pathlib import Path
import math

# Project root directory (where this config.py file is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Dataset paths
# Use DATASET_ROOT_OVERRIDE environment variable to switch between datasets
# Options: "dataset_test" (original), "dataset_augmented" (augmented only), "dataset_combined" (both)
DEFAULT_DATASET = "dataset_combined"  # Use combined dataset by default
DATASET_NAME = os.environ.get("DATASET_ROOT_OVERRIDE", DEFAULT_DATASET)
DATASET_ROOT = PROJECT_ROOT / "0 - DADS dataset extraction" / DATASET_NAME

# Canonical dataset folders (centralized names used across scripts)
EXTRACTION_DIR = PROJECT_ROOT / "0 - DADS dataset extraction"
# Alias expected by some documentation: data extraction root
DATA_EXTRACTION_ROOT = EXTRACTION_DIR
DATASET_DADS_DIR = EXTRACTION_DIR / "dataset_dads"
DATASET_DADS_OFFLINE_DIR = EXTRACTION_DIR / "dataset_DADS_offline"  # Full DADS dataset for offline use
DATASET_AUGMENTED_DIR = EXTRACTION_DIR / "dataset_augmented"
DATASET_COMBINED_DIR = EXTRACTION_DIR / "dataset_combined"
DATASET_TRAIN_DIR = EXTRACTION_DIR / "dataset_train"
DATASET_VAL_DIR = EXTRACTION_DIR / "dataset_val"
DATASET_TEST_DIR = EXTRACTION_DIR / "dataset_test"

# v4 dataset canonical location (versioned and stored inside extraction dir)
# Use V4_ACTIVE_DATASET_ID env var to redirect pipeline consumers to a v4 dataset
# NOTE: Do NOT create the directory at import time - let scripts create it when needed
V4_DATASETS_DIR = EXTRACTION_DIR / "datasets_v4"

# Optional: allow choosing an active v4 dataset via env var. When set,
# DATASET_TRAIN_DIR/VAL/TEST and DATASET_ROOT will point into that v4 dataset
V4_ACTIVE_DATASET_ID = os.environ.get('V4_ACTIVE_DATASET_ID', '').strip()
if V4_ACTIVE_DATASET_ID:
    _v4_root = V4_DATASETS_DIR / V4_ACTIVE_DATASET_ID
    DATASET_TRAIN_DIR = _v4_root / 'train'
    DATASET_VAL_DIR = _v4_root / 'val'
    DATASET_TEST_DIR = _v4_root / 'test'
    DATASET_ROOT = _v4_root

# Config of datasets used for training and validation
AUGMENTATION_CONFIG = EXTRACTION_DIR / 'build_config.json'
# Convenience config entries for augmentation entrypoint and config
# AUGMENTATION_CONFIG points to the single source-of-truth build_config.json
AUGMENTATION_WRAPPER = EXTRACTION_DIR / 'build_dataset_v5.py'

# Canonical pools used by augmentation (can be overridden via env vars)
# Example: export NOISE_POOL_DIR=/mnt/pools/noise
# If the env var is not set, defaults to EXTRACTION_DIR/pools/<name>
NOISE_POOL_DIR = Path(os.environ.get('NOISE_POOL_DIR')) if os.environ.get('NOISE_POOL_DIR') else (EXTRACTION_DIR / 'pools' / 'noise')
RIR_POOL_DIR = Path(os.environ.get('RIR_POOL_DIR')) if os.environ.get('RIR_POOL_DIR') else (EXTRACTION_DIR / 'pools' / 'rirs')
HARDNEG_POOL_DIR = Path(os.environ.get('HARDNEG_POOL_DIR')) if os.environ.get('HARDNEG_POOL_DIR') else (EXTRACTION_DIR / 'pools' / 'hardneg')
# Feature extraction paths
EXTRACTED_FEATURES_DIR = PROJECT_ROOT / "0 - DADS dataset extraction" / "extracted_features"

# Model save directory
MODELS_DIR = PROJECT_ROOT / "0 - DADS dataset extraction" / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

# Results save directory (canonical)
# The project uses the canonical results location inside the repository.
# Scripts MUST import `config.py` to discover this path. Do NOT rely on
# environment variables to override the results location.
RESULTS_DIR = PROJECT_ROOT / "0 - DADS dataset extraction" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CNN Model paths
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
CNN_HISTORY_PATH = RESULTS_DIR / "cnn_history.csv"
CNN_ACC_PATH = RESULTS_DIR / "cnn_accuracy.json"

# RNN Model paths
RNN_MODEL_PATH = MODELS_DIR / "rnn_model.keras"
RNN_HISTORY_PATH = RESULTS_DIR / "rnn_history.csv"
RNN_ACC_PATH = RESULTS_DIR / "rnn_accuracy.json"

# CRNN Model paths
CRNN_MODEL_PATH = MODELS_DIR / "crnn_model.keras"
CRNN_HISTORY_PATH = RESULTS_DIR / "crnn_history.csv"
CRNN_ACC_PATH = RESULTS_DIR / "crnn_accuracy.json"

# Attention-Enhanced CRNN Model paths
ATTENTION_CRNN_MODEL_PATH = MODELS_DIR / "attention_crnn_model.keras"
ATTENTION_CRNN_HISTORY_PATH = RESULTS_DIR / "attention_crnn_history.csv"
ATTENTION_CRNN_ACC_PATH = RESULTS_DIR / "attention_crnn_accuracy.json"

# EfficientNet-B0 Model paths
EFFICIENTNET_MODEL_PATH = MODELS_DIR / "efficientnet_model.keras"
EFFICIENTNET_HISTORY_PATH = RESULTS_DIR / "efficientnet_history.csv"
EFFICIENTNET_ACC_PATH = RESULTS_DIR / "efficientnet_accuracy.json"

# MobileNet-Audio Model paths
MOBILENET_MODEL_PATH = MODELS_DIR / "mobilenet_model.keras"
MOBILENET_HISTORY_PATH = RESULTS_DIR / "mobilenet_history.csv"
MOBILENET_ACC_PATH = RESULTS_DIR / "mobilenet_accuracy.json"

# Conformer Model paths
CONFORMER_MODEL_PATH = MODELS_DIR / "conformer_model.keras"
CONFORMER_HISTORY_PATH = RESULTS_DIR / "conformer_history.csv"
CONFORMER_ACC_PATH = RESULTS_DIR / "conformer_accuracy.json"

# TCN Model paths
TCN_MODEL_PATH = MODELS_DIR / "tcn_model.keras"
TCN_HISTORY_PATH = RESULTS_DIR / "tcn_history.csv"
TCN_ACC_PATH = RESULTS_DIR / "tcn_accuracy.json"

# All models accuracy path
ALL_MODELS_ACC_PATH = RESULTS_DIR / "all_model_acc.json"

# Ensemble/Voting model paths (10 models for late fusion)
ENSEMBLE_MODELS_DIR = MODELS_DIR / "ensemble"
ENSEMBLE_MODELS_DIR.mkdir(exist_ok=True)

ENSEMBLE_MODEL_PATHS = {
    1: ENSEMBLE_MODELS_DIR / "model_1.keras",
    2: ENSEMBLE_MODELS_DIR / "model_2.keras",
    3: ENSEMBLE_MODELS_DIR / "model_3.keras",
    4: ENSEMBLE_MODELS_DIR / "model_4.keras",
    5: ENSEMBLE_MODELS_DIR / "model_5.keras",
    6: ENSEMBLE_MODELS_DIR / "model_6.keras",
    7: ENSEMBLE_MODELS_DIR / "model_7.keras",
    8: ENSEMBLE_MODELS_DIR / "model_8.keras",
    9: ENSEMBLE_MODELS_DIR / "model_9.keras",
    10: ENSEMBLE_MODELS_DIR / "model_10.keras",
}

# Performance calculation paths
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

# Detailed performance metrics directory
PERFORMANCE_DIR = RESULTS_DIR / "performance"
PERFORMANCE_DIR.mkdir(exist_ok=True)

# Visualization output directory
VIZ_OUTPUT_DIR = PROJECT_ROOT / "6 - Visualization" / "outputs"
VIZ_OUTPUT_DIR.mkdir(exist_ok=True)

CNN_PREDICTIONS_PATH = PREDICTIONS_DIR / "cnn_predictions.json"
RNN_PREDICTIONS_PATH = PREDICTIONS_DIR / "rnn_predictions.json"
CRNN_PREDICTIONS_PATH = PREDICTIONS_DIR / "crnn_predictions.json"
ATTENTION_CRNN_PREDICTIONS_PATH = PREDICTIONS_DIR / "attention_crnn_predictions.json"

CNN_SCORES_PATH = PREDICTIONS_DIR / "cnn_scores.json"
RNN_SCORES_PATH = PREDICTIONS_DIR / "rnn_scores.json"
CRNN_SCORES_PATH = PREDICTIONS_DIR / "crnn_scores.json"
ATTENTION_CRNN_SCORES_PATH = PREDICTIONS_DIR / "attention_crnn_scores.json"

# Voting results paths
VOTING_DIR = RESULTS_DIR / "voting"
VOTING_DIR.mkdir(exist_ok=True)

VOTED_RESULTS_PATH = VOTING_DIR / "voted_results.json"
VOTING_WEIGHTS_PATH = VOTING_DIR / "weights.json"
VOTING_FINAL_SCORES_PATH = VOTING_DIR / "FINAL_SCORES.json"

# Individual voted results paths (for ensemble models)
VOTED_PATHS = {
    1: VOTING_DIR / "voted_1.json",
    2: VOTING_DIR / "voted_2.json",
    3: VOTING_DIR / "voted_3.json",
    4: VOTING_DIR / "voted_4.json",
    5: VOTING_DIR / "voted_5.json",
    6: VOTING_DIR / "voted_6.json",
    7: VOTING_DIR / "voted_7.json",
    8: VOTING_DIR / "voted_8.json",
    9: VOTING_DIR / "voted_9.json",
    10: VOTING_DIR / "voted_10.json",
}

SCORES_PATHS = {
    1: VOTING_DIR / "scores_1.json",
    2: VOTING_DIR / "scores_2.json",
    3: VOTING_DIR / "scores_3.json",
    4: VOTING_DIR / "scores_4.json",
    5: VOTING_DIR / "scores_5.json",
    6: VOTING_DIR / "scores_6.json",
    7: VOTING_DIR / "scores_7.json",
    8: VOTING_DIR / "scores_8.json",
    9: VOTING_DIR / "scores_9.json",
    10: VOTING_DIR / "scores_10.json",
}

# Audio parameters
SAMPLE_RATE = 22050
SAMPLE_RATE_HZ = SAMPLE_RATE
"""
Desired audio duration (seconds) used across the pipeline. Set this to
the target temporal length for all generated/processed WAV files.
Example: 4.0 -> all files should be normalized to 4 seconds.
"""

# Add a single source-of-truth duration constant and keep backwards
# compatible integer `DURATION` used by some scripts.
AUDIO_DURATION_S = 4.0

# Backwards-compatible `DURATION` (integer seconds)
DURATION = int(AUDIO_DURATION_S)
NUM_SEGMENTS = 1

# Parallel processing for augmentation
import os as _os
AUGMENTATION_MAX_WORKERS = int(_os.environ.get('AUGMENTATION_WORKERS', _os.cpu_count() or 4))

# TensorFlow CPU threading configuration (optimized for 12-core CPU)
# These can be overridden by environment variables TF_INTRA_OP_THREADS and TF_INTER_OP_THREADS
TF_INTRA_OP_PARALLELISM_THREADS = int(_os.environ.get('TF_INTRA_OP_THREADS', 10))  # Heavy ops (matmul, conv): use most cores
TF_INTER_OP_PARALLELISM_THREADS = int(_os.environ.get('TF_INTER_OP_THREADS', 2))   # Op coordination: keep low

# Training parameters (centralized)
MIN_EPOCHS = 25  # Minimum epochs before early stopping can trigger
MAX_EPOCHS = 250  # Maximum epochs limit
# Reduce default batch size to introduce more gradient noise and improve
# generalization on smaller datasets / limited augmentation.
BATCH_SIZE = 16  # Reduced from 32 to 16 to lower memory/CPU usage
# Learning rate (default for all models)
LEARNING_RATE = 0.0001  # Default learning rate

# WAV write subtype used by augmentation/save routines (must be a valid subtype for soundfile)
# Examples: 'PCM_16', 'FLOAT'
AUDIO_WAV_SUBTYPE = 'FLOAT'

# Dataset generation defaults for v4
DATASET_TARGET_SIZE = int(os.environ.get('DATASET_TARGET_SIZE', '30000'))
DEFAULT_SPLIT_RATIOS = {
    'train': float(os.environ.get('SPLIT_TRAIN_FRAC', 0.8)),
    'val': float(os.environ.get('SPLIT_VAL_FRAC', 0.1)),
    'test': float(os.environ.get('SPLIT_TEST_FRAC', 0.1)),
}
# Default OOD fractions
DEFAULT_OOD_FRACTIONS = {
    'ood_noise_frac': float(os.environ.get('OOD_NOISE_FRAC', 0.15)),
    'ood_rir_frac': float(os.environ.get('OOD_RIR_FRAC', 0.15)),
    'hardneg_frac': float(os.environ.get('HARDNEG_FRAC', 0.10)),
}

# R2 domain randomization defaults (hardware + propagation)
HW_DOMAIN_DEFAULTS = {
    # Tuned defaults for high-quality ADC (24-bit) + good MEMS microphone
    'hpf_hz_range': [20.0, 80.0],
    # Lower noise floor to reflect good MEMS + clean preamp
    'noise_floor_db_range': [-95.0, -70.0],
    # Minimal tilt per octave for high-fidelity MEMS
    'fr_tilt_db_per_oct_range': [-1.0, 1.0],
    'notch_count_range': [0, 1],
    'notch_depth_db_range': [0.5, 2.0],
    # Prefer 24-bit ADC choices for high-quality hardware
    'adc_bits_choices': [24]
}

PROP_DOMAIN_DEFAULTS = {
    # global fallback distance bounds (meters) used when category bounds absent
    'distance_bounds': [50.0, 1000.0],
    'beta_jitter_range': [0.4, 0.8],
    'dry_wet_range': [0.2, 0.6]
}

# Mel extraction parameters (central source of truth)
# Use these values everywhere to ensure consistent features between
# training, extraction and inference.
MFCC_N_MFCC = 20            # number of MFCCs to extract
MEL_DURATION = float(AUDIO_DURATION_S)        # seconds used per example at training/inference
MEL_N_MELS = 64           # number of mel bins
MEL_N_FFT = 2048          # FFT window size
MEL_HOP_LENGTH = 512      # hop length in samples
# Compute number of time frames from sample rate / duration / hop length and segments.
# This keeps features consistent when `MEL_HOP_LENGTH`, `SAMPLE_RATE` or `AUDIO_DURATION_S` change.
_samples_per_track = int(SAMPLE_RATE * float(AUDIO_DURATION_S))
_samples_per_segment = int(_samples_per_track / max(1, NUM_SEGMENTS))
# frames = ceil(samples_per_segment / hop_length)
MEL_TIME_FRAMES = int(math.ceil(_samples_per_segment / float(MEL_HOP_LENGTH)))
MEL_PAD_VALUE = -100.0    # sentinel padding value used in precomputed features

# Feature extraction behaviors
# When True, training extraction will enable SpecAugment by default unless
# overridden on the command line in the extractor scripts.
SPEC_AUGMENT_BY_DEFAULT_FOR_TRAIN = True

# Output filenames for per-split extracted features. Use .format(split=...) to
# produce concrete filenames. The extractors will write e.g. mel_train.json.
MEL_OUTPUT_PATTERN = EXTRACTED_FEATURES_DIR / "mel_{split}.json"
MFCC_OUTPUT_PATTERN = EXTRACTED_FEATURES_DIR / "mfcc_{split}.json"

# Precomputed MEL index paths (one MEL per WAV). Evaluation code expects at
# least `mel_test_index.json` when PRECOMPUTED_ONLY=True.
MEL_TEST_DATA_PATH = EXTRACTED_FEATURES_DIR / "mel_test.json"
MEL_TEST_INDEX_PATH = EXTRACTED_FEATURES_DIR / "mel_test_index.json"
MEL_TRAIN_DATA_PATH = EXTRACTED_FEATURES_DIR / "mel_train.json"
MEL_TRAIN_INDEX_PATH = EXTRACTED_FEATURES_DIR / "mel_train_index.json"
MEL_VAL_DATA_PATH = EXTRACTED_FEATURES_DIR / "mel_val.json"
MEL_VAL_INDEX_PATH = EXTRACTED_FEATURES_DIR / "mel_val_index.json"

# Precomputed MEL NPZ paths (for calibration and analysis)
MEL_TRAIN_NPZ_PATH = EXTRACTED_FEATURES_DIR / "mel_train.npz"
MEL_VAL_NPZ_PATH = EXTRACTED_FEATURES_DIR / "mel_val.npz"
MEL_TEST_NPZ_PATH = EXTRACTED_FEATURES_DIR / "mel_test.npz"

# Auto-calibration settings
# Enable automatic threshold calibration (scripts may choose to run calibration
# on first performance evaluation when this is True).
AUTO_CALIBRATE_THRESHOLDS = True  # Set to True to auto-calibrate on first performance calculation
CALIBRATION_FILE_PATH = RESULTS_DIR / "calibrated_thresholds.json"

# Default calibration behaviour (used by `calibrate_thresholds.py` when run
# non-interactively or by automation). Use 'class_aware' to balance per-class
# precision targets (recommended for reducing ambient false positives).
# CALIBRATION_MODE = 'class_aware'            # 'class_aware' | 'target_recall' | 'max_f1' | 'precision_at_recall' | 'weighted'
# Relaxed calibration defaults (favor more balanced recall/precision)
# CALIBRATION_MIN_PREC_DRONE = 0.70           # Minimum required precision for drone (positive) class (relaxed)
# CALIBRATION_MIN_PREC_AMBIENT = 0.85         # Minimum required precision / NPV for ambient (negative) class (relaxed)
# CALIBRATION_CLASS_ALPHA = 0.40              # Fallback weighting: alpha*prec_drone + (1-alpha)*prec_ambient (favor recall more)
# CALIBRATION_SAVE_TO_VISUALIZER = True       # Write `6 - Visualization/outputs/model_thresholds.json` when saving

# Constraint defaults for threshold calibration
# CALIBRATION_CONSTRAINTS = {
#     'min_recall': 0.90,  # Minimum 90% recall (max 10% false negatives)
#     'min_precision_drone': 0.70,  # Minimum 70% drone precision (PPV)
#     'min_precision_ambient': 0.85  # Minimum 85% ambient precision (NPV)
# }
TARGET_RECALL = 0.95  # Target recall for threshold calibration (fallback when using target_recall mode)

# --- Training parameters (source of truth) --------------------------------
# Batch size for training (used by all trainers when no CLI arg provided)
# With reduced default batch size to introduce more gradient noise and improve
# generalization on smaller datasets / limited augmentation.
BATCH_SIZE = 32  # Default batch size for training

# Learning rate (default for all models)
LEARNING_RATE = 0.0001

# --- Loss / training defaults (source of truth) ---------------------------
# Use these values across trainers; trainers will read these when CLI args
# are not provided so `config.py` is the single source of truth.
LOSS_DEFAULTS = {
    'loss_type': 'focal',           # 'bce' | 'weighted_bce' | 'focal' | 'recall_focused'
    # Focal settings tuned for balanced emphasis on hard examples while
    # avoiding an excessive push to predict positive for all examples.
    'focal_alpha': 0.60,
    'focal_gamma': 2.0,
    # keep static class weight conservative when using focal; dynamic weighting
    # is disabled by default to avoid runaway shifts during early training.
    'class_weight_drone': 2.0,
    'use_dynamic_weight': False
}

# Save training metadata alongside saved models
SAVE_TRAINING_METADATA = True

# Default distance weights (meters -> multiplier) used by
# `distance_weighted_loss`. Includes 700m as requested for long-range.
DEFAULT_DISTANCE_WEIGHTS = {
    700: 12.0,
    500: 10.0,
    350: 8.0,
    200: 5.0,
    100: 2.0,
    50: 1.0,
    0: 1.0
}

# Enforce usage of precomputed features only. When True, any code path that would
# compute MELs on-the-fly should instead read `mel_test_index.json` or fail.
PRECOMPUTED_ONLY = True

# Defaults for ReduceLROnPlateau to be attached by trainers when desired.
REDUCE_LR_ON_PLATEAU = {
    'monitor': 'val_loss',
    'factor': 0.5,
    'patience': 3,
    'min_lr': 1e-7,
    'cooldown': 1,
    'verbose': 1,
    'mode': 'min'
}

# Ensemble voting configuration
# PHASE 2A: Optimized weights based on distance performance analysis
# CRNN performs best at extreme distances (68% @ 100m vs CNN 15%)
ENSEMBLE_WEIGHTS = {
    "CNN": 0.30,   # Good at close range
    "RNN": 0.10,   # Weakest performer (0% across all distances in Phase 1)
    "CRNN": 0.60,  # Best at long distances, prioritize this model
}

# Distance-adaptive weights (optional, requires distance estimation)
DISTANCE_ADAPTIVE_WEIGHTS = False
DISTANCE_WEIGHT_CONFIG = {
    "extreme": {"CNN": 0.2, "RNN": 0.0, "CRNN": 0.8},  # > 300m
    "long": {"CNN": 0.3, "RNN": 0.1, "CRNN": 0.6},     # 150-300m
    "medium": {"CNN": 0.4, "RNN": 0.2, "CRNN": 0.4},   # 75-150m
    "close": {"CNN": 0.4, "RNN": 0.3, "CRNN": 0.3},    # < 75m
}

# Deployment constraints (Raspberry Pi 3B)
RASPBERRY_PI_MODE = False  # Set to True for optimized inference on RPi3
MAX_LATENCY_MS = 3000      # Maximum acceptable latency in milliseconds
ENABLE_GPU = False         # RPi3 doesn't have GPU acceleration
QUANTIZE_MODELS = False    # Enable INT8 quantization for faster inference

# Convert Path objects to strings for compatibility
def get_path_str(path):
    """Convert Path to string with forward slashes for cross-platform compatibility."""
    return str(path).replace(os.sep, "/")


# String versions of paths (for scripts that need strings)
DATASET_ROOT_STR = get_path_str(DATASET_ROOT)
CNN_MODEL_PATH_STR = get_path_str(CNN_MODEL_PATH)
CNN_HISTORY_PATH_STR = get_path_str(CNN_HISTORY_PATH)
CNN_ACC_PATH_STR = get_path_str(CNN_ACC_PATH)
RNN_MODEL_PATH_STR = get_path_str(RNN_MODEL_PATH)
RNN_HISTORY_PATH_STR = get_path_str(RNN_HISTORY_PATH)
RNN_ACC_PATH_STR = get_path_str(RNN_ACC_PATH)
CRNN_MODEL_PATH_STR = get_path_str(CRNN_MODEL_PATH)
CRNN_HISTORY_PATH_STR = get_path_str(CRNN_HISTORY_PATH)
CRNN_ACC_PATH_STR = get_path_str(CRNN_ACC_PATH)
ALL_MODELS_ACC_PATH_STR = get_path_str(ALL_MODELS_ACC_PATH)
CNN_PREDICTIONS_PATH_STR = get_path_str(CNN_PREDICTIONS_PATH)
RNN_PREDICTIONS_PATH_STR = get_path_str(RNN_PREDICTIONS_PATH)
CRNN_PREDICTIONS_PATH_STR = get_path_str(CRNN_PREDICTIONS_PATH)
CNN_SCORES_PATH_STR = get_path_str(CNN_SCORES_PATH)
RNN_SCORES_PATH_STR = get_path_str(RNN_SCORES_PATH)
CRNN_SCORES_PATH_STR = get_path_str(CRNN_SCORES_PATH)
VOTED_RESULTS_PATH_STR = get_path_str(VOTED_RESULTS_PATH)
VOTING_WEIGHTS_PATH_STR = get_path_str(VOTING_WEIGHTS_PATH)
VOTING_FINAL_SCORES_PATH_STR = get_path_str(VOTING_FINAL_SCORES_PATH)

# String versions of feature extraction outputs and index paths
MEL_OUTPUT_PATTERN_STR = get_path_str(MEL_OUTPUT_PATTERN)
MFCC_OUTPUT_PATTERN_STR = get_path_str(MFCC_OUTPUT_PATTERN)
MEL_TEST_DATA_PATH_STR = get_path_str(MEL_TEST_DATA_PATH)
MEL_TEST_INDEX_PATH_STR = get_path_str(MEL_TEST_INDEX_PATH)
MEL_TRAIN_DATA_PATH_STR = get_path_str(MEL_TRAIN_DATA_PATH)
MEL_TRAIN_INDEX_PATH_STR = get_path_str(MEL_TRAIN_INDEX_PATH)
MEL_VAL_DATA_PATH_STR = get_path_str(MEL_VAL_DATA_PATH)
MEL_VAL_INDEX_PATH_STR = get_path_str(MEL_VAL_INDEX_PATH)

# String versions of voted paths dictionaries
VOTED_PATHS_STR = {k: get_path_str(v) for k, v in VOTED_PATHS.items()}
SCORES_PATHS_STR = {k: get_path_str(v) for k, v in SCORES_PATHS.items()}

# ===== CURRICULUM LEARNING CONFIGURATION =====
CURRICULUM_LEARNING = {
    # Enable curriculum learning (can be overridden by CLI --curriculum flag)
    'enabled': False,
    # Strategy: 'linear', 'step', 'exponential', 'none'
    'strategy': 'linear',
    # Number of warmup epochs (start with easy samples)
    'warmup_epochs': 5,
    # Initial fraction of samples to use (0.5 = 50% easiest samples)
    'initial_fraction': 0.25,
    # Growth rate per epoch (for linear strategy)
    'growth_rate': 0.05,  # +5% samples per epoch
    # Step size for 'step' strategy (increase every N epochs)
    'step_size': 10,
    # Tau for 'exponential' strategy (controls curvature)
    'tau': 0.5,
    # Minimum difficulty threshold (0.0-1.0, ignore too-easy samples)
    'min_difficulty': 0.0,
    # Maximum difficulty threshold (1.0 = all samples at the end)
    'max_difficulty': 1.0,
    # Sort order: 'ascending' (easy→hard) or 'descending'
    'sort_order': 'ascending',
    # Verbose logging for curriculum scheduler
    'verbose': True
}

# Mapping trainer script names to model identifiers
TRAINER_MODULES = {
    'CNN_Trainer': 'cnn',
    'RNN_Trainer': 'rnn',
    'CRNN_Trainer': 'crnn',
    'Attention_CRNN_Trainer': 'attention_crnn',
    'EfficientNet_Trainer': 'efficientnet',
    'MobileNet_Trainer': 'mobilenet',
    'Conformer_Trainer': 'conformer',
    'TCN_Trainer': 'tcn'
}

# Output format per trainer (for auto-detection)
TRAINER_OUTPUT_FORMATS = {
    'cnn': 'categorical',           # Dense(2, softmax)
    'rnn': 'categorical',           # Dense(2, softmax)
    'crnn': 'categorical',          # Dense(2, softmax)
    'attention_crnn': 'categorical', # Dense(2, softmax)
    'efficientnet': 'binary',       # Dense(1, sigmoid)
    'mobilenet': 'binary',          # Dense(1, sigmoid)
    'conformer': 'binary',          # Dense(1, sigmoid)
    'tcn': 'binary'                 # Dense(1, sigmoid)
}