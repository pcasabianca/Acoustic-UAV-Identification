import json
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import pandas as pd
from datetime import datetime
from termcolor import colored
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Optimize CPU threads (use config values)
tf.config.threading.set_intra_op_parallelism_threads(config.TF_INTRA_OP_PARALLELISM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(config.TF_INTER_OP_PARALLELISM_THREADS)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth config failed: {e}")

# Import universal feature loader
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from feature_loader import load_mel_features

# Import loss functions
from loss_functions import get_loss_function, get_metrics

# Timer
startTime = datetime.now()

# Paths
MODEL_SAVE = config.PROJECT_ROOT / "0 - DADS dataset extraction" / "saved_models" / "tcn_model.keras"
HISTORY_SAVE = config.RESULTS_DIR / "tcn_history.csv"
ACC_SAVE = config.RESULTS_DIR / "tcn_accuracy.json"

# Training hyperparams
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 32)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0001)


def residual_block(x, dilation_rate, nb_filters, kernel_size):
    """TCN Residual Block with dilated causal convolutions"""
    # Dilated causal convolution
    prev_x = x
    
    # First conv
    x = layers.Conv1D(filters=nb_filters,
                      kernel_size=kernel_size,
                      dilation_rate=dilation_rate,
                      padding='causal',
                      activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Second conv
    x = layers.Conv1D(filters=nb_filters,
                      kernel_size=kernel_size,
                      dilation_rate=dilation_rate,
                      padding='causal',
                      activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # 1x1 conv for residual connection if needed
    if prev_x.shape[-1] != nb_filters:
        prev_x = layers.Conv1D(nb_filters, 1, padding='same')(prev_x)
    
    # Residual connection
    x = layers.Add()([prev_x, x])
    x = layers.Activation('relu')(x)
    
    return x


def build_model(input_shape=None):
    """
    Build TCN (Temporal Convolutional Network) model
    Uses dilated causal convolutions for temporal pattern recognition
    """
    if input_shape is None:
        n_mels = int(getattr(__import__('config'), 'MEL_N_MELS', 44))
        time_frames = int(getattr(__import__('config'), 'MEL_TIME_FRAMES', 90))
        channels = 1
        input_shape = (n_mels, time_frames, channels)

    inputs = layers.Input(shape=input_shape)
    
    # Reshape for 1D convolution (treat frequency as channels, time as sequence)
    # From (n_mels, time_frames, 1) to (time_frames, n_mels) - time steps, features
    x = layers.Reshape((input_shape[1], input_shape[0]))(inputs)
    
    # Initial convolution
    x = layers.Conv1D(filters=64, kernel_size=1, padding='same')(x)
    
    # TCN blocks with increasing dilation rates
    # This creates exponentially growing receptive field
    nb_filters = 64
    kernel_size = 3
    
    # Block 1: dilation_rate = 1
    x = residual_block(x, dilation_rate=1, nb_filters=nb_filters, kernel_size=kernel_size)
    
    # Block 2: dilation_rate = 2
    x = residual_block(x, dilation_rate=2, nb_filters=nb_filters, kernel_size=kernel_size)
    
    # Block 3: dilation_rate = 4
    x = residual_block(x, dilation_rate=4, nb_filters=nb_filters, kernel_size=kernel_size)
    
    # Block 4: dilation_rate = 8
    x = residual_block(x, dilation_rate=8, nb_filters=nb_filters, kernel_size=kernel_size)
    
    # Increase filters for deeper blocks
    nb_filters = 128
    
    # Block 5: dilation_rate = 16
    x = residual_block(x, dilation_rate=16, nb_filters=nb_filters, kernel_size=kernel_size)
    
    # Block 6: dilation_rate = 32
    x = residual_block(x, dilation_rate=32, nb_filters=nb_filters, kernel_size=kernel_size)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='TCN_Audio')
    return model


def load_data():
    """Load training and validation data"""
    print(f"[INFO] Loading validation features (NPZ/JSON auto-detect)")
    val_mels, val_labels, _ = load_mel_features('val')
    if len(val_mels) == 0:
        raise RuntimeError(f"No validation MEL features found")

    print(f"[INFO] Loading training features (NPZ/JSON auto-detect)")
    train_mels, train_labels, _ = load_mel_features('train')
    if len(train_mels) == 0:
        raise RuntimeError(f"No training MEL features found")
    
    print(f"[INFO] Loaded {len(train_mels)} train samples, {len(val_mels)} val samples")
    return train_mels, train_labels, val_mels, val_labels


def prepare_datasets():
    """Prepare datasets with proper dimensions"""
    X_train, y_train, X_validation, y_validation = load_data()
    
    X_test = X_validation
    y_test = y_validation

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


if __name__ == "__main__":
    # Delegate training to the Universal_Curriculum_Trainer. Keep this module
    # focused on model construction via `build_model`.
    import sys
    import subprocess
    from pathlib import Path

    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'Universal_Curriculum_Trainer.py'),
        '--model', 'tcn'
    ]
    # Forward any CLI args the user provided to the universal trainer
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
