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
MODEL_SAVE = config.PROJECT_ROOT / "0 - DADS dataset extraction" / "saved_models" / "mobilenet_model.keras"
HISTORY_SAVE = config.RESULTS_DIR / "mobilenet_history.csv"
ACC_SAVE = config.RESULTS_DIR / "mobilenet_accuracy.json"

# Training hyperparams
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 32)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0001)


def depthwise_separable_block(x, filters, strides=1):
    """Depthwise Separable Convolution block"""
    # Depthwise convolution
    x = layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Pointwise convolution
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    return x


def build_model(input_shape=None):
    """
    Build MobileNet-Audio model
    Lightweight architecture optimized for audio with depthwise separable convolutions
    """
    if input_shape is None:
        n_mels = int(getattr(__import__('config'), 'MEL_N_MELS', 44))
        time_frames = int(getattr(__import__('config'), 'MEL_TIME_FRAMES', 90))
        channels = 1
        input_shape = (n_mels, time_frames, channels)

    inputs = layers.Input(shape=input_shape)
    
    # Initial standard convolution
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Depthwise Separable Convolution blocks
    # Stage 1
    x = depthwise_separable_block(x, 64, strides=1)
    
    # Stage 2
    x = depthwise_separable_block(x, 128, strides=2)
    x = depthwise_separable_block(x, 128, strides=1)
    
    # Stage 3
    x = depthwise_separable_block(x, 256, strides=2)
    x = depthwise_separable_block(x, 256, strides=1)
    
    # Stage 4
    x = depthwise_separable_block(x, 512, strides=2)
    
    # 5 depthwise separable blocks with 512 filters
    for _ in range(5):
        x = depthwise_separable_block(x, 512, strides=1)
    
    # Stage 5
    x = depthwise_separable_block(x, 1024, strides=2)
    x = depthwise_separable_block(x, 1024, strides=1)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='MobileNet_Audio')
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
        '--model', 'mobilenet'
    ]
    # Forward any CLI args the user provided to the universal trainer
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
