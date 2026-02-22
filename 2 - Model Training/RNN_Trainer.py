import json
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from datetime import datetime
from termcolor import colored
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Optimize CPU threads (configurable via config.py)
tf.config.threading.set_intra_op_parallelism_threads(config.TF_INTRA_OP_PARALLELISM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(config.TF_INTER_OP_PARALLELISM_THREADS)

# Enable GPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth config failed: {e}")

# Import universal feature loader (supports NPZ and JSON)
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from feature_loader import load_mel_features

# Import loss functions
from loss_functions import get_loss_function, get_metrics

# Timer.
startTime = datetime.now()

# Path to created json file from mel preprocess and feature extraction script.
TRAIN_DATA_PATH = config.MEL_TRAIN_DATA_PATH_STR
TRAIN_INDEX_PATH = config.MEL_TRAIN_INDEX_PATH_STR
VAL_DATA_PATH = config.MEL_VAL_DATA_PATH_STR
VAL_INDEX_PATH = config.MEL_VAL_INDEX_PATH_STR

# Path to save model.
MODEL_SAVE = config.RNN_MODEL_PATH_STR

# Path to save training history and model accuracy performance at end of training.
HISTORY_SAVE = config.RNN_HISTORY_PATH_STR
ACC_SAVE = config.RNN_ACC_PATH_STR

# Training hyperparams (use config when available)
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 8)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0005)

def load_data():
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    import librosa
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    # Load features using universal loader (auto-detects NPZ or JSON)
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
    X_train, y_train, X_validation, y_validation = load_data()
    X_test = X_validation
    y_test = y_validation

    # RNN expects 3D input: (samples, time_steps, features)
    # Data from load_data() is already 3D: (samples, n_mels=44, time_frames=173)
    # NO transformation needed - unlike CNN which needs 4D for Conv2D
 # Shape: (samples, n_mels, time_frames) - perfect for LSTM

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # Create model.
    # NOTE: LSTM layers cannot use AMD GPU due to ROCm MIOpen limitations
    # This model must run on CPU - the pipeline automatically handles this
    model = keras.Sequential()

    # Reshape from 4D (batch, height, width, channels) to 3D (batch, timesteps, features)
    # LSTM expects 3D: (batch, timesteps, features)
    model.add(keras.layers.Input(shape=input_shape))
    # Flatten spatial dimensions: (h, w, c) → (h, w*c)
    # Use height as timesteps, width*channels as features
    if len(input_shape) == 3:
        height, width, channels = input_shape
        model.add(keras.layers.Reshape((height, width * channels)))
    # If already 2D, keep as is
    
    # 3 Bidirectional LSTM layers
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100)))

    # Dense layers
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='softmax'))

    return model


if __name__ == "__main__":
    # Delegate training to the Universal_Curriculum_Trainer. Keep this module
    # focused on model construction via `build_model`.
    import sys
    import subprocess
    from pathlib import Path

    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'Universal_Curriculum_Trainer.py'),
        '--model', 'rnn'
    ]
    # Forward any CLI args the user provided to the universal trainer
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    # Delegate training to the Universal_Curriculum_Trainer. Keep this module
    # focused on model construction via `build_model`.
    import sys
    import subprocess
    from pathlib import Path

    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'Universal_Curriculum_Trainer.py'),
        '--model', 'rnn'
    ]
    # Forward any CLI args the user provided to the universal trainer
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
