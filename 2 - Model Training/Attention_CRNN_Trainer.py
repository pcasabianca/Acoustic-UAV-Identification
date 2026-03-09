"""
Attention-Enhanced CRNN Trainer
================================

Architecture optimisée pour la détection à longue distance (500m-350m) :
- CNN feature extractor
- Temporal Attention (focus sur time-steps avec signal drone)
- Bidirectional LSTM (contexte temporel)
- Channel Attention (focus sur fréquences drone 500-2000Hz)
- Dense classifier

Améliorations vs CRNN standard :
1. Attention temporelle amplifie segments avec signal faible
2. Attention de canal priorise fréquences drone
3. Deeper feature extraction (4 conv layers vs 3)
4. Residual connections pour gradient flow
"""

import sys
import numpy as np
from pathlib import Path
from termcolor import colored

# Project config
sys.path.append(str(Path(__file__).parent.parent))
import config

# Import universal feature loader (supports NPZ and JSON)
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from feature_loader import load_mel_features

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Limit TensorFlow memory growth to prevent OOM crashes
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth config failed: {e}")

# Optimize CPU threads (configurable via config.py)
tf.config.threading.set_intra_op_parallelism_threads(config.TF_INTRA_OP_PARALLELISM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(config.TF_INTER_OP_PARALLELISM_THREADS)

# Use centralized configuration
TRAIN_DATA_PATH = Path(config.MEL_TRAIN_DATA_PATH)
VAL_DATA_PATH = Path(config.MEL_VAL_DATA_PATH)
MODEL_SAVE = Path(config.ATTENTION_CRNN_MODEL_PATH)
HISTORY_SAVE = Path(config.ATTENTION_CRNN_HISTORY_PATH)

# Training parameters (respect config when present)
# Attention_CRNN is memory-intensive: use config batch size or fallback to safe default
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 8)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0001)

SAMPLE_RATE = config.SAMPLE_RATE
N_MELS = config.MEL_N_MELS
N_FFT = config.MEL_N_FFT
HOP_LENGTH = config.MEL_HOP_LENGTH
TIME_FRAMES = config.MEL_TIME_FRAMES
PAD_VALUE = config.MEL_PAD_VALUE

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
    """Load precomputed mel features and labels, return arrays:
    X_train, y_train, X_val, y_val
    """
    X_train, y_train, X_val, y_val = load_data()

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # Filter out samples with missing labels (None)
    train_mask = np.array([lbl is not None for lbl in y_train])
    val_mask = np.array([lbl is not None for lbl in y_val])
    if not np.all(train_mask):
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        print(f"[INFO] Filtered out {np.sum(~train_mask)} training samples with missing labels")
    if not np.all(val_mask):
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]
        print(f"[INFO] Filtered out {np.sum(~val_mask)} validation samples with missing labels")

    return X_train, y_train, X_val, y_val


def build_attention_crnn(input_shape=None, num_classes=2):
    """
    Build Attention-Enhanced CRNN model.
    
    Architecture:
    1. Deeper CNN (4 layers) with residual connections
    2. Temporal Attention layer
    3. Bidirectional LSTM
    4. Channel Attention layer
    5. Dense classifier
    """
    # Resolve default input shape from centralized config when not provided
    if input_shape is None:
        n_mels = int(getattr(__import__('config'), 'MEL_N_MELS', 44))
        time_frames = int(getattr(__import__('config'), 'MEL_TIME_FRAMES', 90))
        input_shape = (n_mels, time_frames)

    inputs = layers.Input(shape=input_shape + (1,), name='input')
    
    # ============================================================
    # PART 1: Deeper CNN Feature Extractor with Residual
    # ============================================================
    
    # Block 1
    x = layers.Conv2D(32, (5, 5), padding='same', activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.2, name='drop1')(x)
    
    # Block 2
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)
    
    # Block 3 with residual
    residual = x
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    
    # Adjust residual dimensions for addition
    residual = layers.MaxPooling2D((2, 2))(residual)
    residual = layers.Conv2D(128, (1, 1), padding='same')(residual)
    x = layers.Add(name='residual_add1')([x, residual])
    x = layers.Dropout(0.3, name='drop3')(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    x = layers.Dropout(0.4, name='drop4')(x)
    
    # ============================================================
    # PART 2: Reshape for RNN (time-distributed)
    # ============================================================
    
    # Shape after CNN: (batch, freq, time, channels)
    # Reshape to: (batch, time, freq*channels) for LSTM
    shape = x.shape
    x = layers.Reshape((shape[2], shape[1] * shape[3]), name='reshape_for_lstm')(x)
    
    # ============================================================
    # PART 3: Temporal Attention Mechanism
    # ============================================================
    
    # Learn which time-steps contain drone signal
    attention_weights = layers.Dense(1, activation='tanh', name='attention_score')(x)
    attention_weights = layers.Flatten(name='attention_flatten')(attention_weights)
    attention_weights = layers.Activation('softmax', name='attention_softmax')(attention_weights)
    attention_weights = layers.RepeatVector(shape[1] * shape[3], name='attention_repeat')(attention_weights)
    attention_weights = layers.Permute([2, 1], name='attention_permute')(attention_weights)
    
    # Apply attention weights
    x = layers.Multiply(name='attention_applied')([x, attention_weights])
    
    # ============================================================
    # PART 4: Bidirectional LSTM
    # ============================================================
    
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, use_cudnn=False, name='lstm1'),
        name='bi_lstm1'
    )(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3, use_cudnn=False, name='lstm2'),
        name='bi_lstm2'
    )(x)
    
    # ============================================================
    # PART 5: Channel Attention (Frequency Focus)
    # ============================================================
    
    # Global context
    channel_attention = layers.Dense(64, activation='relu', name='channel_dense1')(x)
    channel_attention = layers.Dense(128, activation='sigmoid', name='channel_attention')(channel_attention)
    
    # Apply channel attention
    x = layers.Multiply(name='channel_attention_applied')([x, channel_attention])
    
    # ============================================================
    # PART 6: Classifier
    # ============================================================
    
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='drop_final')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Build model
    model = models.Model(inputs=inputs, outputs=outputs, name='Attention_CRNN')
    
    return model


if __name__ == "__main__":
    # Delegate all training to the Universal_Curriculum_Trainer to centralize
    # curriculum and training logic while keeping this module responsible
    # only for model construction via `build_attention_crnn`.
    import sys
    import subprocess
    from pathlib import Path

    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'Universal_Curriculum_Trainer.py'),
        '--model', 'attention_crnn'
    ]
    # Forward any CLI args the user provided to the universal trainer
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
