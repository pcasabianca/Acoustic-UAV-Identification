import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
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

# Training hyperparams (use config when available)
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 8)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0001)


def load_data():
    print("[INFO] Loading validation features (NPZ/JSON auto-detect)")
    val_mels, val_labels, _ = load_mel_features('val')
    if len(val_mels) == 0:
        raise RuntimeError("No validation MEL features found")

    print("[INFO] Loading training features (NPZ/JSON auto-detect)")
    train_mels, train_labels, _ = load_mel_features('train')
    if len(train_mels) == 0:
        raise RuntimeError("No training MEL features found")

    print(f"[INFO] Loaded {len(train_mels)} train samples, {len(val_mels)} val samples")
    return train_mels, train_labels, val_mels, val_labels


def prepare_datasets():
    X_train, y_train, X_validation, y_validation = load_data()
    X_test = X_validation
    y_test = y_validation

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Reshape CNN output for recurrent layers: (batch, time, features)
    model.add(keras.layers.Permute((2, 1, 3)))
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))

    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(2, activation='softmax'))

    return model


if __name__ == "__main__":
    # Delegate training to the Universal_Curriculum_Trainer. Keep this module
    # focused on model construction via `build_model`.
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'Universal_Curriculum_Trainer.py'),
        '--model', 'crnn'
    ]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
