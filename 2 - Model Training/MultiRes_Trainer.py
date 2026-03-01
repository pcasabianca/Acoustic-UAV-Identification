import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Optimize CPU threads
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from feature_loader import load_mel_features

BATCH_SIZE = getattr(config, 'BATCH_SIZE', 16)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 1e-4)


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


def multires_branch(x, filters, kernel_sizes, name_prefix):
    branches = []
    for idx, k in enumerate(kernel_sizes):
        b = layers.Conv2D(filters, (k, k), padding='same', activation='relu', name=f'{name_prefix}_conv_{idx}')(x)
        b = layers.BatchNormalization(name=f'{name_prefix}_bn_{idx}')(b)
        branches.append(b)
    x = layers.Concatenate(name=f'{name_prefix}_concat')(branches)
    x = layers.Conv2D(filters, (1, 1), padding='same', activation='relu', name=f'{name_prefix}_mix')(x)
    x = layers.MaxPool2D((2, 2), name=f'{name_prefix}_pool')(x)
    return x


def build_model(input_shape=None):
    if input_shape is None:
        n_mels = int(getattr(config, 'MEL_N_MELS', 44))
        time_frames = int(getattr(config, 'MEL_TIME_FRAMES', 90))
        input_shape = (n_mels, time_frames, 1)

    inputs = layers.Input(shape=input_shape, name='multires_input')

    x = multires_branch(inputs, filters=32, kernel_sizes=[3, 5, 7], name_prefix='mr1')
    x = layers.Dropout(0.15, name='mr1_drop')(x)

    x = multires_branch(x, filters=64, kernel_sizes=[3, 5], name_prefix='mr2')
    x = layers.Dropout(0.2, name='mr2_drop')(x)

    x = multires_branch(x, filters=96, kernel_sizes=[3, 5], name_prefix='mr3')
    x = layers.Dropout(0.25, name='mr3_drop')(x)

    x = layers.GlobalAveragePooling2D(name='mr_gap')(x)
    x = layers.Dense(128, activation='relu', name='mr_fc1')(x)
    x = layers.Dropout(0.3, name='mr_fc_drop')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='mr_out')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='MultiRes_Audio')


if __name__ == "__main__":
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'Universal_Curriculum_Trainer.py'),
        '--model', 'multires'
    ]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
