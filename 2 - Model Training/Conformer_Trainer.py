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
from tools.safe_callbacks import SafeEarlyStopping

# Timer
startTime = datetime.now()

# Paths
MODEL_SAVE = config.PROJECT_ROOT / "0 - DADS dataset extraction" / "saved_models" / "conformer_model.keras"
HISTORY_SAVE = config.RESULTS_DIR / "conformer_history.csv"
ACC_SAVE = config.RESULTS_DIR / "conformer_accuracy.json"

# Training hyperparams
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 32)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0001)


def feed_forward_module(x, dim, expansion_factor=4, dropout=0.1, training=False, name_prefix='ffn'):
    """Feed-forward module with GLU activation (Functional API)"""
    inner_dim = dim * expansion_factor
    
    # Layer norm
    x = layers.LayerNormalization(name=f'{name_prefix}_ln')(x)
    
    # Dense with GLU
    x = layers.Dense(inner_dim * 2, name=f'{name_prefix}_dense1')(x)
    
    # GLU activation: split and multiply
    x_gate = layers.Lambda(lambda x: x[:, :, :inner_dim], name=f'{name_prefix}_gate')(x)
    x_linear = layers.Lambda(lambda x: x[:, :, inner_dim:], name=f'{name_prefix}_linear')(x)
    x_gate = layers.Activation('gelu', name=f'{name_prefix}_gelu')(x_gate)
    x = layers.Multiply(name=f'{name_prefix}_glu')([x_gate, x_linear])
    
    x = layers.Dropout(dropout, name=f'{name_prefix}_drop1')(x, training=training)
    x = layers.Dense(dim, name=f'{name_prefix}_dense2')(x)
    x = layers.Dropout(dropout, name=f'{name_prefix}_drop2')(x, training=training)
    
    return x


def convolution_module(x, dim, kernel_size=31, dropout=0.1, training=False, name_prefix='conv'):
    """Depthwise convolution module (Functional API)"""
    # Layer norm
    x = layers.LayerNormalization(name=f'{name_prefix}_ln')(x)
    
    # Pointwise expansion with GLU
    x = layers.Dense(dim * 2, name=f'{name_prefix}_pw1')(x)
    
    # GLU activation
    x_gate = layers.Lambda(lambda x: x[:, :, :dim], name=f'{name_prefix}_gate')(x)
    x_linear = layers.Lambda(lambda x: x[:, :, dim:], name=f'{name_prefix}_linear')(x)
    x_linear = layers.Activation('sigmoid', name=f'{name_prefix}_sigmoid')(x_linear)
    x = layers.Multiply(name=f'{name_prefix}_glu')([x_gate, x_linear])
    
    # Depthwise convolution
    x = layers.DepthwiseConv1D(kernel_size=kernel_size, padding='same', depth_multiplier=1, name=f'{name_prefix}_dw')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn')(x, training=training)
    x = layers.Activation('swish', name=f'{name_prefix}_swish')(x)
    
    # Pointwise projection
    x = layers.Dense(dim, name=f'{name_prefix}_pw2')(x)
    x = layers.Dropout(dropout, name=f'{name_prefix}_drop')(x, training=training)
    
    return x


def conformer_block(x, dim, num_heads=4, ffn_expansion=4, conv_kernel_size=31, dropout=0.1, training=False, block_id=0):
    """
    Conformer block: Macaron-style FFN + MHSA + Convolution + FFN (Functional API)
    """
    # First half-step FFN (scaled by 0.5)
    ffn1_out = feed_forward_module(x, dim, ffn_expansion, dropout, training, name_prefix=f'block{block_id}_ffn1')
    ffn1_out = layers.Lambda(lambda x: 0.5 * x, name=f'block{block_id}_ffn1_scale')(ffn1_out)
    x = layers.Add(name=f'block{block_id}_add1')([x, ffn1_out])
    
    # Multi-head self-attention
    x_norm = layers.LayerNormalization(name=f'block{block_id}_mha_ln')(x)
    mha_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=dim // num_heads,
        dropout=dropout,
        name=f'block{block_id}_mha'
    )(x_norm, x_norm, training=training)
    mha_out = layers.Dropout(dropout, name=f'block{block_id}_mha_drop')(mha_out, training=training)
    x = layers.Add(name=f'block{block_id}_add2')([x, mha_out])
    
    # Convolution module
    conv_out = convolution_module(x, dim, conv_kernel_size, dropout, training, name_prefix=f'block{block_id}_conv')
    x = layers.Add(name=f'block{block_id}_add3')([x, conv_out])
    
    # Second half-step FFN (scaled by 0.5)
    ffn2_out = feed_forward_module(x, dim, ffn_expansion, dropout, training, name_prefix=f'block{block_id}_ffn2')
    ffn2_out = layers.Lambda(lambda x: 0.5 * x, name=f'block{block_id}_ffn2_scale')(ffn2_out)
    x = layers.Add(name=f'block{block_id}_add4')([x, ffn2_out])
    
    # Final layer norm
    x = layers.LayerNormalization(name=f'block{block_id}_ln_final')(x)
    
    return x


def build_model(input_shape=None, num_conformer_blocks=4):
    """
    Build Conformer model (CNN + Transformer hybrid)
    Combines local feature extraction (CNN) with global context (Transformer)
    """
    # Resolve default input_shape from centralized config when not provided
    if input_shape is None:
        n_mels = int(getattr(__import__('config'), 'MEL_N_MELS', 44))
        time_frames = int(getattr(__import__('config'), 'MEL_TIME_FRAMES', 90))
        channels = 1
        input_shape = (n_mels, time_frames, channels)

    inputs = layers.Input(shape=input_shape)

    # Reshape for 1D processing (time as sequence, frequency as features)
    # From (n_mels, time_frames, channels) to (time_frames, n_mels)
    x = layers.Reshape((input_shape[1], input_shape[0]))(inputs)
    
    # Subsampling layer (reduce temporal dimension)
    x = layers.Conv1D(filters=144, kernel_size=3, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters=144, kernel_size=3, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    
    # Linear projection to model dimension
    model_dim = 144
    x = layers.Dense(model_dim)(x)
    
    # Stack of Conformer blocks
    for i in range(num_conformer_blocks):
        x = conformer_block(
            x,
            dim=model_dim,
            num_heads=4,
            ffn_expansion=4,
            conv_kernel_size=31,
            dropout=0.1,
            training=True,
            block_id=i
        )
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Conformer_Audio')
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
        '--model', 'conformer'
    ]
    # Forward any CLI args the user provided to the universal trainer
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
