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


def transformer_block(x, embed_dim, num_heads, mlp_dim, dropout_rate=0.1, prefix='ast'):
    x_norm = layers.LayerNormalization(name=f'{prefix}_ln1')(x)
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=max(8, embed_dim // max(1, num_heads)),
        dropout=dropout_rate,
        name=f'{prefix}_mha'
    )(x_norm, x_norm)
    x = layers.Add(name=f'{prefix}_res1')([x, attn])

    y = layers.LayerNormalization(name=f'{prefix}_ln2')(x)
    y = layers.Dense(mlp_dim, activation='gelu', name=f'{prefix}_mlp1')(y)
    y = layers.Dropout(dropout_rate, name=f'{prefix}_drop1')(y)
    y = layers.Dense(embed_dim, name=f'{prefix}_mlp2')(y)
    y = layers.Dropout(dropout_rate, name=f'{prefix}_drop2')(y)
    x = layers.Add(name=f'{prefix}_res2')([x, y])
    return x


def build_model(input_shape=None, patch_size=(4, 4), embed_dim=128, depth=4, num_heads=4):
    if input_shape is None:
        n_mels = int(getattr(config, 'MEL_N_MELS', 44))
        time_frames = int(getattr(config, 'MEL_TIME_FRAMES', 90))
        input_shape = (n_mels, time_frames, 1)

    inputs = layers.Input(shape=input_shape, name='ast_input')

    # Patch embedding from spectrogram
    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='same',
        name='ast_patch_embed'
    )(inputs)

    # (B, H, W, C) -> (B, tokens, C)
    h_tokens = x.shape[1]
    w_tokens = x.shape[2]
    token_count = int(h_tokens * w_tokens)
    x = layers.Reshape((token_count, embed_dim), name='ast_tokens')(x)

    # Lightweight positional embedding (explicit batch broadcast)
    pos_idx = tf.range(start=0, limit=token_count, delta=1)
    pos = layers.Embedding(
        input_dim=max(256, token_count + 1),
        output_dim=embed_dim,
        name='ast_pos_emb'
    )(pos_idx)
    pos = layers.Lambda(lambda p: tf.expand_dims(p, axis=0), name='ast_pos_expand')(pos)
    x = layers.Add(name='ast_add_pos')([x, pos])

    for i in range(depth):
        x = transformer_block(x, embed_dim, num_heads, mlp_dim=embed_dim * 4, dropout_rate=0.1, prefix=f'ast_blk{i}')

    x = layers.LayerNormalization(name='ast_ln_out')(x)
    x = layers.GlobalAveragePooling1D(name='ast_gap')(x)
    x = layers.Dropout(0.3, name='ast_cls_drop')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='ast_out')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='AST_Audio')


if __name__ == "__main__":
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'Universal_Curriculum_Trainer.py'),
        '--model', 'ast'
    ]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
