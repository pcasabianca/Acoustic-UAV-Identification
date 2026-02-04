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

# Optimize CPU threads
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth config failed: {e}")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'weighted_bce', 'focal', 'recall_focused'],
                        help='Loss function to use')
    parser.add_argument('--epochs', type=int, default=config.MAX_EPOCHS, help='Number of epochs')
    parser.add_argument('--blocks', type=int, default=4, help='Number of Conformer blocks')
    parser.add_argument('--use_dynamic_weight', action='store_true', default=None,
                       help='Enable dynamic positive-class weight scheduler (updates each epoch)')
    parser.add_argument('--dyn_base', type=float, default=1.0, help='Base weight multiplier for dynamic scheduler')
    parser.add_argument('--dyn_beta', type=float, default=0.8, help='EMA beta for dynamic scheduler (smoothing)')
    parser.add_argument('--dyn_min_w', type=float, default=0.01, help='Minimum clamp for dynamic positive weight')
    parser.add_argument('--dyn_max_w', type=float, default=10.0, help='Maximum clamp for dynamic positive weight')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size')
    args = parser.parse_args()

    # Prepare datasets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets()

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape, num_conformer_blocks=args.blocks)

    # Get loss and metrics
    loss_fn = get_loss_function(args.loss)
    metrics = get_metrics()

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # Print model summary
    print("\n" + "="*80)
    print("CONFORMER (CNN + TRANSFORMER HYBRID) MODEL")
    print("="*80)
    model.summary()
    print("="*80 + "\n")

    # Compute balanced class weights (adaptive utility removed)
    class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights_array))
    print(f"[INFO] Class weights (balanced): {class_weights}")

    if class_weights is None:
        class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights_array))
        print(f"[INFO] Class weights (balanced fallback): {class_weights}")
    # Early stopping (monitor val_f1) — ensure defined regardless of class_weights
    from tools.safe_callbacks import SafeEarlyStopping
    early_stop = SafeEarlyStopping(
        start_from_epoch=getattr(args, 'min_epochs', 0),
        monitor='val_f1',
        mode='max',
        patience=10,
        restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    try:
        model_save_str = str(MODEL_SAVE)
    except Exception:
        model_save_str = MODEL_SAVE
    ckpt_f1 = keras.callbacks.ModelCheckpoint(model_save_str, monitor='val_f1', mode='max', save_best_only=True, verbose=1)
    if isinstance(model_save_str, str) and model_save_str.endswith('.keras'):
        ckpt_loss_path = model_save_str.replace('.keras', '.best_val_loss.keras')
    else:
        ckpt_loss_path = str(model_save_str) + '.best_val_loss'
    ckpt_loss = keras.callbacks.ModelCheckpoint(ckpt_loss_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    # Add a SafeEarlyStopping on val_loss as a secondary stopper
    try:
        loss_early_stop = SafeEarlyStopping(start_from_epoch=args.min_epochs, monitor='val_loss', mode='min', patience=8, restore_best_weights=True)
    except Exception:
        loss_early_stop = SafeEarlyStopping(monitor='val_loss', mode='min', patience=8, restore_best_weights=True)
    # We'll prepend ValMetricsCallback so val_f1 is injected before other callbacks run
    callbacks = [early_stop, reduce_lr, loss_early_stop, ckpt_f1, ckpt_loss]

    # Val metrics callback to compute val_f1 and inject into logs
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    class ValMetricsCallback(keras.callbacks.Callback):
        def __init__(self, val_data):
            super().__init__()
            self.x_val, self.y_val = val_data
        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(self.x_val, verbose=0)
            if preds.ndim > 1 and preds.shape[-1] > 1:
                preds = np.argmax(preds, axis=1)
            else:
                preds = (preds.reshape(-1) >= 0.5).astype(int)
            y_true = np.array(self.y_val, dtype=int)
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            except Exception:
                from collections import Counter
                c = Counter(zip(y_true, preds))
                tp = c.get((1,1), 0)
                tn = c.get((0,0), 0)
                fp = c.get((0,1), 0)
                fn = c.get((1,0), 0)
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
            if logs is not None:
                logs['val_precision'] = precision
                logs['val_recall'] = recall
                logs['val_f1'] = f1
            print(f"[EPOCH {epoch}] val_precision={precision:.4f} val_recall={recall:.4f} val_f1={f1:.4f} tp={tp} fp={fp} tn={tn} fn={fn}")
    # Prepend ValMetricsCallback to ensure `val_f1` is available to EarlyStopping/ModelCheckpoint
    val_metrics_cb = ValMetricsCallback((X_validation, y_validation))
    callbacks = [val_metrics_cb] + callbacks

    # Train
    print(f"\n[INFO] Training Conformer with {args.loss} loss, batch_size={BATCH_SIZE}, {args.blocks} blocks")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_validation, y_validation),
        batch_size=BATCH_SIZE,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    # Save model
    MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_SAVE)
    print(f"\n[SUCCESS] Model saved: {MODEL_SAVE}")

    # Save history
    hist_df = pd.DataFrame(history.history)
    HISTORY_SAVE.parent.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(HISTORY_SAVE, index=False)
    print(f"[SUCCESS] History saved: {HISTORY_SAVE}")

    # Evaluate on test set
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = test_results[0]
    test_acc = test_results[1] if len(test_results) > 1 else 0.0
    print(f"\n[RESULTS] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save accuracy
    accuracy_data = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "training_time": str(datetime.now() - startTime),
        "loss_function": args.loss,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_blocks": args.blocks
    }
    with open(ACC_SAVE, 'w') as f:
        json.dump(accuracy_data, f, indent=2)
    print(f"[SUCCESS] Accuracy saved: {ACC_SAVE}")

    print(f"\n[DONE] Total training time: {datetime.now() - startTime}")
