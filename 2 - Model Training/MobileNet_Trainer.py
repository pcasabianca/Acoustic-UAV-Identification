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


def build_model(input_shape=(44, 90, 1)):
    """
    Build MobileNet-Audio model
    Lightweight architecture optimized for audio with depthwise separable convolutions
    """
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'weighted_bce', 'focal', 'recall_focused'],
                        help='Loss function to use')
    parser.add_argument('--epochs', type=int, default=config.MAX_EPOCHS, help='Number of epochs')
    parser.add_argument('--use_dynamic_weight', action='store_true', default=True,
                        help='Enable dynamic positive-class weight scheduler (updates each epoch)')
    parser.add_argument('--dyn_base', type=float, default=1.0,
                        help='Base weight multiplier for dynamic scheduler')
    parser.add_argument('--dyn_beta', type=float, default=0.8,
                        help='EMA beta for dynamic scheduler (smoothing)')
    parser.add_argument('--dyn_min_w', type=float, default=0.01,
                        help='Minimum clamp for dynamic positive weight')
    parser.add_argument('--dyn_max_w', type=float, default=10.0,
                        help='Maximum clamp for dynamic positive weight')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Training batch size')
    args = parser.parse_args()

    # Prepare datasets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets()

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # Get loss and metrics
    loss_fn = get_loss_function(args.loss)
    metrics = get_metrics()

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # Print model summary
    print("\n" + "="*80)
    print("MOBILENET-AUDIO MODEL")
    print("="*80)
    model.summary()
    print("="*80 + "\n")

    # Compute balanced class weights (adaptive utility removed)
    class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights_array))
    print(f"[INFO] Class weights (balanced): {class_weights}")

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    callbacks = [early_stop, reduce_lr]

    # Dynamic scheduler (enabled by default)
    if getattr(args, 'use_dynamic_weight', False):
        try:
            from dynamic_class_weight import DynamicClassWeightCallback
            dyn_cb = DynamicClassWeightCallback((X_validation, y_validation),
                                                base_weight=args.dyn_base,
                                                beta=args.dyn_beta,
                                                min_w=args.dyn_min_w,
                                                max_w=args.dyn_max_w,
                                                batch_size=args.batch_size)
            callbacks.append(dyn_cb)
            print(f"[INFO] Dynamic positive-class weight scheduler ENABLED")
        except Exception as e:
            print(f"[WARN] Could not enable dynamic weight callback: {e}")

    # Train
    print(f"\n[INFO] Training MobileNet-Audio with {args.loss} loss, batch_size={args.batch_size}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_validation, y_validation),
        batch_size=args.batch_size,
        epochs=args.epochs,
        class_weight=(None if getattr(args, 'use_dynamic_weight', False) else class_weights),
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
        "learning_rate": LEARNING_RATE
    }
    with open(ACC_SAVE, 'w') as f:
        json.dump(accuracy_data, f, indent=2)
    print(f"[SUCCESS] Accuracy saved: {ACC_SAVE}")

    print(f"\n[DONE] Total training time: {datetime.now() - startTime}")
