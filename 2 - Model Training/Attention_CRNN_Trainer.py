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

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
# SafeEarlyStopping ensures stable EarlyStopping behavior across callbacks
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from tools.safe_callbacks import SafeEarlyStopping

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
DATASET_DIR = Path(config.PROJECT_ROOT) / "0 - DADS dataset extraction"
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


def main():
    parser = argparse.ArgumentParser(description='Train Attention-Enhanced CRNN with configurable loss function')
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum learning (delegates to Universal_Curriculum_Trainer)')
    parser.add_argument('--curriculum_strategy', type=str, default='linear',
                       choices=['linear', 'step', 'exponential'],
                       help='Curriculum learning strategy (only with --curriculum)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Warmup epochs for curriculum learning (only with --curriculum)')
    parser.add_argument('--loss', type=str, default='focal', choices=['bce', 'weighted_bce', 'focal'],
                       help='Loss function: bce, weighted_bce, or focal (default: focal)')
    parser.add_argument('--class_weight', type=float, default=3.0,
                       help='Class weight for drone class (for weighted_bce, default: 3.0)')
    parser.add_argument('--focal_alpha', type=float, default=0.60,
                       help='Focal loss alpha parameter (default: 0.60 - balanced)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter (default: 2.0 - focus on hard examples)')
    parser.add_argument('--min_epochs', type=int, default=50, help='Minimum epochs before early stopping')
    parser.add_argument('--stratified-validation', action='store_true', help='Enable distance-stratified validation')
    parser.add_argument('--use_class_weight', action='store_true', help='Compute class weights and apply sample_weight to fit')
    parser.add_argument('--max_epochs', type=int, default=config.MAX_EPOCHS, help='Maximum number of epochs to run (default: from config.py)')
    parser.add_argument('--use_dynamic_weight', action='store_true', default=None, help='Enable dynamic positive-class weight scheduler (updates each epoch)')
    parser.add_argument('--dyn_base', type=float, default=1.0, help='Base weight multiplier for dynamic scheduler')
    parser.add_argument('--dyn_beta', type=float, default=0.8, help='EMA beta for dynamic scheduler (smoothing)')
    parser.add_argument('--dyn_min_w', type=float, default=0.01, help='Minimum clamp for dynamic positive weight')
    parser.add_argument('--dyn_max_w', type=float, default=10.0, help='Maximum clamp for dynamic positive weight')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Override batch size for training')
    args = parser.parse_args()
    

    # Intercept curriculum mode and delegate to Universal_Curriculum_Trainer
    if args.curriculum:
        print(colored("[CURRICULUM MODE] Delegating to Universal_Curriculum_Trainer", "cyan"))
        import subprocess
        
        # Build command for Universal_Curriculum_Trainer
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'Universal_Curriculum_Trainer.py'),
            '--model', 'attention_crnn'
        ]
        
        # Forward relevant arguments
        if args.loss:
            cmd.extend(['--loss', args.loss])
        if hasattr(args, 'focal_alpha') and args.focal_alpha:
            cmd.extend(['--focal_alpha', str(args.focal_alpha)])
        if hasattr(args, 'focal_gamma') and args.focal_gamma:
            cmd.extend(['--focal_gamma', str(args.focal_gamma)])
        if hasattr(args, 'batch_size') and args.batch_size:
            cmd.extend(['--batch_size', str(args.batch_size)])
        if hasattr(args, 'min_epochs') and args.min_epochs:
            cmd.extend(['--min_epochs', str(args.min_epochs)])
        if hasattr(args, 'max_epochs') and args.max_epochs:
            cmd.extend(['--max_epochs', str(args.max_epochs)])
        if args.curriculum_strategy:
            cmd.extend(['--curriculum_strategy', args.curriculum_strategy])
        if args.warmup_epochs:
            cmd.extend(['--warmup_epochs', str(args.warmup_epochs)])
        if hasattr(args, 'use_class_weight') and args.use_class_weight:
            cmd.append('--use_class_weight')
        if hasattr(args, 'verbose') and args.verbose is not None:
            cmd.extend(['--verbose', str(args.verbose)])
        # Conformer-specific
        if hasattr(args, 'blocks') and args.blocks:
            cmd.extend(['--blocks', str(args.blocks)])
        
        # Execute and exit
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    print(colored("\n" + "="*70, "magenta"))
    print(colored("  ATTENTION-ENHANCED CRNN TRAINING", "magenta"))
    print(colored("="*70, "magenta"))
    
    # Memory safety check
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_available_gb = mem.available / (1024**3)
        print(colored(f"\n[MEMORY] Available RAM: {mem_available_gb:.2f} GB", "cyan"))
        
        if mem_available_gb < 4.0:
            print(colored(f"[WARNING] Low memory detected! Reducing batch size to 2", "yellow"))
            if args.batch_size > 2:
                args.batch_size = 2
        elif mem_available_gb < 8.0:
            print(colored(f"[WARNING] Limited memory. Using batch size max 4", "yellow"))
            if args.batch_size > 4:
                args.batch_size = 4
    except ImportError:
        print(colored("[WARNING] psutil not available, skipping memory check", "yellow"))
    
    print(colored(f"\n[CONFIG] Batch size: {args.batch_size}", "yellow"))
    print(colored(f"[CONFIG] Loss function: {args.loss}", "yellow"))
    if args.loss == 'weighted_bce':
        print(colored(f"[CONFIG] Drone class weight: {args.class_weight}", "yellow"))
    elif args.loss == 'focal':
        print(colored(f"[CONFIG] Focal alpha: {args.focal_alpha}, gamma: {args.focal_gamma}", "yellow"))
    print(colored(f"[CONFIG] Min epochs: {args.min_epochs}", "yellow"))
    print(colored(f"[CONFIG] Stratified validation: {args.stratified_validation}", "yellow"))
    print()
    
    # Load datasets
    X_train, y_train, X_val, y_val = prepare_datasets()

    # Apply config defaults when CLI args are not specified
    try:
        args.focal_alpha = args.focal_alpha if args.focal_alpha is not None else config.LOSS_DEFAULTS.get('focal_alpha', 0.75)
        args.focal_gamma = args.focal_gamma if args.focal_gamma is not None else config.LOSS_DEFAULTS.get('focal_gamma', 2.0)
        args.class_weight = args.class_weight if args.class_weight is not None else config.LOSS_DEFAULTS.get('class_weight_drone', 3.0)
        if getattr(args, 'use_dynamic_weight', None) is None:
            args.use_dynamic_weight = config.LOSS_DEFAULTS.get('use_dynamic_weight', False)
    except Exception:
        pass

    # Optionally compute class weights and sample weights (Attention trainer uses one-hot labels)
    sample_weight = None
    # Do not apply static sample weights when dynamic scheduler is enabled
    if args.use_class_weight and not getattr(args, 'use_dynamic_weight', False):
        try:
            y_train_int = y_train.astype(int)
            classes = np.unique(y_train_int)
            cw = compute_class_weight('balanced', classes=classes, y=y_train_int)
            class_weight_map = {int(c): float(w) for c, w in zip(classes, cw)}
            # Build sample_weight array matching original training labels
            sample_weight = np.array([class_weight_map[int(lbl)] for lbl in y_train_int])
            print(colored(f"[CONFIG] Using class_weight map: {class_weight_map}", "cyan"))
        except Exception as e:
            print(colored(f"[WARN] Could not compute class_weight/sample_weight: {e}", "yellow"))

    # Add channel dimension for Conv2D if necessary
    if X_train.ndim == 3:
        X_train = X_train[..., np.newaxis]
    if X_val.ndim == 3:
        X_val = X_val[..., np.newaxis]

    # Build model using the actual training shape (like CNN_Trainer does)
    if X_train.ndim != 4:
        raise ValueError(f"X_train must be 4D after adding channel dim, got {X_train.shape}")

    # Convert labels to categorical
    y_train_int = y_train.astype(int)
    y_val_int = y_val.astype(int)
    y_train = keras.utils.to_categorical(y_train_int, num_classes=2)
    y_val = keras.utils.to_categorical(y_val_int, num_classes=2)

    # Build model
    print(colored("\n[INFO] Building Attention-Enhanced CRNN...", "cyan"))
    model = build_attention_crnn(input_shape=X_train.shape[1:3])
    
    # Get loss function based on arguments
    from loss_functions import get_loss_function
    
    if args.loss == 'focal':
        loss_fn = get_loss_function('focal', alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss == 'weighted_bce':
        loss_fn = get_loss_function('weighted_bce', class_weight_drone=args.class_weight)
    elif args.loss == 'recall_focused':
        loss_fn = get_loss_function('recall_focused', fn_penalty=50.0)
        print("[INFO] Using RECALL-FOCUSED LOSS (FN penalty=50x)")
    else:
        loss_fn = 'categorical_crossentropy'
    
    # Compile the network with improved loss and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy']
    )    
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_f1',
        mode='max',
        patience=10,
        start_from_epoch=args.min_epochs,
        restore_best_weights=True
    )
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ckpt_f1 = ModelCheckpoint(
        MODEL_SAVE,
        monitor='val_f1',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    try:
        if str(MODEL_SAVE).endswith('.keras'):
            ckpt_loss_path = str(MODEL_SAVE).replace('.keras', '.best_val_loss.keras')
        else:
            ckpt_loss_path = str(MODEL_SAVE) + '.best_val_loss'
    except Exception:
        ckpt_loss_path = str(MODEL_SAVE) + '.best_val_loss'
    ckpt_loss = ModelCheckpoint(
        ckpt_loss_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # Add a SafeEarlyStopping on val_loss to stop long runs if val_loss plateaus
    try:
        loss_early_stop = SafeEarlyStopping(start_from_epoch=getattr(args, 'min_epochs', 0), monitor='val_loss', mode='min', patience=8, restore_best_weights=True)
    except Exception:
        loss_early_stop = SafeEarlyStopping(monitor='val_loss', mode='min', patience=8, restore_best_weights=True)
    # Distance-stratified validation callback (optional)
    callbacks_list = [early_stopping, lr_reducer, loss_early_stop, ckpt_f1, ckpt_loss]
    
    # Distance-stratified validation
    if args.stratified_validation:
        from distance_stratified_callback import DistanceStratifiedCallback
        stratified_callback = DistanceStratifiedCallback(
            validation_dir=VAL_DATA_PATH,
            model_name="attention_crnn",
            log_dir=str(config.RESULTS_DIR)
        )
        callbacks_list.append(stratified_callback)
        print(colored("[INFO] Distance-stratified validation ENABLED", "green"))

    # Dynamic class-weight scheduler (optional)
    if args.use_dynamic_weight:
        try:
            from dynamic_class_weight import DynamicClassWeightCallback
            # Attention trainer converted y_val to categorical later; use int labels
            y_val_int = np.asarray(y_val).astype(int)
            dyn_cb = DynamicClassWeightCallback((X_val, y_val_int), base_weight=args.dyn_base,
                                                beta=args.dyn_beta, min_w=args.dyn_min_w, max_w=args.dyn_max_w,
                                                batch_size=args.batch_size)
            callbacks_list.append(dyn_cb)
            print(colored("[INFO] Dynamic positive-class weight scheduler ENABLED", "cyan"))
        except Exception as e:
            print(colored(f"[WARN] Could not enable dynamic weight callback: {e}", "yellow"))
    
    # Train
    print(colored("\n[INFO] Starting training...", "cyan"))
    fit_kwargs = dict(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=1000,
        callbacks=callbacks_list,
        verbose=2
    )
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight

    # Add epoch-end validation metrics callback (handles one-hot targets)
    from sklearn.metrics import precision_score, recall_score, confusion_matrix

    class ValMetricsCallback(keras.callbacks.Callback):
        def __init__(self, val_data):
            super().__init__()
            self.x_val, self.y_val = val_data

        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(self.x_val, verbose=0)
            preds = np.argmax(preds, axis=1)
            # y_val may be categorical one-hot or ints
            y = np.array(self.y_val)
            if y.ndim > 1 and y.shape[-1] == 2:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y.astype(int)
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
            print(colored(f"[EPOCH {epoch}] val_precision={precision:.4f} val_recall={recall:.4f} val_f1={f1:.4f} tp={tp} fp={fp} tn={tn} fn={fn}", "magenta"))

    # Prepend ValMetricsCallback so it injects `val_f1` into `logs` before other callbacks
    val_metrics_cb = ValMetricsCallback((X_val, y_val))
    callbacks_list = [val_metrics_cb] + callbacks_list

    # Respect max_epochs argument
    fit_kwargs['epochs'] = args.max_epochs
    history = model.fit(**fit_kwargs)
    
    # Save history
    hist_df = pd.DataFrame(history.history)
    os.makedirs(os.path.dirname(HISTORY_SAVE), exist_ok=True)
    hist_df.to_csv(HISTORY_SAVE)
    
    print(colored("\n" + "="*70, "green"))
    print(colored("  ✓ TRAINING COMPLETED", "green"))
    print(colored("="*70, "green"))
    print(colored(f"Model saved: {MODEL_SAVE}", "cyan"))
    print(colored(f"History saved: {HISTORY_SAVE}", "cyan"))
    print(colored(f"Total epochs: {len(history.history['loss'])}", "yellow"))
    print(colored(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}", "yellow"))

    # Save training metadata if configured
    try:
        if getattr(config, 'SAVE_TRAINING_METADATA', False):
            meta = {
                'loss': args.loss,
                'focal_alpha': float(args.focal_alpha),
                'focal_gamma': float(args.focal_gamma),
                'class_weight': float(args.class_weight) if args.class_weight is not None else None,
                'use_dynamic_weight': bool(getattr(args, 'use_dynamic_weight', False)),
                'model_path': str(MODEL_SAVE),
            }
            meta_path = config.RESULTS_DIR / 'attention_crnn_training_metadata.json'
            with open(meta_path, 'w') as _m:
                json.dump(meta, _m, indent=2)
            print(colored(f"[INFO] Training metadata saved: {meta_path}", "cyan"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
