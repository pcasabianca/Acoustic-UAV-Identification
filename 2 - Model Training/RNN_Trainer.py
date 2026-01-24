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

# Optimize CPU threads for 12-core CPU (use all cores)
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Enable GPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth config failed: {e}")

# Add project root to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

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

    # 3 Bidirectional LSTM layers
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True), input_shape=input_shape))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100)))

    # Dense layers
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='softmax'))

    return model


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RNN model with configurable loss function')
    parser.add_argument('--loss', type=str, default='focal', choices=['bce', 'weighted_bce', 'focal'],
                       help='Loss function: bce, weighted_bce, or focal (default: focal)')
    parser.add_argument('--class_weight', type=float, default=3.0,
                       help='Class weight for drone class (for weighted_bce, default: 3.0)')
    parser.add_argument('--focal_alpha', type=float, default=0.60,
                       help='Focal loss alpha parameter (default: 0.60 - balanced)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter (default: 2.0 - focus on hard examples)')
    parser.add_argument('--min_epochs', type=int, default=config.MIN_EPOCHS,
                       help='Minimum number of epochs before early stopping (default: from config.py)')
    parser.add_argument('--max_epochs', type=int, default=config.MAX_EPOCHS,
                       help='Maximum number of epochs to run (default: from config.py)')
    parser.add_argument('--stratified-validation', action='store_true',
                       help='Enable distance-stratified validation (evaluates each distance separately)')
    parser.add_argument('--use_class_weight', action='store_true',
                       help='Automatically compute and pass class_weight to Keras (balanced)')
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
                       help='Override batch size for training')
    parser.add_argument('--verbose', type=int, default=2,
                       help='Verbosity mode for Keras fit: 0=silent,1=per-batch,2=per-epoch (default: 2)')
    
    args = parser.parse_args()
    
    print(colored(f"\n[CONFIG] Loss function: {args.loss}", "cyan"))
    # Apply config defaults when CLI args are not specified
    try:
        import config as _cfg
        args.focal_alpha = args.focal_alpha if args.focal_alpha is not None else _cfg.LOSS_DEFAULTS.get('focal_alpha', 0.75)
        args.focal_gamma = args.focal_gamma if args.focal_gamma is not None else _cfg.LOSS_DEFAULTS.get('focal_gamma', 2.0)
        args.class_weight = args.class_weight if args.class_weight is not None else _cfg.LOSS_DEFAULTS.get('class_weight_drone', 3.0)
        if not hasattr(args, 'use_dynamic_weight'):
            args.use_dynamic_weight = _cfg.LOSS_DEFAULTS.get('use_dynamic_weight', True)
    except Exception:
        pass
    if args.loss == 'weighted_bce':
        print(colored(f"[CONFIG] Drone class weight: {args.class_weight}", "cyan"))
    elif args.loss == 'focal':
        print(colored(f"[CONFIG] Focal alpha: {args.focal_alpha}, gamma: {args.focal_gamma}", "cyan"))
    print()
    
    # Create train, validation and test sets.
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets()

    # Optionally compute class weights for balanced training
    # adaptive class-weight utility removed; compute balanced class_weight when requested
    class_weight = None

    # Fallback: compute balanced class_weight if requested and adaptive not present
    if class_weight is None and args.use_class_weight:
        try:
            classes = np.unique(y_train)
            cw = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
            print(colored(f"[CONFIG] Using class_weight (balanced): {class_weight}", "cyan"))
        except Exception as e:
            print(colored(f"[WARN] Could not compute class_weight: {e}", "yellow"))

    # Early stopping with minimum epochs.
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=args.min_epochs)

    # Checkpoint.
    checkpoint = keras.callbacks.ModelCheckpoint(MODEL_SAVE, monitor='val_loss',
                                                  mode='min', save_best_only=True, verbose=1)
    
    # Distance-stratified validation callback (optional)
    callbacks_list = [callback, checkpoint]
    if args.stratified_validation:
        from distance_stratified_callback import DistanceStratifiedCallback
        stratified_callback = DistanceStratifiedCallback(
            validation_dir=VAL_DATA_PATH,
            model_name="rnn",
            log_dir=str(config.RESULTS_DIR)
        )
        callbacks_list.append(stratified_callback)
        print(colored("[INFO] Distance-stratified validation ENABLED", "green"))

    # Build the RNN network.
    # Data shape: (samples, n_mels, time_frames) = (N, 44, 173)
    # Input shape for LSTM: (time_steps, features) = (44, 173)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # Get loss function based on arguments
    if args.loss == 'focal':
        loss_fn = get_loss_function('focal', alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss == 'weighted_bce':
        loss_fn = get_loss_function('weighted_bce', class_weight_drone=args.class_weight)
    else:
        loss_fn = 'sparse_categorical_crossentropy'

    # Validate labels vs selected loss
    try:
        import numpy as _np
        y_arr = _np.array(y_train)
        is_onehot = (y_arr.ndim > 1 and y_arr.shape[-1] == 2)
        if is_onehot and isinstance(loss_fn, str) and loss_fn == 'sparse_categorical_crossentropy':
            raise RuntimeError("Label format appears one-hot but loss is 'sparse_categorical_crossentropy'. Convert labels to ints or choose a categorical-compatible loss.")
    except Exception as _e:
        print(colored(f"[WARN] Label vs loss validation: {_e}", "yellow"))
    
    # Compile the network with improved loss and metrics
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimiser,
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.summary()

    # Train the RNN.
    # Add epoch-end validation metrics callback for precision/recall/confusion
    from sklearn.metrics import precision_score, recall_score, confusion_matrix

    class ValMetricsCallback(keras.callbacks.Callback):
        def __init__(self, val_data):
            super().__init__()
            self.x_val, self.y_val = val_data

        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(self.x_val, verbose=0)
            preds = np.argmax(preds, axis=1)
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
            print(colored(f"[EPOCH {epoch}] val_precision={precision:.4f} val_recall={recall:.4f} tp={tp} fp={fp} tn={tn} fn={fn}", "magenta"))

    callbacks_list.append(ValMetricsCallback((X_validation, y_validation)))
    if args.use_dynamic_weight:
        try:
            from dynamic_class_weight import DynamicClassWeightCallback
            dyn_cb = DynamicClassWeightCallback((X_validation, y_validation),
                                                base_weight=args.dyn_base,
                                                beta=args.dyn_beta,
                                                min_w=args.dyn_min_w,
                                                max_w=args.dyn_max_w,
                                                batch_size=args.batch_size)
            callbacks_list.append(dyn_cb)
            print(colored("[INFO] Dynamic positive-class weight scheduler ENABLED", "cyan"))
        except Exception as e:
            print(colored(f"[WARN] Could not enable dynamic weight callback: {e}", "yellow"))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        callbacks=callbacks_list,
        class_weight=(None if getattr(args, 'use_dynamic_weight', False) else class_weight),
        verbose=int(getattr(args, 'verbose', 2))
    )

    # Save history.
    hist = pd.DataFrame(history.history)

    # Save to csv:
    hist_csv = HISTORY_SAVE
    with open(hist_csv, mode='w') as f:
        hist.to_csv(f)

    print(colored("Paper 8's model has been trained and its training history has been saved to {}.".format(hist_csv), "green"))

    # Evaluate the RNN on the test set.
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # Timer output.
    time = datetime.now() - startTime
    print(time)

    # Save model accuracies on test set.
    accuracy = {
        "model_acc": [],
        "total_train_time": [],
    }

    accuracy["model_acc"].append(test_accuracy)
    accuracy["total_train_time"].append(str(time))

    with open(ACC_SAVE, "w") as fp:
        json.dump(accuracy, fp, indent=4)

    # Save training metadata if configured
    try:
        import config as _cfg
        if getattr(_cfg, 'SAVE_TRAINING_METADATA', False):
            import json as _json
            meta = {
                'loss': args.loss,
                'focal_alpha': float(args.focal_alpha),
                'focal_gamma': float(args.focal_gamma),
                'class_weight': float(args.class_weight) if args.class_weight is not None else None,
                'use_dynamic_weight': bool(getattr(args, 'use_dynamic_weight', False)),
                'model_path': str(MODEL_SAVE),
            }
            meta_path = _cfg.RESULTS_DIR / 'rnn_training_metadata.json'
            with open(meta_path, 'w') as _m:
                _json.dump(meta, _m, indent=2)
            print(colored(f"[INFO] Training metadata saved: {meta_path}", "cyan"))
    except Exception:
        pass
