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

# Dimensions for input data
N_MELS = config.MEL_N_MELS
TIME_FRAMES = config.MEL_TIME_FRAMES

# Path to created json file from mel preprocess and feature extraction script.
TRAIN_DATA_PATH = config.MEL_TRAIN_DATA_PATH_STR
TRAIN_INDEX_PATH = config.MEL_TRAIN_INDEX_PATH_STR
VAL_DATA_PATH = config.MEL_VAL_DATA_PATH_STR
VAL_INDEX_PATH = config.MEL_VAL_INDEX_PATH_STR

# Path to save model.
MODEL_SAVE = config.CRNN_MODEL_PATH_STR

# Path to save training history and model accuracy performance at end of training.
HISTORY_SAVE = config.CRNN_HISTORY_PATH_STR
ACC_SAVE = config.CRNN_ACC_PATH_STR

# Training hyperparams (use config when available)
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 8)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0001)


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

    # 3D array.
    X_train = X_train[..., np.newaxis]  # 4-dim array: (# samples, # time steps, # coefficients, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # Create model.
    model = keras.Sequential()

    # 1st convolutional layer.
    model.add(keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=input_shape))
        # 16 kernels, and 5x5 grid size of kernel.
    model.add(keras.layers.MaxPool2D((5, 5), strides=(2, 2), padding='same'))
        # Pooling size 5x5.
    model.add(keras.layers.BatchNormalization())
        # Batch Normalization allows model to be more accurate and computations are faster.

    # Resize for RNN part.
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(keras.layers.Reshape((model.output_shape[1], resize_shape)))

    # RNN layer.
    model.add(keras.layers.LSTM(32, input_shape=input_shape, return_sequences=True, use_cudnn=False))

    # Flatten the output and feed into dense layer.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
        # 32 neurons.
    model.add(keras.layers.Dropout(0.3))
        # Reduces chances of over fitting.

    # Output layer that uses softmax activation.
    model.add(keras.layers.Dense(2, activation='softmax'))
        # 2 neurons --> depends on how many categories we want to predict.

    return model


def predict(model, X, y):
    # Random prediction post-training.
    X = X[np.newaxis, ...]

    prediction = model.predict(X)

    # Extract index with max value.
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CRNN model with configurable loss function')
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
    parser.add_argument('--use_dynamic_weight', action='store_true', default=None,
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
        if getattr(args, 'use_dynamic_weight', None) is None:
            args.use_dynamic_weight = _cfg.LOSS_DEFAULTS.get('use_dynamic_weight', False)
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

    # Early stopping (monitor val_f1) and learning-rate reducer (monitor val_loss)
    from tools.safe_callbacks import SafeEarlyStopping
    callback = SafeEarlyStopping(start_from_epoch=args.min_epochs, monitor='val_f1', mode='max', patience=10, restore_best_weights=True)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    # Primary checkpoint: best by val_f1. Secondary checkpoint: archive best val_loss.
    ckpt_f1 = keras.callbacks.ModelCheckpoint(MODEL_SAVE, monitor='val_f1', mode='max', save_best_only=True, verbose=1)
    try:
        if str(MODEL_SAVE).endswith('.keras'):
            ckpt_loss_path = str(MODEL_SAVE).replace('.keras', '.best_val_loss.keras')
        else:
            ckpt_loss_path = str(MODEL_SAVE) + '.best_val_loss'
    except Exception:
        ckpt_loss_path = str(MODEL_SAVE) + '.best_val_loss'
    ckpt_loss = keras.callbacks.ModelCheckpoint(ckpt_loss_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    # Add a SafeEarlyStopping on val_loss to stop on validation loss plateau as a fallback
    try:
        safe_es_loss = SafeEarlyStopping(start_from_epoch=args.min_epochs, monitor='val_loss', mode='min', patience=8, restore_best_weights=True)
    except Exception:
        safe_es_loss = SafeEarlyStopping(monitor='val_loss', mode='min', patience=8, restore_best_weights=True)
    # Distance-stratified validation callback (optional)
    callbacks_list = [callback, lr_reducer, safe_es_loss, ckpt_f1, ckpt_loss]
    if args.stratified_validation:
        from distance_stratified_callback import DistanceStratifiedCallback
        stratified_callback = DistanceStratifiedCallback(
            validation_dir=VAL_DATA_PATH,
            model_name="crnn",
            log_dir=str(config.RESULTS_DIR)
        )
        callbacks_list.append(stratified_callback)
        print(colored("[INFO] Distance-stratified validation ENABLED", "green"))

    # Dynamic class-weight scheduler (optional)
    if args.use_dynamic_weight:
        try:
            from dynamic_class_weight import DynamicClassWeightCallback
            dyn_cb = DynamicClassWeightCallback((X_validation, y_validation), base_weight=1.0,
                                                beta=0.8, min_w=0.5, max_w=8.0,
                                                batch_size=BATCH_SIZE)
            callbacks_list.append(dyn_cb)
            print(colored("[INFO] Dynamic positive-class weight scheduler ENABLED", "cyan"))
        except Exception as e:
            print(colored(f"[WARN] Could not enable dynamic weight callback: {e}", "yellow"))

    # Build the CRNN network.
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
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
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.summary()

    # Train the CRNN.
    # Callback: compute validation precision/recall and confusion per epoch for visibility
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
                # Fallback if confusion matrix shape differs
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
    val_metrics_cb = ValMetricsCallback((X_validation, y_validation))
    callbacks_list = [val_metrics_cb] + callbacks_list
    # Dynamic callback attachment (use CLI params)
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
            meta_path = _cfg.RESULTS_DIR / 'crnn_training_metadata.json'
            with open(meta_path, 'w') as _m:
                _json.dump(meta, _m, indent=2)
            print(colored(f"[INFO] Training metadata saved: {meta_path}", "cyan"))
    except Exception:
        pass

    print(
        colored("CRNN model has been trained and its training history has been saved to {}.".format(hist_csv), "green"))

    # Evaluate the CNN on the test set.
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # Timer output.
    time = datetime.now() - startTime
    print(time)

    # Make prediction on a random sample.
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)

    # Save model accuracies on test set (for weight calculations later on).
    accuracy = {
        "model_acc": [],
        "model_loss": [],
        "total_train_time": [],
    }

    accuracy["model_acc"].append(test_accuracy)
    accuracy["model_loss"].append(test_error)
    accuracy["total_train_time"].append(str(time))

    with open(ACC_SAVE, "w") as fp:
        json.dump(accuracy, fp, indent=4)