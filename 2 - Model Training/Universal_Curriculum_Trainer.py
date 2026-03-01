#!/usr/bin/env python3
"""
Universal Curriculum Learning Trainer

Implements curriculum learning by ordering training samples by difficulty score.
Progressively exposes the model to harder samples during training.

Curriculum Strategy:
  - Start with easy samples (low difficulty scores)
  - Gradually introduce medium difficulty samples
  - Finally include hard samples in later epochs
  
Usage:
  python Universal_Curriculum_Trainer.py --model cnn --curriculum_strategy linear
  python Universal_Curriculum_Trainer.py --model efficientnet --curriculum_strategy step --batch_size 16
"""
import json
import numpy as np
import argparse
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from datetime import datetime
from termcolor import colored
import sys
from pathlib import Path
from typing import Tuple, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Optimize CPU threads (use config values)
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

# Import feature loader and loss functions
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from feature_loader import load_mel_features_with_metadata
from loss_functions import get_loss_function, get_metrics

# Import build_model functions from original trainers
# We'll dynamically import these based on model_type to avoid circular dependencies
_MODEL_BUILDERS = {}  # Cache for imported build functions

# Timer
startTime = datetime.now()

# Paths
TRAIN_DATA_PATH = config.MEL_TRAIN_DATA_PATH_STR
VAL_DATA_PATH = config.MEL_VAL_DATA_PATH_STR

# Default hyperparams
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 8)
LEARNING_RATE = getattr(config, 'LEARNING_RATE', 0.0001)


class CurriculumScheduler(keras.callbacks.Callback):
    """
    Callback to implement curriculum learning during training.
    
    Progressively increases the difficulty of training samples across epochs.
    """
    
    def __init__(self, 
                 difficulty_scores: np.ndarray,
                 total_epochs: int,
                 labels: Optional[np.ndarray] = None,
                 strategy: str = 'linear',
                 warmup_epochs: int = 5,
                 initial_fraction: float = 0.5,
                 growth_rate: float = 0.02,
                 verbose: int = 1):
        """
        Args:
            difficulty_scores: Array of difficulty scores for each training sample
            total_epochs: Total number of epochs planned (for reaching 100%)
            strategy: 'linear', 'step', or 'exponential'
            warmup_epochs: Number of initial epochs to use easy samples only
            initial_fraction: Initial fraction of samples (e.g., 0.25 = 25%)
            growth_rate: Growth rate per epoch for linear strategy (e.g., 0.05 = 5%)
            verbose: Verbosity level (0=silent, 1=progress)
        """
        super().__init__()
        self.difficulty_scores = difficulty_scores
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.initial_fraction = initial_fraction
        self.growth_rate = growth_rate
        self.verbose = verbose
        
        # Sort indices by difficulty (ascending)
        self.sorted_indices = np.argsort(difficulty_scores)
        self.total_samples = len(difficulty_scores)

        # Optional: labels for stratified selection within curriculum
        self.labels = None if labels is None else np.asarray(labels)
        self.classwise_sorted = None
        if self.labels is not None:
            # build per-class sorted index lists (preserve per-class difficulty ordering)
            classes = np.unique(self.labels)
            self.classwise_sorted = {}
            for c in classes:
                idxs = np.where(self.labels == c)[0]
                # sort these indices by difficulty_scores
                sorted_local = idxs[np.argsort(difficulty_scores[idxs])]
                self.classwise_sorted[c] = list(sorted_local)
        
    def on_epoch_begin(self, epoch, logs=None):
        """Compute the subset of samples to use for this epoch."""
        # Note: warmup_epochs sert seulement pour early stopping,
        # PAS pour bloquer la progression curriculum
        
        if self.strategy == 'linear':
            # Linear growth: initial_fraction → 100% with growth_rate per epoch
            # Example: 25% + 5% per epoch = 100% at epoch 15
            current_fraction = min(1.0, self.initial_fraction + epoch * self.growth_rate)
            num_samples = int(self.total_samples * current_fraction)
            
        elif self.strategy == 'step':
            # Step growth: add samples in discrete jumps from initial_fraction
            quarter = max(1, self.total_epochs // 4)
            if epoch < quarter:
                num_samples = int(self.total_samples * self.initial_fraction)
            elif epoch < 2 * quarter:
                num_samples = int(self.total_samples * (self.initial_fraction + (1.0 - self.initial_fraction) * 0.33))
            elif epoch < 3 * quarter:
                num_samples = int(self.total_samples * (self.initial_fraction + (1.0 - self.initial_fraction) * 0.67))
            else:
                num_samples = self.total_samples
                
        elif self.strategy == 'exponential':
            # Exponential growth: slow at first from initial_fraction, then rapid
            if self.total_epochs <= 1:
                progress = 1.0
            else:
                progress = min(1.0, epoch / (self.total_epochs - 1))
            # Scale from initial_fraction to 1.0 with exponential curve
            current_fraction = self.initial_fraction + (1.0 - self.initial_fraction) * (progress ** 2)
            num_samples = int(self.total_samples * current_fraction)
            
        else:
            # Default: use all samples
            num_samples = self.total_samples
        
        self.current_num_samples = num_samples

        # If labels were provided, perform stratified per-class selection preserving
        # per-class difficulty ordering, and interleave classes to avoid mono-class blocks
        if self.classwise_sorted is not None and num_samples < self.total_samples:
            classes = list(self.classwise_sorted.keys())
            n_classes = len(classes)

            # Base quota per class, distribute remainder starting from first class
            base = num_samples // n_classes
            remainder = num_samples % n_classes
            quotas = {c: base for c in classes}
            for i, c in enumerate(classes):
                if i < remainder:
                    quotas[c] += 1

            # Select up to quota from each class (preserve class-local ordering)
            selected_per_class = {}
            total_selected = 0
            for c in classes:
                avail = len(self.classwise_sorted[c])
                take = min(quotas[c], avail)
                selected_per_class[c] = self.classwise_sorted[c][:take]
                total_selected += take

            # If we couldn't fill num_samples (some classes short), fill remainder from other classes
            if total_selected < num_samples:
                needed = num_samples - total_selected
                # iterate classes and take more where possible
                for c in classes:
                    avail = len(self.classwise_sorted[c]) - len(selected_per_class[c])
                    if avail <= 0:
                        continue
                    add = min(avail, needed)
                    selected_per_class[c] = self.classwise_sorted[c][:len(selected_per_class[c]) + add]
                    needed -= add
                    if needed == 0:
                        break

            # Interleave selections round-robin to mix classes
            pointers = {c: 0 for c in classes}
            mixed = []
            while len(mixed) < num_samples:
                progressed = False
                for c in classes:
                    lst = selected_per_class.get(c, [])
                    p = pointers[c]
                    if p < len(lst) and len(mixed) < num_samples:
                        mixed.append(int(lst[p]))
                        pointers[c] += 1
                        progressed = True
                if not progressed:
                    break

            self.current_indices = np.array(mixed, dtype=int)
        else:
            self.current_indices = self.sorted_indices[:num_samples]
        
        # Les infos de curriculum sont affichées dans le header principal
        # (pas besoin de message supplémentaire ici)


def load_data_with_difficulty() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training and validation data with difficulty metadata.
    
    Returns:
        train_mels, train_labels, train_difficulty, val_mels, val_labels, val_difficulty
    """
    print(colored("[INFO] Loading validation features with metadata...", "cyan"))
    val_mels, val_labels, val_metadata = load_mel_features_with_metadata('val')
    
    if len(val_mels) == 0:
        raise RuntimeError("No validation MEL features found")
    
    print(colored("[INFO] Loading training features with metadata...", "cyan"))
    train_mels, train_labels, train_metadata = load_mel_features_with_metadata('train')
    
    if len(train_mels) == 0:
        raise RuntimeError("No training MEL features found")
    
    # Extract difficulty scores
    train_difficulty = np.array([m.get('difficulty', 0.5) for m in train_metadata])
    val_difficulty = np.array([m.get('difficulty', 0.5) for m in val_metadata])
    
    print(colored(
        f"[INFO] Loaded {len(train_mels)} train samples, {len(val_mels)} val samples",
        "green"
    ))
    print(colored(
        f"[INFO] Train difficulty: mean={np.mean(train_difficulty):.3f}, "
        f"std={np.std(train_difficulty):.3f}, range=[{np.min(train_difficulty):.3f}, {np.max(train_difficulty):.3f}]",
        "green"
    ))
    
    return train_mels, train_labels, train_difficulty, val_mels, val_labels, val_difficulty


def prepare_datasets_curriculum(curriculum_strategy: str = 'linear'):
    """Prepare datasets for curriculum learning."""
    train_mels, train_labels, train_difficulty, val_mels, val_labels, _ = load_data_with_difficulty()
    
    # Add channel dimension
    train_mels = train_mels[..., np.newaxis]
    val_mels = val_mels[..., np.newaxis]
    
    return train_mels, train_labels, train_difficulty, val_mels, val_labels


def get_model_builder(model_type: str):
    """
    Dynamically import and return the build_model function from the appropriate trainer.
    
    Args:
        model_type: One of 'cnn', 'rnn', 'crnn', 'attention_crnn', 'efficientnet',
               'mobilenet', 'conformer', 'tcn', 'ast', 'multires'
    
    Returns:
        build_model function from the original trainer
    """
    global _MODEL_BUILDERS
    
    if model_type in _MODEL_BUILDERS:
        return _MODEL_BUILDERS[model_type]
    
    # Import the appropriate trainer module
    trainer_dir = Path(__file__).parent
    
    if model_type == 'cnn':
        import importlib.util
        spec = importlib.util.spec_from_file_location("cnn_trainer", trainer_dir / "CNN_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['cnn'] = module.build_model
        
    elif model_type == 'rnn':
        import importlib.util
        spec = importlib.util.spec_from_file_location("rnn_trainer", trainer_dir / "RNN_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['rnn'] = module.build_model
        
    elif model_type == 'crnn':
        import importlib.util
        spec = importlib.util.spec_from_file_location("crnn_trainer", trainer_dir / "CRNN_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['crnn'] = module.build_model
        
    elif model_type == 'attention_crnn':
        import importlib.util
        spec = importlib.util.spec_from_file_location("attention_crnn_trainer", trainer_dir / "Attention_CRNN_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['attention_crnn'] = module.build_attention_crnn
        
    elif model_type == 'efficientnet':
        import importlib.util
        spec = importlib.util.spec_from_file_location("efficientnet_trainer", trainer_dir / "EfficientNet_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['efficientnet'] = module.build_model
        
    elif model_type == 'mobilenet':
        import importlib.util
        spec = importlib.util.spec_from_file_location("mobilenet_trainer", trainer_dir / "MobileNet_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['mobilenet'] = module.build_model
        
    elif model_type == 'conformer':
        import importlib.util
        spec = importlib.util.spec_from_file_location("conformer_trainer", trainer_dir / "Conformer_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['conformer'] = module.build_model
        
    elif model_type == 'tcn':
        import importlib.util
        spec = importlib.util.spec_from_file_location("tcn_trainer", trainer_dir / "TCN_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['tcn'] = module.build_model

    elif model_type == 'ast':
        import importlib.util
        spec = importlib.util.spec_from_file_location("ast_trainer", trainer_dir / "AST_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['ast'] = module.build_model

    elif model_type == 'multires':
        import importlib.util
        spec = importlib.util.spec_from_file_location("multires_trainer", trainer_dir / "MultiRes_Trainer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODEL_BUILDERS['multires'] = module.build_model
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: "
                        f"cnn, rnn, crnn, attention_crnn, efficientnet, mobilenet, conformer, tcn, ast, multires")
    
    return _MODEL_BUILDERS[model_type]


def build_model_from_trainer(input_shape: Tuple, model_type: str, **kwargs):
    """
    Build model using the original trainer's build_model function.
    
    Args:
        input_shape: Input shape (varies by model - some use channels, some don't)
        model_type: Model architecture identifier
        **kwargs: Additional arguments for specific trainers (e.g., num_conformer_blocks)
    
    Returns:
        Keras model built by the original trainer
    """
    build_fn = get_model_builder(model_type)
    
    # Handle different input shape requirements
    # Some models expect (n_mels, time_frames, channels), others (n_mels, time_frames)
    if model_type in ['conformer', 'attention_crnn']:
        # These models don't use channel dimension in input_shape parameter
        if len(input_shape) == 3:
            input_shape_adjusted = input_shape[:-1]  # Remove channel
        else:
            input_shape_adjusted = input_shape
    else:
        input_shape_adjusted = input_shape
    
    # Call build_model with appropriate signature
    try:
        if model_type == 'conformer':
            # Conformer has optional num_conformer_blocks parameter
            num_blocks = kwargs.get('num_conformer_blocks', 4)
            model = build_fn(input_shape=input_shape_adjusted, num_conformer_blocks=num_blocks)
        elif model_type == 'attention_crnn':
            # Attention_CRNN has num_classes parameter
            num_classes = kwargs.get('num_classes', 2)
            model = build_fn(input_shape=input_shape_adjusted, num_classes=num_classes)
        else:
            # Standard build_model(input_shape)
            model = build_fn(input_shape_adjusted)
        
        return model
    except Exception as e:
        print(colored(f"[ERROR] Failed to build {model_type} model: {e}", "red"))
        raise


def main():
    parser = argparse.ArgumentParser(description='Universal Curriculum Learning Trainer')
    
    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                       choices=['cnn', 'rnn', 'crnn', 'attention_crnn', 'efficientnet', 
                       'mobilenet', 'conformer', 'tcn', 'ast', 'multires'],
                       help='Model architecture to train')
    
    # Model-specific parameters
    parser.add_argument('--blocks', type=int, default=6,
                       help='Number of conformer blocks (conformer only)')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of output classes (for attention_crnn)')
    
    # Curriculum learning parameters
    parser.add_argument('--curriculum_strategy', type=str, default='linear',
                       choices=['linear', 'step', 'exponential', 'none'],
                       help='Curriculum learning strategy')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs with easy samples only')
    
    # Loss function
    parser.add_argument('--loss', type=str, default='focal',
                       choices=['bce', 'weighted_bce', 'focal'],
                       help='Loss function')
    parser.add_argument('--focal_alpha', type=float, default=0.60,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--min_epochs', type=int, default=config.MIN_EPOCHS,
                       help='Minimum epochs before early stopping')
    parser.add_argument('--max_epochs', type=int, default=config.MAX_EPOCHS,
                       help='Maximum epochs to train')
    
    # Other options
    parser.add_argument('--use_class_weight', action='store_true',
                       help='Use balanced class weights')
    parser.add_argument('--verbose', type=int, default=2,
                       help='Verbosity: 0=silent, 1=per-batch, 2=per-epoch')
    
    args = parser.parse_args()
    
    print(colored("\n" + "="*80, "magenta"))
    print(colored("  UNIVERSAL CURRICULUM LEARNING TRAINER", "magenta"))
    print(colored("="*80 + "\n", "magenta"))
    
    print(colored(f"[CONFIG] Model: {args.model}", "cyan"))
    print(colored(f"[CONFIG] Curriculum strategy: {args.curriculum_strategy}", "cyan"))
    print(colored(f"[CONFIG] Loss function: {args.loss}", "cyan"))
    print(colored(f"[CONFIG] Batch size: {args.batch_size}", "cyan"))
    print()
    
    # Load data with difficulty scores
    X_train, y_train, difficulty_train, X_val, y_val = prepare_datasets_curriculum(args.curriculum_strategy)
    
    # Build model using original trainer's build function
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model_from_trainer(
        input_shape, 
        args.model,
        num_conformer_blocks=args.blocks,
        num_classes=args.num_classes
    )
    
    # Detect output format from model
    output_format = config.TRAINER_OUTPUT_FORMATS.get(args.model, 'categorical')
    print(colored(f"[INFO] Model output format: {output_format}", "cyan"))
    
    # Configure loss function - use Focal Loss from config for better performance on hard examples
    # Focal Loss synergizes with curriculum learning by focusing on difficult samples
    focal_alpha = config.LOSS_DEFAULTS.get('focal_alpha', 0.60)
    focal_gamma = config.LOSS_DEFAULTS.get('focal_gamma', 2.0)
    
    loss_fn = get_loss_function('focal', alpha=focal_alpha, gamma=focal_gamma)
    
    print(colored(f"[INFO] Using loss: Focal Loss (α={focal_alpha}, γ={focal_gamma}) - curriculum mode", "yellow"))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # Use simple metrics that work for both binary and categorical
    # (Precision/Recall/AUC from get_metrics() are binary-only and cause dimension issues)
    metrics_list = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)
    
    # Compute class weights if requested
    class_weight = None
    if args.use_class_weight:
        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
        print(colored(f"[CONFIG] Class weights: {class_weight}", "cyan"))
    
    # Setup callbacks
    from tools.safe_callbacks import CurriculumAwareEarlyStopping, ConfusionMatrixLogger
    
    # Callback qui persiste entre les appels fit() pour curriculum
    early_stopping_cb = CurriculumAwareEarlyStopping(
        start_from_epoch=args.min_epochs,
        monitor='val_accuracy',
        mode='max',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Callback pour afficher la matrice de confusion
    confusion_cb = ConfusionMatrixLogger(
        validation_data=(X_val, y_val),
        output_format=output_format,
        verbose=1,
        predict_batch_size=min(8, args.batch_size)
    )
    
    callbacks_list = [
        early_stopping_cb,
        confusion_cb,
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Add curriculum scheduler if strategy is not 'none'
    curriculum_scheduler = None
    if args.curriculum_strategy != 'none':
        # Use config.CURRICULUM_LEARNING parameters
        curr_config = config.CURRICULUM_LEARNING
        curriculum_scheduler = CurriculumScheduler(
            difficulty_scores=difficulty_train,
            labels=y_train,
            total_epochs=args.max_epochs,
            strategy=args.curriculum_strategy,
            warmup_epochs=args.warmup_epochs,
            initial_fraction=curr_config['initial_fraction'],
            growth_rate=curr_config['growth_rate'],
            verbose=1
        )
        print(colored(f"[INFO] Curriculum learning ENABLED: {args.curriculum_strategy}", "green"))
        print(colored(f"[INFO] Initial fraction: {curr_config['initial_fraction']*100:.0f}%, Growth: {curr_config['growth_rate']*100:.0f}%/epoch", "cyan"))
    else:
        print(colored("[INFO] Curriculum learning DISABLED (standard training)", "yellow"))
    
    # Model save paths (nom standard sans suffix curriculum)
    model_name = f"{args.model}_model"
    model_save_path = config.MODELS_DIR / f"{model_name}.keras"
    history_save_path = config.RESULTS_DIR / f"{args.model}_history.csv"
    
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        str(model_save_path),
        monitor='val_accuracy',  # Changed from val_f1
        mode='max',
        save_best_only=True,
        verbose=0  # Silencieux - on affiche nous-mêmes les infos
    )
    callbacks_list.append(checkpoint_cb)
    
    # Train model with curriculum learning (epoch by epoch)
    print(colored("\n[TRAINING] Starting curriculum learning...\n", "green"))
    
    # Initialize empty history - will dynamically capture all metrics from callbacks
    all_history = {}
    
    if curriculum_scheduler is not None:
        # Curriculum: train epoch by epoch with increasing difficulty
        print(colored(f"\n[CURRICULUM] Training for {args.max_epochs} epochs (min {args.min_epochs} before early stopping)\n", "cyan"))
        
        for epoch in range(args.max_epochs):
            # Compute curriculum indices for this epoch
            curriculum_scheduler.on_epoch_begin(epoch, None)
            curr_indices = curriculum_scheduler.current_indices
            
            # Filter training data
            X_train_curr = X_train[curr_indices]
            y_train_curr = y_train[curr_indices]
            
            # Calculer stats de difficulté
            avg_diff = np.mean(difficulty_train[curr_indices]) if difficulty_train is not None else 0
            max_diff = np.max(difficulty_train[curr_indices]) if difficulty_train is not None else 0
            
            # Header unifié avec info warmup si applicable
            print(colored(f"{'='*80}", "cyan"))
            if epoch < args.min_epochs:
                print(colored(f"EPOCH {epoch + 1}/{args.max_epochs} (Warmup {epoch + 1}/{args.min_epochs})", "cyan", attrs=['bold']))
            else:
                print(colored(f"EPOCH {epoch + 1}/{args.max_epochs}", "cyan", attrs=['bold']))
            
            print(colored(f"Samples: {len(curr_indices)}/{len(X_train)} ({100*len(curr_indices)/len(X_train):.1f}%) | ", "cyan") +
                  colored(f"Avg difficulty: {avg_diff:.3f} | Max: {max_diff:.3f}", "yellow"))
            print(colored(f"{'='*80}", "cyan"))
            
            # Train for 1 epoch (verbose=0 pour éviter les messages Keras confus)
            epoch_start = datetime.now()
            history = model.fit(
                X_train_curr, y_train_curr,
                validation_data=(X_val, y_val),
                batch_size=args.batch_size,
                epochs=1,
                callbacks=callbacks_list,
                class_weight=class_weight,
                verbose=0  # On gère nous-mêmes l'affichage
            )
            epoch_duration = datetime.now() - epoch_start
            
            # Accumulate ALL metrics dynamically (captures callbacks metrics like TP/TN/FP/FN)
            for key, values in history.history.items():
                if key not in all_history:
                    all_history[key] = []
                all_history[key].extend(values)
            
            # Afficher métriques sur une ligne
            loss = history.history.get('loss', [0])[0]
            acc = history.history.get('accuracy', [0])[0]
            val_loss = history.history.get('val_loss', [0])[0]
            val_acc = history.history.get('val_accuracy', [0])[0]
            print(colored(f"loss: {loss:.4f} - accuracy: {acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} | Duration: {epoch_duration}", "white"))
            
            # Le callback early stopping affichera son propre message ici
            
            print(colored(f"{'='*80}\n", "cyan"))
            
            # Check si early stopping a demandé l'arrêt
            if model.stop_training:
                print(colored(f"[INFO] Training stopped at epoch {epoch + 1}/{args.max_epochs}\n", "yellow", attrs=['bold']))
                break
    else:
        # Standard training (no curriculum)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=args.batch_size,
            epochs=args.max_epochs,
            callbacks=callbacks_list,
            class_weight=class_weight,
            verbose=args.verbose
        )
        all_history = history.history
    
    # Save training history
    history_df = pd.DataFrame(all_history)
    history_df.to_csv(history_save_path, index=False)
    print(colored(f"\n[SAVED] Training history: {history_save_path}", "green"))
    print(colored(f"[SAVED] Best model: {model_save_path}", "green"))
    
    # Training summary
    duration = datetime.now() - startTime
    print(colored("\n" + "="*80, "magenta"))
    print(colored("  TRAINING COMPLETE", "magenta"))
    print(colored("="*80, "magenta"))
    print(colored(f"Duration: {duration}", "cyan"))
    print(colored(f"Best val_accuracy: {max(all_history.get('val_accuracy', [0])):.4f}", "cyan"))
    print(colored(f"Best val_loss: {min(all_history.get('val_loss', [999])):.4f}", "cyan"))
    print()


if __name__ == '__main__':
    main()
