import time
import numpy as np
from tensorflow import keras
from termcolor import colored
from sklearn.metrics import confusion_matrix

class SafeEarlyStopping(keras.callbacks.EarlyStopping):
    """
    Extension of Keras EarlyStopping that supports a `start_from_epoch` argument.
    Training will ignore early-stopping until the current epoch index is >= start_from_epoch.
    """
    def __init__(self, start_from_epoch=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_from_epoch = int(start_from_epoch or 0)

    def on_epoch_end(self, epoch, logs=None):
        # Only delegate to the parent after start_from_epoch
        if epoch < self._start_from_epoch:
            return
        return super().on_epoch_end(epoch, logs)

class TimeStopping(keras.callbacks.Callback):
    """
    Stop training after a given number of seconds have elapsed.
    """
    def __init__(self, max_seconds, verbose=0):
        super().__init__()
        self.max_seconds = float(max_seconds)
        self.verbose = verbose
        self._start_time = None

    def on_train_begin(self, logs=None):
        self._start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self._start_time is None:
            return
        elapsed = time.time() - self._start_time
        if elapsed >= self.max_seconds:
            self.model.stop_training = True
            if self.verbose:
                print(f"[TimeStopping] Stopping training after {elapsed:.1f}s >= {self.max_seconds}s")


class CurriculumAwareEarlyStopping(keras.callbacks.Callback):
    """
    Early stopping compatible avec curriculum learning.
    Track le nombre global d'epochs même avec appels fit() multiples.
    
    Paramètres:
    -----------
    start_from_epoch : int
        Nombre minimum d'epochs avant que l'early stopping puisse se déclencher
    monitor : str
        Métrique à surveiller (ex: 'val_accuracy', 'val_loss')
    mode : str
        'min' pour minimiser la métrique, 'max' pour maximiser
    patience : int
        Nombre d'epochs sans amélioration avant d'arrêter
    restore_best_weights : bool
        Si True, restaure les meilleurs poids à la fin
    verbose : int
        0 = silencieux, 1 = messages de progression
    """
    def __init__(self, start_from_epoch=0, monitor='val_loss', mode='min', 
                 patience=10, restore_best_weights=True, verbose=1):
        super().__init__()
        self.start_from_epoch = start_from_epoch
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # État global (persiste entre les appels fit())
        self.global_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Appelé à la fin de chaque epoch Keras (mais on track les epochs globaux)."""
        # Incrémenter le compteur global (persiste entre les appels fit())
        self.global_epoch += 1
        
        # Récupérer la métrique
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # Phase warmup (avant start_from_epoch)
        if self.global_epoch < self.start_from_epoch:
            # Track best même en warmup
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
                if self.verbose:
                    print(colored(f"[Warmup] Best {self.monitor}: {current:.4f} (new best)", "green"))
            else:
                self.wait += 1
                if self.verbose:
                    epochs_since = self.wait
                    print(colored(f"[Warmup] Best {self.monitor}: {self.best:.4f} (no improvement, {epochs_since} {'epoch' if epochs_since == 1 else 'epochs'} since best)", "yellow"))
            return
        
        # Phase active (early stopping peut se déclencher)
        if self.monitor_op(current, self.best):
            if self.verbose:
                print(colored(f"[EarlyStopping] ✓ {self.monitor} improved: {self.best:.4f} → {current:.4f} (patience reset)", "green"))
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.verbose:
                print(colored(f"[EarlyStopping] ✗ No improvement ({current:.4f}), patience {self.wait}/{self.patience}", "yellow"))
            
            if self.wait >= self.patience:
                self.stopped_epoch = self.global_epoch
                self.model.stop_training = True
                if self.verbose:
                    print(colored(f"[EarlyStopping] ⚠ STOPPING (no improvement for {self.patience} epochs)", "red", attrs=['bold']))
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    if self.verbose:
                        print(colored(f"[EarlyStopping] Restored best weights", "cyan"))
    
    def reset(self):
        """Reset le callback pour un nouveau training."""
        self.global_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        if self.mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf


class ConfusionMatrixLogger(keras.callbacks.Callback):
    """
    Callback qui affiche la matrice de confusion (TP, TN, FP, FN) à chaque epoch.
    Compatible avec les modèles binaires et catégoriels.
    """
    def __init__(self, validation_data, output_format='categorical', verbose=1):
        """
        Args:
            validation_data: Tuple (X_val, y_val)
            output_format: 'binary' ou 'categorical'
            verbose: 0=silent, 1=afficher
        """
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.output_format = output_format
        self.verbose = verbose
    
    def on_epoch_end(self, epoch, logs=None):
        if not self.verbose:
            return
        
        # Prédictions
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        
        # Conversion selon format
        if self.output_format == 'binary':
            # Sortie binaire (sigmoïde)
            y_pred = (y_pred_proba.flatten() > 0.5).astype(int)
            y_true = self.y_val.flatten()
        else:
            # Sortie catégorielle (softmax)
            y_pred = np.argmax(y_pred_proba, axis=1)
            # Si y_val est one-hot, convertir
            if len(self.y_val.shape) > 1 and self.y_val.shape[1] > 1:
                y_true = np.argmax(self.y_val, axis=1)
            else:
                y_true = self.y_val.flatten()
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            # Binaire: [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = cm.ravel()
            
            # Calculer métriques
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Affichage
            print(colored(f"[Confusion] TP={tp:4d} TN={tn:4d} FP={fp:4d} FN={fn:4d} | "
                         f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}", "cyan"))
        else:
            # Multi-classe: afficher juste les vrais/faux
            correct = np.trace(cm)
            total = np.sum(cm)
            print(colored(f"[Confusion] Correct={correct}/{total} ({100*correct/total:.1f}%)", "cyan"))
