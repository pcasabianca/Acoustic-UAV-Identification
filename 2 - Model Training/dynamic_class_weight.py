import numpy as np
import tensorflow as tf
from tensorflow import keras

from loss_functions import set_dynamic_neg_weight, DYNAMIC_NEG_WEIGHT


class DynamicClassWeightCallback(keras.callbacks.Callback):
    """Callback to update a global positive-class weight each epoch.

    The callback evaluates the model on a provided validation set at the end
    of each epoch, computes FP and FN counts, and updates
    `loss_functions.DYNAMIC_POS_WEIGHT` using an EMA-smoothed ratio:

        candidate = base_weight * (fn + 1) / (fp + 1)
        new = beta * current + (1-beta) * candidate
        new = clip(new, min_w, max_w)

    Defaults are tuned to keep the positive-class weight within [0.1, 0.9]
    and the `beta` parameter controls inertia (EMA smoothing).
    """

    def __init__(self, val_data, base_weight=1.0, beta=0.9, min_w=0.01, max_w=10.0,
                 batch_size=64, warmup_epochs=3, max_delta=2.0, min_pred_samples=10,
                 rate_ema_alpha=0.8):
        super().__init__()
        self.x_val, self.y_val = val_data
        self.base_weight = float(base_weight)
        # EMA inertia (higher => slower updates)
        self.beta = float(beta)
        # clamp bounds for positive-class weight
        self.min_w = float(min_w)
        self.max_w = float(max_w)
        self.batch_size = int(batch_size)
        # defensive tuning
        self.warmup_epochs = int(warmup_epochs)
        # max multiplicative change allowed per epoch (e.g. 2.0 -> at most double)
        self.max_delta = float(max_delta)
        # minimum number of validation samples required to compute updates
        self.min_pred_samples = int(min_pred_samples)
        # smoothing for FP/FN rates (to reduce noisy per-epoch swings)
        self.rate_ema_alpha = float(rate_ema_alpha)
        self.ema_fp_rate = None
        self.ema_fn_rate = None
        # initialize with current negative-class weight (global state)
        try:
            self.current = float(DYNAMIC_NEG_WEIGHT)
        except Exception:
            self.current = float(self.base_weight)

    def on_train_begin(self, logs=None):
        try:
            # reset global state to base to avoid cross-run contamination
            set_dynamic_neg_weight(float(self.base_weight))
            self.current = float(DYNAMIC_NEG_WEIGHT)
        except Exception:
            self.current = float(self.base_weight)

    def on_epoch_end(self, epoch, logs=None):
        # respect warmup: don't update until enough epochs passed
        if epoch + 1 <= self.warmup_epochs:
            print(f"[DynamicClassWeight] Epoch {epoch+1}: warmup (<= {self.warmup_epochs}), skipping update")
            return
        # Predict on validation set
        preds = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)

        # Handle both softmax(2) outputs and scalar probabilities
        if len(preds.shape) > 1 and preds.shape[-1] > 1:
            preds_pos = preds[:, 1]
            pred_labels = (preds_pos >= 0.5).astype(int)
        else:
            preds_pos = preds.flatten()
            pred_labels = (preds_pos >= 0.5).astype(int)

        # Handle y_val that may be one-hot encoded or integer labels
        y_arr = np.array(self.y_val)
        if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
            # one-hot -> argmax
            y_true = np.argmax(y_arr, axis=1).astype(int)
        else:
            y_true = y_arr.reshape(-1).astype(int)

        # Compute confusion counts
        tp = int(np.sum((pred_labels == 1) & (y_true == 1)))
        tn = int(np.sum((pred_labels == 0) & (y_true == 0)))
        fp = int(np.sum((pred_labels == 1) & (y_true == 0)))
        fn = int(np.sum((pred_labels == 0) & (y_true == 1)))

        # defensive checks: ensure enough samples
        n_samples = len(y_true)
        preds_pos_mean = float(np.mean(preds_pos)) if n_samples > 0 else 0.0

        if n_samples < self.min_pred_samples:
            print(f"[DynamicClassWeight] Epoch {epoch+1}: insufficient val samples ({n_samples}), skipping update")
            return

        # detect degenerate predictions: all predicted same class or probabilities collapsed
        degenerate = False
        if (tp + fp) == 0 or (tn + fn) == 0:
            degenerate = True
        if preds_pos_mean <= 1e-6 or preds_pos_mean >= 1.0 - 1e-6:
            degenerate = True

        # compute normalized error rates to account for class imbalance
        eps = 1e-6
        positives = tp + fn
        negatives = tn + fp
        # rates: FN rate = FN / positives, FP rate = FP / negatives
        fn_rate = float(fn) / float(positives) if positives > 0 else 0.0
        fp_rate = float(fp) / float(negatives) if negatives > 0 else 0.0

        # maintain EMA of rates to stabilize updates across epochs
        if self.ema_fp_rate is None:
            self.ema_fp_rate = fp_rate
        else:
            self.ema_fp_rate = self.rate_ema_alpha * self.ema_fp_rate + (1.0 - self.rate_ema_alpha) * fp_rate
        if self.ema_fn_rate is None:
            self.ema_fn_rate = fn_rate
        else:
            self.ema_fn_rate = self.rate_ema_alpha * self.ema_fn_rate + (1.0 - self.rate_ema_alpha) * fn_rate

        smooth_fp = float(self.ema_fp_rate)
        smooth_fn = float(self.ema_fn_rate)

        # We update the negative-class weight (penalize false positives).
        # Candidate negative weight increases when FP rate > FN rate:
        #   candidate_neg = base_weight * (smoothed_fp + eps) / (smoothed_fn + eps)
        # This raises the penalty on negative-class examples when FP is high,
        # which discourages predicting positive for negatives.
        candidate = self.base_weight * float((smooth_fp + eps) / (smooth_fn + eps))

        if degenerate:
            # do not apply large swings when model predicts a single class; nudge towards base
            # use a conservative candidate (average with base)
            candidate = 0.5 * (candidate + self.base_weight)
            print(f"[DynamicClassWeight] Epoch {epoch+1}: degenerate preds detected (mean={preds_pos_mean:.4f}), using conservative candidate={candidate:.4f}")

        # EMA step (inertia)
        new_val = self.beta * float(self.current) + (1.0 - self.beta) * float(candidate)

        # clamp multiplicative change relative to current to avoid runaway
        if self.current > 0:
            ratio = new_val / float(self.current)
            if ratio > self.max_delta:
                new_val = float(self.current) * self.max_delta
            elif ratio < 1.0 / float(self.max_delta):
                new_val = float(self.current) / self.max_delta

        # hard bounds
        new_val = max(self.min_w, min(self.max_w, new_val))

        # update global variable (negative-class weight)
        set_dynamic_neg_weight(new_val)
        self.current = new_val

        # Log changes with diagnostics
        print(f"[DynamicClassWeight] Epoch {epoch+1}: tp={tp} fp={fp} tn={tn} fn={fn} mean_pred={preds_pos_mean:.4f} -> new_neg_weight={new_val:.4f}")
