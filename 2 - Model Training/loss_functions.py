"""
Utility functions for improved loss functions and training techniques.

This module provides:
- Focal Loss: Focus on hard-to-classify examples
- Class Weighted BCE: Balance class importance
- Distance Weighted Loss: Prioritize hard distances (500m, 350m)
- Additional metrics for better evaluation
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Global dynamic positive-class weight. Trainers/callbacks can update this at
# runtime via `set_dynamic_pos_weight()` to steer the loss toward higher
# or lower emphasis on the positive (drone) class.
# Default 1.0 means no re-weighting (positive weight == negative weight).
DYNAMIC_POS_WEIGHT = 1.0
# Dynamic negative-class weight (used to penalize false positives)
DYNAMIC_NEG_WEIGHT = 1.0

def set_dynamic_pos_weight(value: float):
    """Set the global dynamic positive-class weight used by weighted losses.

    Args:
        value: positive-class weight (float > 0). Typical values >1 increase
               emphasis on positive (drone) class.
    """
    global DYNAMIC_POS_WEIGHT
    try:
        DYNAMIC_POS_WEIGHT = float(value)
    except Exception:
        DYNAMIC_POS_WEIGHT = 1.0


def set_dynamic_neg_weight(value: float):
    """Set the global dynamic negative-class weight used by weighted losses.

    Args:
        value: negative-class weight (float > 0). Values >1 increase penalty
               for negative-class examples (i.e., penalize false positives).
    """
    global DYNAMIC_NEG_WEIGHT
    try:
        DYNAMIC_NEG_WEIGHT = float(value)
    except Exception:
        DYNAMIC_NEG_WEIGHT = 1.0


def recall_focused_loss(fn_penalty=50.0):
    """
    Custom loss that heavily penalizes False Negatives (missed drones).
    
    Loss = BCE + fn_penalty * FN_weight
    
    This forces the model to prioritize recall (detecting ALL drones)
    even at the cost of false positives.
    
    Args:
        fn_penalty: Multiplier for false negative penalty (default: 50.0)
                   Higher = more aggressive drone detection
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    def loss_fn(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
            y_true = tf.cast(y_true, tf.float32)
        
        # Standard BCE
        bce = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # False Negative penalty: when y_true=1 (drone) but y_pred→0 (predicted no-drone)
        # Extract drone class (index 1)
        y_true_drone = y_true[:, 1]  # True label for drone
        y_pred_drone = y_pred[:, 1]  # Predicted probability for drone
        
        # FN occurs when: true=drone (1) but predicted=no-drone (low probability)
        # Penalty = y_true_drone * (1 - y_pred_drone)
        # If drone but predicted 0 → penalty = 1.0
        # If drone but predicted 1 → penalty = 0.0
        fn_weight = y_true_drone * (1.0 - y_pred_drone)
        
        # Combined loss
        total_loss = bce + fn_penalty * fn_weight
        
        return total_loss
    
    return loss_fn


def focal_loss(alpha=0.75, gamma=2.0):
    """
    Focal Loss for addressing class imbalance and focusing on hard examples.
    
    Focal Loss = -α * (1 - p_t)^γ * log(p_t)
    
    where p_t is the model's estimated probability for the true class.
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               Higher alpha = more weight on positive class (drones)
        gamma: Focusing parameter >= 0. Higher gamma = more focus on hard examples
               gamma = 0 reduces to standard Cross-Entropy
               gamma = 2 is typical
    
    Returns:
        Loss function compatible with Keras model.compile()
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
            y_true = tf.cast(y_true, tf.float32)
        
        # Calculate categorical cross entropy
        bce = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Calculate p_t (probability of true class)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # If a dynamic positive-class weight has been set at runtime, prefer
        # using it to compute an effective alpha. The mapping used here is:
        #   dynamic_alpha = w_pos / (1 + w_pos)
        # where w_pos is the positive-class weight. This maps weight>1 to
        # alpha in (0.5, 1.0).
        try:
            dyn_w = float(DYNAMIC_POS_WEIGHT)
        except Exception:
            dyn_w = 1.0

        if dyn_w is not None and dyn_w != 1.0:
            dynamic_alpha = dyn_w / (1.0 + dyn_w)
            effective_alpha = dynamic_alpha
        else:
            effective_alpha = alpha

        # Build per-sample alpha_t from one-hot y_true using effective alpha
        alpha_tensor = y_true * effective_alpha + (1.0 - y_true) * (1.0 - effective_alpha)
        # Reduce to scalar per-sample
        alpha_factor = tf.reduce_sum(alpha_tensor, axis=-1)

        # Modulating factor
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        focal = alpha_factor * modulating_factor * bce

        # Apply dynamic per-class weights multiplicatively so runtime updates
        # to `DYNAMIC_POS_WEIGHT` and `DYNAMIC_NEG_WEIGHT` have an immediate effect.
        try:
            dyn_pos = float(DYNAMIC_POS_WEIGHT)
        except Exception:
            dyn_pos = 1.0
        try:
            dyn_neg = float(DYNAMIC_NEG_WEIGHT)
        except Exception:
            dyn_neg = 1.0

        # y_true is one-hot at this point
        y_true_pos = y_true[:, 1]
        weights = y_true_pos * dyn_pos + (1.0 - y_true_pos) * dyn_neg

        weighted_focal = weights * focal

        return tf.reduce_mean(weighted_focal)
    
    return loss_fn


def weighted_binary_crossentropy(class_weight_drone=3.0):
    """
    Weighted Binary Cross-Entropy Loss.
    
    Applies different weights to positive (drone) and negative (no-drone) classes.
    
    Args:
        class_weight_drone: Weight for drone class (positive class)
                           class_weight_no_drone is implicitly 1.0
                           Higher values = more penalty for missing drones
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    def loss_fn(y_true, y_pred):
        # Support both single-prob outputs and 2-class softmax outputs
        # If y_pred has shape (batch, 2) assume softmax and extract positive class
        if tf.rank(y_pred) == 2 and tf.shape(y_pred)[-1] == 2:
            # Use probability of positive (drone) class
            y_pred_pos = y_pred[:, 1]
            # Convert y_true to scalar positive indicator if one-hot
            if tf.rank(y_true) > 1 and tf.shape(y_true)[-1] == 2:
                y_true_pos = y_true[:, 1]
            else:
                y_true_pos = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        else:
            # y_pred is scalar probability shape (batch,)
            y_pred_pos = tf.reshape(y_pred, [-1])
            y_true_pos = tf.cast(tf.reshape(y_true, [-1]), tf.float32)

        # Binary cross entropy between scalar labels and positive-class prob
        bce = keras.losses.binary_crossentropy(y_true_pos, y_pred_pos)

        # Determine effective positive-class weight: prefer runtime-updated
        # `DYNAMIC_POS_WEIGHT` if caller left class_weight_drone as None.
        effective_pos_weight = class_weight_drone
        try:
            if class_weight_drone is None:
                effective_pos_weight = float(DYNAMIC_POS_WEIGHT)
        except Exception:
            effective_pos_weight = 1.0

        # Apply class weights: positive label gets effective_pos_weight, negative gets
        # the runtime-updated `DYNAMIC_NEG_WEIGHT` (to penalize false positives).
        try:
            dyn_neg = float(DYNAMIC_NEG_WEIGHT)
        except Exception:
            dyn_neg = 1.0

        weights = y_true_pos * effective_pos_weight + (1.0 - y_true_pos) * dyn_neg

        weighted_bce = weights * bce

        return tf.reduce_mean(weighted_bce)
    
    return loss_fn


def get_loss_function(loss_type='bce', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: One of 'bce', 'weighted_bce', 'focal', 'recall_focused'
        **kwargs: Additional arguments for the loss function
                  - class_weight_drone: for 'weighted_bce' (default: 3.0)
                  - alpha: for 'focal' (default: 0.75)
                  - gamma: for 'focal' (default: 2.0)
                  - fn_penalty: for 'recall_focused' (default: 50.0)
    
    Returns:
        Loss function
    
    Examples:
        >>> loss = get_loss_function('focal', alpha=0.75, gamma=2.0)
        >>> loss = get_loss_function('weighted_bce', class_weight_drone=3.0)
        >>> loss = get_loss_function('bce')  # Standard BCE
    """
    if loss_type == 'bce':
        return 'binary_crossentropy'
    
    elif loss_type == 'weighted_bce':
        class_weight = kwargs.get('class_weight_drone', 3.0)
        return weighted_binary_crossentropy(class_weight_drone=class_weight)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.75)
        gamma = kwargs.get('gamma', 2.0)
        return focal_loss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'recall_focused':
        fn_penalty = kwargs.get('fn_penalty', 50.0)
        return recall_focused_loss(fn_penalty=fn_penalty)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: 'bce', 'weighted_bce', 'focal', 'recall_focused'")


def get_metrics():
    """
    Get standard metrics for binary classification.
    
    Returns:
        List of metrics to pass to model.compile()
    """
    # Simple standard metrics that work with scalar labels (batch,)
    # and sigmoid outputs (batch, 1)
    return [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]


def get_metrics_OLD_COMPLEX():
    """
    DEPRECATED: Complex metrics with one-hot handling (kept for reference).
    Use get_metrics() instead for simpler, more robust metrics.
    """
    class _PositiveClassMetric(keras.metrics.Metric):
        """Wrapper metric that delegates to a binary metric but feeds it
        the positive-class probability extracted from a 2-class softmax.
        """
        def __init__(self, metric_cls, name, **kwargs):
            super().__init__(name=name)
            # instantiate the underlying metric (Precision, Recall, etc.)
            self._inner = metric_cls(name=name + "_inner", **kwargs)

        def update_state(self, y_true, y_pred, sample_weight=None):
            # Safely handle Tensor shapes inside graph mode using tf.cond
            def _y_true_pos():
                return y_true[:, 1]

            def _y_true_scalar():
                return tf.reshape(y_true, [-1])

            rank_true = tf.rank(y_true)
            shape_true = tf.shape(y_true)
            is_one_hot = tf.logical_and(tf.equal(rank_true, 2), tf.equal(shape_true[-1], 2))
            y_true_pos = tf.cond(is_one_hot, _y_true_pos, _y_true_scalar)

            # extract positive-class probability from softmax outputs
            def _y_pred_pos():
                return y_pred[:, 1]

            def _y_pred_scalar():
                return tf.reshape(y_pred, [-1])

            rank_pred = tf.rank(y_pred)
            shape_pred = tf.shape(y_pred)
            is_softmax2 = tf.logical_and(tf.equal(rank_pred, 2), tf.equal(shape_pred[-1], 2))
            y_pred_pos = tf.cond(is_softmax2, _y_pred_pos, _y_pred_scalar)

            # cast labels to appropriate dtype
            y_true_pos = tf.cast(y_true_pos, tf.float32)

            self._inner.update_state(y_true_pos, y_pred_pos, sample_weight)

        def result(self):
            return self._inner.result()

        def reset_states(self):
            self._inner.reset_states()

    return [
        'accuracy',
        _PositiveClassMetric(keras.metrics.Precision, name='precision'),
        _PositiveClassMetric(keras.metrics.Recall, name='recall'),
        _PositiveClassMetric(keras.metrics.AUC, name='auc'),
        _PositiveClassMetric(keras.metrics.TruePositives, name='tp'),
        _PositiveClassMetric(keras.metrics.FalsePositives, name='fp'),
        _PositiveClassMetric(keras.metrics.TrueNegatives, name='tn'),
        _PositiveClassMetric(keras.metrics.FalseNegatives, name='fn')
    ]


# Convenience functions for common configurations
def get_focal_loss_for_drones(alpha=0.75, gamma=2.0):
    """
    Get focal loss optimized for drone detection.
    
    Args:
        alpha: 0.75 means 75% weight on drones, 25% on no-drones
        gamma: 2.0 focuses on hard examples
    
    Returns:
        Focal loss function
    """
    return focal_loss(alpha=alpha, gamma=gamma)


def get_weighted_loss_for_drones(weight=3.0):
    """
    Get weighted BCE optimized for drone detection.
    
    Args:
        weight: How much more important drones are than no-drones
                3.0 = drones are 3x more important
    
    Returns:
        Weighted BCE loss function
    """
    return weighted_binary_crossentropy(class_weight_drone=weight)


if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    print("Testing loss functions...")
    
    # Dummy data
    y_true = np.array([[1], [0], [1], [0]])  # 2 drones, 2 no-drones
    y_pred = np.array([[0.9], [0.1], [0.6], [0.4]])  # Predictions
    
    # Test BCE
    bce_loss = keras.losses.BinaryCrossentropy()
    print(f"BCE Loss: {bce_loss(y_true, y_pred):.4f}")
    
    # Test Weighted BCE
    weighted_loss = get_weighted_loss_for_drones(weight=3.0)
    print(f"Weighted BCE Loss (weight=3.0): {weighted_loss(y_true, y_pred):.4f}")
    
    # Test Focal Loss
    focal = get_focal_loss_for_drones(alpha=0.75, gamma=2.0)
    print(f"Focal Loss (α=0.75, γ=2.0): {focal(y_true, y_pred):.4f}")


def distance_weighted_loss(distance_weights=None, base_loss='bce', alpha=0.75, gamma=0.0):
    """
    Distance-Weighted Loss Function.
    
    Applies different weights to samples based on their distance to prioritize
    hard-to-detect extreme distances (500m, 350m).
    
    Args:
        distance_weights: Dict mapping distance (in meters) to weight multiplier
                         Example: {500: 10.0, 350: 8.0, 200: 5.0, 100: 2.0, 50: 1.0}
                         If None, uses default weights
        base_loss: Base loss function ('bce', 'focal')
        alpha: Focal loss alpha (if base_loss='focal')
        gamma: Focal loss gamma (if base_loss='focal')
    
    Returns:
        Loss function compatible with Keras model.compile()
        
    Usage:
        This requires passing distance labels alongside training.
        Use fit() with custom training loop or data generator that provides:
        - y_true: [batch_size, 2] where y_true[:, 0] = class label (0/1)
                                        y_true[:, 1] = distance label (500/350/200/100/50)
    """
    if distance_weights is None:
        # Default weights: prioritize extreme distances
        distance_weights = {
            700: 12.0,  # 12x weight for 700m (long-range)
            500: 10.0,  # 10x weight for 500m
            350: 8.0,   # 8x weight for 350m
            200: 5.0,   # 5x weight for 200m
            100: 2.0,   # 2x weight for 100m
            50: 1.0,    # 1x weight for 50m (baseline)
            0: 1.0      # 1x weight for ambient/background (no distance)
        }
    
    def loss_fn(y_true, y_pred):
        """
        y_true: [batch_size, 2] - [:, 0] = class label, [:, 1] = distance (m)
        y_pred: [batch_size, 2] - softmax output
        """
        # Extract class labels and distances
        class_labels = y_true[:, 0]
        distance_labels = y_true[:, 1]
        
        # Calculate base loss
        if base_loss == 'focal':
            # Focal loss calculation
            epsilon = keras.backend.epsilon()
            y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Convert to one-hot
            class_labels_int = tf.cast(class_labels, tf.int32)
            class_labels_onehot = tf.one_hot(class_labels_int, depth=2)
            class_labels_onehot = tf.cast(class_labels_onehot, tf.float32)
            
            # Cross entropy
            bce = keras.losses.categorical_crossentropy(class_labels_onehot, y_pred_clipped)
            
            # Focal modulation
            p_t = tf.reduce_sum(class_labels_onehot * y_pred_clipped, axis=-1)
            focal_weight = tf.pow(1.0 - p_t, gamma)
            base_loss_value = alpha * focal_weight * bce
        else:
            # Standard BCE
            epsilon = keras.backend.epsilon()
            y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Get predicted probability for positive class
            y_pred_pos = y_pred_clipped[:, 1]
            
            # Binary cross entropy
            base_loss_value = -(class_labels * tf.math.log(y_pred_pos) + 
                               (1 - class_labels) * tf.math.log(1 - y_pred_pos))
        
        # Map distances to weights
        # Create a lookup table
        distance_values = tf.constant(list(distance_weights.keys()), dtype=tf.float32)
        weight_values = tf.constant(list(distance_weights.values()), dtype=tf.float32)
        
        # For each sample, find its weight based on distance
        # Use broadcasting to find closest distance match
        distance_labels_expanded = tf.expand_dims(distance_labels, 1)  # [batch, 1]
        distance_values_expanded = tf.expand_dims(distance_values, 0)  # [1, n_distances]
        
        # Find closest distance (for fuzzy matching)
        distances_diff = tf.abs(distance_labels_expanded - distance_values_expanded)
        closest_idx = tf.argmin(distances_diff, axis=1)
        
        # Get weights for each sample
        sample_weights = tf.gather(weight_values, closest_idx)
        
        # Apply distance weights to loss
        weighted_loss = base_loss_value * sample_weights
        
        return tf.reduce_mean(weighted_loss)
    
    return loss_fn


def get_loss_function_with_distance(loss_type='distance_weighted', **kwargs):
    """
    Get a distance-aware loss function.
    
    Args:
        loss_type: Type of loss ('distance_weighted', 'focal', 'bce')
        **kwargs: Additional arguments passed to loss function
        
    Returns:
        Loss function
    """
    if loss_type == 'distance_weighted':
        return distance_weighted_loss(**kwargs)
    elif loss_type == 'focal':
        return focal_loss(**kwargs)
    elif loss_type == 'weighted_bce':
        return weighted_binary_crossentropy(**kwargs)
    else:
        return 'sparse_categorical_crossentropy'
    
    print("\nAll tests passed!")
