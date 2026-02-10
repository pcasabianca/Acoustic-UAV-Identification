"""
Prediction and ensemble voting logic.
"""

import numpy as np
from typing import Dict, Tuple


class Predictor:
    """Handles model predictions and ensemble decision making."""
    
    def __init__(self, config: dict, models: dict, feature_extractor, logger=None):
        """
        Initialize predictor.
        
        Args:
            config: Configuration dictionary
            models: Dictionary of loaded models
            feature_extractor: FeatureExtractor instance
            logger: Logger instance (optional)
        """
        self.config = config
        self.models = models
        self.feature_extractor = feature_extractor
        self.logger = logger
    
    def predict_all(self, features: np.ndarray, reload_thresholds: bool = False) -> Dict[str, float]:
        """
        Run inference on all loaded models.
        
        Args:
            features: MEL spectrogram
            reload_thresholds: Whether to reload thresholds before prediction
        
        Returns:
            Dict of {model_name: drone_probability}
        """
        predictions = {}

        for model_name, model_info in self.models.items():
            try:
                # Adapt features to the model's expected input shape
                model = model_info['model']
                X_model = self.feature_extractor.adapt_to_model(features, model)

                # Get prediction
                pred = model.predict(X_model, verbose=0)
                
                # Extract drone probability (handle different architectures)
                if pred.shape[1] == 2:
                    # Softmax with 2 classes (CNN, CRNN, Attention-CRNN)
                    drone_prob = float(pred[0][1])
                elif pred.shape[1] == 1:
                    # Sigmoid (RNN)
                    drone_prob = float(pred[0][0])
                else:
                    if self.logger:
                        self.logger.error(f"Unexpected prediction shape for {model_name}: {pred.shape}")
                    drone_prob = 0.0
                
                predictions[model_name] = drone_prob
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Prediction failed for {model_name}: {e}")
                predictions[model_name] = 0.0
        
        return predictions
    
    def make_decision(self, predictions: Dict[str, float]) -> Tuple[bool, Dict]:
        """
        Make final detection decision based on model predictions.
        
        Args:
            predictions: Dict of {model_name: drone_probability}
        
        Returns:
            Tuple of (is_drone: bool, details: dict)
        """
        # Count votes
        votes = []
        details = {}
        probabilities = []
        
        for model_name, prob in predictions.items():
            # Get threshold from model info
            threshold = self.models[model_name].get('threshold', 0.5)
            is_drone = prob >= threshold
            votes.append(is_drone)
            probabilities.append(prob)
            
            details[model_name] = {
                'probability': prob,
                'threshold': threshold,
                'vote': 'DRONE' if is_drone else 'NO_DRONE'
            }
        
        # Voting strategy
        strategy = self.config['detection']['voting_strategy']
        
        if strategy == 'majority':
            # Majority vote
            is_drone = sum(votes) > len(votes) / 2
        elif strategy == 'unanimous':
            # All must agree
            is_drone = all(votes)
        elif strategy == 'any':
            # At least one detection
            is_drone = any(votes)
        else:
            # Default to majority
            is_drone = sum(votes) > len(votes) / 2
        
        # Calculate statistics
        probs_array = np.array(probabilities)
        mean_confidence = float(np.mean(probs_array))
        std_confidence = float(np.std(probs_array))
        min_confidence = float(np.min(probs_array))
        max_confidence = float(np.max(probs_array))
        
        # Add summary
        details['_summary'] = {
            'votes_for_drone': sum(votes),
            'total_models': len(votes),
            'strategy': strategy,
            'final_decision': 'DRONE' if is_drone else 'NO_DRONE',
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence
        }
        
        return is_drone, details
