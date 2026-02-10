"""
Model loading and management with timeout handling.
"""

import json
import signal
import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional


class TimeoutError(Exception):
    """Raised when model loading times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutError("Model loading timeout")


class ModelLoader:
    """Handles loading and managing TensorFlow models."""
    
    def __init__(self, config: dict, logger=None):
        """
        Initialize model loader.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger
        self.models = {}
    
    def load_thresholds(self) -> Dict[str, float]:
        """
        Load thresholds from external thresholds file.
        
        Returns:
            Dict mapping calibrated_key to threshold value
        """
        thresholds_file = self.config['detection'].get('thresholds_file', './thresholds.json')
        thresholds_path = Path(thresholds_file)
        
        if not thresholds_path.exists():
            if self.logger:
                self.logger.warning(f"Thresholds file not found: {thresholds_path}")
            return {}
        
        try:
            with open(thresholds_path, 'r') as f:
                data = json.load(f)
            
            # Extract thresholds from file
            thresholds = {}
            for model_key, model_data in data.get('models', {}).items():
                thresholds[model_key] = model_data.get('threshold', 0.5)
            
            return thresholds
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load thresholds: {e}")
            return {}
    
    def load_single_model(self, model_path: Path, safe_mode: bool = True, 
                         timeout: int = 30) -> Optional[tf.keras.Model]:
        """
        Load a single Keras model with timeout protection.
        
        Args:
            model_path: Path to .keras file
            safe_mode: Use safe mode for deserialization
            timeout: Timeout in seconds (0 = no timeout)
            
        Returns:
            Loaded model or None if failed
        """
        try:
            # Set timeout if supported (Unix only)
            if timeout > 0 and hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            try:
                if safe_mode:
                    model = tf.keras.models.load_model(model_path, compile=False)
                else:
                    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                
                if timeout > 0 and hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout
                
                return model
                
            except TimeoutError:
                if timeout > 0 and hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                if self.logger:
                    self.logger.error(f"Model loading timeout (>{timeout}s): {model_path.name}")
                return None
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load {model_path.name}: {e}")
            return None
    
    def load_all(self) -> Dict:
        """
        Load all enabled models from configuration.
        
        Returns:
            Dictionary of loaded models with metadata
        """
        models_dir = Path(self.config['models']['directory'])
        enabled_models = self.config['detection']['enabled_models']
        available_models = self.config['models']['available_models']
        
        if self.logger:
            self.logger.info("Loading models...")
        
        # Load thresholds from external file
        calibrated_thresholds = self.load_thresholds()
        
        # Track loaded files to avoid duplicates
        loaded_files = set()
        
        for model_name in enabled_models:
            if model_name not in available_models:
                if self.logger:
                    self.logger.warning(f"Model {model_name} not in available models, skipping")
                continue
            
            model_info = available_models[model_name]
            model_path = models_dir / model_info['filename']
            
            if not model_path.exists():
                if self.logger:
                    self.logger.warning(f"Model file not found: {model_path}, skipping")
                continue
            
            # Check if model requires safe_mode=False
            safe_mode = model_info.get('safe_mode', True)
            
            if self.logger:
                self.logger.info(f"  Loading {model_name} from {model_path.name} ({model_path.stat().st_size / (1024*1024):.1f} MB)...")
                if not safe_mode:
                    self.logger.info(f"    Using safe_mode=False for {model_name}")
            
            # Load model with timeout
            model = self.load_single_model(model_path, safe_mode=safe_mode, timeout=30)
            
            if model is None:
                continue
            
            if self.logger:
                self.logger.info(f"    Model loaded, initializing...")
            
            # Get threshold from calibration file
            calibrated_key = model_info.get('calibrated_key', model_name)
            threshold = calibrated_thresholds.get(calibrated_key, 0.5)
            
            self.models[model_name] = {
                'model': model,
                'threshold': threshold,
                'calibrated_key': calibrated_key,
                'description': model_info['description']
            }
            loaded_files.add(model_path.name)
            
            if self.logger:
                self.logger.info(f"  [OK] Loaded {model_name}: {model_info['description']} (t={threshold:.2f})")
        
        if not self.models:
            raise RuntimeError("No models loaded! Check configuration and model files.")
        
        return self.models
    
    def get_models(self) -> Dict:
        """Get loaded models dictionary."""
        return self.models
    
    def reload_thresholds(self) -> None:
        """Reload thresholds and update loaded models."""
        calibrated_thresholds = self.load_thresholds()
        for model_name, model_info in self.models.items():
            calibrated_key = model_info.get('calibrated_key', model_name)
            if calibrated_key in calibrated_thresholds:
                model_info['threshold'] = calibrated_thresholds[calibrated_key]
