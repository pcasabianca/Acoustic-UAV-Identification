"""
Feature extraction and adaptation for audio processing.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf


class FeatureExtractor:
    """Handles MEL spectrogram extraction and feature adaptation."""
    
    def __init__(self, config: dict, logger=None):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger
    
    def extract_mel(self, audio_path: Path) -> Optional[np.ndarray]:
        """
        Extract MEL spectrogram features from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            MEL spectrogram features or None if failed
        """
        try:
            # Load audio
            sr = self.config['audio']['sample_rate']
            duration = self.config['audio']['duration_seconds']
            
            # Convert Path to string for librosa
            audio, _ = librosa.load(str(audio_path), sr=sr, duration=duration)
            
            # Ensure exact duration
            expected_samples = int(sr * duration)
            if len(audio) < expected_samples:
                # Pad if too short
                audio = np.pad(audio, (0, expected_samples - len(audio)), mode='constant')
            elif len(audio) > expected_samples:
                # Trim if too long
                audio = audio[:expected_samples]

            # Normalize audio peak to target (default 0.95) if enabled in config
            if self.config.get('feature_extraction', {}).get('normalization', False):
                target_peak = self.config.get('audio', {}).get('normalize_peak', 0.95)
                peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
                if peak > 0 and target_peak > 0:
                    scale = float(target_peak) / peak
                    audio = audio * scale
                    if self.logger:
                        self.logger.debug(f"[AUDIO NORMALIZE] Applied scale {scale:.4f} to set peak {target_peak}")
            
            # Extract MEL spectrogram
            mel_config = self.config['feature_extraction']['mel_spectrogram']
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=mel_config['n_mels'],
                n_fft=mel_config['n_fft'],
                hop_length=mel_config['hop_length'],
                fmin=mel_config['fmin'],
                fmax=mel_config['fmax']
            )
            
            # Convert to log scale
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            return mel_db
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Feature extraction failed for {audio_path}: {e}")
            return None
    
    def adapt_to_model(self, features: np.ndarray, model: tf.keras.Model) -> np.ndarray:
        """
        Adapt MEL features to model's expected input shape.
        
        Args:
            features: Raw MEL spectrogram
            model: TensorFlow model
            
        Returns:
            Adapted features ready for inference
        """
        mel = np.squeeze(features) if features.ndim != 2 else features
        input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
        rank = len(input_shape) if input_shape else 4
        
        # Conv2D: (batch, H, W, C)
        if rank == 4:
            _, H, W, _ = input_shape
            H = int(H) if H else mel.shape[0]
            W = int(W) if W else mel.shape[1]
            mel_adapted = self._pad_crop_2d(mel, H, W)
            return mel_adapted[np.newaxis, :, :, np.newaxis]
        
        # RNN: (batch, H, W)
        elif rank == 3:
            _, H, W = input_shape
            H = int(H) if H else mel.shape[0]
            W = int(W) if W else mel.shape[1]
            mel_adapted = self._pad_crop_2d(mel, H, W)
            return mel_adapted[np.newaxis, :, :]
        
        # Fallback: flatten
        return mel.flatten()[np.newaxis, :]
    
    def _pad_crop_2d(self, arr: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Pad or center-crop array to target (H, W).
        
        Args:
            arr: 2D array
            H: Target height
            W: Target width
            
        Returns:
            Resized array
        """
        h, w = arr.shape
        fill = float(arr.min()) if arr.size > 0 else 0.0
        
        # Height
        if h < H:
            pad = H - h
            arr = np.pad(arr, ((pad//2, pad - pad//2), (0, 0)), constant_values=fill)
        elif h > H:
            arr = arr[(h-H)//2:(h-H)//2 + H, :]
        
        # Width
        h, w = arr.shape
        if w < W:
            pad = W - w
            arr = np.pad(arr, ((0, 0), (pad//2, pad - pad//2)), constant_values=fill)
        elif w > W:
            arr = arr[:, (w-W)//2:(w-W)//2 + W]
        
        return arr
