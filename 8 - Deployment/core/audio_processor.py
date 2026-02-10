"""
Audio file processing and monitoring.
"""

import time
import json
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class AudioProcessor:
    """Handles audio file processing and directory monitoring."""
    
    def __init__(self, config: dict, feature_extractor, predictor, logger=None):
        """
        Initialize audio processor.
        
        Args:
            config: Configuration dictionary
            feature_extractor: FeatureExtractor instance
            predictor: Predictor instance
            logger: Logger instance (optional)
        """
        self.config = config
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.logger = logger
        self.processed_files = set()
    
    def inspect_file(self, audio_path: Path) -> None:
        """
        Read audio file and log diagnostic information.
        
        Args:
            audio_path: Path to audio file
        """
        try:
            data, sr = sf.read(str(audio_path), always_2d=False)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to inspect {audio_path.name}: {e}")
            return

        # Determine channels
        if hasattr(data, 'ndim') and data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1] if hasattr(data, 'shape') else 1

        # Ensure numpy array
        arr = np.asarray(data)
        num_samples = arr.shape[0]
        duration = float(num_samples) / float(sr) if sr > 0 else 0.0

        # For multi-channel, compute metrics on mean across channels
        if arr.ndim > 1:
            mono = np.mean(arr, axis=1)
        else:
            mono = arr

        peak = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size > 0 else 0.0

        # dBFS
        rms_db = 20.0 * np.log10(rms) if rms > 0 else float('-inf')
        peak_db = 20.0 * np.log10(peak) if peak > 0 else float('-inf')

        # Clipping detection
        clipping = bool(np.any(np.abs(mono) >= 0.9999))

        if self.logger:
            self.logger.info(
                f"[AUDIO INSPECT] {audio_path.name} | dur: {duration:.2f}s | sr: {sr} | ch: {channels} | "
                f"peak: {peak:.4f} ({peak_db:.1f} dB) | rms: {rms:.4f} ({rms_db:.1f} dB) | "
                f"clipping: {clipping}"
            )
    
    def process_file(self, audio_path: Path) -> Optional[Dict]:
        """
        Process a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detection result dictionary or None if failed
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Processing: {audio_path.name}")
        
        # Inspect audio file
        self.inspect_file(audio_path)
        
        # Extract features
        features = self.feature_extractor.extract_mel(audio_path)
        if features is None:
            return None
        
        # Reload thresholds if configured
        if self.config['detection'].get('reload_thresholds_every_prediction', False):
            if hasattr(self.predictor, 'reload_thresholds'):
                self.predictor.reload_thresholds()
        
        # Get predictions
        predictions = self.predictor.predict_all(features)
        
        # Make decision
        is_drone, details = self.predictor.make_decision(predictions)
        
        processing_time = time.time() - start_time
        
        # Build result
        result = {
            'timestamp': datetime.now().strftime(self.config['output']['timestamp_format']),
            'file': str(audio_path.name),
            'detection': 'DRONE' if is_drone else 'NO_DRONE',
            'predictions': predictions,
            'details': details,
            'processing_time_ms': int(processing_time * 1000)
        }
        
        # Log result
        self._log_detection(result)
        
        # Save if configured
        if self.config['output']['save_predictions']:
            self._save_prediction(result)
        
        return result
    
    def _log_detection(self, result: Dict) -> None:
        """Log detection result."""
        if not self.logger:
            return
        
        is_drone = result['detection'] == 'DRONE'
        
        # Prefix
        if is_drone:
            prefix = "[!] ALERT"
            level = logging.WARNING
        else:
            prefix = "[+] CLEAR"
            level = logging.INFO
        
        # Main message
        msg = f"{prefix} | {result['file']} | {result['detection']}"
        
        # Add votes summary
        summary = result['details'].get('_summary', {})
        votes = summary.get('votes_for_drone', 0)
        total = summary.get('total_models', 0)
        msg += f" | Votes: {votes}/{total}"
        
        # Log with appropriate level
        self.logger.log(level, msg)
        
        # Show statistics if enabled
        if self.config['output'].get('show_prediction_statistics', False):
            mean_conf = summary.get('mean_confidence', 0)
            std_conf = summary.get('std_confidence', 0)
            min_conf = summary.get('min_confidence', 0)
            max_conf = summary.get('max_confidence', 0)
            self.logger.info(f"    [STATS] Mean: {mean_conf:.2%} | Std: {std_conf:.2%} | Range: [{min_conf:.2%}, {max_conf:.2%}]")
        
        # Detailed model results - ALWAYS show for transparency
        for model_name, model_result in result['details'].items():
            if model_name == '_summary':
                continue
            prob = model_result['probability']
            threshold = model_result['threshold']
            vote = model_result['vote']
            self.logger.info(f"    {model_name}: {prob:.2%} (threshold: {threshold:.2f}) -> {vote}")
    
    def _save_prediction(self, result: Dict) -> None:
        """Save prediction to JSON file."""
        predictions_file = Path(self.config['output']['predictions_file'])
        predictions_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load existing predictions
            if predictions_file.exists():
                try:
                    with open(predictions_file, 'r') as f:
                        predictions = json.load(f)
                except json.JSONDecodeError as e:
                    # JSON file is corrupted, create backup and start fresh
                    if self.logger:
                        self.logger.warning(f"Corrupted predictions file detected: {e}")
                        self.logger.warning(f"Creating backup and starting fresh...")
                    
                    # Backup corrupted file
                    backup_path = predictions_file.with_suffix('.json.corrupted')
                    predictions_file.rename(backup_path)
                    
                    if self.logger:
                        self.logger.info(f"Backup saved to: {backup_path}")
                    
                    # Start with empty list
                    predictions = []
            else:
                predictions = []
            
            # Append new result
            predictions.append(result)
            
            # Save with atomic write (write to temp file then rename)
            temp_file = predictions_file.with_suffix('.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            # Atomic rename (safer)
            temp_file.replace(predictions_file)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save prediction: {e}")
    
    def scan_directory(self, directory: Path, pattern: str = "*.wav", reprocess_all: bool = True) -> List[Path]:
        """
        Scan directory for audio files.
        
        Args:
            directory: Directory to scan
            pattern: File pattern to match
            reprocess_all: If True, return all files (ignore processed tracking)
            
        Returns:
            List of audio files
        """
        all_files = sorted(Path(directory).glob(pattern))
        
        if reprocess_all:
            # Return all files every cycle
            return all_files
        else:
            # Only return files not yet processed
            new_files = [f for f in all_files if f not in self.processed_files]
            return new_files
    
    def mark_processed(self, audio_path: Path) -> None:
        """Mark a file as processed."""
        self.processed_files.add(audio_path)
    
    def should_delete(self, audio_path: Path) -> bool:
        """Check if file should be deleted after processing."""
        return self.config['audio'].get('delete_after_processing', False)
