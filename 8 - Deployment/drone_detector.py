#!/usr/bin/env python3
"""
Real-Time Drone Detection System for Raspberry Pi
Monitors audio files and performs inference using trained models.

Author: Acoustic UAV Identification Team
Version: 1.0
"""

import os
import sys
import json
import time
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class DroneDetector:
    """Real-time drone detection system."""
    
    def __init__(self, config_path: str = "deployment_config.json"):
        """
        Initialize detector.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.models = {}
        self.detection_history = []
        
        # Setup logging
        self.setup_logging()
        
        # Load models
        self.load_models()
        
        # Initialize directories
        self.setup_directories()
        
        self.logger.info("=" * 70)
        self.logger.info("[*] DRONE DETECTION SYSTEM INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        self.logger.info(f"Monitoring: {self.config['audio']['input_directory']}")
        self.logger.info(f"Scan interval: {self.config['detection']['scan_interval_seconds']}s")
        self.logger.info("=" * 70)
    
    def load_config(self) -> dict:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def reload_config(self):
        """Reload configuration (for runtime updates)."""
        try:
            new_config = self.load_config()
            self.config = new_config
            self.logger.info("✓ Configuration reloaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload config: {e}")
            return False
    
    def cleanup_old_logs(self, log_dir: Path, days: int = 7):
        """
        Remove log files older than specified days.
        
        Args:
            log_dir: Directory containing log files
            days: Keep logs from last N days
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (days * 86400)  # days in seconds
            
            removed_count = 0
            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                print(f"[CLEANUP] Removed {removed_count} old log file(s)")
        except Exception as e:
            print(f"[WARN] Log cleanup failed: {e}")
    
    def setup_logging(self):
        """Configure logging."""
        log_dir = Path(self.config['output']['log_directory'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Cleanup old logs (keep only last 7 days)
        self.cleanup_old_logs(log_dir, days=7)
        
        log_level = getattr(logging, self.config['output']['log_level'])
        
        # Create logger
        self.logger = logging.getLogger('DroneDetector')
        self.logger.setLevel(log_level)
        
        # File handler
        log_file = log_dir / f"detector_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(log_level)
        
        # Console handler (force UTF-8 on Windows)
        import sys
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        if self.config['output']['console_output']:
            self.logger.addHandler(ch)
    
    def setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.config['audio']['input_directory'],
            self.config['output']['log_directory'],
        ]
        
        if self.config.get('recording', {}).get('save_recordings'):
            dirs.append(self.config['recording']['recordings_directory'])
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_thresholds(self) -> Dict[str, float]:
        """Load thresholds from external thresholds file.
        
        Returns:
            Dict mapping calibrated_key to threshold value
        """
        thresholds_file = self.config['detection'].get('thresholds_file', './thresholds.json')
        thresholds_path = Path(thresholds_file)
        
        if not thresholds_path.exists():
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
            self.logger.error(f"Failed to load thresholds: {e}")
            return {}
    
    def load_models(self):
        """Load trained models from disk."""
        models_dir = Path(self.config['models']['directory'])
        enabled_models = self.config['detection']['enabled_models']
        available_models = self.config['models']['available_models']
        
        self.logger.info("Loading models...")
        
        # Load thresholds from external file
        calibrated_thresholds = self.load_thresholds()
        
        for model_name in enabled_models:
            if model_name not in available_models:
                self.logger.warning(f"Model {model_name} not in available models, skipping")
                continue
            
            model_info = available_models[model_name]
            model_path = models_dir / model_info['filename']
            
            if not model_path.exists():
                self.logger.warning(f"Model file not found: {model_path}, skipping")
                continue
            
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Get threshold from calibration file using calibrated_key
                calibrated_key = model_info.get('calibrated_key', model_name)
                threshold = calibrated_thresholds.get(calibrated_key, 0.5)
                
                self.models[model_name] = {
                    'model': model,
                    'threshold': threshold,
                    'calibrated_key': calibrated_key,
                    'description': model_info['description']
                }
                self.logger.info(f"  [OK] Loaded {model_name}: {model_info['description']} (t={threshold:.2f})")
            except Exception as e:
                self.logger.error(f"  ✗ Failed to load {model_name}: {e}")
        
        if not self.models:
            raise RuntimeError("No models loaded! Check configuration and model files.")
    
    def extract_mel_features(self, audio_path: Path) -> Optional[np.ndarray]:
        """
        Extract MEL spectrogram features from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            MEL spectrogram features (44, 90) or None if failed
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

            # Normalize audio peak to target (default 0.95)
            try:
                target_peak = self.config.get('audio', {}).get('normalize_peak', 0.95)
            except Exception:
                target_peak = 0.95

            try:
                peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
                if peak > 0 and target_peak > 0:
                    scale = float(target_peak) / peak
                    audio = audio * scale
                    self.logger.debug(f"[AUDIO NORMALIZE] Applied scale {scale:.4f} to set peak {target_peak}")
            except Exception as e:
                self.logger.debug(f"Failed to normalize audio: {e}")
            
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
            
            # Convert to log scale (ref=np.max pour compatibilité avec l'entraînement)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # NOTE: Pas de normalisation pour préserver les différences SNR entre distances
            # La normalisation détruirait l'information critique pour la détection
            
            return mel_db
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {audio_path}: {e}")
            return None
    
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Run inference on all loaded models.
        
        Args:
            features: MEL spectrogram (44, 90)
        
        Returns:
            Dict of {model_name: drone_probability}
        """
        # Reload thresholds if configured
        if self.config['detection'].get('reload_thresholds_every_prediction', False):
            calibrated_thresholds = self.load_thresholds()
            for model_name, model_info in self.models.items():
                calibrated_key = model_info.get('calibrated_key', model_name)
                if calibrated_key in calibrated_thresholds:
                    model_info['threshold'] = calibrated_thresholds[calibrated_key]
        
        # Prepare input (add batch and channel dimensions)
        X = features[np.newaxis, ..., np.newaxis]  # (1, 44, 90, 1)
        
        predictions = {}
        
        for model_name, model_info in self.models.items():
            try:
                # Get prediction
                pred = model_info['model'].predict(X, verbose=0)
                
                # Extract drone probability (handle different architectures)
                if pred.shape[1] == 2:
                    # Softmax with 2 classes (CNN, CRNN, Attention-CRNN)
                    drone_prob = float(pred[0][1])
                elif pred.shape[1] == 1:
                    # Sigmoid (RNN)
                    drone_prob = float(pred[0][0])
                else:
                    self.logger.error(f"Unexpected prediction shape for {model_name}: {pred.shape}")
                    drone_prob = 0.0
                
                predictions[model_name] = drone_prob
                
            except Exception as e:
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
        
        for model_name, prob in predictions.items():
            # Get threshold from model info (loaded from thresholds.json)
            threshold = self.models[model_name].get('threshold', 0.5)
            is_drone = prob >= threshold
            votes.append(is_drone)
            
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
            # All models must agree
            is_drone = all(votes)
        elif strategy == 'any':
            # Any model detecting is enough
            is_drone = any(votes)
        else:
            # Default: majority
            is_drone = sum(votes) > len(votes) / 2
        
        details['final_decision'] = 'DRONE' if is_drone else 'NO_DRONE'
        details['votes_for_drone'] = sum(votes)
        details['total_votes'] = len(votes)
        details['strategy'] = strategy
        
        return is_drone, details
    
    def process_audio_file(self, audio_path: Path) -> Optional[Dict]:
        """
        Process single audio file through complete pipeline.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Detection result dict or None if failed
        """
        self.logger.info(f"Processing: {audio_path.name}")

        # Inspect audio file for basic health metrics (duration, sr, RMS, peak, clipping)
        try:
            self.inspect_audio_file(audio_path)
        except Exception as e:
            self.logger.warning(f"Audio inspection failed for {audio_path.name}: {e}")
        
        start_time = time.time()
        
        # Extract features
        features = self.extract_mel_features(audio_path)
        if features is None:
            return None
        
        # Run inference
        predictions = self.predict(features)
        
        # Make decision
        is_drone, details = self.make_decision(predictions)
        
        # Calculate processing time
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
        self.log_detection(result)
        
        # Save if configured
        if self.config['output']['save_predictions']:
            self.save_prediction(result)
        
        return result

    def inspect_audio_file(self, audio_path: Path):
        """Read audio file and log diagnostic information useful to detect corrupt/buggy files.

        Logs: sample rate, duration, channels, peak amplitude, RMS (and dB), clipping indicator.
        """
        try:
            data, sr = sf.read(str(audio_path), always_2d=False)
        except Exception as e:
            raise

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

        # dBFS (0 dB = full scale). Handle zeros safely.
        rms_db = 20.0 * np.log10(rms) if rms > 0 else float('-inf')
        peak_db = 20.0 * np.log10(peak) if peak > 0 else float('-inf')

        # Clipping detection: values near or at full-scale (>= 0.9999)
        clipping = bool(np.any(np.abs(mono) >= 0.9999))

        # Compute normalized metrics using same target as feature extraction (do not modify file)
        try:
            target_peak = float(self.config.get('audio', {}).get('normalize_peak', 0.95))
        except Exception:
            target_peak = 0.95

        if peak > 0 and target_peak > 0:
            scale = target_peak / peak
            norm_peak = peak * scale
            norm_rms = rms * scale
            norm_peak_db = 20.0 * np.log10(norm_peak) if norm_peak > 0 else float('-inf')
            norm_rms_db = 20.0 * np.log10(norm_rms) if norm_rms > 0 else float('-inf')
        else:
            scale = 0.0
            norm_peak = 0.0
            norm_rms = 0.0
            norm_peak_db = float('-inf')
            norm_rms_db = float('-inf')

        # Log concise diagnostics with both original and normalized values
        self.logger.info(
            f"[AUDIO INSPECT] {audio_path.name} | dur: {duration:.2f}s | sr: {sr} | ch: {channels} | "
            f"orig_peak: {peak:.4f} ({peak_db:.1f} dB) | orig_rms: {rms:.4f} ({rms_db:.1f} dB) | "
            f"norm_peak: {norm_peak:.4f} ({norm_peak_db:.1f} dB) | norm_rms: {norm_rms:.4f} ({norm_rms_db:.1f} dB) | "
            f"scale: {scale:.4f} | clipping: {clipping}"
        )
    
    def log_detection(self, result: Dict):
        """Log detection result."""
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
        
        # Add confidence if configured
        if self.config['alerts']['include_confidence']:
            avg_conf = np.mean(list(result['predictions'].values()))
            msg += f" | Avg Confidence: {avg_conf:.2%}"
        
        # Add model breakdown if configured
        if self.config['alerts']['include_model_breakdown']:
            votes_str = f" | Votes: {result['details']['votes_for_drone']}/{result['details']['total_votes']}"
            msg += votes_str
        
        # Add processing time
        msg += f" | {result['processing_time_ms']}ms"
        
        self.logger.log(level, msg)
        
        # Detailed breakdown - ALWAYS show for both DRONE and CLEAR
        if self.config['alerts']['include_model_breakdown']:
            for model_name, model_details in result['details'].items():
                if model_name not in ['final_decision', 'votes_for_drone', 'total_votes', 'strategy']:
                    self.logger.info(
                        f"    {model_name}: {model_details['probability']:.2%} "
                        f"(threshold: {model_details['threshold']:.2f}) -> {model_details['vote']}"
                    )
    
    def save_prediction(self, result: Dict):
        """Save prediction to JSON file."""
        predictions_file = Path(self.config['output']['predictions_file'])
        
        # Load existing predictions
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
        else:
            predictions = []
        
        # Append new result
        predictions.append(result)
        
        # Keep only last 1000 predictions
        predictions = predictions[-1000:]
        
        # Save
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
    
    def scan_and_process(self):
        """Scan input directory and process new audio files."""
        input_dir = Path(self.config['audio']['input_directory'])
        file_pattern = self.config['audio']['file_pattern']
        
        # Find audio files
        audio_files = list(input_dir.glob(file_pattern))
        
        if not audio_files:
            return
        
        self.logger.info(f"Found {len(audio_files)} audio file(s) to process")
        
        for audio_file in audio_files:
            # Process file
            result = self.process_audio_file(audio_file)
            
            # Delete if configured
            if result and self.config['audio']['delete_after_processing']:
                try:
                    audio_file.unlink()
                    self.logger.debug(f"Deleted processed file: {audio_file.name}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {audio_file.name}: {e}")
    
    def run_continuous(self):
        """Run continuous monitoring loop."""
        scan_interval = self.config['detection']['scan_interval_seconds']
        
        self.logger.info("Starting continuous monitoring...")
        self.logger.info(f"Press Ctrl+C to stop")
        
        try:
            iteration = 0
            while True:
                # Scan and process
                self.scan_and_process()
                
                # Periodic cleanup (every 100 iterations ~8 minutes at 5s interval)
                iteration += 1
                if iteration % 100 == 0:
                    self.cleanup_old_audio_files()
                
                # Wait for next scan
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("Monitoring stopped by user")
            self.logger.info("=" * 70)
    
    def cleanup_old_audio_files(self):
        """Remove old audio files from input directory (safety cleanup)."""
        try:
            input_dir = Path(self.config['audio']['input_directory'])
            current_time = time.time()
            cutoff_time = current_time - 300  # Remove files older than 5 minutes
            
            removed_count = 0
            for audio_file in input_dir.glob("*.wav"):
                if audio_file.stat().st_mtime < cutoff_time:
                    audio_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                self.logger.debug(f"Cleaned up {removed_count} old audio file(s)")
        except Exception as e:
            self.logger.warning(f"Audio cleanup failed: {e}")
    
    def run_single_file(self, audio_path: str):
        """Process single audio file (for testing)."""
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            self.logger.error(f"File not found: {audio_path}")
            return
        
        self.logger.info("=" * 70)
        self.logger.info("SINGLE FILE DETECTION MODE")
        self.logger.info("=" * 70)
        
        result = self.process_audio_file(audio_path)
        
        if result:
            self.logger.info("=" * 70)
            self.logger.info("DETECTION COMPLETE")
            self.logger.info("=" * 70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Drone Detection System')
    parser.add_argument('--config', type=str, default='deployment_config.json',
                        help='Path to configuration file')
    parser.add_argument('--file', type=str, default=None,
                        help='Process single file (test mode)')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuous monitoring (default)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DroneDetector(config_path=args.config)
    
    if args.file:
        # Single file mode
        detector.run_single_file(args.file)
    else:
        # Continuous monitoring mode
        detector.run_continuous()


if __name__ == "__main__":
    main()
