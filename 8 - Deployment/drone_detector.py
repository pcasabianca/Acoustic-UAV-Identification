#!/usr/bin/env python3
"""
Real-Time Drone Detection System for Raspberry Pi
Monitors audio files and performs inference using trained models.

Author: Acoustic UAV Identification Team
Version: 2.0 (Refactored)
"""

import os
import sys
import time
import warnings
import threading
from pathlib import Path
from typing import Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Suppress Cryptography deprecation warnings (paramiko)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='paramiko')
warnings.filterwarnings('ignore', message='.*TripleDES.*')
warnings.filterwarnings('ignore', message='.*Blowfish.*')

# Import core modules
from core import ConfigManager, ModelLoader, FeatureExtractor, Predictor, AudioProcessor
from utils import setup_logger, ensure_directories


class DroneDetector:
    """Real-time drone detection system (refactored)."""
    
    def __init__(self, config_path: str = "deployment_config.json"):
        """
        Initialize detector.
        
        Args:
            config_path: Path to configuration JSON file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Setup logging
        log_dir = Path(self.config['output']['log_directory'])
        log_level = self.config['output']['log_level']
        console_output = self.config['output']['console_output']
        self.logger = setup_logger('DroneDetector', log_dir, log_level, console_output)
        
        # Recording thread
        self.recorder_thread = None
        self.stop_recording = threading.Event()
        
        # Initialize directories
        self._setup_directories()
        
        # Load models
        self.model_loader = ModelLoader(self.config, self.logger)
        self.models = self.model_loader.load_all()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.config, self.logger)
        
        # Initialize predictor
        self.predictor = Predictor(self.config, self.models, self.feature_extractor, self.logger)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            self.config, 
            self.feature_extractor, 
            self.predictor, 
            self.logger
        )
        
        # Startup banner
        self.logger.info("=" * 70)
        self.logger.info("[*] DRONE DETECTION SYSTEM INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        self.logger.info(f"Monitoring: {self.config['audio']['input_directory']}")
        self.logger.info(f"Scan interval: {self.config['detection']['scan_interval_seconds']}s")
        self.logger.info("=" * 70)
    
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.config['audio']['input_directory'],
            self.config['output']['log_directory'],
        ]
        
        if self.config.get('recording', {}).get('save_recordings'):
            dirs.append(self.config['recording']['recordings_directory'])
        
        ensure_directories([Path(d) for d in dirs])
    
    def reload_config(self) -> bool:
        """
        Reload configuration from disk.
        
        Returns:
            True if successful
        """
        if self.config_manager.reload():
            self.config = self.config_manager.config
            self.logger.info("✓ Configuration reloaded successfully")
            return True
        else:
            self.logger.error("Failed to reload configuration")
            return False
    
    def process_file(self, audio_path: Path) -> Optional[dict]:
        """
        Process a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detection result or None
        """
        result = self.audio_processor.process_file(audio_path)
        
        # Delete if configured
        if result and self.audio_processor.should_delete(audio_path):
            try:
                audio_path.unlink()
                self.logger.debug(f"Deleted: {audio_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to delete {audio_path.name}: {e}")
        
        return result
    
    def _recording_thread(self):
        """Background thread for audio recording."""
        try:
            from audio_recorder import AudioRecorder
            recorder = AudioRecorder(self.config_manager.config_path)
            
            self.logger.info("[Recording] Thread started - continuous recording")
            
            while not self.stop_recording.is_set():
                try:
                    # Record audio (this blocks for duration_seconds)
                    recorder.record_single()
                    
                    # No wait - immediately record next chunk
                    # This ensures continuous recording without gaps
                    
                except Exception as e:
                    self.logger.error(f"[Recording] Error: {e}")
                    time.sleep(0.5)
            
            recorder.cleanup()
            self.logger.info("[Recording] Thread stopped")
            
        except ImportError:
            self.logger.error("[Recording] pyaudio not installed, recording unavailable")
        except Exception as e:
            self.logger.error(f"[Recording] Fatal error: {e}")
    
    def start_recording(self):
        """Start background recording thread."""
        if self.recorder_thread is None or not self.recorder_thread.is_alive():
            self.stop_recording.clear()
            self.recorder_thread = threading.Thread(target=self._recording_thread, daemon=True)
            self.recorder_thread.start()
            time.sleep(1)  # Give thread time to start
    
    def stop_recording_thread(self):
        """Stop background recording thread."""
        if self.recorder_thread and self.recorder_thread.is_alive():
            self.logger.info("[Recording] Stopping...")
            self.stop_recording.set()
            self.recorder_thread.join(timeout=3)
    
    def run_continuous(self, with_recording: bool = False):
        """Run continuous monitoring mode.
        
        Args:
            with_recording: If True, start background audio recording
        """
        input_dir = Path(self.config['audio']['input_directory'])
        scan_interval = self.config['detection']['scan_interval_seconds']
        file_pattern = self.config['audio']['file_pattern']
        
        self.logger.info("Starting continuous monitoring...")
        self.logger.info(f"Directory: {input_dir}")
        self.logger.info(f"Pattern: {file_pattern}")
        self.logger.info(f"Interval: {scan_interval}s")
        
        # Start recording if requested
        if with_recording:
            self.logger.info("Recording: ENABLED")
            self.start_recording()
        else:
            self.logger.info("Recording: DISABLED (manual files only)")
        
        self.logger.info("Press Ctrl+C to stop")
        self.logger.info("-" * 70)
        
        try:
            while True:
                cycle_start = time.time()
                
                # Scan for files (reprocess all every cycle)
                files = self.audio_processor.scan_directory(input_dir, file_pattern, reprocess_all=True)
                
                if files:
                    for audio_path in files:
                        # Process file
                        self.process_file(audio_path)
                
                # Calculate elapsed time and wait only the remaining time
                elapsed = time.time() - cycle_start
                remaining = scan_interval - elapsed
                
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    self.logger.debug(f"Processing took {elapsed:.1f}s (>{scan_interval}s), starting next cycle immediately")
                
        except KeyboardInterrupt:
            self.logger.info("\nMonitoring stopped by user")
            # Stop recording thread if running
            if with_recording:
                self.stop_recording_thread()
    
    def run_single(self, audio_path: str):
        """
        Process a single audio file.
        
        Args:
            audio_path: Path to audio file
        """
        path = Path(audio_path)
        
        if not path.exists():
            self.logger.error(f"File not found: {audio_path}")
            return
        
        # Enable statistics for test mode
        original_stats_setting = self.config['output'].get('show_prediction_statistics', False)
        self.config['output']['show_prediction_statistics'] = True
        
        # Disable file deletion for test mode
        original_delete_setting = self.config['audio'].get('delete_after_processing', False)
        self.config['audio']['delete_after_processing'] = False
        
        self.logger.info(f"Processing single file: {path.name}")
        self.logger.info("-" * 70)
        
        result = self.process_file(path)
        
        if result:
            self.logger.info("-" * 70)
            self.logger.info("Processing complete")
        
        # Restore original settings
        self.config['output']['show_prediction_statistics'] = original_stats_setting
        self.config['audio']['delete_after_processing'] = original_delete_setting


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Drone Detection System')
    parser.add_argument('--config', default='deployment_config.json',
                       help='Path to config file')
    parser.add_argument('--file', type=str,
                       help='Process a single audio file')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous monitoring mode')
    parser.add_argument('--with-recording', action='store_true',
                       help='Enable background audio recording')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DroneDetector(args.config)
    
    if args.file:
        # Single file mode - process and EXIT
        detector.run_single(args.file)
    else:
        # Continuous monitoring mode (default or explicit --continuous)
        detector.run_continuous(with_recording=args.with_recording)


if __name__ == "__main__":
    main()
