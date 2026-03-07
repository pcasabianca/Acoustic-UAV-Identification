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
from queue import Queue, Empty
from pathlib import Path
from typing import Optional
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Suppress Cryptography deprecation warnings (paramiko)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='paramiko')
warnings.filterwarnings('ignore', message='.*TripleDES.*')
warnings.filterwarnings('ignore', message='.*Blowfish.*')

# Import core modules and new orchestrators
import json
from utils.logger import setup_logger
from utils.file_utils import ensure_directories
from utils.orchestrator_global import OrchestratorGlobal
from utils.audio_buffer import AudioBuffer


class DroneDetector:
    """Minimal front-end that delegates to `OrchestratorGlobal`.

    This class keeps the CLI and logging surface but defers all pipeline
    logic to the orchestrator implemented in `utils/orchestrator_global.py`.
    """

    def __init__(self, config_path: str = "deployment_config.json", save_to_disk: bool = False, csv_log_path: Optional[str] = None):
        # load configuration using utils (simple JSON load)
        self.config_path = config_path
        try:
            with open(config_path, 'r') as _f:
                self.config = json.load(_f)
        except Exception as e:
            raise RuntimeError(f'Failed to load config {config_path}: {e}')

        # Setup logging
        log_dir = Path(self.config['output']['log_directory'])
        log_level = self.config['output']['log_level']
        console_output = self.config['output']['console_output']
        self.logger = setup_logger('DroneDetector', log_dir, log_level, console_output)

        # Ensure directories exist
        self._setup_directories()

        # Create orchestrator and expose it (propagate disk/log options)
        self.global_orch = OrchestratorGlobal(config_path=config_path, logger=self.logger,
                              save_to_disk=save_to_disk, csv_log_path=csv_log_path)

        self.logger.info("=" * 70)
        self.logger.info("[*] DRONE DETECTION SYSTEM INITIALIZED (orchestrated)")
        self.logger.info("=" * 70)

    def _setup_directories(self):
        dirs = [
            self.config['audio']['input_directory'],
            self.config['output']['log_directory'],
        ]
        ensure_directories([Path(d) for d in dirs])

    def run_once(self):
        """Run one record->feature->predict cycle via orchestrator and log results."""
        out = self.global_orch.run_once()
        # Print concise summary
        self.logger.info(f"Single run predictions: {out.get('predictions')}")
        return out

    def run_continuous(self):
        """Start the orchestrator continuous pipeline (blocks until Ctrl-C)."""
        self.global_orch.run_continuous()

    def run_single(self, audio_path: str):
        """Process a single WAV file through the orchestrator's feature + models."""
        p = Path(audio_path)
        if not p.exists():
            self.logger.error(f"File not found: {audio_path}")
            return
        try:
            buf = AudioBuffer.from_wav_file(str(p))
            feats = self.global_orch.feature_extractor.extract_mel(audio_array=buf.array, sr=buf.sr, mel_overrides={'n_mels': self.global_orch.desired_n_mels})
            from utils.predictor_utils import run_models
            res = run_models(self.global_orch.models, self.global_orch.feature_extractor, feats, logger=self.logger)
            self.logger.info(f"File run predictions: {res.get('predictions')}")
            return res
        except Exception as e:
            self.logger.error(f"Failed to run single file: {e}")
            return None


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
                       help='Enable background audio recording (default: Pi mode, RAM storage)')
    parser.add_argument('--laptop', action='store_true',
                       help='Use laptop mode (PyAudio) instead of Pi mode (SPI)')
    parser.add_argument('--save-to-disk', action='store_true',
                       help='Save recordings to disk instead of RAM')
    parser.add_argument('--log-csv', dest='log_csv', type=str, default=None,
                       help='Append predictions to CSV file (path)')
    
    args = parser.parse_args()
    
    # Initialize detector and propagate flags to orchestrator
    detector = DroneDetector(args.config, save_to_disk=args.save_to_disk, csv_log_path=args.log_csv)
    
    if args.file:
        # Single file mode - process and EXIT
        detector.run_single(args.file)
    else:
        # Continuous monitoring mode (default)
        # The orchestrator manages recording and pipeline; simply start it.
        detector.run_continuous()


if __name__ == "__main__":
    main()
