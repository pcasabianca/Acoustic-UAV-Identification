#!/usr/bin/env python3
"""
Audio Recorder for Raspberry Pi
Continuously records audio from microphone and saves to input directory.

Usage:
    python3 audio_recorder.py --duration 4 --interval 5
"""

import os
import sys
import json
import time
import wave
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import pyaudio
except ImportError:
    print("[ERROR] pyaudio not installed!")
    print("Install with: pip install pyaudio")
    print("On Raspberry Pi: sudo apt-get install python3-pyaudio")
    sys.exit(1)


class AudioRecorder:
    """Records audio from microphone."""
    
    def __init__(self, config_path: str = "deployment_config.json"):
        """Initialize recorder."""
        self.config = self.load_config(config_path)
        
        # Audio parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.duration = self.config['audio']['duration_seconds']
        self.channels = self.config['audio']['channels']
        self.output_dir = Path(self.config['audio']['input_directory'])
        
        # Recording config
        self.device_index = self.config['recording'].get('device_index', None)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        
        print("=" * 70)
        print("AUDIO RECORDER FOR DRONE DETECTION")
        print("=" * 70)
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Duration: {self.duration} seconds")
        print(f"Channels: {self.channels}")
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device_index if self.device_index is not None else 'Default'}")
        print("=" * 70)
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def list_devices(self):
        """List available audio input devices."""
        print("\nAvailable Audio Devices:")
        print("-" * 70)
        
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
                print(f"      Sample Rate: {int(info['defaultSampleRate'])} Hz")
                print(f"      Input Channels: {info['maxInputChannels']}")
                print()
    
    def record_audio(self) -> np.ndarray:
        """
        Record audio from microphone.
        
        Returns:
            Audio data as numpy array
        """
        try:
            # Open stream
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024
            )
            
            print(f"[REC] Recording {self.duration} seconds...", end='', flush=True)
            
            # Calculate number of frames
            frames_per_buffer = 1024
            num_frames = int(self.sample_rate * self.duration / frames_per_buffer)
            
            # Record
            frames = []
            for _ in range(num_frames):
                data = stream.read(frames_per_buffer, exception_on_overflow=False)
                frames.append(data)
            
            print(" Done!")
            
            # Stop stream
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            return audio_array
            
        except Exception as e:
            print(f"[ERROR] Recording failed: {e}")
            return None
    
    def save_audio(self, audio: np.ndarray, filename: str):
        """
        Save audio to WAV file with normalization.
        
        Args:
            audio: Audio data as numpy array
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Optionally normalize audio before saving (controlled by config.feature_extraction.normalization)
        if self.config.get('feature_extraction', {}).get('normalization', False):
            max_val = np.abs(audio).max()
            if max_val > 0:
                # Normalize to 80% of max to avoid clipping
                audio = audio * (0.8 / max_val)
                print(f"[NORM] Normalized audio (gain: {0.8/max_val:.1f}x)")
        
        # Convert back to int16
        audio_int = (audio * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int.tobytes())
        
        print(f"[SAVE] Saved: {output_path.name}")
    
    def run_continuous(self, interval: int = None):
        """
        Run continuous recording with interval.
        
        Args:
            interval: Seconds between recordings (default from config)
        """
        if interval is None:
            interval = self.config['detection']['scan_interval_seconds']
        
        print(f"\nStarting continuous recording (interval: {interval}s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            recording_count = 0
            
            while True:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.wav"
                
                # Record
                audio = self.record_audio()
                
                if audio is not None:
                    # Delete old recordings BEFORE saving new one
                    # This ensures the detector finishes processing before deletion
                    for old_file in self.output_dir.glob("recording_*.wav"):
                        try:
                            old_file.unlink()
                            print(f"[CLEANUP] Removed old: {old_file.name}")
                        except Exception:
                            pass  # File in use, will be deleted next time
                    
                    # Save new recording
                    self.save_audio(audio, filename)
                    recording_count += 1
                    print(f"[OK] Total recordings: {recording_count}\n")
                
                # No wait - record continuously every 4 seconds
                # (removed sleep to get new recording immediately)
                
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print(f"Recording stopped. Total recordings: {recording_count}")
            print("=" * 70)
    
    def record_single(self, filename: str = None):
        """Record single audio file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
        
        audio = self.record_audio()
        
        if audio is not None:
            self.save_audio(audio, filename)
            print("[OK] Recording complete!")
    
    def cleanup(self):
        """Cleanup resources."""
        self.pa.terminate()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Audio Recorder for Drone Detection')
    parser.add_argument('--config', type=str, default='deployment_config.json',
                        help='Path to configuration file')
    parser.add_argument('--duration', type=int, default=None,
                        help='Recording duration in seconds (overrides config)')
    parser.add_argument('--interval', type=int, default=None,
                        help='Interval between recordings (overrides config)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio device index (use --list-devices to see options)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices')
    parser.add_argument('--single', action='store_true',
                        help='Record single file (test mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for single recording')
    
    args = parser.parse_args()
    
    # Initialize recorder
    recorder = AudioRecorder(config_path=args.config)
    
    # Override config if specified
    if args.duration:
        recorder.duration = args.duration
    if args.device is not None:
        recorder.device_index = args.device
    
    try:
        if args.list_devices:
            # List devices
            recorder.list_devices()
        elif args.single:
            # Single recording
            recorder.record_single(args.output)
        else:
            # Continuous recording
            recorder.run_continuous(interval=args.interval)
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    main()
