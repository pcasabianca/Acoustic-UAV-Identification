"""Laptop recorder wrapper that exposes the same API as RaspberryRecorder.

Uses `audio_recorder.AudioRecorderLaptop` when available, otherwise falls
back to a synthetic generator for testing. Respects `deployment_config.json`.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any
import io
import numpy as np

try:
    from audio_recorder import AudioRecorderLaptop
    _HAS_PYAUDIO = True
except Exception:
    AudioRecorderLaptop = None
    _HAS_PYAUDIO = False


class LaptopRecorder:
    def __init__(self, config_path: str = 'deployment_config.json', use_ram: bool = True, logger=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        audio_conf = self.config.get('audio', {})
        self.sample_rate = int(audio_conf.get('sample_rate', 22050))
        self.duration = float(audio_conf.get('duration_seconds', 4.0))
        self.channels = int(audio_conf.get('channels', 1))
        self.use_ram = bool(use_ram)

        if _HAS_PYAUDIO:
            # AudioRecorderLaptop may accept a logger in future; pass-through if available
            try:
                self.rec = AudioRecorderLaptop(config_path=config_path, use_ram=use_ram)
            except Exception:
                self.rec = None
        else:
            self.rec = None

    def record_audio(self) -> Optional[np.ndarray]:
        if self.rec is not None:
            return self.rec.record_audio()
        # synthetic fallback
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    def get_audio_bytes(self, audio: np.ndarray) -> bytes:
        if self.rec is not None:
            return self.rec.get_audio_bytes(audio)
        # generate WAV bytes via wav_utils
        from wav_utils import wav_bytes
        return wav_bytes(audio, int(self.sample_rate), channels=self.channels)

    def save_audio(self, audio: np.ndarray, filename: str) -> str:
        # Save to recordings dir
        rec_conf = self.config.get('recording', {})
        out_dir = Path(rec_conf.get('recordings_directory', './recordings'))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        with open(path, 'wb') as f:
            f.write(self.get_audio_bytes(audio))
        return str(path)

    def cleanup(self):
        if self.rec is not None:
            try:
                self.rec.cleanup()
            except Exception:
                pass


if __name__ == '__main__':
    # quick smoke
    r = LaptopRecorder()
    a = r.record_audio()
    print('recorded', a.shape)
    b = r.get_audio_bytes(a)
    print('bytes len', len(b))