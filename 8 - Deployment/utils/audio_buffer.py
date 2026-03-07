"""AudioBuffer helper: unified in-memory audio representation.

Provides construction from WAV bytes, from file, or from numpy arrays,
and utilities to convert to WAV bytes/BytesIO, save to disk, and inspect
metadata (samplerate, channels, nframes, size_bytes). Honors
`deployment_config.json` audio `sample_rate` and `duration_seconds` when
normalizing/padding/trimming arrays if `apply_config_duration=True`.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import io
import json

import numpy as np

from .wav_utils import wav_bytes, read_wav


class AudioBuffer:
    """In-memory audio buffer abstraction.

    Internally stores audio as float32 numpy array in [-1, 1] and samplerate.
    """

    def __init__(self, array: np.ndarray, samplerate: int, channels: int = 1):
        self.array = np.asarray(array, dtype=np.float32)
        self.sr = int(samplerate)
        self.channels = int(channels)

    @classmethod
    def from_wav_file(cls, path: str, apply_config_duration: bool = True, config_path: str = 'deployment_config.json') -> 'AudioBuffer':
        samples, sr, channels = read_wav(path)
        buf = cls(samples, sr, channels)
        if apply_config_duration:
            buf._apply_config_duration(config_path)
        return buf

    @classmethod
    def from_wav_bytes(cls, data: bytes, apply_config_duration: bool = True, config_path: str = 'deployment_config.json') -> 'AudioBuffer':
        # Read from bytes using wave via BytesIO (reuse logic similar to read_wav)
        import wave
        buf = io.BytesIO(data)
        with wave.open(buf, 'rb') as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sampwidth != 2:
            raise NotImplementedError('Only 16-bit WAV supported')

        arr = np.frombuffer(raw, dtype=np.int16)
        if channels > 1:
            arr = arr.reshape((-1, channels))

        float_arr = arr.astype(np.float32) / 32768.0
        instance = cls(float_arr, sr, channels)
        if apply_config_duration:
            instance._apply_config_duration(config_path)
        return instance

    @classmethod
    def from_array(cls, array: np.ndarray, samplerate: int, channels: int = 1, apply_config_duration: bool = True, config_path: str = 'deployment_config.json') -> 'AudioBuffer':
        inst = cls(array, samplerate, channels)
        if apply_config_duration:
            inst._apply_config_duration(config_path)
        return inst

    def to_wav_bytes(self) -> bytes:
        return wav_bytes(self.array, self.sr, channels=self.channels)

    def to_bytesio(self) -> io.BytesIO:
        b = io.BytesIO(self.to_wav_bytes())
        b.seek(0)
        return b

    def save(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(self.to_wav_bytes())
        return path

    def nframes(self) -> int:
        return int(self.array.shape[0])

    def size_bytes(self) -> int:
        return len(self.to_wav_bytes())

    def info(self) -> Dict[str, Any]:
        return {'samplerate': int(self.sr), 'channels': int(self.channels), 'nframes': self.nframes(), 'bytes': self.size_bytes()}

    def _apply_config_duration(self, config_path: str = 'deployment_config.json') -> None:
        p = Path(config_path)
        if not p.exists():
            return
        try:
            with open(p, 'r') as f:
                cfg = json.load(f)
        except Exception:
            return

        target_sr = int(cfg.get('audio', {}).get('sample_rate', self.sr))
        duration = float(cfg.get('audio', {}).get('duration_seconds', self.nframes() / float(self.sr)))

        # Resample if samplerate differs (use simple librosa if available, else no-op)
        if target_sr != self.sr:
            try:
                import librosa
                self.array = librosa.resample(self.array.astype(np.float32), orig_sr=self.sr, target_sr=target_sr)
                self.sr = target_sr
            except Exception:
                # if librosa not available, skip resample
                pass

        expected = int(self.sr * duration)
        if self.array.shape[0] < expected:
            self.array = np.pad(self.array, (0, expected - self.array.shape[0]), mode='constant')
        elif self.array.shape[0] > expected:
            self.array = self.array[:expected]


if __name__ == '__main__':
    # Quick smoke tests
    import numpy as _np
    # synthetic tone 1s @ 24000
    sr = 24000
    t = _np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    a = 0.5 * _np.sin(2 * _np.pi * 440.0 * t).astype(_np.float32)
    buf = AudioBuffer.from_array(a, samplerate=sr)
    print('info:', buf.info())
    b = buf.to_wav_bytes()
    print('wav bytes len:', len(b))
    buf2 = AudioBuffer.from_wav_bytes(b)
    print('reloaded info:', buf2.info())