"""Simple feature extraction utilities mirroring `core.feature_extractor`.

Provides a lightweight `extract_mel` function that respects parameters
in `deployment_config.json` and operates on either a file path or a
numpy audio array. It returns a log-scaled MEL spectrogram (dB).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import json
import warnings

import numpy as np
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    _LIBROSA_AVAILABLE = False


def _load_config(config_path: str = 'deployment_config.json') -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')
    with p.open('r') as f:
        return json.load(f)


def extract_mel(audio_path: Optional[str] = None,
                audio_array: Optional[np.ndarray] = None,
                sr: Optional[int] = None,
                config_path: str = 'deployment_config.json',
                mel_overrides: Optional[dict] = None) -> np.ndarray:
    """Extract MEL spectrogram (dB) from `audio_path` or `audio_array`.

    Args:
        audio_path: optional path to WAV file (takes priority)
        audio_array: numpy array of audio samples (float32 in [-1,1])
        sr: sample rate for the provided array (ignored when loading path)
        config_path: path to deployment_config.json

    Returns:
        MEL spectrogram in dB (numpy.ndarray)
    """
    cfg = _load_config(config_path)
    audio_conf = cfg.get('audio', {})
    feat_conf = cfg.get('feature_extraction', {})
    mel_conf = feat_conf.get('mel_spectrogram', {})
    if mel_overrides:
        # apply overrides explicitly provided (e.g., n_mels)
        mel_conf = {**mel_conf, **mel_overrides}

    target_sr = int(audio_conf.get('sample_rate', 22050))
    duration = float(audio_conf.get('duration_seconds', 4.0))

    if not _LIBROSA_AVAILABLE:
        raise RuntimeError('librosa is required for extract_mel. Install with: pip install librosa')

    try:
        if audio_path is not None:
            # Load from file to the configured sample rate and duration
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                y, _ = librosa.load(str(audio_path), sr=target_sr, duration=duration)
        elif audio_array is not None:
            y = np.asarray(audio_array)
            if sr is None:
                sr = target_sr
        else:
            raise ValueError('Either audio_path or audio_array must be provided')

        # If we used audio_path, sr already equals target_sr; otherwise use provided sr
        if audio_path is None:
            sr = int(sr or target_sr)

        # Ensure exact duration: pad or trim
        expected = int(sr * duration)
        if y.shape[0] < expected:
            y = np.pad(y, (0, expected - y.shape[0]), mode='constant')
        elif y.shape[0] > expected:
            y = y[:expected]

        # Optional normalization
        if feat_conf.get('normalization', False):
            target_peak = float(audio_conf.get('normalize_peak', 0.95))
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak > 0 and target_peak > 0:
                y = y * (target_peak / peak)

        # Compute mel spectrogram
        n_mels = int(mel_conf.get('n_mels', 44))
        n_fft = int(mel_conf.get('n_fft', 2048))
        hop_length = int(mel_conf.get('hop_length', 512))
        fmin = float(mel_conf.get('fmin', 20.0))
        fmax = float(mel_conf.get('fmax', sr // 2))

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                           n_fft=n_fft, hop_length=hop_length,
                                           fmin=fmin, fmax=fmax)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db

    except Exception:
        raise


def pad_crop_2d(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Pad or center-crop 2D array to shape (H, W)."""
    h, w = arr.shape
    fill = float(arr.min()) if arr.size else 0.0
    # Height
    if h < H:
        pad = H - h
        arr = np.pad(arr, ((pad // 2, pad - pad // 2), (0, 0)), constant_values=fill)
    elif h > H:
        arr = arr[(h - H) // 2:(h - H) // 2 + H, :]
    # Width
    h, w = arr.shape
    if w < W:
        pad = W - w
        arr = np.pad(arr, ((0, 0), (pad // 2, pad - pad // 2)), constant_values=fill)
    elif w > W:
        arr = arr[:, (w - W) // 2:(w - W) // 2 + W]
    return arr


class FeatureExtractor:
    """Compatibility FeatureExtractor (inspired by core.feature_extractor.FeatureExtractor).

    Provides `extract_mel` and `adapt_to_model(features, model)` so other
    modules (predictor_utils, orchestrators) can rely on the same behavior
    as the legacy `core` implementation.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        self.config = config or _load_config()
        self.logger = logger

    def extract_mel(self, audio_path: Optional[str] = None, audio_array: Optional[np.ndarray] = None,
                    sr: Optional[int] = None, mel_overrides: Optional[dict] = None) -> np.ndarray:
        return extract_mel(audio_path=audio_path, audio_array=audio_array, sr=sr,
                           config_path=None if self.config is None else 'deployment_config.json',
                           mel_overrides=mel_overrides)

    def adapt_to_model(self, features: np.ndarray, model) -> np.ndarray:
        """Adapt MEL features to the model's expected input shape.

        Mirrors the logic from `core.feature_extractor.FeatureExtractor.adapt_to_model`.
        """
        mel = np.squeeze(features) if features.ndim != 2 else features
        # model.input_shape may be a tuple or a list; normalize
        input_shape = None
        try:
            if isinstance(model.input_shape, list):
                input_shape = model.input_shape[0]
            else:
                input_shape = model.input_shape
        except Exception:
            input_shape = None

        rank = len(input_shape) if input_shape else 4

        if rank == 4:
            _, H, W, _ = input_shape
            H = int(H) if H else mel.shape[0]
            W = int(W) if W else mel.shape[1]
            mel_adapted = pad_crop_2d(mel, H, W)
            return mel_adapted[np.newaxis, :, :, np.newaxis].astype(np.float32, copy=False)
        elif rank == 3:
            _, H, W = input_shape
            H = int(H) if H else mel.shape[0]
            W = int(W) if W else mel.shape[1]
            mel_adapted = pad_crop_2d(mel, H, W)
            return mel_adapted[np.newaxis, :, :].astype(np.float32, copy=False)

        # Fallback: flatten to vector
        return mel.flatten()[np.newaxis, :].astype(np.float32, copy=False)


if __name__ == '__main__':
    # Quick smoke test
    import numpy as np
    sr = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    mel = extract_mel(audio_array=audio, sr=sr)
    print('mel shape:', mel.shape)