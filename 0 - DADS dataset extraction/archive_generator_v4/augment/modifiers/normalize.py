from typing import Tuple, Dict, Any
import numpy as np


def apply(audio: np.ndarray, sr: int, rng, meta: dict, cfg: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize audio peak to `max_amplitude` in cfg (default 0.95)."""
    debug = {}
    if audio is None:
        return audio, debug

    max_amp = 0.95
    if isinstance(cfg, dict):
        max_amp = float(cfg.get('max_amplitude', max_amp))

    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak <= 0:
        return audio, debug

    if peak > max_amp:
        gain = max_amp / peak
        audio = audio * gain
        debug['normalized_peak_before'] = float(peak)
        debug['normalized_gain'] = float(gain)
    else:
        debug['normalized_peak_before'] = float(peak)
        debug['normalized_gain'] = 1.0

    return audio, debug
