from typing import Tuple, Dict, Any
import math
import numpy as np


def _rms(x: np.ndarray) -> float:
    if x is None or x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(float) ** 2)))


def apply(audio: np.ndarray, sr: int, rng, meta: dict, cfg: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mix the provided `audio` with additional sources supplied via `meta['mix_sources']`.

    Contract:
    - `audio` and each source in `meta['mix_sources']` are numpy arrays (floats), already normalized.
    - `meta['mix_sources']` is a list of dicts: {'audio': np.ndarray, 'gain': float} where `gain` is a linear multiplier.
    - This modifier does NOT perform any file I/O.
    """
    debug: Dict[str, Any] = {}

    mix_list = meta.get('mix_sources') if isinstance(meta, dict) else None
    if not mix_list:
        return audio, debug

    # determine target length
    lengths = [m['audio'].size for m in mix_list if m.get('audio') is not None]
    if audio is not None:
        lengths.append(audio.size)
    if not lengths:
        return audio, debug
    target_len = int(max(lengths))

    def _ensure_len(arr: np.ndarray):
        if arr is None:
            return np.zeros(target_len, dtype=float)
        a = arr.astype(float)
        if a.size == target_len:
            return a
        if a.size > target_len:
            return a[:target_len]
        # pad with zeros
        pad = np.zeros(target_len - a.size, dtype=float)
        return np.concatenate([a, pad])

    base = _ensure_len(audio) if audio is not None else np.zeros(target_len, dtype=float)
    rms_before = _rms(base)

    total_sources = 0
    applied = []

    # If target SNR is specified relative to `audio` (drone signal), adjust ambient gains
    target_snr_db = None
    if isinstance(meta, dict):
        target_snr_db = meta.get('target_snr_db')

    # Prepare per-source arrays and initial gains
    src_entries = []
    for m in mix_list:
        src = m.get('audio')
        gain = float(m.get('gain', 1.0))
        distance_m = m.get('distance_m', None)
        if src is None:
            continue
        s = _ensure_len(src)
        src_entries.append({'audio': s, 'gain': gain, 'rms': _rms(s), 'distance_m': (float(distance_m) if distance_m is not None else None)})

    # If a target SNR is requested and base signal exists, scale ambient gains so that
    # total_noise_power = signal_power / (10^(SNR/10)). Distribute scaling proportionally.
    if target_snr_db is not None and audio is not None and audio.size > 0 and src_entries:
        sig_power = float(np.mean(base ** 2))
        desired_noise_power = sig_power / (10.0 ** (float(target_snr_db) / 10.0)) if sig_power > 0 else 0.0
        # current total noise power (with initial gains)
        current_noise_power = 0.0
        for e in src_entries:
            current_noise_power += (e['gain'] ** 2) * float(np.mean(e['audio'] ** 2))
        if current_noise_power > 0 and desired_noise_power > 0:
            scale = (desired_noise_power / current_noise_power) ** 0.5
        else:
            scale = 0.0
        for e in src_entries:
            e['gain'] = float(e['gain'] * scale)
        debug['target_snr_db'] = float(target_snr_db)
        debug['snr_scale'] = float(scale)

    out = base.copy()
    for e in src_entries:
        out += e['audio'] * float(e['gain'])
        total_sources += 1
        eff_gain = float(e['gain'])
        eff_atten_dB = None
        if eff_gain > 0:
            eff_atten_dB = 20.0 * math.log10(eff_gain)
        applied.append({
            'gain': eff_gain,
            'src_len': int(e['audio'].size),
            'src_rms': float(e['rms']),
            'requested_distance_m': e.get('distance_m'),
            'effective_atten_dB': (float(eff_atten_dB) if eff_atten_dB is not None else None),
            'used_for_gain': (True if target_snr_db is None else False),
        })

    rms_after = _rms(out)
    # compute resulting SNR between base (signal) and noise (out - base)
    resulting_snr_db = None
    try:
        sig_power = float(np.mean(base ** 2)) if base is not None else 0.0
        noise = out - base if base is not None else out
        noise_power = float(np.mean(noise ** 2)) if noise is not None else 0.0
        if sig_power > 0 and noise_power >= 0:
            # avoid division by zero
            if noise_power == 0:
                resulting_snr_db = float('inf')
            else:
                resulting_snr_db = 10.0 * math.log10(sig_power / noise_power)
    except Exception:
        resulting_snr_db = None

    debug.update({
        'mixed_sources': total_sources,
        'mix_details': applied,
        'rms_before': rms_before,
        'rms_after': rms_after,
        'resulting_snr_db': (float(resulting_snr_db) if resulting_snr_db is not None and resulting_snr_db != float('inf') else ("inf" if resulting_snr_db == float('inf') else None)),
    })
    return out, debug
