import numpy as np

def _design_freq_response(n, sr, f_low, f_high, kind='low'):
    # frequency bins
    freqs = np.fft.rfftfreq(n, 1.0/sr)
    H = np.ones_like(freqs)
    if kind == 'low':
        H[freqs > f_low] = 0.0
        # smooth transition
        bw = max(1.0, 0.05 * f_low)
        mask = (freqs > f_low - bw) & (freqs <= f_low + bw)
        H[mask] = 0.5 * (1 + np.cos(np.pi * (freqs[mask] - (f_low - bw)) / (2*bw)))
    elif kind == 'high':
        H[freqs < f_high] = 0.0
        bw = max(1.0, 0.05 * f_high)
        mask = (freqs >= f_high - bw) & (freqs < f_high + bw)
        H[mask] = 0.5 * (1 - np.cos(np.pi * (freqs[mask] - (f_high - bw)) / (2*bw)))
    elif kind == 'band':
        H[:] = 0.0
        bw_low = max(1.0, 0.05 * f_low)
        bw_high = max(1.0, 0.05 * f_high)
        mask = (freqs >= max(0, f_low - bw_low)) & (freqs <= f_high + bw_high)
        # windowed passband
        H[mask] = 1.0
    return H

def _apply_freq_filter(x, sr, kind, cutoff1, cutoff2=None):
    n = len(x)
    N = 1 << (int(np.ceil(np.log2(n))) + 1)
    X = np.fft.rfft(x, n=N)
    if kind == 'band':
        H = _design_freq_response(N, sr, cutoff1, cutoff2, kind='band')
    else:
        H = _design_freq_response(N, sr, cutoff1, cutoff2, kind=kind)
    Y = X * H
    y = np.fft.irfft(Y, n=N)[:n]
    return y.astype(float)

def apply(audio, sr, rng, meta, cfg):
    """Filters modifier: supports 'low', 'high', 'band' via frequency-domain multiplication.

    params:
      - type: 'low'|'high'|'band'
      - cutoff: number or [min,max]
      - cutoff2: for band upper cutoff
    """
    params = cfg.get('params', {}) if isinstance(cfg, dict) else {}
    ftype = params.get('type', 'low')
    cutoff = params.get('cutoff', 3000.0)
    if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
        cutoff = float(rng.uniform(float(cutoff[0]), float(cutoff[1])))
    else:
        cutoff = float(cutoff)

    cutoff2 = params.get('cutoff2', None)
    if cutoff2 is not None and isinstance(cutoff2, (list, tuple)) and len(cutoff2) == 2:
        cutoff2 = float(rng.uniform(float(cutoff2[0]), float(cutoff2[1])))
    elif cutoff2 is not None:
        cutoff2 = float(cutoff2)

    if audio is None:
        return audio, {'applied': False}

    try:
        out = _apply_freq_filter(audio, sr, 'low' if ftype == 'low' else ('high' if ftype == 'high' else 'band'), cutoff, cutoff2)
        return out, {'applied': True, 'type': ftype, 'cutoff': cutoff, 'cutoff2': cutoff2}
    except Exception as e:
        return audio, {'applied': False, 'error': str(e)}
