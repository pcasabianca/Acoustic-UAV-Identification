import numpy as np
import math

def _apply_jitter(x, sr, max_jitter_ms, rng):
    # small resampling jitter: apply random slight time scaling segments
    n = x.size
    if n == 0:
        return x
    max_jitter = max(0.0, float(max_jitter_ms)) / 1000.0
    # single global jitter factor
    factor = 1.0 + rng.uniform(-max_jitter, max_jitter)
    new_n = max(1, int(round(n * factor)))
    idx = np.linspace(0, n-1, new_n)
    y = np.interp(idx, np.arange(n), x)
    # resample back to n
    idx2 = np.linspace(0, new_n-1, n)
    y2 = np.interp(idx2, np.arange(new_n), y)
    return y2

def _dropout(x, sr, drop_prob, max_ms, rng):
    n = x.size
    if n == 0 or drop_prob <= 0.0:
        return x, []
    expected = drop_prob * (n / float(sr))
    n_drops = int(rng.poisson(expected)) if expected>0 else 0
    drops = []
    for _ in range(n_drops):
        start = int(rng.integers(0, n))
        length = int(min(n - start, max(1, int(round((max_ms/1000.0) * sr)))))
        x[start:start+length] = 0.0
        drops.append({'start': start, 'length': length})
    return x, drops

def _quantize_bits(x, bits):
    if bits is None or bits >= 32:
        return x
    peak = max(1e-9, float(np.max(np.abs(x))))
    y = x / peak
    levels = 2 ** bits
    yq = np.round((y + 1.0) * (levels/2 - 1)) / (levels/2 - 1) - 1.0
    return (yq * peak).astype(float)

def apply(audio, sr, rng, meta, cfg):
    """ADC artifacts: jitter, dropouts, quantization.

    params:
      - max_jitter_ms: float
      - dropout_prob_per_s: float
      - dropout_max_ms: int
      - bits: int
    """
    params = cfg.get('params', {}) if isinstance(cfg, dict) else {}
    max_jitter_ms = float(params.get('max_jitter_ms', 0.0))
    drop_prob = float(params.get('dropout_prob_per_s', 0.0))
    drop_max_ms = int(params.get('dropout_max_ms', 10))
    bits = params.get('bits', None)

    if audio is None:
        return audio, {'applied': False}

    x = audio.astype(float)
    debug = {'applied': True}
    try:
        if max_jitter_ms and max_jitter_ms > 0.0:
            x = _apply_jitter(x, sr, max_jitter_ms, rng)
            debug['jitter_ms'] = max_jitter_ms
    except Exception as e:
        debug['jitter_error'] = str(e)

    try:
        drops = []
        if drop_prob and drop_prob > 0.0:
            x, drops = _dropout(x, sr, drop_prob, drop_max_ms, rng)
            debug['dropouts'] = drops
    except Exception as e:
        debug['drop_error'] = str(e)

    try:
        if bits is not None:
            x = _quantize_bits(x, int(bits))
            debug['quant_bits'] = int(bits)
    except Exception as e:
        debug['quant_error'] = str(e)

    return x.astype(float), debug
