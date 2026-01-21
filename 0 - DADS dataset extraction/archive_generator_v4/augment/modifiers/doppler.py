import numpy as np

def apply(audio, sr, rng, meta, cfg):
    """Simple Doppler modifier: applies time-warping corresponding to constant radial velocity.

    params:
      - v_m_s: velocity (positive toward observer increases pitch)
      - dir: 1 or -1 (direction sign)
      - c: speed of sound (default 343)
    This implementation performs a time-warp (resample with time mapping) and
    optional amplitude scaling to approximate distance change.
    """
    params = cfg.get('params', {}) if isinstance(cfg, dict) else {}
    v = float(params.get('v_m_s', 0.0))
    direction = int(params.get('dir', 1))
    c = float(params.get('c', 343.0))

    if audio is None or abs(v) < 1e-6:
        return audio, {'applied': False}

    # observed frequency scaling factor
    factor = c / (c - direction * v)

    # compute time-warp mapping: output time t_out -> input time t_in = t_out / factor
    n = len(audio)
    t_in = np.arange(n) / float(sr)
    t_src = t_in / factor
    # map back to sample indices
    idx = t_src * sr
    idx_clipped = np.clip(idx, 0, n-1)
    out = np.interp(idx_clipped, np.arange(n), audio).astype(float)

    # simple amplitude adjustment approx (inverse distance law for constant closing speed)
    # not physically exact but gives perceptual change
    amp = 1.0
    try:
        d0 = float(params.get('d0', 10.0))
        # assume ending distance changes by v * duration
        duration = n / float(sr)
        d1 = max(0.1, d0 - direction * v * duration)
        amp = (d0 / d1) if d1 > 0 else 1.0
    except Exception:
        amp = 1.0
    out *= float(amp)
    return out, {'applied': True, 'v_m_s': v, 'dir': direction, 'factor': factor, 'amp': amp}
