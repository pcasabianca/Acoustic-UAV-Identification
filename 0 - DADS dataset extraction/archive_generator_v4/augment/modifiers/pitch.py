import numpy as np
from .time_stretch import apply as time_stretch_apply

def _resample_linear(x, factor):
    if factor == 1.0:
        return x
    n = len(x)
    new_n = max(1, int(round(n * factor)))
    if n == 0:
        return x
    old_idx = np.arange(n)
    new_idx = np.linspace(0, n-1, new_n)
    return np.interp(new_idx, old_idx, x).astype(float)

def apply(audio, sr, rng, meta, cfg):
    """Pitch shift by semitones while preserving duration (using resample + phase-vocoder).

    cfg['params']:
      - semitones: float or [min,max]
    """
    params = cfg.get('params', {}) if isinstance(cfg, dict) else {}
    sem = params.get('semitones', 0.0)
    if isinstance(sem, (list, tuple)) and len(sem) == 2:
        smin, smax = float(sem[0]), float(sem[1])
        sem = float(rng.uniform(smin, smax))
    else:
        sem = float(sem)

    if audio is None or abs(sem) < 1e-6:
        return audio, {'applied': False}

    # pitch shift factor
    factor = 2.0 ** (sem / 12.0)
    # resample to change pitch
    res = _resample_linear(audio, factor)
    # time-stretch back to original duration
    rate = 1.0 / factor
    ts_cfg = {'params': {'rate': rate}}
    stretched, dbg = time_stretch_apply(res, sr, rng, meta, ts_cfg)
    return stretched, {'applied': True, 'semitones': sem, 'factor': factor, 'rms_before': float(np.sqrt((audio**2).mean())), 'rms_after': float(np.sqrt((stretched**2).mean())), 'time_stretch_dbg': dbg}
