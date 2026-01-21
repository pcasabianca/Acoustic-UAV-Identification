import numpy as np

def _lowpass_freqshape(n, sr, cutoff):
    N = 1 << (int(np.ceil(np.log2(n))) + 1)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    H = 1.0 / (1.0 + (freqs / float(cutoff))**4)
    return H

def apply(audio, sr, rng, meta, cfg):
    """Simulate occlusion by applying a low-pass + attenuation envelope.

    params:
      - cutoff_hz: number or [min,max]
      - attenuation_db: number or [min,max]
      - event_prob: probability that occlusion occurs
      - event_ms: duration in ms
    """
    params = cfg.get('params', {}) if isinstance(cfg, dict) else {}
    cutoff = params.get('cutoff_hz', 3000.0)
    att_db = params.get('attenuation_db', -6.0)
    event_prob = float(params.get('event_prob', 1.0))
    event_ms = int(params.get('event_ms', 500))

    if audio is None:
        return audio, {'applied': False}

    x = audio.astype(float)
    n = x.size
    # decide event
    if hasattr(rng, 'random'):
        r = float(rng.random())
    else:
        r = np.random.rand()
    if r > event_prob:
        return x, {'applied': False}

    # choose cutoff and attenuation
    if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
        cutoff = float(rng.uniform(float(cutoff[0]), float(cutoff[1])))
    else:
        cutoff = float(cutoff)
    if isinstance(att_db, (list, tuple)) and len(att_db) == 2:
        att_db = float(rng.uniform(float(att_db[0]), float(att_db[1])))
    else:
        att_db = float(att_db)

    # create lowpass shape and apply in freq domain
    N = 1 << (int(np.ceil(np.log2(n))) + 1)
    X = np.fft.rfft(x, n=N)
    H = _lowpass_freqshape(n, sr, cutoff)
    Y = X * H
    y = np.fft.irfft(Y, n=N)[:n]

    # attenuation envelope (ms -> samples)
    L = n
    dur = int(min(n, max(1, int(round((event_ms/1000.0) * sr)))))
    start = int(rng.integers(0, max(1, n - dur))) if n > dur else 0
    env = np.ones(n, dtype=float)
    fade = np.linspace(1.0, 10 ** (att_db/20.0), dur)
    env[start:start+dur] = fade
    out = x * env + y * (1.0 - env)

    debug = {'applied': True, 'cutoff_hz': cutoff, 'atten_db': att_db, 'start_sample': int(start), 'duration_samples': int(dur)}
    return out.astype(float), debug
