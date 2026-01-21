import numpy as np
import math

def _colored_noise(n, color, rng):
    # white noise then simple 1/f^alpha shaping for pink-like noise
    w = rng.normal(0.0, 1.0, size=n)
    if color == 'white' or abs(color) < 1e-9:
        return w
    # FFT-based shaping
    N = 1 << (int(np.ceil(np.log2(n))) + 1)
    W = np.fft.rfft(w, n=N)
    # use provided sample rate if available via RNG context; default to 44100
    # (caller will resample/scale as needed). We cannot access sr here,
    # so keep original behaviour but avoid hard-coded frequency vector errors by
    # deriving frequency spacing from sample count approximate 1/N.
    freqs = np.fft.rfftfreq(N, 1.0 / float(max(22050, 44100)))
    # shape ~ 1/f^(color)
    H = 1.0 / np.maximum(freqs ** float(color), 1e-6)
    H[0] = 0.0
    Y = W * H
    y = np.fft.irfft(Y, n=N)[:n]
    # normalize
    y = y / (np.sqrt((y**2).mean()) + 1e-12)
    return y

def apply(audio, sr, rng, meta, cfg):
    """Hardware noise: overlay broadband or colored ambient noise.

    params:
      - noise_type: 'white'|'pink' or numeric color exponent
      - snr_db: target SNR (signal relative to added noise)
      - level_db: absolute noise level relative to signal (instead of snr)
    """
    params = cfg.get('params', {}) if isinstance(cfg, dict) else {}
    noise_type = params.get('noise_type', 'pink')
    snr_db = params.get('snr_db', None)
    level_db = params.get('level_db', None)

    if audio is None:
        return audio, {'applied': False}

    x = audio.astype(float)
    n = x.size
    color = 1.0 if noise_type == 'pink' else 0.0
    try:
        color = float(noise_type) if isinstance(noise_type, (int, float, str)) and str(noise_type).replace('.','',1).isdigit() else color
    except Exception:
        pass

    noise = _colored_noise(n, color, rng)

    # scale noise
    sig_rms = math.sqrt(float((x**2).mean())) if n>0 else 0.0
    if snr_db is not None and sig_rms > 0:
        target_noise_rms = sig_rms / (10 ** (float(snr_db) / 20.0))
        cur = math.sqrt(float((noise**2).mean()))
        noise = noise * (target_noise_rms / (cur + 1e-12))
    elif level_db is not None and sig_rms > 0:
        # level_db relative to signal RMS
        target_noise_rms = sig_rms * (10 ** (float(level_db) / 20.0))
        cur = math.sqrt(float((noise**2).mean()))
        noise = noise * (target_noise_rms / (cur + 1e-12))
    else:
        # default low-level noise
        noise *= 0.001 * sig_rms if sig_rms>0 else noise

    out = x + noise
    debug = {'applied': True, 'noise_type': noise_type, 'snr_db': snr_db, 'level_db': level_db, 'noise_rms': float(math.sqrt(float((noise**2).mean())))}
    return out.astype(float), debug
