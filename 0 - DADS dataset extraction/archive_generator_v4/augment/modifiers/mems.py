import numpy as np
import math

def _rfft_eq(n, sr, tilt_db_per_octave=0.0, hf_rolloff_hz=None):
    freqs = np.fft.rfftfreq(n, 1.0/sr)
    H = np.ones_like(freqs)
    # spectral tilt: +db per octave
    if abs(tilt_db_per_octave) > 1e-9:
        # compute octaves relative to 1 kHz
        ref = 1000.0
        # avoid extreme amplification from near-zero freqs by clamping
        freqs_clamped = np.maximum(freqs, ref * (2 ** -8))
        octaves = np.log2(freqs_clamped / ref)
        # clamp octave range to reasonable bounds to avoid huge gains
        octaves = np.clip(octaves, -8.0, 8.0)
        H *= 10 ** ((tilt_db_per_octave * octaves) / 20.0)
    # HF rolloff (simple first-order)
    if hf_rolloff_hz is not None:
        roll = 1.0 / (1.0 + (freqs / float(hf_rolloff_hz))**2)
        H *= roll
    return H

def _apply_eq(x, sr, tilt_db_per_octave=0.0, hf_rolloff_hz=None):
    n = len(x)
    if n == 0:
        return x
    N = 1 << (int(np.ceil(np.log2(n))) + 1)
    X = np.fft.rfft(x, n=N)
    H = _rfft_eq(N, sr, tilt_db_per_octave, hf_rolloff_hz)
    Y = X * H
    y = np.fft.irfft(Y, n=N)[:n]
    return y.astype(float)

def _rms(x):
    x = np.asarray(x, dtype=float)
    return math.sqrt(float((x**2).mean())) if x.size else 0.0

def _add_noise_for_snr(x, target_snr_db, rng):
    # add white Gaussian noise to achieve target SNR (signal power / noise power)
    sig_rms = _rms(x)
    if sig_rms <= 0 or target_snr_db is None:
        return x, 0.0
    target_noise_rms = sig_rms / (10 ** (target_snr_db / 20.0))
    noise = rng.normal(0.0, 1.0, size=x.shape)
    # scale noise to target RMS
    cur_n_rms = math.sqrt(float((noise**2).mean()))
    if cur_n_rms <= 0:
        return x, 0.0
    noise = noise * (target_noise_rms / cur_n_rms)
    out = x + noise
    return out, float(target_noise_rms)

def _quantize(x, bits):
    if bits is None or bits >= 32:
        return x
    # assume x in [-1,1]
    q_levels = 2 ** bits
    y = np.clip(x, -1.0, 1.0)
    y = np.round((y + 1.0) * (q_levels/2 - 1)) / (q_levels/2 - 1) - 1.0
    return y.astype(float)

def apply(audio, sr, rng, meta, cfg):
    """MEMS microphone simulation modifier.

    cfg['params'] may contain:
      - tilt_db_per_octave: float (spectral tilt)
      - hf_rolloff_hz: float
      - snr_floor_db: float (if present, ensure noise floor at this SNR)
      - bit_depth: int (quantization)
      - adc_noise_db: float (additional ADC noise RMS in dB relative to signal)
      - glitch_rate: probability per second
      - glitch_max_ms: max glitch length in ms
    """
    params = cfg.get('params', {}) if isinstance(cfg, dict) else {}
    tilt = float(params.get('tilt_db_per_octave', 0.0))
    hf = params.get('hf_rolloff_hz', None)
    snr_floor = params.get('snr_floor_db', None)
    bits = params.get('bit_depth', None)
    adc_noise_db = params.get('adc_noise_db', None)
    glitch_rate = float(params.get('glitch_rate', 0.0))
    glitch_max_ms = int(params.get('glitch_max_ms', 10))

    if audio is None:
        return audio, {'applied': False}

    x = audio.astype(float)
    rms_before = _rms(x)

    debug = {'applied': True}

    # apply EQ simulating MEMS capsule response
    try:
        x = _apply_eq(x, sr, tilt_db_per_octave=tilt, hf_rolloff_hz=hf)
        debug['eq'] = {'tilt_db_per_octave': tilt, 'hf_rolloff_hz': hf}
    except Exception as e:
        debug['eq_error'] = str(e)

    # ensure noise floor
    try:
        if snr_floor is not None:
            # add noise so that signal/noise <= snr_floor
            # compute required noise RMS
            target_noise_rms = rms_before / (10 ** (float(snr_floor) / 20.0)) if rms_before > 0 else 1e-6
            noise = rng.normal(0.0, 1.0, size=x.shape)
            cur_n_rms = math.sqrt(float((noise**2).mean()))
            noise = noise * (target_noise_rms / (cur_n_rms + 1e-12))
            x = x + noise
            debug['snr_floor_db'] = float(snr_floor)
    except Exception as e:
        debug['snr_floor_error'] = str(e)

    # ADC noise addition
    try:
        if adc_noise_db is not None:
            # interpret as noise RMS in dB relative to full-scale? treat as relative to signal RMS
            sig_rms = _rms(x)
            noise_rms = sig_rms / (10 ** (float(adc_noise_db) / 20.0)) if sig_rms > 0 else 1e-6
            noise = rng.normal(0.0, 1.0, size=x.shape)
            cur_n_rms = math.sqrt(float((noise**2).mean()))
            noise = noise * (noise_rms / (cur_n_rms + 1e-12))
            x = x + noise
            debug['adc_noise_db'] = float(adc_noise_db)
    except Exception as e:
        debug['adc_noise_error'] = str(e)

    # quantization
    try:
        if bits is not None:
            # normalize to -1..1 by peak
            peak = max(1e-9, float(np.max(np.abs(x))))
            xq = x / peak
            xq = _quantize(xq, int(bits))
            x = xq * peak
            debug['bit_depth'] = int(bits)
    except Exception as e:
        debug['quant_error'] = str(e)

    # glitches
    try:
        if glitch_rate and glitch_rate > 0.0:
            duration_s = float(x.size) / float(sr)
            expected = glitch_rate * duration_s
            n_glitches = int(rng.poisson(expected)) if expected > 0 else 0
            glitches = []
            for _ in range(n_glitches):
                start = int(rng.integers(0, x.size))
                length = int(min(x.size - start, max(1, int(round((glitch_max_ms / 1000.0) * sr)))))
                x[start:start+length] = 0.0
                glitches.append({'start': start, 'length': length})
            debug['glitches'] = glitches
    except Exception as e:
        debug['glitch_error'] = str(e)

    rms_after = _rms(x)
    debug['rms_before'] = float(rms_before)
    debug['rms_after'] = float(rms_after)

    return x.astype(float), debug
