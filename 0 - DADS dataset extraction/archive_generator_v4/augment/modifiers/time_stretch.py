import numpy as np

def _stft(x, win_size=2048, hop=512, window=None):
    if window is None:
        window = np.hanning(win_size)
    pad = win_size - hop
    x = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + (len(x) - win_size) // hop
    frames = np.stack([x[i*hop:i*hop+win_size] * window for i in range(n_frames)])
    return np.fft.rfft(frames, axis=1)

def _istft(stft_matrix, win_size=2048, hop=512, window=None):
    if window is None:
        window = np.hanning(win_size)
    n_frames = stft_matrix.shape[0]
    out_len = n_frames * hop + win_size
    out = np.zeros(out_len)
    win_sum = np.zeros(out_len)
    frames = np.fft.irfft(stft_matrix, n=win_size, axis=1)
    for i in range(n_frames):
        start = i*hop
        out[start:start+win_size] += frames[i] * window
        win_sum[start:start+win_size] += window**2
    # avoid divide by zero
    small = win_sum < 1e-8
    win_sum[small] = 1.0
    out /= win_sum
    # remove padding
    return out

def apply(audio, sr, rng, meta, cfg):
    """Time-stretch modifier using a simple phase-vocoder.

    Params in cfg['params']:
      - rate: float or [min,max] range (rate <1 slower, >1 faster)
      - prob: probability (handled at pipeline level typically)
    """
    params = cfg.get('params', {}) if isinstance(cfg, dict) else {}
    rate = params.get('rate', 1.0)
    if isinstance(rate, (list, tuple)) and len(rate) == 2:
        rmin, rmax = float(rate[0]), float(rate[1])
        rate = float(rng.uniform(rmin, rmax))
    else:
        rate = float(rate)

    if rate == 1.0 or audio is None:
        return audio, {'applied': False}

    # analysis/synthesis hop sizes
    win = 2048
    hop_a = win // 4
    hop_s = max(1, int(round(hop_a * rate)))

    S = _stft(audio, win_size=win, hop=hop_a)
    # magnitude and phase
    mag = np.abs(S)
    phase = np.angle(S)

    # time-stretch via phase vocoder: create new time axis
    n_frames = S.shape[0]
    t_steps = np.arange(0, n_frames, rate)
    n_frames_s = len(t_steps)

    out_stft = np.zeros((n_frames_s, S.shape[1]), dtype=complex)
    # initial phase
    prev_phase = phase[0].copy()
    phase_adv = 2.0 * np.pi * (hop_a * np.arange(S.shape[1]) / float(win))
    for i, t in enumerate(t_steps):
        left = int(np.floor(t))
        right = min(left + 1, n_frames - 1)
        frac = t - left
        mag_frame = (1-frac) * mag[left] + frac * mag[right]
        inst_phase = (1-frac) * phase[left] + frac * phase[right]
        # phase propagation
        delta = inst_phase - prev_phase
        delta = delta - np.round(delta / (2*np.pi)) * 2*np.pi
        true_freq = phase_adv + delta / hop_a
        prev_phase = prev_phase + hop_s * true_freq
        out_stft[i] = mag_frame * np.exp(1j * prev_phase)

    out = _istft(out_stft, win_size=win, hop=hop_s)
    return out.astype(float), {'applied': True, 'rate': rate, 'rms_before': float(np.sqrt((audio**2).mean())) if audio is not None else None, 'rms_after': float(np.sqrt((out**2).mean()))}
