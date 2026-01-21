from typing import Tuple, Dict, Any
import numpy as np
import math


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # FFT convolution, returns full convolution
    na = a.size
    nb = b.size
    n = 1 << (int(np.ceil(np.log2(na + nb - 1))))
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    fc = fa * fb
    c = np.fft.irfft(fc, n)
    return c[: na + nb - 1]


def _synthesize_ir(rt60: float, sr: int, length_s: float = None) -> np.ndarray:
    # Create a simple exponentially decaying noise IR matching rt60
    if rt60 is None or rt60 <= 0:
        rt60 = 0.5
    if length_s is None:
        length_s = min(max(0.5, rt60 * 3.0), 5.0)
    L = int(length_s * sr)
    if L <= 0:
        L = int(sr * 0.5)
    t = np.arange(L) / float(sr)
    # exponential decay envelope
    # amplitude A(t) ~ exp(-6.91 * t / rt60) so at t=rt60 amplitude down by e^-6.91 ~= 0.001
    env = np.exp(-6.91 * t / float(rt60))
    noise = np.random.normal(0.0, 1.0, size=L)
    ir = noise * env
    # normalize energy
    e = np.sqrt(np.mean(ir ** 2))
    if e > 0:
        ir = ir / e
    return ir


def _estimate_rt60_schroeder(x: np.ndarray, sr: int) -> float | None:
    # Schroeder integration method
    if x is None or x.size == 0:
        return None
    e = np.cumsum(x[::-1] ** 2)[::-1]
    if e.max() <= 0:
        return None
    edb = 10.0 * np.log10(e / e.max())
    # find linear region between -5 dB and -35 dB (or as available)
    idx = np.where((edb <= -5.0) & (edb >= -35.0))[0]
    if idx.size < 2:
        idx = np.where((edb <= -5.0) & (edb >= -20.0))[0]
        if idx.size < 2:
            return None
    t = np.arange(edb.size) / float(sr)
    y = edb[idx]
    x_t = t[idx]
    # linear fit
    A = np.vstack([x_t, np.ones_like(x_t)]).T
    try:
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return None
    # slope m is dB per second; RT60 = -60 / m
    if m >= 0:
        return None
    rt60 = -60.0 / m
    return float(rt60)


def apply(audio: np.ndarray, sr: int, rng, meta: dict, cfg: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply reverb to a single audio array. Pure function, no I/O.

    meta may contain 'ir' (np.ndarray) or 'ir_id' to select IR.
    cfg should contain reverb parameters under key 'reverb' or scenario may pass params in meta.
    """
    debug: Dict[str, Any] = {}
    params = None
    if isinstance(meta, dict) and 'reverb' in meta:
        params = meta.get('reverb')
    if params is None and isinstance(cfg, dict):
        params = cfg.get('reverb', None)
    if params is None:
        return audio, debug

    mode = params.get('mode', 'post')
    rtype = params.get('type', 'synthetic')
    rt60_target = float(params.get('rt60_target', 0.5))
    wet = float(params.get('wet', 0.5))
    dry = float(params.get('dry', 0.5))
    pre_delay_ms = float(params.get('pre_delay_ms', 0.0))

    # choose or synthesize IR
    ir = None
    if isinstance(meta, dict) and meta.get('ir') is not None:
        ir = meta.get('ir')
    elif rtype == 'synthetic':
        # use RNG seed from provided rng if possible to deterministic IRs
        try:
            # create reproducible local RNG state
            seed_state = None
            if hasattr(rng, 'integers'):
                seed_state = int(rng.integers(0, 2 ** 31 - 1))
            else:
                seed_state = int(np.floor(abs(np.random.RandomState().rand() * 1e9)))
            _rng = np.random.RandomState(seed_state)
            # synthesize using local RNG
            # temporarily set global rng for synthesis
            old_rand = np.random.get_state()
            np.random.set_state(_rng.get_state())
            ir = _synthesize_ir(rt60_target, sr)
            np.random.set_state(old_rand)
        except Exception:
            ir = _synthesize_ir(rt60_target, sr)
    else:
        # no IR available
        ir = _synthesize_ir(rt60_target, sr)

    # apply pre-delay if requested
    if pre_delay_ms and pre_delay_ms > 0:
        pd = int(sr * (pre_delay_ms / 1000.0))
        if pd > 0:
            ir = np.concatenate([np.zeros(pd, dtype=float), ir])

    # convolution
    try:
        conv = _fft_convolve(audio.astype(float), ir.astype(float))
        conv = conv[: audio.size]
    except Exception:
        # fallback to naive convolution
        conv = np.convolve(audio.astype(float), ir.astype(float))[: audio.size]

    out = dry * audio.astype(float) + wet * conv

    # diagnostics
    rt60_est = _estimate_rt60_schroeder(out, sr)
    # tail energy ratio: energy in last 25% vs total
    L = out.size
    tail = out[int(L * 0.75) :]
    total_e = float(np.mean(out ** 2)) if out.size else 0.0
    tail_e = float(np.mean(tail ** 2)) if tail.size else 0.0
    tail_energy_ratio = (tail_e / total_e) if total_e > 0 else None

    debug.update({
        'reverb_applied': True,
        'mode': mode,
        'rt60_target': float(rt60_target),
        'rt60_estimated': (float(rt60_est) if rt60_est is not None else None),
        'wet': wet,
        'dry': dry,
        'tail_energy_ratio': (float(tail_energy_ratio) if tail_energy_ratio is not None else None),
    })
    return out, debug
