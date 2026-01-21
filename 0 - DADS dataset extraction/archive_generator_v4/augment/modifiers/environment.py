import numpy as np
import math

def _atten_gain_from_dist(d, ref=1.0, alpha=1.0):
    # avoid log domain errors
    d = max(d, 1e-6)
    att_db = -20.0 * alpha * math.log10(max(d / ref, 1e-6))
    return float(10 ** (att_db / 20.0)), float(att_db)


def _cos_fade(n, fade_in, fade_out):
    env = np.ones(n, dtype=float)
    if fade_in > 0:
        t = np.linspace(0.0, 1.0, min(n, fade_in))
        env[:fade_in] = 0.5 - 0.5 * np.cos(np.pi * t)
    if fade_out > 0:
        t = np.linspace(0.0, 1.0, min(n, fade_out))
        env[-fade_out:] = 0.5 + 0.5 * np.cos(np.pi * t)
    return env


def apply(audio, sr, rng, meta, env_cfg):
    """Environment wrapper modifier (Phase 1 simple implementation).

    env_cfg (dict) may contain:
      - trajectory: {start_m, end_m, duration_s}
      - appear: {fade_in_ms, fade_out_ms}
      - modifiers: [ {name: 'doppler', params: {...}}, ... ]
      - distance: {ref_m, alpha}

    Returns (audio_out, env_debug).
    If env_cfg is falsy, returns (audio, None).
    """
    if not env_cfg or audio is None:
        return audio, None

    n = int(audio.size)
    duration = float(n) / float(sr) if sr > 0 else 0.0

    # trajectory
    traj = env_cfg.get('trajectory', {}) if isinstance(env_cfg, dict) else {}
    start_m = float(traj.get('start_m', traj.get('distance_m', 10.0)))
    end_m = float(traj.get('end_m', start_m))
    dur_t = float(traj.get('duration_s', duration)) if traj else duration

    # distance params
    dist_cfg = env_cfg.get('distance', {}) if isinstance(env_cfg, dict) else {}
    ref_m = float(dist_cfg.get('ref_m', 1.0))
    alpha = float(dist_cfg.get('alpha', 1.0))

    # build per-sample distance (linear interpolation over requested duration)
    t = np.arange(n) / float(sr) if n > 0 else np.array([])
    if dur_t <= 0.0:
        frac = np.zeros_like(t)
    else:
        frac = np.clip(t / dur_t, 0.0, 1.0)
    d = start_m + frac * (end_m - start_m)

    # compute gain envelope from distance
    gains = np.ones(n, dtype=float)
    try:
        for i in range(n):
            g, _ = _atten_gain_from_dist(float(d[i]), ref=ref_m, alpha=alpha)
            gains[i] = g
    except Exception:
        gains = np.ones(n, dtype=float)

    # appearance envelope
    appear = env_cfg.get('appear', {}) if isinstance(env_cfg, dict) else {}
    fade_in_ms = int(appear.get('fade_in_ms', 0))
    fade_out_ms = int(appear.get('fade_out_ms', 0))
    fade_in = int(min(n, max(0, int(round(fade_in_ms * sr / 1000.0)))))
    fade_out = int(min(n, max(0, int(round(fade_out_ms * sr / 1000.0)))))
    env = _cos_fade(n, fade_in, fade_out)

    out = audio.astype(float) * gains * env

    # apply listed modifiers (name & optional params)
    mods_dbg = []
    mods = env_cfg.get('modifiers', []) if isinstance(env_cfg, dict) else []
    for mi, m in enumerate(mods):
        try:
            name = m.get('name') if isinstance(m, dict) else str(m)
            params = m.get('params', {}) if isinstance(m, dict) else {}
            # dynamic import from augment.modifiers
            mod = __import__('augment.modifiers.' + name, fromlist=['*'])
            seed = int(rng.integers(0, 2**31 - 1)) if hasattr(rng, 'integers') else int(rng.random() * 1e9)
            mod_rng = np.random.default_rng(seed)
            # if the modifier params contain a top-level 'chain' (e.g. hardware_wrapper),
            # pass them as the modifier cfg directly. Otherwise wrap under {'params': ...}
            if isinstance(params, dict) and 'chain' in params:
                mod_cfg = params
            else:
                mod_cfg = {'params': params}
            out, dbg = mod.apply(out, sr, mod_rng, meta, mod_cfg)
            mods_dbg.append({'name': name, 'applied': True, 'debug': dbg})
        except Exception as e:
            mods_dbg.append({'name': name if 'name' in locals() else str(m), 'applied': False, 'error': str(e)})

    # collect debug
    try:
        rms_before = float(np.sqrt((audio.astype(float)**2).mean())) if audio is not None else None
        rms_after = float(np.sqrt((out.astype(float)**2).mean())) if out is not None else None
    except Exception:
        rms_before = None
        rms_after = None

    env_debug = {
        'applied': True,
        'trajectory': {'start_m': start_m, 'end_m': end_m, 'duration_s': dur_t},
        'distance': {'ref_m': ref_m, 'alpha': alpha},
        'appear': {'fade_in_ms': fade_in_ms, 'fade_out_ms': fade_out_ms},
        'modifiers': mods_dbg,
        'rms_before': rms_before,
        'rms_after': rms_after,
    }

    return out.astype(float), env_debug
