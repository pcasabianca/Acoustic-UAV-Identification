from typing import Tuple, Dict, Any
import numpy as np
from pathlib import Path

from .modifiers.attenuate import apply as attenuate_apply
from pathlib import Path as _Path
import sys as _sys
# ensure project root on sys.path for importing tools
_root = _Path(__file__).resolve().parents[3]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
from tools.audio_utils import load_audio_file, ensure_duration


def apply(audio: np.ndarray, sr: int, rng, meta: dict, cfg: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Wrapper that samples a distance, computes attenuation, mixes ambient noise deterministically.

    cfg expected keys under 'distance' block; fallback defaults used.
    """
    debug = {}
    if audio is None:
        return audio, debug

    # distance sampling: allow per-call override via meta['requested_distance_m'] or meta['distance']
    dist_cfg = cfg.get('distance', {}) if isinstance(cfg, dict) else {}
    if isinstance(meta, dict) and isinstance(meta.get('distance'), dict):
        # scenario-level distance block supplied via meta
        dist_cfg = meta.get('distance')

    # if caller requested an exact distance, honour it
    dist = None
    try:
        if isinstance(meta, dict) and meta.get('requested_distance_m') is not None:
            dist = float(meta.get('requested_distance_m'))
    except Exception:
        dist = None

    if dist is None:
        dmin, dmax = 1.0, 30.0
        if 'range_m' in dist_cfg:
            try:
                dmin, dmax = float(dist_cfg['range_m'][0]), float(dist_cfg['range_m'][1])
            except Exception:
                pass

        mode = dist_cfg.get('mode', 'linear')
        if mode == 'log':
            import math
            ldmin, ldmax = math.log(dmin), math.log(dmax)
            dist = float(np.exp(rng.uniform(ldmin, ldmax)))
        else:
            dist = float(rng.uniform(dmin, dmax))

    debug['distance_m'] = float(dist)

    # If caller specified a desired drone source level (SPL at 1m), apply source-level gain
    try:
        # Interpret `drone_spl_db` as a delta gain in dB (relative gain), not an absolute SPL.
        if isinstance(meta, dict) and 'drone_spl_db' in meta and audio is not None:
            gain_db = float(meta.get('drone_spl_db'))
            gain_linear = 10.0 ** (gain_db / 20.0)
            # clamp excessive gains to avoid numerical explosions in uncalibrated digital domain
            max_gain_linear = float(cfg.get('max_gain_linear', 50.0)) if isinstance(cfg, dict) else 50.0
            if gain_linear > max_gain_linear:
                gain_linear = max_gain_linear
                # record that clamp occurred (store applied db)
                applied_db = 20.0 * math.log10(gain_linear) if gain_linear > 0 else float('-inf')
                debug['drone_spl_db_clamped_to_db'] = float(applied_db)
            audio = audio * gain_linear
            # keep debug keys for compatibility but store the sampled and applied relative gains
            debug['drone_spl_db_set'] = float(gain_db)
            debug['drone_source_gain_db'] = float(20.0 * math.log10(gain_linear) if gain_linear>0 else float('-inf'))
    except Exception:
        pass

    # attenuation (dB): att_dB = -20 * alpha * log10(dist/ref)
    ref = float(dist_cfg.get('ref_m', 1.0))
    alpha = float(dist_cfg.get('alpha', 1.8))
    import math
    att_dB = -20.0 * alpha * math.log10(max(dist / ref, 1e-6))
    debug['atten_dB'] = float(att_dB)

    # apply attenuation via attenuate modifier
    audio, att_meta = attenuate_apply(audio, sr, rng, meta, {'atten_dB': att_dB})
    debug.update(att_meta)

    # mix ambient if configured: load files here (wrapper) and delegate actual array mixing to modifiers/mix.py
    ambient_cfg = dist_cfg.get('noise', {})
    ambient_dir = ambient_cfg.get('sources_dir') if isinstance(ambient_cfg, dict) else None
    target_snr_db = float(ambient_cfg.get('snr_db_base', 10.0)) if isinstance(ambient_cfg, dict) else 10.0
    snr_jitter = float(ambient_cfg.get('snr_jitter_db', 0.0)) if isinstance(ambient_cfg, dict) else 0.0

    if ambient_dir:
        try:
            src_dir = Path(ambient_dir)
            files = []
            if src_dir.exists():
                for ext in ('*.wav', '*.flac', '*.ogg'):
                    files.extend(list(src_dir.rglob(ext)))
            if files:
                src = rng.choice(files)
                noise = load_audio_file(src, sr, duration=float(cfg.get('audio_parameters', {}).get('target_duration_sec', 4.0)))
                if noise is not None:
                    noise = ensure_duration(noise, sr, float(cfg.get('audio_parameters', {}).get('target_duration_sec', 4.0)), cfg.get('audio_parameters', {}).get('crossfade_duration_sec', 0.1))
                    # compute scaling for target SNR
                    if snr_jitter > 0:
                        target_snr_db = target_snr_db + float(rng.normal(0, snr_jitter))

                    sig_power = float(np.mean(audio**2)) if audio.size > 0 else 0.0
                    noise_power = float(np.mean(noise**2)) if noise.size > 0 else 0.0
                    if sig_power > 0 and noise_power > 0:
                        target_noise_power = sig_power / (10.0 ** (target_snr_db / 10.0))
                        scale = (target_noise_power / noise_power) ** 0.5
                        # prepare meta for mixing modifier (no file I/O in modifier)
                        mix_meta = dict(meta) if isinstance(meta, dict) else {}
                        mix_meta['mix_sources'] = [{'audio': noise, 'gain': float(scale)}]
                        mix_meta['ambient_source'] = str(src)
                        mix_meta['target_snr_db'] = float(target_snr_db)
                        # call pure-array mixer modifier
                        try:
                            from .modifiers.mix import apply as mix_apply
                            audio, mix_debug = mix_apply(audio, sr, rng, mix_meta, cfg)
                            debug.update(mix_debug if isinstance(mix_debug, dict) else {})
                        except Exception:
                            # fallback to naive add
                            audio = audio + noise * float(scale)
                            debug.update({'ambient_source': str(src), 'target_snr_db': float(target_snr_db)})
        except Exception:
            pass

    return audio, debug
