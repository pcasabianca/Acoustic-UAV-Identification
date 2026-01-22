from .registry import register
import numpy as np

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None


def _require_scipy():
    if resample_poly is None:
        raise ImportError("scipy is required for doppler_segmented (pip install scipy)")


@register("doppler_segmented")
def doppler_segmented(audio, params, ctx=None):
    """
    Doppler approximation via segmented resampling using resample_poly.
    Returns (audio_out, meta).
    """
    _require_scipy()
    v = float(params.get("v_mps", 0.0))
    direction = int(params.get("direction", 1))
    c = float(params.get("c_mps", 343.0))
    segments = int(params.get("segments", 32))
    max_ratio = float(params.get("max_ratio", 1.05))
    profile = params.get("profile", "linear")

    if segments < 1:
        segments = 1
    # compute target ratio and clamp
    r_target = c / (c - direction * v) if (c - direction * v) != 0 else 1.0
    r_target = max(1.0 / max_ratio, min(max_ratio, r_target))
    N = len(audio)
    if N == 0:
        return audio, {}

    seg_len = int(np.ceil(N / segments))
    out_parts = []
    ratios = []
    # build ratio profile between 1 and r_target
    if profile == "ease_in_out":
        t = np.linspace(0, 1, segments)
        w = t * t * (3 - 2 * t)
    else:
        w = np.linspace(0, 1, segments)

    for i in range(segments):
        ratio = 1.0 + (r_target - 1.0) * w[i]
        ratio = max(1.0 / max_ratio, min(max_ratio, ratio))
        ratios.append(ratio)

    from math import gcd
    for i in range(segments):
        start = i * seg_len
        seg = audio[start: start + seg_len]
        if seg.size == 0:
            continue
        # resample_poly requires integer up/down; approximate ratio as up/down
        up = int(np.round(ratios[i] * 100))
        down = 100
        g = gcd(up, down)
        up //= g
        down //= g
        try:
            seg_rs = resample_poly(seg, up, down)
        except Exception:
            seg_rs = seg
        out_parts.append(np.asarray(seg_rs, dtype="float32"))

    out = np.concatenate(out_parts) if out_parts else np.array([], dtype="float32")
    out = out[:N] if len(out) >= N else np.pad(out, (0, N - len(out)))
    meta = {
        "doppler_v_mps": v,
        "doppler_direction": direction,
        "doppler_ratio_min": float(min(ratios)) if ratios else 1.0,
        "doppler_ratio_max": float(max(ratios)) if ratios else 1.0,
        "doppler_segments": segments,
    }
    return out.astype("float32"), meta
