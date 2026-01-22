from .registry import register
import math
import numpy as np


@register("distance_atten")
def distance_atten(audio, params, ctx=None):
    """
    Attenuate amplitude by ~1/r law.
    params: {"distance_m": float, "ref_m": float=1.0}
    Returns audio, meta with distance_m, distance_gain, distance_db
    """
    distance = float(params.get("distance_m", 1.0))
    ref_m = float(params.get("ref_m", 1.0))
    gain = ref_m / max(distance, 1e-6)
    out = audio * gain
    db = 20.0 * math.log10(gain) if gain > 0 else -300.0
    return out, {"distance_m": distance, "distance_gain": gain, "distance_db": db}


@register("trajectory_atten")
def trajectory_atten(audio, params, ctx=None):
    # params: distance_start_m, distance_end_m, ref_m=1.0, curve
    start = float(params.get("distance_start_m", params.get("distance_m", 1.0)))
    end = float(params.get("distance_end_m", start))
    ref_m = float(params.get("ref_m", 1.0))
    curve = params.get("curve", "linear")
    min_d = float(params.get("min_distance_m", 1e-3))
    N = len(audio)
    if N <= 0:
        return audio, {}
    t = np.linspace(0.0, 1.0, N)
    if curve == "ease_in_out":
        t = t * t * (3 - 2 * t)
    dist = start + (end - start) * t
    gain = ref_m / np.maximum(dist, min_d)
    out = audio * gain
    gain_min = float(np.min(gain))
    gain_max = float(np.max(gain))
    db_min = 20.0 * math.log10(gain_min) if gain_min > 0 else -300.0
    db_max = 20.0 * math.log10(gain_max) if gain_max > 0 else -300.0
    return out.astype("float32"), {
        "distance_start_m": start,
        "distance_end_m": end,
        "ref_m": ref_m,
        "gain_min": gain_min,
        "gain_max": gain_max,
        "distance_min": float(np.min(dist)),
        "distance_max": float(np.max(dist)),
        "db_min": db_min,
        "db_max": db_max,
    }
