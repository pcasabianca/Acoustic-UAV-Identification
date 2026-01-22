from .registry import register
import numpy as np
from .utils_rng import get_rng


@register("dropout")
def dropout(audio, params, ctx=None):
    # params: prob, window_ms, atten_db
    prob = float(params.get("prob", 0.05))
    window_ms = float(params.get("window_ms", 50.0))
    atten_db = float(params.get("atten_db", -60.0))
    sr = int(params.get("sr", 22050))
    rng = get_rng(ctx)
    N = len(audio)
    w = max(1, int(window_ms * sr / 1000.0))
    out = audio.copy()
    i = 0
    while i < N:
        if rng.random() < prob:
            # apply attenuation over window
            end = min(N, i + w)
            factor = 10 ** (atten_db / 20.0)
            out[i:end] = out[i:end] * factor
        i += w
    return out.astype("float32"), {"dropout_prob": prob}


@register("softclip_tanh")
def softclip_tanh(audio, params, ctx=None):
    # params: drive_db
    drive_db = float(params.get("drive_db", 0.0))
    drive = 10 ** (drive_db / 20.0)
    x = audio * drive
    out = np.tanh(x) / max(1e-12, np.max(np.abs(np.tanh(x))))
    return out.astype("float32"), {"softclip_drive_db": drive_db}
