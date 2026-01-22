import numpy as np
from .registry import register
from .utils_rng import get_rng


@register("add_white_noise")
def add_white_noise(audio, params, ctx=None):
    rms_dbfs = float(params.get("noise_rms_dbfs", -60.0))
    rng = get_rng(ctx)
    # target RMS amplitude
    target_rms = 10 ** (rms_dbfs / 20.0)
    # generate white noise with unit std, scale to unit RMS ~1/sqrt(2) for normal? use direct normalization
    noise = rng.normal(size=audio.shape).astype("float32")
    cur_rms = float(np.sqrt(np.mean(noise ** 2)))
    if cur_rms == 0:
        scaled = noise
    else:
        scaled = noise * (target_rms / cur_rms)
    out = audio + scaled
    return out, {"noise_rms_dbfs": rms_dbfs}


@register("adc_quantize")
def adc_quantize(audio, params, ctx=None):
    bits = int(params.get("bits", 16))
    clip = bool(params.get("clip", True))
    if clip:
        audio = np.clip(audio, -1.0, 1.0)
    levels = 2 ** bits
    # map to integer range
    ints = np.round((audio + 1.0) * (levels - 1) / 2.0)
    ints = np.clip(ints, 0, levels - 1)
    out = (ints * 2.0 / (levels - 1)) - 1.0
    return out.astype("float32"), {"adc_bits": bits}


@register("mems_hp")
def mems_hp(audio, params, ctx=None):
    # convenience wrapper around highpass op
    from .ops_filters import highpass

    out, meta = highpass(audio, params, ctx=ctx)
    return out, meta
"""Placeholder for MEMS-simulation ops (Step 3).

Real implementations will model MEMS coloration/noise/hp filtering.
"""

def TODO_mems_op(params):
    raise NotImplementedError("MEMS op to be implemented in Step 3")
