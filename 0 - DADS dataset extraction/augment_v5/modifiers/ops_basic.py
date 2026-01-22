from ..audio_utils_v5 import trim_or_pad, normalize_peak, audio_stats
from .registry import register


@register("trim_or_pad")
def op_trim_or_pad(audio, params, ctx=None):
    # params: {"target_samples": int}
    ts = params.get("target_samples")
    if ts is None:
        raise ValueError("trim_or_pad requires target_samples param")
    out = trim_or_pad(audio, int(ts))
    return out, {}


@register("normalize_peak")
def op_normalize_peak(audio, params, ctx=None):
    target_dbfs = params.get("target_dbfs", -1.0)
    out = normalize_peak(audio, target_dbfs=target_dbfs)
    return out, {}


@register("stats")
def op_stats(audio, params, ctx=None):
    sr = params.get("sr")
    if sr is None:
        return audio, {}
    return audio, audio_stats(audio, sr)
