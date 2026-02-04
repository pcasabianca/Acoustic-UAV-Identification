try:
    from scipy.signal import butter, lfilter, iirnotch
except Exception:
    butter = None
    lfilter = None
    iirnotch = None

from .registry import register


def _require_scipy():
    if butter is None:
        raise ImportError("scipy is required for ops_filters (pip install scipy)")


@register("highpass")
def highpass(audio, params, ctx=None):
    _require_scipy()
    # If hz is not provided, skip the op and return audio unchanged
    if params is None or params.get("hz") is None:
        return audio, {}
    hz = float(params["hz"])
    sr = int(params["sr"])
    order = int(params.get("order", 4))
    wn = hz / (sr / 2.0)
    if not (0.0 < wn < 1.0):
        # invalid cutoff frequency for digital filter; skip op
        return audio, {"highpass_hz_skipped": hz}
    b, a = butter(order, wn, btype="high")
    out = lfilter(b, a, audio)
    return out, {"highpass_hz": hz}


@register("lowpass")
def lowpass(audio, params, ctx=None):
    _require_scipy()
    if params is None or params.get("hz") is None:
        return audio, {}
    hz = float(params["hz"])
    sr = int(params["sr"])
    order = int(params.get("order", 4))
    wn = hz / (sr / 2.0)
    if not (0.0 < wn < 1.0):
        # invalid cutoff frequency for digital filter; skip op
        return audio, {"lowpass_hz_skipped": hz}
    b, a = butter(order, wn, btype="low")
    out = lfilter(b, a, audio)
    return out, {"lowpass_hz": hz}


@register("notch")
def notch(audio, params, ctx=None):
    _require_scipy()
    if params is None or params.get("f0") is None:
        return audio, {}
    f0 = float(params["f0"])
    q = float(params.get("q", 30.0))
    sr = int(params["sr"])
    b, a = iirnotch(f0 / (sr / 2.0), q)
    out = lfilter(b, a, audio)
    return out, {"notch_f0": f0, "notch_q": q}
"""Placeholder for filter ops (HP/LP/notch).

Implementations to be added in Step 3.
"""

def TODO_filter_op(params):
    raise NotImplementedError("filter ops to be implemented in Step 3")
