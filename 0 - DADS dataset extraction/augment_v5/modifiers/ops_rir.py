import numpy as np
from .registry import register


@register("convolve_rir")
def convolve_rir(audio, params, ctx=None):
    rir = params.get("rir")
    mode = params.get("mode", "same")
    normalize = bool(params.get("normalize_rir", False))
    if rir is None:
        raise ValueError("convolve_rir requires 'rir' param (list or array)")
    r = np.array(rir, dtype="float32")
    if normalize and r.size:
        r = r / np.sqrt(np.sum(r ** 2))
    # use 'full' conv then slice so that index 0 aligns with rir[0]
    conv_full = np.convolve(audio, r, mode="full")
    if mode == "full":
        out = conv_full
    else:
        # 'same' -> take the first len(audio) samples so impulse maps to rir at start
        out = conv_full[: len(audio)]
    return out.astype("float32"), {"rir_len": int(len(r)), "mode": mode}
"""Placeholder for RIR convolution ops.

To be implemented in Step 3.
"""

def TODO_rir_op(params):
    raise NotImplementedError("RIR ops to be implemented in Step 3")
