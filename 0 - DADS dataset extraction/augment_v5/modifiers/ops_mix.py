from .registry import register
import numpy as np


def mix_sum(parts):
    if not parts:
        return np.array([], dtype="float32")
    out = np.zeros_like(parts[0])
    for p in parts:
        out = out + p
    return out


@register("mix_sum_ctx")
def mix_sum_ctx(audio, params, ctx=None):
    parts = ctx.get("parts") if ctx else None
    if parts is None:
        raise ValueError("mix_sum_ctx expects ctx['parts'] list of arrays")
    out = mix_sum(parts)
    return out, {}
"""Placeholder for mix ops (e.g. mix_snr, mix_sum).

Step 3 will implement ops to control SNR mixing and related behaviors.
"""

def TODO_mix_op(params):
    raise NotImplementedError("mix ops to be implemented in Step 3")
