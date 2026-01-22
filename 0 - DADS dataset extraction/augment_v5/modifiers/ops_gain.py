from .registry import register


@register("gain_db")
def gain_db(audio, params, ctx=None):
    # params: {"db": float}
    db = float(params.get("db", 0.0))
    factor = 10 ** (db / 20.0)
    out = audio * factor
    return out, {"gain_db": db}


@register("gain_linear")
def gain_linear(audio, params, ctx=None):
    # params: {"factor": float}
    factor = float(params.get("factor", 1.0))
    out = audio * factor
    return out, {"gain_factor": factor}
