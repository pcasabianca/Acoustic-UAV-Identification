import numpy as np


def get_rng(ctx):
    if ctx is None:
        raise ValueError("ctx required for RNG")
    if "rng" in ctx and ctx["rng"] is not None:
        return ctx["rng"]
    if "seed" in ctx and ctx["seed"] is not None:
        return np.random.default_rng(int(ctx["seed"]))
    raise ValueError("ctx must contain 'seed' or 'rng' for deterministic ops")
