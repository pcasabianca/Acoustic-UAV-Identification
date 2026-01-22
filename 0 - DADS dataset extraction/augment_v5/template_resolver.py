import random
import math


def resolve_value(v, rng: random.Random):
    # primitives
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, list):
        return [resolve_value(x, rng) for x in v]
    if isinstance(v, dict):
        if "dist" in v:
            d = v["dist"]
            if d == "uniform":
                lo = float(v["min"]) if "min" in v else 0.0
                hi = float(v["max"]) if "max" in v else 1.0
                return rng.uniform(lo, hi)
            if d == "choice":
                vals = v.get("values", [])
                if not vals:
                    raise ValueError("choice dist requires values")
                return rng.choice(vals)
            if d == "normal":
                mu = float(v.get("mu", 0.0))
                sigma = float(v.get("sigma", 1.0))
                return rng.gauss(mu, sigma)
            raise ValueError(f"unknown dist: {d}")
        # otherwise resolve recursively
        out = {}
        for k, val in v.items():
            out[k] = resolve_value(val, rng)
        return out
    raise ValueError("Unsupported value type in template")


def instantiate_chain(chain_template, rng: random.Random):
    out = []
    if not chain_template:
        return out
    for item in chain_template:
        if not isinstance(item, dict) or "op" not in item:
            raise ValueError("chain items must be dicts with 'op'")
        params = item.get("params", {})
        resolved = resolve_value(params, rng)
        # ensure no 'dist' remain
        def check_no_dist(x):
            if isinstance(x, dict):
                if "dist" in x:
                    raise ValueError("dist found after instantiation")
                for vv in x.values():
                    check_no_dist(vv)
            elif isinstance(x, list):
                for vv in x:
                    check_no_dist(vv)

        check_no_dist(resolved)
        out.append({"op": item["op"], "params": resolved})
    return out
