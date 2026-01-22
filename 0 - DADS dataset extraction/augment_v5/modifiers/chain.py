from .registry import OP_REGISTRY


def validate_chain(chain):
    if not isinstance(chain, list):
        raise ValueError("chain must be a list")
    for op in chain:
        if not isinstance(op, dict) or "op" not in op:
            raise ValueError("each chain item must be a dict with an 'op' key")
        if op["op"] not in OP_REGISTRY:
            raise ValueError(f"unknown op {op['op']}")


def apply_chain(audio, chain, ctx=None):
    """Apply a sequence of ops to audio. Each op is a dict {"op": name, "params": {...}}.

    Returns (audio, meta_updates) where meta_updates is a merged dict from ops.
    """
    meta = {}
    if not chain:
        return audio, meta
    for op in chain:
        name = op.get("op")
        params = op.get("params", {})
        fn = OP_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown op {name}")
        audio, m = fn(audio, params, ctx)
        if m:
            meta.update(m)
    return audio, meta
