def run_pipeline(audio, sr, rng, meta, modifiers, cfg):
    cur = audio
    meta_acc = {}
    for mod in modifiers:
        # support modifier as module with `apply` or as a callable function
        if hasattr(mod, 'apply') and callable(getattr(mod, 'apply')):
            cur, delta = mod.apply(cur, sr, rng, meta, cfg)
        elif callable(mod):
            cur, delta = mod(cur, sr, rng, meta, cfg)
        else:
            raise TypeError(f"Modifier {mod} is not callable and has no 'apply'")
        if delta:
            meta_acc.update(delta)
    return cur, meta_acc
