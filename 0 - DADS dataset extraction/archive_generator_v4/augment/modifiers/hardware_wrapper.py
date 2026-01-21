import numpy as np

def apply(audio, sr, rng, meta, cfg):
    """Hardware simulation wrapper: runs mems, hardware_noise, adc_artifacts, occlusion in order.

    cfg expected structure:
      { 'chain': [ {'name':'mems','params':{...}}, ... ] }
    """
    if audio is None:
        return audio, None

    chain = cfg.get('chain', []) if isinstance(cfg, dict) else []
    dbg = []
    out = audio.astype(float)
    for i, item in enumerate(chain):
        try:
            name = item.get('name') if isinstance(item, dict) else str(item)
            params = item.get('params', {}) if isinstance(item, dict) else {}
            mod = __import__('augment.modifiers.' + name, fromlist=['*'])
            seed = int(rng.integers(0, 2**31 - 1)) if hasattr(rng, 'integers') else int(rng.random() * 1e9)
            mod_rng = np.random.default_rng(seed)
            out, d = mod.apply(out, sr, mod_rng, meta, {'params': params})
            dbg.append({'name': name, 'applied': True, 'debug': d})
        except Exception as e:
            dbg.append({'name': name if 'name' in locals() else str(item), 'applied': False, 'error': str(e)})
    return out.astype(float), {'applied': True, 'chain_debug': dbg}
