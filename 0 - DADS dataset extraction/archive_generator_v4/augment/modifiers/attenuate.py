from typing import Tuple, Dict, Any
import numpy as np


def apply(audio: np.ndarray, sr: int, rng, meta: dict, cfg: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply attenuation in dB to simulate distance.

    Expects cfg to contain 'atten_dB' numeric value (negative for attenuation),
    or 'distance' block will be used by caller to compute it.
    """
    debug = {}
    if audio is None:
        return audio, debug

    att_db = None
    # allow caller to pass requested attenuation in cfg
    if isinstance(cfg, dict) and 'atten_dB' in cfg:
        try:
            att_db = float(cfg['atten_dB'])
        except Exception:
            att_db = None

    if att_db is None:
        # nothing to do
        return audio, debug

    # convert dB to linear
    gain = 10.0 ** (att_db / 20.0)
    out = audio * gain
    debug['atten_dB'] = float(att_db)
    debug['atten_linear'] = float(gain)
    return out, debug
