import numpy as np
import pytest
import importlib
from augment_v5.modifiers import OP_REGISTRY

has_scipy = importlib.util.find_spec("scipy") is not None


@pytest.mark.skipif(not has_scipy, reason="scipy may be missing in CI; skip when absent")
def test_lowpass_and_highpass():
    lp = OP_REGISTRY.get("lowpass")
    hp = OP_REGISTRY.get("highpass")
    assert lp and hp
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    high = np.sin(2 * np.pi * 3000 * t)
    low = np.sin(2 * np.pi * 200 * t)
    mix = high + low
    out_lp, _ = lp(mix.astype('float32'), {"hz": 500.0, "sr": sr})
    # high freq should be attenuated
    assert np.std(out_lp) < np.std(mix)
