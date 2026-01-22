import numpy as np
import pytest
import importlib
from augment_v5.modifiers import OP_REGISTRY


has_scipy = importlib.util.find_spec("scipy") is not None


@pytest.mark.skipif(not has_scipy, reason="scipy may be missing in CI; skip when absent")
def test_doppler_v0_identity():
    fn = OP_REGISTRY.get("doppler_segmented")
    audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 2048))
    out, meta = fn(audio.astype('float32'), {"v_mps": 0.0, "direction": 1, "segments": 8, "sr": 2048}, ctx=None)
    assert np.allclose(out, audio[:len(out)], atol=1e-5)


@pytest.mark.skipif(not has_scipy, reason="scipy may be missing in CI; skip when absent")
def test_doppler_direction_differs():
    fn = OP_REGISTRY.get("doppler_segmented")
    t = np.linspace(0, 1, 2048)
    s = np.sin(2 * np.pi * 1000.0 * t)
    out_a, _ = fn(s.astype('float32'), {"v_mps": 30.0, "direction": 1, "segments": 8, "sr": 2048}, ctx=None)
    out_r, _ = fn(s.astype('float32'), {"v_mps": 30.0, "direction": -1, "segments": 8, "sr": 2048}, ctx=None)
    # FFT peaks should differ
    fa = np.abs(np.fft.rfft(out_a))
    fr = np.abs(np.fft.rfft(out_r))
    assert np.argmax(fa) != np.argmax(fr)