import numpy as np
from augment_v5.modifiers import OP_REGISTRY


def test_add_white_noise_deterministic():
    fn = OP_REGISTRY.get("add_white_noise")
    audio = np.zeros(100, dtype="float32")
    out1, _ = fn(audio, {"noise_rms_dbfs": -20.0}, ctx={"seed": 123})
    out2, _ = fn(audio, {"noise_rms_dbfs": -20.0}, ctx={"seed": 123})
    assert np.allclose(out1, out2)


def test_adc_quantize_levels():
    fn = OP_REGISTRY.get("adc_quantize")
    audio = np.linspace(-1, 1, 256, dtype="float32")
    out, meta = fn(audio, {"bits": 8}, ctx=None)
    assert meta["adc_bits"] == 8
    # unique levels should be <= 2^8
    assert len(np.unique(out)) <= 2 ** 8
