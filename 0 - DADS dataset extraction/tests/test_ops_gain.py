import numpy as np
from augment_v5.modifiers import OP_REGISTRY


def test_gain_db():
    fn = OP_REGISTRY.get("gain_db")
    audio = np.ones(32, dtype="float32") * 0.5
    out, meta = fn(audio, {"db": 6.0}, ctx=None)
    factor = 10 ** (6.0 / 20.0)
    assert np.allclose(out, audio * factor, atol=1e-8)
    assert "gain_db" in meta
