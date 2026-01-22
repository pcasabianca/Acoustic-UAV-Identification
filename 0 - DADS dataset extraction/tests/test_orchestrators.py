import numpy as np
from augment_v5.modifiers import OP_REGISTRY


def test_filters_orch_basic():
    fn = OP_REGISTRY.get("filters_orch")
    assert fn is not None
    audio = np.ones(128, dtype="float32") * 0.1
    out, meta = fn(audio, {"sr": 22050}, ctx=None)
    assert out.shape == audio.shape
