import numpy as np
import pytest
from augment_v5.modifiers import OP_REGISTRY


def test_convolve_rir_impulse():
    fn = OP_REGISTRY.get("convolve_rir")
    impulse = np.zeros(32, dtype="float32")
    impulse[0] = 1.0
    rir = [0.5, 0.25, 0.125]
    out, meta = fn(impulse, {"rir": rir, "mode": "same"}, ctx=None)
    # output should contain rir energy
    assert pytest.approx(0.5, rel=1e-6) == out[0]
    assert meta["rir_len"] == len(rir)
