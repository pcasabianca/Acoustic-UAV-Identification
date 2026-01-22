import pytest
import numpy as np
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from augment_v5.modifiers.chain import apply_chain
from augment_v5.modifiers import OP_REGISTRY

has_scipy = importlib.util.find_spec("scipy") is not None


def make_sig(sr=22050, dur=1.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (np.sin(2 * np.pi * 440.0 * t)).astype('float32')


def test_hardware_orch_meta_contains_noise_and_adc():
    sig = make_sig()
    chain = [{"op": "hardware_orch", "params": {"noise_rms_dbfs": -50, "adc_bits": 8, "hp_hz": 100, "sr": 22050}}]
    out, meta = apply_chain(sig, chain, {"seed": 1})
    assert "noise_rms_dbfs" in meta or "noise_rms_dbfs" in out
    assert meta.get("adc_bits") == 8


def test_drone_orch_meta_contains_flutter():
    sig = make_sig()
    chain = [{"op": "drone_orch", "params": {"flutter_db": 1.5, "flutter_hz": 0.7, "sr": 22050}}]
    out, meta = apply_chain(sig, chain, {"seed": 2})
    assert "flutter_db" in meta
    assert "flutter_phase" in meta


@pytest.mark.skipif(not has_scipy, reason="scipy missing")
def test_env_orch_with_doppler_produces_doppler_meta():
    sig = make_sig()
    chain = [{"op": "env_orch", "params": {"distance_mode": "static", "distance_m": 80.0, "sr": 22050, "doppler": {"enabled": True, "v_mps": 12.0, "direction": 1, "segments": 8, "max_ratio": 1.02}}}]
    out, meta = apply_chain(sig, chain, {"seed": 3})
    # doppler_segmented should add doppler_* fields
    assert any(k.startswith("doppler_") for k in meta.keys())
