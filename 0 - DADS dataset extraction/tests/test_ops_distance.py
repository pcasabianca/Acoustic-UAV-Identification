import numpy as np
from augment_v5.modifiers import OP_REGISTRY


def test_distance_scaling_exact():
    fn = OP_REGISTRY.get("distance_atten")
    assert fn is not None
    audio = np.ones(64, dtype="float32") * 0.1
    out, meta = fn(audio, {"distance_m": 10.0, "ref_m": 1.0}, ctx=None)
    gain = 1.0 / 10.0
    assert np.allclose(out, audio * gain, atol=1e-8)
    assert meta["distance_m"] == 10.0
    assert abs(meta["distance_gain"] - gain) < 1e-12
    assert abs(meta["distance_db"] - (20.0 * np.log10(gain))) < 1e-9


def test_distance_min_distance():
    fn = OP_REGISTRY.get("distance_atten")
    audio = np.ones(16, dtype="float32") * 0.2
    out, meta = fn(audio, {"distance_m": 0.0, "ref_m": 1.0}, ctx=None)
    # should not be NaN or Inf
    assert np.isfinite(out).all()
    assert np.isfinite(meta["distance_gain"]) and np.isfinite(meta["distance_db"]) 


def test_trajectory_equals_static_when_start_eq_end():
    fn_static = OP_REGISTRY.get("distance_atten")
    fn_traj = OP_REGISTRY.get("trajectory_atten")
    audio = np.ones(64, dtype="float32") * 0.05
    out_s, meta_s = fn_static(audio, {"distance_m": 20.0, "ref_m": 1.0}, ctx=None)
    out_t, meta_t = fn_traj(audio, {"distance_start_m": 20.0, "distance_end_m": 20.0, "ref_m": 1.0}, ctx=None)
    assert np.allclose(out_s, out_t, atol=1e-8)
    assert abs(meta_t["gain_min"] - meta_t["gain_max"]) < 1e-12
