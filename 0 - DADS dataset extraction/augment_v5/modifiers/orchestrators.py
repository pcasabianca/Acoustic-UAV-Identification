from .registry import register
from .chain import apply_chain
from .utils_rng import get_rng
import numpy as np

@register("filters_orch")
def filters_orch(audio, params, ctx=None):
    sub = []
    sr = params.get("sr")
    if params.get("hp_hz") is not None:
        sub.append({"op": "highpass", "params": {"hz": params.get("hp_hz"), "sr": sr}})
    if params.get("lp_hz") is not None:
        sub.append({"op": "lowpass", "params": {"hz": params.get("lp_hz"), "sr": sr}})
    for n in params.get("notches", []):
        if n.get("f0") is not None:
            sub.append({"op": "notch", "params": {"f0": n.get("f0"), "q": n.get("q", 30), "sr": sr}})
    out, meta = apply_chain(audio, sub, ctx)
    return out, meta


@register("env_orch")
def env_orch(audio, params, ctx=None):
    sub = []
    # distance: either static or trajectory
    if params.get("distance_mode") == "trajectory":
        sub.append({"op": "trajectory_atten", "params": {"distance_start_m": params.get("distance_start_m"), "distance_end_m": params.get("distance_end_m"), "ref_m": params.get("ref_m", 1.0), "curve": params.get("curve", "linear")}})
    elif params.get("distance_m") is not None:
        sub.append({"op": "distance_atten", "params": {"distance_m": params.get("distance_m"), "ref_m": params.get("ref_m", 1.0)}})
    if params.get("air_cutoff_hz") is not None:
        sub.append({"op": "lowpass", "params": {"hz": params.get("air_cutoff_hz"), "sr": params.get("sr")}})
    # optional doppler
    if params.get("doppler", {}).get("enabled"):
        d = params.get("doppler", {})
        sub.append({"op": "doppler_segmented", "params": {"v_mps": d.get("v_mps", 0.0), "direction": d.get("direction", 1), "segments": d.get("segments", 32), "max_ratio": d.get("max_ratio", 1.05), "sr": params.get("sr")}})
    out, meta = apply_chain(audio, sub, ctx)
    # ensure distance/doppler keys from params are present in meta for downstream validation
    if params.get("distance_mode") == "trajectory":
        if "distance_start_m" not in meta and params.get("distance_start_m") is not None:
            meta["distance_start_m"] = float(params.get("distance_start_m"))
        if "distance_end_m" not in meta and params.get("distance_end_m") is not None:
            meta["distance_end_m"] = float(params.get("distance_end_m"))
    elif params.get("distance_m") is not None:
        meta.setdefault("distance_m", float(params.get("distance_m")))
    d = params.get("doppler", {}) or {}
    if d.get("enabled"):
        meta.setdefault("doppler_v_mps", float(d.get("v_mps", 0.0)))
        meta.setdefault("doppler_direction", int(d.get("direction", 1)))
    return out, meta


@register("hardware_orch")
def hardware_orch(audio, params, ctx=None):
    sub = []
    if "hp_hz" in params:
        if params.get("hp_hz") is not None:
            sub.append({"op": "highpass", "params": {"hz": params.get("hp_hz"), "sr": params.get("sr")}})
    if "noise_rms_dbfs" in params:
        sub.append({"op": "add_white_noise", "params": {"noise_rms_dbfs": params["noise_rms_dbfs"]}})
    if "adc_bits" in params:
        sub.append({"op": "adc_quantize", "params": {"bits": params["adc_bits"]}})
    out, meta = apply_chain(audio, sub, ctx)
    # expose hardware params in meta
    if "noise_rms_dbfs" in params:
        meta.setdefault("noise_rms_dbfs", float(params["noise_rms_dbfs"]))
    if "adc_bits" in params:
        meta.setdefault("adc_bits", int(params["adc_bits"]))
    return out, meta


@register("drone_orch")
def drone_orch(audio, params, ctx=None):
    # amplitude flutter: slow sinusoidal amplitude modulation
    flutter_db = float(params.get("flutter_db", 0.0))
    flutter_hz = float(params.get("flutter_hz", 0.5))
    sr = int(params.get("sr", 22050))
    N = len(audio)
    if flutter_db <= 0 or N == 0:
        return audio, {}
    rng = None
    try:
        rng = get_rng(ctx)
    except Exception:
        rng = None
    t = np.arange(N) / float(sr)
    phase = float(rng.random() * 2.0 * np.pi) if rng is not None else 0.0
    amp = 10 ** ((flutter_db * 0.5 * np.sin(2 * np.pi * flutter_hz * t + phase)) / 20.0)
    out = audio * amp
    meta = {"flutter_db": flutter_db, "flutter_hz": flutter_hz, "flutter_phase": phase}
    return out.astype("float32"), meta


@register("ambient_orch")
def ambient_orch(audio, params, ctx=None):
    sub = []
    sr = params.get("sr")
    if params.get("hp_hz") is not None:
        sub.append({"op": "highpass", "params": {"hz": params.get("hp_hz"), "sr": sr}})
    if params.get("lp_hz") is not None:
        sub.append({"op": "lowpass", "params": {"hz": params.get("lp_hz"), "sr": sr}})
    for n in params.get("notches", []):
        if n.get("f0") is not None:
            sub.append({"op": "notch", "params": {"f0": n.get("f0"), "q": n.get("q", 30), "sr": sr}})
    if params.get("dropout", {}).get("enabled"):
        d = params.get("dropout", {})
        sub.append({"op": "dropout", "params": {"prob": d.get("prob", 0.05), "window_ms": d.get("window_ms", 50.0), "atten_db": d.get("atten_db", -60.0), "sr": sr}})
    if params.get("softclip", {}).get("enabled"):
        s = params.get("softclip", {})
        sub.append({"op": "softclip_tanh", "params": {"drive_db": s.get("drive_db", 0.0)}})

    out, meta = apply_chain(audio, sub, ctx)
    return out, meta
"""Placeholders for orchestrator ops (TODO: implement in Step 3).

These modules register macro-ops that expand to sequences of smaller ops.
Currently only TODO stubs are provided to document the planned API.
"""

# TODO: implement orchestrators that expand to registered ops

def TODO_orchestrator_example(cfg):
    raise NotImplementedError("orchestrators are planned for Step 3")
