import json
from pathlib import Path
import numpy as np
from .audio_utils_v5 import load_audio
from .modifiers.chain import apply_chain
from .io_atomic import atomic_write_wav

# ensure ops are registered
from . import modifiers  # side-effect: imports ops modules and populates registry


def process_shard(tasks, dataset_dir, shard_path, audio_cfg, mix_cfg):
    dataset_dir = Path(dataset_dir)
    shard_path = Path(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    sr = int(audio_cfg["sample_rate"])
    dur_s = float(audio_cfg["duration_s"])
    target_samples = int(sr * dur_s)

    with shard_path.open("w") as out:
        for t in tasks:
            parts = []
            role_parts = {}
            for c in t["components"]:
                a, s_sr = load_audio(c["source_path"], target_sr=sr)
                a, src_meta = apply_chain(a, c.get("chain", []), {"seed": t.get("seed"), "sample_id": t.get("sample_id"), "role": c.get("role")})
                # attach any per-source meta back into the component dict so it's saved in metadata
                if src_meta:
                    for k, v in src_meta.items():
                        c[k] = v
                parts.append((c.get("role"), a))
                role_parts.setdefault(c.get("role"), []).append(a)

            # compute per-role sums before global mixing to estimate SNR
            mix = np.zeros(target_samples, dtype="float32")
            for role, arr in parts:
                mix = mix + arr

            # estimate rms per role (drone vs ambient)
            def _rms_arr(a):
                return float(np.sqrt(np.mean(a.astype('float64') ** 2))) if a is not None and len(a) > 0 else 0.0

            drone_sum = None
            ambient_sum = None
            if 'drone' in role_parts:
                drone_sum = np.sum(np.stack(role_parts['drone']), axis=0) if len(role_parts['drone']) > 0 else np.zeros(target_samples, dtype='float32')
            if 'ambient' in role_parts:
                ambient_sum = np.sum(np.stack(role_parts['ambient']), axis=0) if len(role_parts['ambient']) > 0 else np.zeros(target_samples, dtype='float32')

            rms_drone = _rms_arr(drone_sum) if drone_sum is not None else 0.0
            rms_ambient = _rms_arr(ambient_sum) if ambient_sum is not None else 0.0
            eps = 1e-12
            snr_est_db = 20.0 * np.log10((rms_drone + eps) / (rms_ambient + eps)) if (rms_drone is not None) else None

            # apply global_chain (should include normalize_peak and stats)
            mix, meta_updates = apply_chain(mix, t.get("global_chain", []), {"seed": t.get("seed"), "sample_id": t.get("sample_id")})

            out_path = dataset_dir / t["output_relpath"]
            atomic_write_wav(out_path, mix, sr)

            # meta_updates are expected to contain audio stats (flat dict)
            audio_meta = meta_updates or {}

            meta = {
                "split": t["split"],
                "label": t["label"],
                "scenario": t["scenario"],
                "path": t["output_relpath"],
                "sources": t["components"],
                "audio": {"sample_rate": sr, "duration_s": dur_s, **audio_meta},
            }
            # attach SNR metadata when available
            if snr_est_db is not None:
                meta["snr_est_db"] = float(snr_est_db)
                meta["rms_drone"] = float(rms_drone)
                meta["rms_ambient"] = float(rms_ambient)
                # also copy into components where relevant
                for c in meta["sources"]:
                    if c.get("role") == 'drone':
                        c.setdefault("rms_drone", float(rms_drone))
                    if c.get("role") == 'ambient':
                        c.setdefault("rms_ambient", float(rms_ambient))
            out.write(json.dumps(meta) + "\n")
