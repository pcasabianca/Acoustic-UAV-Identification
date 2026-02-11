import json
import time
import math
from pathlib import Path
import numpy as np
from .audio_utils_v5 import load_audio
from .modifiers.chain import apply_chain
from .io_atomic import atomic_write_wav
from .difficulty import compute_difficulty_score

# ensure ops are registered
from . import modifiers  # side-effect: imports ops modules and populates registry


def process_shard(tasks, dataset_dir, shard_path, audio_cfg, mix_cfg):
    dataset_dir = Path(dataset_dir)
    shard_path = Path(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    sr = int(audio_cfg["sample_rate"])
    dur_s = float(audio_cfg["duration_s"])
    target_samples = int(sr * dur_s)
    total = len(tasks)
    # Prepare progress reporting every 5% (including 0%) with ETA
    report_step = 5
    next_report_pct = 0
    processed = 0
    start_t = time.time()

    with shard_path.open("w") as out:
        # initial 0% report
        if total > 0:
            print(f"[PROGRESS] shard={shard_path.name} processed=0/{total} (0%) ETA=estimating", flush=True)
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
            
            # Compute difficulty score before finalizing metadata
            difficulty_cfg = mix_cfg.get('difficulty', {})
            if difficulty_cfg.get('enabled', True):
                # Build metadata dict for difficulty computation
                diff_meta = {
                    'snr_est_db': snr_est_db,
                    'sources': t["components"],
                }
                # Extract distance if available in scenario
                if 'distance_m' in t:
                    diff_meta['distance_m'] = t['distance_m']
                
                difficulty_score = compute_difficulty_score(mix, diff_meta, t["label"], difficulty_cfg)
                meta['difficulty'] = float(difficulty_score)
            
            # attach SNR metadata when available
            if snr_est_db is not None:
                # preserve legacy top-level fields
                meta["snr_est_db"] = float(snr_est_db)
                meta["rms_drone"] = float(rms_drone)
                meta["rms_ambient"] = float(rms_ambient)
                # compute per-drone RMS values when individual arrays are available
                drone_rms_list = []
                if 'drone' in role_parts:
                    for arr in role_parts['drone']:
                        drone_rms_list.append(_rms_arr(arr))
                # build a debug dict containing the canonical resulting_snr_db key
                debug = {
                    'resulting_snr_db': float(snr_est_db),
                    'resulting_snr_db_per_drone': [float(v) for v in drone_rms_list],
                    'rms_drone': float(rms_drone),
                    'rms_ambient': float(rms_ambient)
                }
                meta['debug'] = debug
                # also annotate individual source entries with per-source RMS when possible
                for idx, c in enumerate(meta["sources"]):
                    role = c.get("role")
                    if role == 'drone':
                        # try to map drone sources to drone_rms_list by order
                        try:
                            c.setdefault('rms', float(drone_rms_list.pop(0)))
                        except Exception:
                            c.setdefault('rms', float(rms_drone))
                    elif role == 'ambient':
                        c.setdefault('rms', float(rms_ambient))
            out.write(json.dumps(meta) + "\n")
            # progress update
            processed += 1
            if total > 0:
                pct = int(math.floor((processed / total) * 100))
                # report at each multiple of report_step (including 0 already printed)
                if pct >= next_report_pct:
                    now = time.time()
                    elapsed = now - start_t
                    avg = elapsed / processed if processed > 0 else 0
                    remaining = max(0, total - processed)
                    eta_sec = int(round(remaining * avg))
                    # format ETA
                    h = eta_sec // 3600
                    m = (eta_sec % 3600) // 60
                    s = eta_sec % 60
                    eta_str = f"{h:02d}:{m:02d}:{s:02d}"
                    print(f"[PROGRESS] shard={shard_path.name} processed={processed}/{total} ({pct}%) ETA={eta_str}", flush=True)
                    # advance next_report_pct to the next multiple > pct
                    next_report_pct = ((pct // report_step) + 1) * report_step
