import json
from pathlib import Path
import math
import numpy as np
try:
    import soundfile as sf
except Exception:
    sf = None


def validate_split(dataset_dir, audio_cfg):
    dataset_dir = Path(dataset_dir)
    meta = dataset_dir / "metadata.jsonl"
    if not meta.exists():
        raise FileNotFoundError(f"Missing metadata: {meta}")
    lines = list(meta.open())
    wav_count = sum(1 for _ in dataset_dir.rglob("*.wav"))
    if wav_count != len(lines):
        raise ValueError(f"Mismatch wav ({wav_count}) vs metadata ({len(lines)}) in {dataset_dir}")

    sr = int(audio_cfg["sample_rate"])
    dur = float(audio_cfg["duration_s"])
    tol = 1.0 / sr
    for l in lines:
        j = json.loads(l)
        p = dataset_dir / j["path"]
        if not p.exists():
            raise FileNotFoundError(p)
        if sf is None:
            continue
        data, s = sf.read(str(p), dtype="float32")
        if s != sr:
            raise ValueError(f"SR mismatch {p}: {s} != {sr}")
        if abs(len(data) / s - dur) > tol:
            raise ValueError(f"Duration mismatch {p}")
        # check NaN/Inf
        if not np.isfinite(data).all():
            raise ValueError(f"NaN/Inf detected in {p}")


def run_all_splits(root_dir, cfg):
    root_dir = Path(root_dir)
    audio_cfg = cfg.get("audio", {})
    # validate each split and collect source_hashes for anti-leakage
    hash_map = {}
    for split in ("dataset_train", "dataset_val", "dataset_test"):
        sd = root_dir / split
        if sd.exists():
            validate_split(sd, audio_cfg)
            meta = sd / "metadata.jsonl"
            for l in meta.open():
                j = json.loads(l)
                for s in j.get("sources", []):
                    h = s.get("source_hash")
                    if h:
                        hash_map.setdefault(h, set()).add(split)

    # anti-leakage: no hash should be in >1 split
    leaks = [h for h, ss in hash_map.items() if len(ss) > 1]
    if leaks:
        raise ValueError(f"Source leakage detected across splits for hashes: {leaks}")

    # run distribution checks using validation cfg if present
    if isinstance(cfg, dict) and ("targets" in cfg or "validation" in cfg):
        validate_cfg = cfg
    else:
        validate_cfg = {}
    validate_distribution(root_dir, validate_cfg)


def validate_distribution(root_dir, validate_cfg=None):
    root_dir = Path(root_dir)
    vcfg = validate_cfg or {}
    # support either validation keys or a targets subsection
    targets = vcfg.get("targets") if isinstance(vcfg.get("targets"), dict) else vcfg
    # defaults
    label0_target = float(targets.get("label0_ratio", targets.get("label0_target", 0.30)))
    tol_label0 = float(targets.get("tol_label0", targets.get("tol_label0", 0.02)))
    far_frac_min = float(targets.get("far_frac_min", targets.get("far_frac_min", 0.25)))
    moving_frac_min = float(targets.get("moving_frac_label1", targets.get("moving_frac_min", targets.get("moving_ratio_label1", 0.50))))
    doppler_frac_min = float(targets.get("doppler_frac_label1", targets.get("doppler_frac_min", 0.40)))
    strong_noise_dbfs = float(targets.get("strong_noise_dbfs", -55.0))
    strict = bool(targets.get("strict", False))
    snr_bins = targets.get("snr_bins_db") or []
    snr_bin_weights = targets.get("snr_bin_weights_label1") or []
    # collect metadata across splits
    all_meta = []
    for split in ("dataset_train", "dataset_val", "dataset_test"):
        sd = root_dir / split
        if sd.exists():
            meta = sd / "metadata.jsonl"
            if meta.exists():
                for l in meta.open():
                    j = json.loads(l)
                    j["_split"] = split
                    all_meta.append(j)

    if not all_meta:
        return

    total = len(all_meta)
    label0 = [m for m in all_meta if int(m.get("label", 0)) == 0]
    frac0 = len(label0) / total
    if abs(frac0 - label0_target) > tol_label0:
        print(f"WARNING: Label0 fraction {frac0:.3f} outside tolerance around {label0_target:.2f} (tol={tol_label0})")

    # distances bins for label1
    label1 = [m for m in all_meta if int(m.get("label", 0)) == 1]
    if label1:
        far_count = 0
        moving_count = 0
        doppler_count = 0
        strong_noise = 0
        snr_list = []
        for m in label1:
            srcs = m.get("sources", [])
            # flags we'll try to infer from multiple possible metadata locations
            found_far = False
            found_moving = False
            found_doppler = False
            found_strong_noise = False

            # helper to check numeric-like values safely
            def _num(v):
                try:
                    return float(v)
                except Exception:
                    return None

            # 1) top-level fields
            dstart = _num(m.get("distance_start_m"))
            dend = _num(m.get("distance_end_m"))
            if dstart is not None or dend is not None:
                if dstart is not None and dend is not None and dstart != dend:
                    found_moving = True
                # check far bin on endpoints
                if (dstart is not None and 150 <= dstart <= 300) or (dend is not None and 150 <= dend <= 300):
                    found_far = True

            # 2) audio-level doppler/noise indicators
            audio = m.get("audio", {}) or {}
            if audio.get("doppler_ratio_max"):
                found_doppler = True
            if _num(audio.get("noise_rms_dbfs")) is not None and _num(audio.get("noise_rms_dbfs")) > strong_noise_dbfs:
                found_strong_noise = True

            # 3) inspect each source and its chain for distance/doppler info
            for s in srcs:
                # direct keys on source
                sd = _num(s.get("distance_m"))
                if sd is not None and not found_far:
                    if 150 <= sd <= 300:
                        found_far = True

                if s.get("doppler_v_mps") is not None:
                    found_doppler = True

                # source-level noise
                snd = _num(s.get("noise_rms_dbfs"))
                if snd is not None and snd > -55:
                    found_strong_noise = True

                # chain entries may encode params from ops like distance_atten, trajectory_atten, doppler_segmented
                for ch in s.get("chain", []) if isinstance(s.get("chain", []), list) else []:
                    op = ch.get("op")
                    params = ch.get("params", {}) or {}
                    # distance params
                    pd = params.get("distance_m")
                    if pd is not None:
                        pdn = _num(pd)
                        if pdn is not None and 150 <= pdn <= 300:
                            found_far = True
                    # trajectory params
                    ps = _num(params.get("distance_start_m"))
                    pe = _num(params.get("distance_end_m"))
                    if ps is not None or pe is not None:
                        if ps is not None and pe is not None and ps != pe:
                            found_moving = True
                        if (ps is not None and 150 <= ps <= 300) or (pe is not None and 150 <= pe <= 300):
                            found_far = True
                    # doppler ops
                    if op and ("doppler" in op or op.startswith("doppler")):
                        found_doppler = True

            if found_far:
                far_count += 1
            if found_moving:
                moving_count += 1
            if found_doppler:
                doppler_count += 1
            if found_strong_noise:
                strong_noise += 1
            # collect snr bins for label1
            snr = None
            if 'snr_est_db' in m:
                try:
                    snr = float(m.get('snr_est_db'))
                except Exception:
                    snr = None
            if snr is not None:
                snr_list.append(snr)

        # Distribution checks: report or raise if strict
        failures = []
        if len(label1) > 0:
            if far_count / len(label1) < far_frac_min:
                failures.append(f"Far bin [150-300] under-represented among label1 (<{far_frac_min*100:.0f}%)")
            if moving_count / len(label1) < moving_frac_min:
                failures.append(f"Moving ratio among label1 < {moving_frac_min*100:.0f}%")
            if doppler_count / len(label1) < doppler_frac_min:
                failures.append(f"Doppler usage among label1 < {doppler_frac_min*100:.0f}%")
            if strong_noise / len(label1) < float(targets.get('strong_noise_ratio', 0.0)):
                failures.append('Strong noise fraction below target')

            # SNR bin coverage check when configured
            if snr_bins and snr_bin_weights and len(snr_list) > 0:
                # compute histogram of snr_list into bins defined by snr_bins boundaries
                import numpy as _np

                bins = snr_bins
                hist, _ = _np.histogram(_np.array(snr_list), bins=bins)
                # normalize
                hist_frac = hist.astype(float) / hist.sum() if hist.sum() > 0 else hist
                # compare to target weights if same length
                if len(hist_frac) == len(snr_bin_weights):
                    for i, (h, target_w) in enumerate(zip(hist_frac, snr_bin_weights)):
                        if h + 1e-6 < target_w:
                            failures.append(f"SNR bin {i} under target: {h:.2f} < {target_w:.2f}")

        if failures:
            msg = "; ".join(failures)
            if strict:
                raise ValueError("Distribution validation failed: " + msg)
            else:
                print("WARNING:", msg)
