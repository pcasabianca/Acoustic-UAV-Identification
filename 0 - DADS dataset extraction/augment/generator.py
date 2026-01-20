from pathlib import Path
import json
import sys
import logging
from datetime import datetime, timezone
import hashlib

from .io import atomic_write_json
# reuse existing audio utilities from project for loading/resampling
from pathlib import Path as _Path
import sys as _sys
# ensure workspace root on sys.path so `tools` can be imported
_root = _Path(__file__).resolve().parents[2]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
from tools.audio_utils import load_audio_file
from .validation import validate_build_config
from .writer import SplitWriter
from .rng import rng_for_key
from . import pipeline
from .distance import apply as distance_apply
from .modifiers import normalize as normalize_mod
import os
import multiprocessing as mp
from typing import Dict, Any
import time


def process_task(task: Dict[str, Any]):
    """Top-level worker function: load offline source (if any), run pipeline, return processed audio and metadata."""
    try:
        from .rng import rng_for_key as _rng_for_key
        from tools.audio_utils import load_audio_file as _load_audio_file
        from .modifiers import normalize as _normalize_mod
        # Initialize commonly-used variables so worker exceptions are informative
        # and NameError doesn't occur when some paths do not set these values.
        rng = None
        try:
            rng = _rng_for_key(globals().get('master_seed', 0), task.get('seed_key') if isinstance(task, dict) else '')
        except Exception:
            try:
                rng = _rng_for_key(globals().get('master_seed', 0), '')
            except Exception:
                rng = None

        base_audio = None
        # ensure `scenario` is defined to avoid NameError when referenced later
        scenario = task.get('scenario') if isinstance(task, dict) else None
        # load offline source audio if provided (so generated samples are not silent)
        if isinstance(task, dict) and task.get('offline_src'):
            try:
                base_audio = _load_audio_file(task.get('offline_src'), task.get('sr'))
            except Exception:
                base_audio = None
        # prepare a deterministic ambient/drone offline pool so we can assemble mix_sources
        try:
            offline_dir = globals().get('out') / 'dataset_DADS_offline'
            if offline_dir is not None and offline_dir.exists():
                # build pools per label (0=ambient,1=drone)
                pool0 = []
                pool1 = []
                for ext in ('*.wav', '*.flac', '*.ogg'):
                    pool0.extend([str(p) for p in (offline_dir / '0').rglob(ext)]) if (offline_dir / '0').exists() else None
                    pool1.extend([str(p) for p in (offline_dir / '1').rglob(ext)]) if (offline_dir / '1').exists() else None
                # fallback: scan root offline dir and split by filename containing _0_ or _1_
                if not pool0 or not pool1:
                    files = []
                    for ext in ('*.wav', '*.flac', '*.ogg'):
                        files.extend(list(Path(offline_dir).rglob(ext)))
                    for p in files:
                        n = p.name
                        if f"_0_" in n:
                            pool0.append(str(p))
                        if f"_1_" in n:
                            pool1.append(str(p))
                offline_pool_label = {0: pool0, 1: pool1}
            else:
                offline_pool_label = {0: [], 1: []}
        except Exception:
            offline_pool_label = {0: [], 1: []}
        drone_parts = []
        drone_debug = []
        ambient_debug = []
        mix_meta = {}
        mix_sources = []
        scenario_snr = None

        # allocations are consumed by the main process; workers receive a single task to process
        mix_debug = {}
        try:
            from .modifiers import mix as _mix_mod
            from .modifiers import reverb as _reverb_mod
            # If we have a base audio (offline sample) and an ambient pool, pick an ambient
            try:
                if base_audio is not None and isinstance(task, dict):
                    # pick one ambient deterministically for this task
                    pool_label = offline_pool_label.get(0, [])
                    if pool_label:
                        try:
                            pick_rng = _rng_for_key(globals().get('master_seed', 0), task.get('seed_key', '') + ':ambient_pick')
                            ambient_path = str(pick_rng.choice(pool_label))
                            ambient_audio = _load_audio_file(ambient_path, task.get('sr'))
                            if ambient_audio is not None:
                                # sample ambient gain delta from scenario if present
                                amb_gain_db = None
                                try:
                                    agr = scenario.get('ambient_spl_db_range') if isinstance(scenario, dict) else None
                                    if isinstance(agr, (list, tuple)) and len(agr) >= 2:
                                        low = float(agr[0]); high = float(agr[1])
                                        amb_gain_db = float(pick_rng.uniform(low, high))
                                except Exception:
                                    amb_gain_db = None
                                if amb_gain_db is None:
                                    lin_gain = 1.0
                                else:
                                    # convert dB delta to linear and clamp
                                    lin_gain = float(10.0 ** (amb_gain_db / 20.0))
                                    max_gain_linear = float(task.get('cfg', {}).get('max_gain_linear', 50.0) if isinstance(task.get('cfg', {}), dict) else 50.0)
                                    if lin_gain > max_gain_linear:
                                        lin_gain = max_gain_linear
                                mix_sources.append({'audio': ambient_audio, 'gain': lin_gain})
                                ad = {'source': Path(ambient_path).name}
                                if amb_gain_db is not None:
                                    ad['ambient_spl_db_set'] = float(amb_gain_db)
                                    ad['ambient_source_gain_db'] = float(amb_gain_db)
                                ambient_debug.append(ad)
                        except Exception:
                            pass
            except Exception:
                pass

            # assemble drone parts and additional ambients from the `scenario` definition
            try:
                if isinstance(scenario, dict):
                    # drones
                    for di, _ in enumerate(scenario.get('drones', []) if isinstance(scenario.get('drones', []), list) else []):
                        pool_drone = offline_pool_label.get(1, [])
                        if not pool_drone:
                            break
                        try:
                            pick_rng = _rng_for_key(globals().get('master_seed', 0), task.get('seed_key', '') + f':drone_pick:{di}')
                            drone_path = str(pick_rng.choice(pool_drone))
                            drone_audio = _load_audio_file(drone_path, task.get('sr'))
                            if drone_audio is None:
                                continue
                            try:
                                from .distance import apply as _distance_apply
                                # Prepare meta for distance.apply: allow scenario to specify drone_spl_db_range
                                local_meta = {}
                                try:
                                    dsr = scenario.get('drone_spl_db_range') if isinstance(scenario, dict) else None
                                    if isinstance(dsr, (list, tuple)) and len(dsr) >= 2:
                                        low = float(dsr[0]); high = float(dsr[1])
                                        drone_spl_db = float(pick_rng.uniform(low, high))
                                        local_meta['drone_spl_db'] = drone_spl_db
                                except Exception:
                                    pass
                                # pass requested distance from scenario if available
                                try:
                                    dlist = scenario.get('drones') if isinstance(scenario, dict) else None
                                    requested_distance = None
                                    if isinstance(dlist, (list, tuple)) and len(dlist) > 0:
                                        # prefer element-specific value when present
                                        if di < len(dlist):
                                            val = dlist[di]
                                            if isinstance(val, (list, tuple)) and len(val) >= 2:
                                                low = float(val[0]); high = float(val[1])
                                                requested_distance = float(pick_rng.uniform(low, high))
                                            else:
                                                requested_distance = float(val)
                                        else:
                                            # fallback: treat list as a range if it contains two numeric entries
                                            try:
                                                if len(dlist) >= 2 and all(isinstance(x, (int, float)) for x in dlist):
                                                    low = float(min(dlist)); high = float(max(dlist))
                                                    requested_distance = float(pick_rng.uniform(low, high))
                                            except Exception:
                                                requested_distance = None
                                    if requested_distance is not None:
                                        local_meta['requested_distance_m'] = float(requested_distance)
                                except Exception:
                                    pass
                                processed_drone, ddbg = _distance_apply(drone_audio, task.get('sr'), pick_rng, local_meta, task.get('cfg', {}))
                            except Exception:
                                processed_drone = drone_audio
                                ddbg = {}
                            drone_parts.append((processed_drone, 1.0))
                            entry = {'source': Path(drone_path).name}
                            if isinstance(ddbg, dict):
                                entry.update(ddbg)
                            # record sampled drone_spl_db if provided
                            try:
                                if isinstance(local_meta, dict) and 'drone_spl_db' in local_meta:
                                    entry['drone_spl_db'] = float(local_meta['drone_spl_db'])
                            except Exception:
                                pass
                            drone_debug.append(entry)
                        except Exception:
                            continue

                    # ambients (additional to any base_audio)
                    for ai, _ in enumerate(scenario.get('ambients', []) if isinstance(scenario.get('ambients', []), list) else []):
                        pool_amb = offline_pool_label.get(0, [])
                        if not pool_amb:
                            break
                        try:
                            pick_rng = _rng_for_key(globals().get('master_seed', 0), task.get('seed_key', '') + f':ambient_pick:{ai}')
                            amb_path = str(pick_rng.choice(pool_amb))
                            amb_audio = _load_audio_file(amb_path, task.get('sr'))
                            if amb_audio is None:
                                continue
                            # sample ambient gain delta from scenario if present
                            amb_gain_db = None
                            try:
                                agr = scenario.get('ambient_spl_db_range') if isinstance(scenario, dict) else None
                                if isinstance(agr, (list, tuple)) and len(agr) >= 2:
                                    low = float(agr[0]); high = float(agr[1])
                                    amb_gain_db = float(pick_rng.uniform(low, high))
                            except Exception:
                                amb_gain_db = None
                            if amb_gain_db is None:
                                lin_gain = 1.0
                            else:
                                lin_gain = float(10.0 ** (amb_gain_db / 20.0))
                                max_gain_linear = float(task.get('cfg', {}).get('max_gain_linear', 50.0) if isinstance(task.get('cfg', {}), dict) else 50.0)
                                if lin_gain > max_gain_linear:
                                    lin_gain = max_gain_linear
                            mix_sources.append({'audio': amb_audio, 'gain': lin_gain})
                            ad = {'source': Path(amb_path).name}
                            if amb_gain_db is not None:
                                ad['ambient_spl_db_set'] = float(amb_gain_db)
                                ad['ambient_source_gain_db'] = float(amb_gain_db)
                            ambient_debug.append(ad)
                        except Exception:
                            continue

                    # scenario-level SNR override
                    if scenario.get('snr_db') is not None:
                        scenario_snr = scenario.get('snr_db')
            except Exception:
                pass

            if scenario_snr is not None and base_audio is not None and drone_parts:
                # find the strongest drone (max rms * gain)
                strongest = None
                strongest_power = -1.0
                for (a, g), dbg in zip(drone_parts, drone_debug):
                    r = float(_mix_mod._rms(a)) * float(g) if a is not None else 0.0
                    if r > strongest_power:
                        strongest_power = r
                        strongest = (a, g)
                ref_audio = strongest[0] if strongest is not None else base_audio
                # ask mix.apply to scale ambients to meet target SNR relative to ref_audio
                mix_meta['target_snr_db'] = float(scenario_snr)
                # ensure mix.apply knows which ambient sources to mix/scale
                mix_meta['mix_sources'] = mix_sources
                processed_ref_sum, mix_debug = _mix_mod.apply(ref_audio, task['sr'], rng, mix_meta, task['cfg'])
                # extract scaled ambient (processed_ref_sum - ref_audio)
                import numpy as _np
                def _ensure_len_local(arr, L):
                    if arr is None:
                        return _np.zeros(L, dtype=float)
                    a = arr.astype(float)
                    if a.size == L:
                        return a
                    if a.size > L:
                        return a[:L]
                    return _np.pad(a, (0, L - a.size))
                L = 0
                if ref_audio is not None:
                    L = max(L, int(ref_audio.size))
                if processed_ref_sum is not None:
                    L = max(L, int(processed_ref_sum.size))
                if L == 0:
                    ambient_scaled = _np.zeros(int(task['sr'] * task['duration']), dtype=float)
                else:
                    ref = _ensure_len_local(ref_audio, L)
                    scaled_total = _ensure_len_local(processed_ref_sum, L)
                    ambient_scaled = scaled_total - ref
                # now add ambient_scaled to full drone mix (base_audio)
                if base_audio is None:
                    processed_audio = ambient_scaled
                else:
                    ba = base_audio.astype(float) if base_audio is not None else _np.zeros(0, dtype=float)
                    L2 = max(ba.size, ambient_scaled.size)
                    if ba.size < L2:
                        ba = _np.pad(ba, (0, L2 - ba.size))
                    if ambient_scaled.size < L2:
                        ambient_scaled = _np.pad(ambient_scaled, (0, L2 - ambient_scaled.size))
                    processed_audio = ba + ambient_scaled
            else:
                # ensure mix.apply receives the assembled ambient sources
                mix_meta['mix_sources'] = mix_sources
                processed_audio, mix_debug = _mix_mod.apply(base_audio, task['sr'], rng, mix_meta, task['cfg'])
        except Exception as e:
            # fallback: naive sum
            import numpy as _np
            processed_audio = base_audio.copy() if base_audio is not None else _np.zeros(int(task['sr'] * task['duration']), dtype=float)
            for m in mix_sources:
                a = m.get('audio')
                g = float(m.get('gain', 1.0))
                if a is None:
                    continue
                aa = a.astype(float)
                if aa.size < processed_audio.size:
                    aa = _np.pad(aa, (0, processed_audio.size - aa.size))
                processed_audio += aa * g
            mix_debug = {'fallback_mix_error': str(e)}

        # final normalize -- REMOVED: preserve raw post-mix levels (keeps attenuation/gains intact)
        try:
            # Normalization intentionally skipped to preserve applied attenuation and relative levels.
            nd = {}
            # apply post-mix reverb if requested by scenario
            try:
                rev_cfg = None
                if isinstance(scenario, dict) and scenario.get('reverb') is not None:
                    rev_cfg = scenario.get('reverb')
                elif isinstance(task.get('cfg', {}), dict) and task['cfg'].get('reverb') is not None:
                    rev_cfg = task['cfg'].get('reverb')
                if rev_cfg is not None and rev_cfg.get('mode', 'post') == 'post':
                    mix_meta_rev = dict(task.get('train_meta', {}))
                    mix_meta_rev['reverb'] = rev_cfg
                    processed_audio, rev_debug = _reverb_mod.apply(processed_audio, task['sr'], rng, mix_meta_rev, task['cfg'])
                    # merge reverb debug into mix_debug
                    if isinstance(mix_debug, dict):
                        mix_debug['reverb_debug'] = rev_debug
                    else:
                        mix_debug = {'reverb_debug': rev_debug}
            except Exception:
                pass
            # collate debug
            debug = {'drone_parts': drone_debug, 'ambient_parts': ambient_debug}
            debug.update(mix_debug if isinstance(mix_debug, dict) else {})

            # Ensure each drone debug entry records the requested distance explicitly
            try:
                for d in debug.get('drone_parts', []):
                    # prefer explicit requested_distance_m if present, else fall back to sampled distance_m
                    rd = None
                    try:
                        rd = d.get('requested_distance_m', None)
                    except Exception:
                        rd = None
                    if rd is None:
                        try:
                            rd = d.get('distance_m', None)
                        except Exception:
                            rd = None
                    d['requested_distance_m'] = (float(rd) if rd is not None else None)
            except Exception:
                pass

            # Summarize applied augmentations heuristically for provenance
            try:
                appl = []
                # attenuation/distance
                if any((isinstance(d, dict) and (d.get('atten_dB') is not None or d.get('distance_m') is not None)) for d in debug.get('drone_parts', [])):
                    appl.append('attenuate')
                # mixing / snr scaling
                if isinstance(debug.get('snr_scale', None), (int, float)) or isinstance(debug.get('target_snr_db', None), (int, float)):
                    appl.append('snr_scaling')
                # reverb
                if debug.get('reverb_debug') is not None:
                    appl.append('reverb')
                # normalization
                if isinstance(nd, dict) and nd.get('normalized_gain') is not None:
                    appl.append('normalize')
                # record list (dedup)
                debug['applied_augmentations'] = list(dict.fromkeys(appl))
            except Exception:
                debug['applied_augmentations'] = []
            # compute resulting SNR (drone vs rest) when possible and attach to debug
            try:
                import numpy as _np, math as _math
                # sum drone parts into one reference signal (align lengths)
                if drone_parts:
                    # determine target length
                    L = int(processed_audio.size) if hasattr(processed_audio, 'size') else len(processed_audio)
                    def _ensure_len_local(arr, L):
                        if arr is None:
                            return _np.zeros(L, dtype=float)
                        a = _np.asarray(arr).astype(float)
                        if a.size == L:
                            return a
                        if a.size > L:
                            return a[:L]
                        return _np.pad(a, (0, L - a.size))
                    drone_sum = _np.zeros(L, dtype=float)
                    for (darr, g) in drone_parts:
                        try:
                            darr_a = _ensure_len_local(darr, L)
                            drone_sum += darr_a * float(g)
                        except Exception:
                            continue
                    # compute noise as processed_audio - drone_sum
                    proc = _ensure_len_local(processed_audio, L)
                    # account for post-normalization gain applied to `processed_audio`
                    try:
                        normalized_gain = float(nd.get('normalized_gain', 1.0))
                    except Exception:
                        normalized_gain = 1.0
                    # scale the drone reference by the same normalization gain
                    try:
                        drone_sum = drone_sum * normalized_gain
                    except Exception:
                        pass
                    # indicate whether any drone parts were present for this sample
                    try:
                        debug['drone_present'] = len(drone_parts) > 0
                    except Exception:
                        debug['drone_present'] = False
                    noise = proc - drone_sum
                    # Only compute resulting SNR when at least one drone part was present
                    try:
                        drone_present = bool(debug.get('drone_present'))
                    except Exception:
                        drone_present = False
                    if not drone_present:
                        resulting_snr = None
                    else:
                        drone_power = float(_np.mean(drone_sum**2)) if drone_sum.size > 0 else 0.0
                        noise_power = float(_np.mean(noise**2)) if noise.size > 0 else 0.0
                        if noise_power <= 0 and drone_power <= 0:
                            resulting_snr = None
                        elif noise_power <= 0:
                            resulting_snr = float('inf')
                        else:
                            resulting_snr = 10.0 * _math.log10(max(drone_power, 1e-20) / max(noise_power, 1e-20))
                    if resulting_snr is not None:
                        debug['resulting_snr_db'] = float(resulting_snr)
            except Exception:
                pass
        except Exception as e:
            debug = {'error': str(e)}

        # Attempt to write output files from the worker process to avoid
        # transferring large numpy arrays back to the master via multiprocessing.
        try:
            from .writer import SplitWriter as _SplitWriter
            _out = globals().get('out', None)
            if _out is not None:
                w = _SplitWriter(_out / task['split'])
                # write wav and append metadata atomically from worker
                try:
                    w.write_wav(str(task['label']), task['filename'], processed_audio, task['sr'])
                    debug_meta = {'master_seed': globals().get('master_seed')}
                    debug_meta.update(debug if isinstance(debug, dict) else {})
                    if task.get('offline_src'):
                        try:
                            from pathlib import Path as _P
                            debug_meta['offline_source'] = _P(task.get('offline_src')).name
                        except Exception:
                            debug_meta['offline_source'] = task.get('offline_src')
                    w.append_jsonl(task.get('train_meta', {}), debug_meta)
                    return {
                        'written': True,
                        'filename': task['filename'],
                        'split': task['split'],
                        'label': task['label'],
                        'train_meta': task.get('train_meta', {}),
                        'debug_meta': debug_meta,
                    }
                except Exception:
                    # fall through to returning in-memory result if worker write fails
                    pass
        except Exception:
            pass

        # Fallback: return small result without huge arrays so master can write it
        return {
            'filename': task['filename'],
            'split': task['split'],
            'label': task['label'],
            'processed_audio': processed_audio,
            'train_meta': task['train_meta'],
            'debug': debug,
            'offline_source': task.get('offline_src'),
        }
    except Exception as e:
        # Return a compact task summary to avoid huge log dumps when tasks contain large lists
        summary = None
        try:
            summary = {
                'filename': task.get('filename'),
                'split': task.get('split'),
                'label': task.get('label'),
                'seed_key': task.get('seed_key'),
            }
        except Exception:
            summary = None
        return {'error': str(e), 'task_summary': summary}


def _sha256(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open('rb') as fh:
        while True:
            b = fh.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_build(build_config_path: str, out_dir: str, dry_run=False, show_progress=False, total=None, seed=None, num_workers=None, quiet=False):
    cfg_path = Path(build_config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    validate_build_config(cfg_path, cfg)

    # configure logging according to quiet flag
    if quiet:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    info_name = cfg.get('output', {}).get('info_filename', 'augmentation_samples.jsonl')

    # helper: robust write to handle PermissionError (try chmod+retry)
    def _safe_write_text(path_obj, text, encoding='utf-8'):
        try:
            path_obj.write_text(text, encoding=encoding)
            return
        except PermissionError:
            try:
                import os, stat
                os.chmod(path_obj, stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            except Exception:
                pass
            try:
                path_obj.write_text(text, encoding=encoding)
                return
            except Exception as e:
                raise

    # prepare split dirs and meta files
    splits = ('dataset_train', 'dataset_val', 'dataset_test')
    for s in splits:
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        meta = d / info_name
        # Always truncate the meta file so runs are idempotent and overwrite previous
        _safe_write_text(meta, '', encoding='utf-8')

    # determine totals (use total_samples and split_ratio)
    # allow caller override via `total` argument
    total_samples = int(cfg.get('output', {}).get('total_samples', 20))
    if total is not None:
        try:
            total_samples = int(total)
        except Exception:
            pass
    split_ratio = cfg.get('output', {}).get('split_ratio', [0.8, 0.1, 0.1])

    def _counts(total):
        train = int(total * split_ratio[0])
        val = int(total * split_ratio[1])
        test = int(total * split_ratio[2])
        rem = total - (train + val + test)
        train += rem
        return (train, val, test)

    per_split_counts = _counts(total_samples)

    master_seed = int(cfg.get('advanced', {}).get('random_seed', 0))
    sr = int(cfg.get('audio_parameters', {}).get('sample_rate', 22050))
    duration = float(cfg.get('audio_parameters', {}).get('target_duration_sec', cfg.get('audio_parameters', {}).get('duration_s', 4.0)))
    class_props = cfg.get('class_proportions', {'drone': 0.5, 'no_drone': 0.5})

    # expose key variables as module-level globals so worker `process_task` can access them
    # this relies on fork semantics on Linux; it's acceptable for this project
    globals()['cfg'] = cfg
    globals()['splits'] = splits
    globals()['master_seed'] = master_seed
    globals()['sr'] = sr
    globals()['duration'] = duration
    globals()['out'] = out

    # Generate deterministic allocations and build the exact task list from them.
    try:
        from .compute_allocations import compute_allocations as _compute_allocations
        alloc_path = _compute_allocations(cfg_path, None, total_override=total_samples)
        allocations = json.loads(Path(alloc_path).read_text(encoding='utf-8'))
    except Exception as e:
        raise SystemExit(f"Failed to compute/load allocations.json: {e}")

    # validate allocations splits match our expected splits
    meta = allocations.get('meta', {})
    alloc_splits = meta.get('splits') if isinstance(meta, dict) else None
    if alloc_splits is None or list(alloc_splits) != list(splits):
        hint = (
            f"allocations.json splits {alloc_splits} do not match expected splits {list(splits)}\n"
            f"allocations file: {alloc_path}\n"
            "Possible fixes:\n"
            "  - Add explicit split names in your build config under `output.splits`, e.g.\n"
            "      \"output\": { \"splits\": [\"dataset_train\", \"dataset_val\", \"dataset_test\"] }\n"
            "  - Or regenerate `allocations.json` by running `python augment/compute_allocations.py --config build_config.json --total 1000`\n"
            "The generator requires deterministic allocations that use the canonical dataset split names.\n"
        )
        raise SystemExit(hint)

    # Map expected output split names to allocation keys.
    # We enforce canonical split names in allocations.json, so this is an identity mapping.
    expected_to_alloc = {s: s for s in splits}

    # consume allocations to create deterministic tasks: per-split, per-scenario counts
    alloc_scenarios = allocations.get('scenarios', {})
    scenarios = cfg.get('augmentation_scenarios', []) if isinstance(cfg, dict) else []
    scenario_names = [ (s.get('name') if isinstance(s, dict) else None) or (s.get('id') if isinstance(s, dict) else None) or f"scn_{i}" for i, s in enumerate(scenarios) ]

    global_idx = 0
    tasks = []
    offline_dir = out / 'dataset_DADS_offline'
    offline_files_all = []
    offline_pool_label = {0: [], 1: []}
    if offline_dir.exists():
        for ext in ('*.wav', '*.flac', '*.ogg'):
            offline_files_all.extend(list(offline_dir.rglob(ext)))
        for lbl in (0, 1):
            lbl_dir = offline_dir / str(lbl)
            if lbl_dir.exists():
                for ext in ('*.wav', '*.flac', '*.ogg'):
                    offline_pool_label[lbl].extend([str(p) for p in lbl_dir.rglob(ext)])
            else:
                offline_pool_label[lbl].extend([str(p) for p in offline_files_all if f"_{lbl}_" in p.name])

    for split_name in splits:
        for i, scn in enumerate(scenarios):
            scn_name = scenario_names[i]
            sc_entry = alloc_scenarios.get(scn_name) or alloc_scenarios.get(str(i))
            if not sc_entry:
                continue
            # look up allocation count using mapped allocation split name
            alloc_split_name = expected_to_alloc.get(split_name)
            try:
                count = int(sc_entry.get('per_split', {}).get(alloc_split_name, 0))
            except Exception:
                count = 0
            has_drone = bool(scn.get('drones')) if isinstance(scn, dict) else False
            class_label = 1 if has_drone else 0
            class_name = 'drone' if has_drone else 'no_drone'
            for j in range(count):
                seed_key_full = f"{class_name}:{split_name}:{global_idx}"
                filename = f"dads_sample_{global_idx:06d}.wav"
                train_meta = {
                    'filename': str(Path(str(class_label)) / filename),
                    'label': class_label,
                    'class_name': class_name,
                    'seed_key': seed_key_full,
                    'split': split_name,
                }
                # Include deterministic scenario identifier in train_meta so downstream
                # tools (testers/visualizers) can consume `metadata.scenario` directly.
                # Use the scenario name if available, otherwise fall back to an index-based id.
                try:
                    scn_name = scn.get('name') if isinstance(scn, dict) and scn.get('name') else None
                except Exception:
                    scn_name = None
                train_meta['scenario'] = scn_name or f'scn_{i}'
                src = None
                pool_label = offline_pool_label.get(class_label, [])
                if pool_label:
                    try:
                        rng_pick = rng_for_key(master_seed, seed_key_full + ':pick')
                        src = str(rng_pick.choice(pool_label))
                    except Exception:
                        src = None

                tasks.append({
                    'master_seed': master_seed,
                    'seed_key': seed_key_full,
                    'label': class_label,
                    'class_name': class_name,
                    'filename': filename,
                    'split': split_name,
                    'offline_src': src,
                    'scenario': scn,
                    'train_meta': train_meta,
                    'sr': sr,
                    'duration': duration,
                    'cfg': cfg,
                })
                global_idx += 1

    # prepare writers for main process
    writers = {s: SplitWriter(out / s, meta_name=info_name) for s in splits}

    # worker function runs as top-level `process_task` so it's pickleable by multiprocessing

    # choose number of workers (allow caller override via `num_workers` arg)
    if num_workers is None:
        num_workers = int(cfg.get('advanced', {}).get('num_workers', max(1, (os.cpu_count() or 1))))
    else:
        try:
            num_workers = int(num_workers)
        except Exception:
            num_workers = int(cfg.get('advanced', {}).get('num_workers', max(1, (os.cpu_count() or 1))))

    # main process writes outputs and metadata
    total_tasks = len(tasks)
    start_ts = time.time()
    written = 0
    errors = 0
    written_per_split = {}

    # helper to format ETA
    def _format_eta(remaining, rate):
        if rate <= 0:
            return '??:??:??'
        secs = int(remaining / rate)
        return time.strftime('%H:%M:%S', time.gmtime(secs))

    # progress reporting thresholds (every 5%)
    pct_step = 5
    next_pct = pct_step

    # create iterator of results so we can stream and report progress
    iterator = None
    pool = None
    try:
        if num_workers <= 1:
            iterator = (process_task(t) for t in tasks)
        else:
            pool = mp.Pool(processes=num_workers)
            iterator = pool.imap_unordered(process_task, tasks)

        for r in iterator:
            # handle interruptible cancellation gracefully
            if r is None:
                continue
            if 'error' in r:
                errors += 1
                try:
                    err = r.get('error')
                    # prefer compact summary produced by worker
                    task_info = r.get('task_summary') or (r.get('task') if isinstance(r.get('task'), dict) else {})
                    # ensure task_info is compact for logging
                    if isinstance(task_info, dict):
                        task_summary = {
                            'filename': task_info.get('filename'),
                            'split': task_info.get('split'),
                            'label': task_info.get('label'),
                            'seed_key': task_info.get('seed_key'),
                        }
                    else:
                        task_summary = None
                    seed = task_summary.get('seed_key') if isinstance(task_summary, dict) else None
                    logging.error("[ERROR] %s -- seed=%s -- task=%s", err, seed, task_summary)
                except Exception:
                    pass
                continue
            # If the worker already wrote the file, it will return {'written': True}
            if r.get('written'):
                written += 1
                split = r.get('split')
                written_per_split[split] = written_per_split.get(split, 0) + 1
            else:
                split = r['split']
                writer = writers.get(split)
                try:
                    writer.write_wav(str(r['label']), r['filename'], r['processed_audio'], sr)
                    debug_meta = {'master_seed': master_seed}
                    debug_meta.update(r.get('debug', {}) if isinstance(r.get('debug', {}), dict) else {})
                    if r.get('offline_source'):
                        try:
                            from pathlib import Path as _P
                            debug_meta['offline_source'] = _P(r.get('offline_source')).name
                        except Exception:
                            debug_meta['offline_source'] = r.get('offline_source')
                    writer.append_jsonl(r['train_meta'], debug_meta)
                    written += 1
                    written_per_split[split] = written_per_split.get(split, 0) + 1
                except Exception:
                    errors += 1

            # report progress every pct_step
            pct_done = (written / total_tasks) * 100 if total_tasks > 0 else 100.0
            if pct_done >= next_pct or written == total_tasks:
                elapsed = time.time() - start_ts
                rate = written / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total_tasks - written)
                eta = _format_eta(remaining, rate)
                logging.info("[PROGRESS] %d%% — %d/%d written — ETA %s — %.2f samples/s", int(pct_done), written, total_tasks, eta, rate)
                next_pct += pct_step
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # main process finished writing — summarize results
    build_info = {
        'dataset_id': out.name,
        'generator': 'augment.generator',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'git_sha': None,
        'sha256_build_config': _sha256(cfg_path),
        'return_code': 0,
        'totals_requested': {'total_samples': total_samples},
        'generated': {'total_files': global_idx}
    }

    atomic_write_json(out / 'build_info.json', build_info)
    # human-friendly summary (logged so it can be silenced)
    try:
        logging.info('\n' + '='*72)
        logging.info(' AUGMENTATION BUILD SUMMARY')
        logging.info('='*72)
        logging.info('Output root: %s', out)
        logging.info('Build config: %s', cfg_path)
        logging.info('Requested total samples: %d', total_samples)
        logging.info('Tasks created: %d', total_tasks)
        logging.info('Successfully written: %d', written)
        for s in splits:
            cnt = written_per_split.get(s, 0)
            logging.info('  - %s: %d', s, cnt)
        logging.info('Errors during generation: %d', errors)
        logging.info('\nMetadata files:')
        for s in splits:
            logging.info('  - %s', (out / s / info_name))
        logging.info('Build info: %s', (out / 'build_info.json'))
        aug_report = Path(__file__).resolve().parents[2] / 'tools' / 'augmentation_report.json'
        if aug_report.exists():
            logging.info('Augmentation report: %s', aug_report)
        logging.info('='*72 + '\n')
    except Exception:
        pass

    return 0
