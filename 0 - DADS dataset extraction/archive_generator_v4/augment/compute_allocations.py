#!/usr/bin/env python3
"""
Deterministically compute scenario -> split allocations for augmentation builds.

Writes `allocations.json` in this augment folder. Uses floor + largest-remainder
rounding with deterministic tie-breaks based on the master seed.

Usage: python compute_allocations.py --config build_config.json
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def _name_for_scn(scn, idx):
    if isinstance(scn, dict):
        return scn.get('name') or scn.get('id') or f"scn_{idx}"
    return f"scn_{idx}"


def compute_allocations(cfg_path: Path, out_path: Path = None, total_override: int = None):
    cfg_path = Path(cfg_path)
    base = cfg_path.parent
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    scenarios = cfg.get('augmentation_scenarios', [])
    if not scenarios:
        raise SystemExit('No augmentation_scenarios in config')

    # Allow caller to override total_samples (e.g. generator requested N samples)
    if total_override is not None:
        total_samples = int(total_override)
    else:
        total_samples = int(cfg.get('output', {}).get('total_samples', 0))
        if total_samples <= 0:
            raise SystemExit('output.total_samples must be > 0')

    split_ratio = cfg.get('output', {}).get('split_ratio', [0.7, 0.15, 0.15])
    splits = cfg.get('output', {}).get('splits', ['dataset_train', 'dataset_val', 'dataset_test'])
    if len(split_ratio) != len(splits):
        raise SystemExit('split_ratio length must match splits')

    # scenario weights
    weights = [float(s.get('proportion', 1.0)) for s in scenarios]
    total_w = sum(weights)
    if total_w <= 0:
        weights = [1.0 for _ in weights]
        total_w = sum(weights)

    float_targets = [total_samples * (w / total_w) for w in weights]
    floors = [math.floor(x) for x in float_targets]
    remainder = [ft - f for ft, f in zip(float_targets, floors)]
    assigned = sum(floors)
    remaining = total_samples - assigned

    master_seed = int(cfg.get('advanced', {}).get('random_seed', cfg.get('advanced', {}).get('master_seed', 42)))
    rng = np.random.default_rng(master_seed)

    # deterministic tie-breaker rnd for each scenario
    tiebreak = [float(rng.random()) for _ in scenarios]

    order = sorted(range(len(scenarios)), key=lambda i: (-remainder[i], -tiebreak[i]))
    extras = [0] * len(scenarios)
    for i in order[:remaining]:
        extras[i] += 1

    scenario_totals = [floors[i] + extras[i] for i in range(len(scenarios))]

    # now split each scenario total across splits using floor + remainder
    split_ratios = [float(r) for r in split_ratio]
    sum_sr = sum(split_ratios)
    if sum_sr <= 0:
        split_ratios = [1.0 / len(split_ratios) for _ in split_ratios]
    else:
        split_ratios = [r / sum_sr for r in split_ratios]

    scenarios_out = {}
    per_split_totals = {s: 0 for s in splits}

    for idx, scn in enumerate(scenarios):
        scn_name = _name_for_scn(scn, idx)
        sc_total = int(scenario_totals[idx])
        float_allocs = [sc_total * r for r in split_ratios]
        floors_s = [math.floor(x) for x in float_allocs]
        rem_s = [fa - fl for fa, fl in zip(float_allocs, floors_s)]
        remsum = sc_total - sum(floors_s)

        # deterministic tie-breakers per scenario using seeded RNG
        rng2 = np.random.default_rng(master_seed + idx)
        tieb = [float(rng2.random()) for _ in splits]
        order_s = sorted(range(len(splits)), key=lambda i: (-rem_s[i], -tieb[i]))
        extras_s = [0] * len(splits)
        for j in order_s[:remsum]:
            extras_s[j] += 1

        per_split = {splits[j]: floors_s[j] + extras_s[j] for j in range(len(splits))}
        for k, v in per_split.items():
            per_split_totals[k] += v

        scenarios_out[scn_name] = {
            'total': sc_total,
            'per_split': per_split,
            'weight': float(weights[idx]),
        }

    out = {
        'meta': {
            'total_samples': total_samples,
            'splits': splits,
            'split_ratio': split_ratio,
            'master_seed': master_seed,
        },
        'scenarios': scenarios_out,
        'per_split_totals': per_split_totals,
    }

    out_path = Path(out_path) if out_path is not None else (Path(__file__).resolve().parent / 'allocations.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', '-c', default='build_config.json')
    p.add_argument('--total', '-t', type=int, default=None, help='Override total_samples for allocation computation')
    p.add_argument('--out', '-o', default=None)
    args = p.parse_args()
    cfg = Path(args.config)
    if not cfg.exists():
        # try to resolve relative to this folder
        cfg = Path(__file__).resolve().parents[1] / 'build_config.json'
    out = compute_allocations(cfg, args.out, total_override=args.total)
    print(f'Wrote allocations to: {out}')


if __name__ == '__main__':
    main()
