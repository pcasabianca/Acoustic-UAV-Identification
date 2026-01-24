#!/usr/bin/env python3
"""Plugin-based orchestrator for visualization generation.

This module discovers `viz_*.py` plugins under the same `scripts/`
directory and runs them with a shared `VizContext`.
"""
from pathlib import Path
import importlib.util
import logging
from datetime import datetime
from typing import List

import config
from tools.io.perf_index import scan_perf_dir, best_for_model_split
from tools.viz_core import VizContext, VizResult

logger = logging.getLogger(__name__)


def _ensure_run_dirs(base_out: Path):
    # base_out is run directory
    dirs = {
        'perf_vs_snr': base_out / 'perf_vs_snr',
        'training': base_out / 'training',
        'summary': base_out / 'summary',
        'manifests': base_out / 'manifests'
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _discover_plugins(scripts_dir: Path) -> List[object]:
    plugins = []
    for p in scripts_dir.iterdir():
        if not p.is_file():
            continue
        if p.name == 'viz_all.py':
            continue
        if p.name.startswith('viz_') and p.suffix == '.py':
            spec = importlib.util.spec_from_file_location(p.stem, str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # require run function
            if hasattr(mod, 'VIZ_NAME') and hasattr(mod, 'run'):
                plugins.append(mod)
    # sort by VIZ_ORDER if present
    plugins.sort(key=lambda m: getattr(m, 'VIZ_ORDER', 100))
    return plugins


def run(models: List[str] = None, splits=('val','test'), perf_dir=None, results_dir=None, out_dir=None,
        thresholds_path=None, snr_bins=None, metric='accuracy_drone', strict=False, logger=None):
    """Run orchestration: write summary and call plugins.

    Returns a dict similar to previous API for compatibility.
    """
    perf_dir = Path(perf_dir or config.PERFORMANCE_DIR)
    results_dir = Path(results_dir or config.RESULTS_DIR)

    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    if out_dir:
        base_out = Path(out_dir)
    else:
        base_out = Path(config.VIZ_OUTPUT_DIR) / f'run_{run_id}'
    dirs = _ensure_run_dirs(base_out)

    perfs = scan_perf_dir(perf_dir, recursive=True, strict=False)

    models_list = [m.upper() for m in (models or [])]
    if not models_list:
        models_list = sorted({perf.get('metadata', {}).get('model') for perf in perfs if perf.get('metadata', {}).get('model')})

    # write a basic summary perf_index.csv and best_models JSON so plugins can rely on it
    selected = []
    from collections import defaultdict
    import csv, json
    for model in models_list:
        for split in splits:
            perf = best_for_model_split(perfs, model, split=split, metric=metric) or best_for_model_split(perfs, model, split=split, metric='f1_score')
            if perf:
                selected.append(perf)
            else:
                msg = f'No perf found for {model}/{split}'
                if strict:
                    raise RuntimeError(msg)
                (logger or logging.getLogger()).warning(msg)

    # write summary
    csv_path = dirs['summary'] / 'perf_index.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['model', 'split', 'threshold', 'tp', 'tn', 'fp', 'fn'])
        for perf in selected:
            md = perf.get('metadata', {})
            gm = perf.get('global_metrics', {})
            w.writerow([md.get('model'), md.get('split'), md.get('threshold'), gm.get('tp'), gm.get('tn'), gm.get('fp'), gm.get('fn')])

    artifacts = {'perf_vs_snr': [], 'training': [], 'summary': [str(csv_path)], 'heatmaps': [], 'thresholds': [], 'pr_roc': [], 'benchmark': []}

    # prepare VizContext
    ctx = VizContext(
        run_id=run_id,
        repo_root=Path.cwd(),
        perf_dir=perf_dir,
        results_dir=results_dir,
        out_dir=base_out,
        thresholds_path=Path(thresholds_path) if thresholds_path else None,
        perfs=perfs,
        models=models_list,
        splits=list(splits),
        strict=bool(strict),
        logger=logger or logging.getLogger('viz')
    )

    # discover and run plugins
    scripts_dir = Path(__file__).resolve().parent
    plugins = _discover_plugins(scripts_dir)
    for mod in plugins:
        try:
            res: VizResult = mod.run(ctx, args=None)
            if res and isinstance(res, VizResult):
                for p in res.artifacts:
                    sp = str(p)
                    if 'perf_vs_snr' in sp:
                        artifacts['perf_vs_snr'].append(sp)
                    elif '/heatmaps/' in sp or 'heatmap__' in sp:
                        artifacts['heatmaps'].append(sp)
                    elif '/thresholds/' in sp or 'thresholds__' in sp:
                        artifacts['thresholds'].append(sp)
                    elif '/pr_roc/' in sp or 'pr_roc__' in sp:
                        artifacts['pr_roc'].append(sp)
                    elif '/training/' in sp or 'training__' in sp:
                        artifacts['training'].append(sp)
                    elif '/benchmark/' in sp or 'confusions' in sp:
                        artifacts['benchmark'].append(sp)
                    else:
                        # fallback: add to summary
                        artifacts['summary'].append(sp)
        except Exception as e:
            logger and logger.warning('Plugin %s failed: %s', getattr(mod, 'VIZ_NAME', mod.__name__), e)

    return {'artifacts': artifacts, 'selected_count': len(selected), 'out_dir': str(base_out)}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--perf-dir', default=str(config.PERFORMANCE_DIR))
    p.add_argument('--results-dir', default=str(config.RESULTS_DIR))
    p.add_argument('--out-dir', default=None)
    p.add_argument('--models', nargs='*', default=None)
    args = p.parse_args()
    res = run(models=args.models, perf_dir=args.perf_dir, results_dir=args.results_dir, out_dir=args.out_dir)
    print(res)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--perf-dir', default=config.PERFORMANCE_DIR)
    p.add_argument('--results-dir', default=config.RESULTS_DIR)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--models', nargs='*', default=None)
    args = p.parse_args()
    res = run(models=args.models, perf_dir=args.perf_dir, results_dir=args.results_dir, out_dir=args.out_dir)
    print(res)
