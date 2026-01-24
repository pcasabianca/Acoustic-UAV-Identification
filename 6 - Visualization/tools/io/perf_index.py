from pathlib import Path
from typing import List, Optional
from .perf_loader import load_perf
import logging
import math

logger = logging.getLogger(__name__)


def scan_perf_dir(perf_dir: str, recursive: bool = True, strict: bool = False) -> List[dict]:
    p = Path(perf_dir)
    results = []
    scanned = 0
    loaded = 0
    skipped = 0
    pattern = '**/*.json' if recursive else '*.json'
    for f in sorted(p.rglob('*.json') if recursive else p.glob('*.json')):
        scanned += 1
        fname = f.name.lower()
        # skip known non-performance JSONs
        if fname in ('best_results_summary.json', 'calibrated_thresholds.json') or 'training_metadata' in fname:
            logger.debug('Skipping non-perf file: %s', f)
            skipped += 1
            continue
        try:
            perf = load_perf(str(f), strict=strict)
            perf['_path'] = str(f)
            results.append(perf)
            loaded += 1
        except Exception as e:
            logger.warning('Skipping file %s: %s', f, str(e))
            skipped += 1
            continue
    logger.info('Scanned %d files, loaded %d perf files, skipped %d', scanned, loaded, skipped)
    return results


def query(perfs: List[dict], model: Optional[str] = None, split: Optional[str] = None, threshold: Optional[float] = None):
    out = []
    model_up = model.upper() if model else None
    for perf in perfs:
        md = perf.get('metadata', {})
        perf_model = md.get('model')
        if model_up and (perf_model is None or perf_model.upper() != model_up):
            continue
        if split and md.get('split') != split:
            continue
        if threshold is not None:
            try:
                if float(md.get('threshold')) != float(threshold):
                    continue
            except Exception:
                continue
        out.append(perf)
    return out


def best_for_model_split(perfs: List[dict], model: str, split: str = 'val', metric: str = 'f1_score') -> Optional[dict]:
    candidates = query(perfs, model=model, split=split)
    best = None
    best_val = -math.inf
    for perf in candidates:
        gm = perf.get('global_metrics', {})
        val = gm.get(metric)
        if val is None:
            # try class metrics fallback
            val = gm.get('f1') if metric == 'f1_score' else None
        try:
            valf = float(val) if val is not None else -math.inf
        except Exception:
            valf = -math.inf
        if valf > best_val:
            best_val = valf
            best = perf
    return best


def closest_threshold(perfs: List[dict], model: str, split: str, target_threshold: float) -> Optional[dict]:
    candidates = query(perfs, model=model, split=split)
    if not candidates:
        return None
    best = None
    best_diff = float('inf')
    for perf in candidates:
        try:
            t = float(perf.get('metadata', {}).get('threshold'))
            diff = abs(t - float(target_threshold))
            if diff < best_diff:
                best_diff = diff
                best = perf
        except Exception:
            continue
    return best
