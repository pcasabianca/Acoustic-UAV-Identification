import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def infer_metadata_from_filename(path: Path) -> Dict[str, Any]:
    """Infer `model`, `split`, and `threshold` from filename while preserving
    underscores in model names (e.g., attention_crnn).
    Patterns supported: model_split[_t0.55].json, model__split__t0.55.json
    """
    name = path.stem
    tokens = [t for t in name.replace('__', '_').split('_') if t]
    split_tokens = {'train', 'val', 'test'}
    out: Dict[str, Any] = {}
    split_idx = None
    thr = None
    for i, t in enumerate(tokens):
        low = t.lower()
        if low in split_tokens:
            split_idx = i
            out['split'] = low
            # check next token for threshold like t0.55
            if i + 1 < len(tokens) and tokens[i+1].lower().startswith('t'):
                try:
                    thr = float(tokens[i+1][1:])
                    out['threshold'] = thr
                except Exception:
                    pass
            break
        if low.startswith('t') and len(low) > 1:
            # token itself is threshold and no explicit split found
            try:
                thr = float(low[1:])
                out['threshold'] = thr
            except Exception:
                pass
    # model is everything before the split token (or everything if no split)
    if split_idx is not None:
        model_tokens = tokens[:split_idx]
    else:
        # if last token is threshold strip it
        if tokens and tokens[-1].lower().startswith('t'):
            model_tokens = tokens[:-1]
        else:
            model_tokens = tokens
    if model_tokens:
        out['model'] = '_'.join(model_tokens).upper()
    return out


def recompute_pred_label(prob: float, threshold: float) -> int:
    return int(float(prob) >= float(threshold))


def is_perf_json(raw: dict) -> Tuple[bool, Optional[str]]:
    """Quick heuristic to determine whether a loaded JSON looks like a
    performance JSON. Returns (ok, reason) where reason is None when ok.
    """
    if not isinstance(raw, dict):
        return False, 'not_a_dict'
    if 'predictions' not in raw or not isinstance(raw['predictions'], list):
        return False, 'predictions_missing_or_not_list'
    gm = raw.get('global_metrics') or raw.get('metrics') or {}
    if not isinstance(gm, dict):
        return False, 'global_metrics_missing_or_invalid'
    # ensure first few predictions contain prob and true label
    preds = raw.get('predictions', [])
    for p in preds[:5]:
        if not isinstance(p, dict):
            return False, 'prediction_not_object'
        if 'true_label' not in p:
            return False, 'true_label_missing'
        if 'prob_class1' not in p and 'probability' not in p and 'prob' not in p:
            return False, 'probability_missing'
    return True, None


def load_perf(path: str, strict: bool = True) -> Dict[str, Any]:
    p = Path(path)
    raw = json.loads(p.read_text())

    ok, reason = is_perf_json(raw)
    if not ok:
        if strict:
            raise ValueError(f'Not a performance JSON: {reason}')
        logger.warning('Skipping non-performance JSON %s: %s', str(p), reason)
        raise ValueError(f'Non-performance JSON: {reason}')

    meta = raw.get('metadata', {}) or {}
    inferred = infer_metadata_from_filename(p)
    model = meta.get('model') or inferred.get('model')
    split = meta.get('split') or inferred.get('split')
    if split is None:
        if strict:
            raise ValueError('split_missing_in_metadata_and_filename')
        split = 'val'
    threshold = None
    if 'threshold' in meta:
        threshold = meta.get('threshold')
    elif 'threshold' in inferred:
        threshold = inferred.get('threshold')
    else:
        threshold = 0.5

    preds = raw.get('predictions', [])
    normalized: List[Dict[str, Any]] = []
    missing_probs = 0
    for rec in preds:
        prob = None
        if 'prob_class1' in rec:
            prob = rec.get('prob_class1')
        elif 'probability' in rec:
            prob = rec.get('probability')
        elif 'prob' in rec:
            prob = rec.get('prob')

        if prob is None:
            if strict:
                raise ValueError(f'missing_prob_in_prediction: {rec.get("filename")}')
            missing_probs += 1
            logger.warning('Skipping prediction without prob in %s: %s', str(p), rec.get('filename'))
            continue

        try:
            prob = float(prob)
        except Exception:
            if strict:
                raise ValueError(f'prob_not_numeric: {rec.get("filename")}')
            logger.warning('Non-numeric prob, skipping: %s', rec.get('filename'))
            continue
        if not (0.0 <= prob <= 1.0):
            if strict:
                raise ValueError(f'prob_out_of_range: {rec.get("filename")} -> {prob}')
            logger.warning('prob out of range, skipping: %s -> %s', rec.get('filename'), prob)
            continue

        true = rec.get('true_label')
        if true is None:
            if strict:
                raise ValueError(f'missing_true_label: {rec.get("filename")}')
            logger.warning('Missing true_label, skipping: %s', rec.get('filename'))
            continue
        try:
            true = int(true)
        except Exception:
            if strict:
                raise ValueError(f'true_label_not_int: {rec.get("filename")}')
            logger.warning('true_label not int, skipping: %s', rec.get('filename'))
            continue
        if true not in (0, 1):
            if strict:
                raise ValueError(f'true_label_invalid: {rec.get("filename")} -> {true}')
            logger.warning('Invalid true_label, skipping: %s -> %s', rec.get('filename'), true)
            continue

        # compute pred_label if missing
        if 'pred_label' in rec and rec.get('pred_label') is not None:
            pred_label = int(rec.get('pred_label'))
        else:
            pred_label = recompute_pred_label(prob, threshold)

        md = rec.get('metadata', {}) or {}
        # extract snr and distance if present
        snr = None
        rd = md.get('raw_debug') or {}
        if isinstance(rd, dict):
            snr = rd.get('resulting_snr_db')
        snr = md.get('resulting_snr_db') or snr
        distance = md.get('distance_m') or md.get('distance')

        normalized.append({
            'filename': rec.get('filename'),
            'true_label': true,
            'prob_class1': prob,
            'pred_label': int(pred_label),
            'metadata': md,
            'scenario': rec.get('subcategory') or md.get('scenario'),
            'snr_db': snr,
            'distance_m': distance,
        })

    total = raw.get('global_metrics', {}).get('total')
    if total is None:
        total = raw.get('metadata', {}).get('total_samples')
    if total is None:
        total = len(normalized)

    # strict validation: counts and confusion consistency if present
    if strict:
        if len(normalized) != total:
            raise ValueError(f'predictions_count_mismatch: {len(normalized)} != {total}')
        gm = raw.get('global_metrics', {})
        if all(k in gm for k in ('tp', 'tn', 'fp', 'fn')):
            if (gm['tp'] + gm['tn'] + gm['fp'] + gm['fn']) != total:
                raise ValueError('global_confusion_count_mismatch')

    global_metrics = raw.get('global_metrics', {})

    out: Dict[str, Any] = {
        'metadata': {'model': model, 'split': split, 'threshold': float(threshold), 'total_samples': total},
        'global_metrics': global_metrics,
        'predictions': normalized,
        '_raw': raw
    }

    return out
