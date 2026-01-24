from typing import List, Dict, Any, Tuple
import math


def parse_bins_from_str(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [float(p) for p in parts]


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def filter_drone_only(perf: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [p for p in perf.get('predictions', []) if p.get('true_label') == 1]


def bin_by_snr(preds: List[Dict[str, Any]], bins: List[float]):
    bins_sorted = sorted(bins)
    out = {i: [] for i in range(len(bins_sorted) - 1)}
    for p in preds:
        md = p.get('metadata', {}) or {}
        # nested raw_debug
        snr = None
        rd = md.get('raw_debug') or md.get('raw_debug', {})
        if isinstance(rd, dict):
            snr = rd.get('resulting_snr_db')
        if snr is None:
            snr = md.get('resulting_snr_db') or md.get('snr_db')
        if snr is None:
            continue
        snr = safe_float(snr)
        if snr is None:
            continue
        # find bin
        for i in range(len(bins_sorted) - 1):
            if bins_sorted[i] <= snr < bins_sorted[i+1]:
                out[i].append(p)
                break
    return out


def metric_over_bins(binned: Dict[int, List[Dict[str, Any]]], metric: str = 'recall'):
    # compute simple metric per bin: recall = tp/(tp+fn)
    results = {}
    for i, preds in binned.items():
        tp = sum(1 for p in preds if p.get('true_label') == 1 and p.get('pred_label') == 1)
        fn = sum(1 for p in preds if p.get('true_label') == 1 and p.get('pred_label') == 0)
        tn = sum(1 for p in preds if p.get('true_label') == 0 and p.get('pred_label') == 0)
        fp = sum(1 for p in preds if p.get('true_label') == 0 and p.get('pred_label') == 1)
        total = tp + fn + tn + fp
        if total == 0:
            val = math.nan
        elif metric == 'recall':
            val = tp / (tp + fn) if (tp + fn) > 0 else math.nan
        elif metric == 'precision':
            val = tp / (tp + fp) if (tp + fp) > 0 else math.nan
        elif metric == 'accuracy':
            val = (tp + tn) / total if total > 0 else math.nan
        elif metric == 'mean_prob':
            # mean predicted probability in bin
            probs = [safe_float(p.get('prob_class1')) for p in preds]
            probs = [v for v in probs if v is not None]
            val = sum(probs) / len(probs) if probs else math.nan
        else:
            val = math.nan
        results[i] = {'metric': val, 'counts': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'n': total}}
    return results


def filter_drone_only_with_mean_prob(perf: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float]:
    preds = [p for p in perf.get('predictions', []) if p.get('true_label') == 1]
    probs = [safe_float(p.get('prob_class1')) for p in preds]
    probs = [p for p in probs if p is not None]
    mean_prob = sum(probs) / len(probs) if probs else math.nan
    return preds, mean_prob
