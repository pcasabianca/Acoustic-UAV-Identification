#!/usr/bin/env python3
"""Aggregate augmentation JSONL into a short report JSON/CSV.
Usage: python tools/report_augmentation.py <jsonl> [out.json]
Writes summary JSON to stdout and optionally to out.json
"""
import sys
import json
from pathlib import Path
from statistics import mean, median, pstdev

def safe_num(x):
    try:
        return float(x)
    except Exception:
        return None

def hist(values, bins=10):
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mn == mx:
        return [{"range":"%g"%mn, "count": len(values)}]
    step = (mx - mn) / bins
    buckets = [0]*bins
    for v in values:
        if v is None:
            continue
        idx = int((v - mn) / (mx - mn) * bins)
        if idx < 0: idx = 0
        if idx >= bins: idx = bins-1
        buckets[idx] += 1
    out = []
    for i,c in enumerate(buckets):
        lo = mn + i*step
        hi = lo + step
        out.append({"range":"[%g,%g)"%(lo,hi), "count": c})
    return out

def main():
    if len(sys.argv) < 2:
        print('Usage: report_augmentation.py <jsonl> [out.json]')
        return 2
    p = Path(sys.argv[1])
    outp = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if not p.exists():
        print('File not found:', p)
        return 2

    total = 0
    with_env = 0
    with_chain = 0
    snrs = []
    rms_after = []

    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        debug = obj.get('debug_meta') or obj.get('debug')
        if not debug:
            continue
        # env_debug
        drone_parts = debug.get('drone_parts', [])
        ambient_parts = debug.get('ambient_parts', [])
        found_env = False
        for part in (drone_parts + ambient_parts):
            if part and part.get('env_debug'):
                with_env += 1
                found_env = True
                break
        # chain_debug
        if debug.get('chain_debug') and isinstance(debug.get('chain_debug'), list) and len(debug.get('chain_debug'))>0:
            with_chain += 1
        else:
            # nested search
            for part in (drone_parts + ambient_parts):
                ed = part.get('env_debug') if isinstance(part, dict) else None
                if ed and isinstance(ed.get('modifiers'), list):
                    for m in ed.get('modifiers'):
                        d = m.get('debug')
                        if d and d.get('chain_debug') and isinstance(d.get('chain_debug'), list) and len(d.get('chain_debug'))>0:
                            with_chain += 1
                            break
                    else:
                        continue
                    break
        rs = debug.get('resulting_snr_db')
        rsv = safe_num(rs)
        if rsv is not None:
            snrs.append(rsv)
        ra = safe_num(debug.get('rms_after'))
        if ra is not None:
            rms_after.append(ra)

    summary = {
        'total': total,
        'with_env_debug': with_env,
        'with_chain_debug_nonempty': with_chain,
        'snr_count': len(snrs),
        'snr_stats': None,
        'rms_count': len(rms_after),
        'rms_stats': None,
    }
    if snrs:
        summary['snr_stats'] = {
            'min': min(snrs), 'max': max(snrs), 'mean': mean(snrs), 'median': median(snrs), 'stdev': pstdev(snrs) if len(snrs)>1 else 0.0
        }
    if rms_after:
        summary['rms_stats'] = {
            'min': min(rms_after), 'max': max(rms_after), 'mean': mean(rms_after), 'median': median(rms_after), 'stdev': pstdev(rms_after) if len(rms_after)>1 else 0.0,
            'hist': hist(rms_after, bins=10)
        }

    print(json.dumps(summary, indent=2))
    if outp:
        outp.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
