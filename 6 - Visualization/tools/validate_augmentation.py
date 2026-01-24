#!/usr/bin/env python3
"""Validate augmentation JSONL for required debug fields and value ranges.
Usage: python tools/validate_augmentation.py <path/to/augmentation_samples.jsonl>
Exits with code 0 if no severe issues, 1 otherwise. Prints a short report.
"""
import sys
import json
from pathlib import Path

def safe_get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d

def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def main():
    if len(sys.argv) < 2:
        print('Usage: validate_augmentation.py <jsonl>')
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print('File not found:', p)
        return 2

    total = 0
    missing_debug = 0
    missing_resulting_snr = 0
    bad_rms = 0
    env_debug_count = 0
    chain_nonempty = 0
    issues = []

    for i, line in enumerate(p.read_text(encoding='utf-8').splitlines()):
        if not line.strip():
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception as e:
            issues.append((i+1, 'json_parse_error', str(e)))
            continue
        debug = obj.get('debug_meta') or obj.get('debug')
        if debug is None:
            missing_debug += 1
            issues.append((i+1, 'missing_debug', 'no debug_meta or debug field'))
            continue
        # resulting_snr_db
        rs = debug.get('resulting_snr_db')
        if rs is None or not is_number(rs):
            missing_resulting_snr += 1
            issues.append((i+1, 'missing_resulting_snr', str(rs)))
        # rms_after
        rms_after = debug.get('rms_after')
        if rms_after is None or not is_number(rms_after):
            issues.append((i+1, 'missing_rms_after', str(rms_after)))
        else:
            try:
                r = float(rms_after)
                if r != r or r < 0 or r > 1e4:
                    bad_rms += 1
                    issues.append((i+1, 'rms_out_of_range', r))
            except Exception:
                issues.append((i+1, 'rms_parse_error', str(rms_after)))
        # env_debug presence
        drone_parts = debug.get('drone_parts', [])
        ambient_parts = debug.get('ambient_parts', [])
        found_env = False
        for part in (drone_parts + ambient_parts):
            if part and part.get('env_debug'):
                env_debug_count += 1
                found_env = True
                break
        # chain_debug presence
        # chain_debug may be nested inside env_debug.modifiers[...] or directly in debug
        if debug.get('chain_debug'):
            if isinstance(debug.get('chain_debug'), list) and len(debug.get('chain_debug'))>0:
                chain_nonempty += 1
        else:
            # search nested
            def find_chain(parts):
                for p in parts:
                    ed = p.get('env_debug') if isinstance(p, dict) else None
                    if ed and isinstance(ed.get('modifiers'), list):
                        for m in ed.get('modifiers'):
                            d = m.get('debug')
                            if d and d.get('chain_debug') and isinstance(d.get('chain_debug'), list) and len(d.get('chain_debug'))>0:
                                return True
                return False
            try:
                if find_chain(drone_parts) or find_chain(ambient_parts):
                    chain_nonempty += 1
            except Exception:
                pass

    print('Validation report for', p)
    print('  total lines:', total)
    print('  missing debug entries:', missing_debug)
    print('  missing resulting_snr_db:', missing_resulting_snr)
    print('  env_debug occurrences (any part):', env_debug_count)
    print('  samples with non-empty chain_debug:', chain_nonempty)
    print('  rms flagged out-of-range:', bad_rms)
    if issues:
        print('\nSample issues (up to 20):')
        for it in issues[:20]:
            print(' ', it)
    exit_code = 0 if (missing_debug==0 and missing_resulting_snr==0 and bad_rms==0) else 1
    return exit_code

if __name__ == '__main__':
    raise SystemExit(main())
