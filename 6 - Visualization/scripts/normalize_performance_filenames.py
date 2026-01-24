#!/usr/bin/env python3
"""
Normalize performance JSON filenames to canonical names without timestamps.
For each JSON in config.PERFORMANCE_DIR this script will:
- Read metadata.model, metadata.split, metadata.threshold
- Write canonical file: {model_lower}_{split}_t{threshold:.2f}.json (overwriting)
- Remove other files that share the same canonical key (optional: here we remove)
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

perf_dir = Path(config.PERFORMANCE_DIR)
if not perf_dir.exists():
    print(f"Performance dir not found: {perf_dir}")
    sys.exit(1)

files = list(perf_dir.glob('*.json'))
print(f"Found {len(files)} json files in {perf_dir}")

# Group by canonical key
groups = {}
for p in files:
    try:
        with open(p, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skipping unreadable {p.name}: {e}")
        continue

    meta = data.get('metadata', {})
    model = meta.get('model')
    split = meta.get('split')
    threshold = meta.get('threshold')

    if model is None or split is None or threshold is None:
        print(f"Skipping {p.name}: missing metadata fields")
        continue

    key = (model, split, float(threshold))
    # Keep the newest file by mtime
    prev = groups.get(key)
    if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
        groups[key] = p

# Now write canonical files and remove other variants
for (model, split, threshold), chosen in groups.items():
    canonical_name = f"{model.lower()}_{split}.json"
    canonical_path = perf_dir / canonical_name

    try:
        with open(chosen, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read chosen file {chosen.name}: {e}")
        continue

    # Write canonical file (overwrite)
    with open(canonical_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Wrote canonical: {canonical_name}")

# Remove other files that map to same keys (keep the canonical)
for p in files:
    try:
        with open(p, 'r') as f:
            data = json.load(f)
    except Exception:
        continue
    meta = data.get('metadata', {})
    model = meta.get('model')
    split = meta.get('split')
    threshold = meta.get('threshold')
    if model is None or split is None or threshold is None:
        continue
    canonical_name = f"{model.lower()}_{split}.json"
    canonical_path = perf_dir / canonical_name
    if p.resolve() == canonical_path.resolve():
        continue
    # Remove dated file
    try:
        p.unlink()
        print(f"Removed old file: {p.name}")
    except Exception as e:
        print(f"Could not remove {p.name}: {e}")

print("Normalization complete.")
