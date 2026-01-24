#!/usr/bin/env python3
"""
Quick diagnostic: inspect precomputed MEL feature JSONs used by trainers.
Prints shapes, label dtypes, unique values and a small sample.
Run: .venv/bin/python tools/inspect_trainers_data.py
"""
import json
import numpy as np
import sys
from pathlib import Path
# ensure project root is on path so config can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

paths = {
    'train': config.MEL_TRAIN_DATA_PATH,
    'val': config.MEL_VAL_DATA_PATH,
    'test': config.MEL_TEST_DATA_PATH,
}

for split, p in paths.items():
    p = Path(p)
    print(f"\n--- {split.upper()} : {p} ---")
    if not p.exists():
        print("MISSING")
        continue
    with open(p, 'r') as f:
        data = json.load(f)
    mels = np.array(data.get('mel', []))
    labels = np.array(data.get('labels', []))
    print(f"mels: shape={mels.shape}, dtype={mels.dtype}")
    print(f"labels: shape={labels.shape}, dtype={labels.dtype}")
    # Inspect label contents
    if labels.size == 0:
        print("no labels")
        continue
    # If labels are nested (one-hot), try to detect
    if labels.ndim > 1:
        print("Labels appear multi-dimensional (maybe one-hot). Sample:")
        print(labels[:5])
        try:
            ints = np.argmax(labels, axis=1)
            print("Converted to ints via argmax: ", np.unique(ints, return_counts=True))
        except Exception as e:
            print("Could not convert to ints via argmax:", e)
    else:
        print("Labels appear 1D integers/floats. Unique counts:")
        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique.tolist(), counts.tolist())))
        print("Sample labels:", labels[:50].tolist()[:50])

    # Print a few sample filenames if present in metadata
    index_path = str(p).replace('.json', '_index.json')
    ip = Path(index_path)
    if ip.exists():
        try:
            with open(ip, 'r') as f:
                idx = json.load(f)
            print(f"Index entries: {len(idx)}. Sample keys:", list(idx.keys())[:5])
        except Exception as e:
            print("Could not read index file:", e)

print("\nDiagnostic completed.")
