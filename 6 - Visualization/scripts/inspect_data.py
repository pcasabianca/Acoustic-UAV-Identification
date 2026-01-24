#!/usr/bin/env python3
import json
import numpy as np
import sys
from pathlib import Path
# add project root so `import config` works when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def load(path):
    p = Path(path)
    if not p.exists():
        print(f"MISSING: {p}")
        return None
    with open(p, 'r') as f:
        return json.load(f)

train = load(config.MEL_TRAIN_DATA_PATH_STR)
val = load(config.MEL_VAL_DATA_PATH_STR)

print('Train present:', train is not None)
print('Val present:', val is not None)

for name, data in (('train', train), ('val', val)):
    if data is None:
        continue
    mels = np.array(data.get('mel', []))
    labels = np.array(data.get('labels', []))
    print(f"--- {name} ---")
    print('mel shape:', mels.shape)
    print('labels shape:', labels.shape)
    # label types
    if labels.size > 0:
        print('labels sample (first 10):', labels.flat[:10])
        # Check if one-hot
        if labels.ndim > 1 and labels.shape[-1] == 2:
            print('Detected one-hot labels; unique rows count:', np.unique(labels.reshape(labels.shape[0], -1), axis=0).shape[0])
            yints = np.argmax(labels, axis=1)
        else:
            # flatten and unique
            try:
                yints = labels.astype(int).reshape(-1)
            except Exception:
                yints = np.array([int(x[0]) if hasattr(x, '__len__') else int(x) for x in labels])
        print('label unique counts:', {int(k): int(v) for k,v in zip(*np.unique(yints, return_counts=True))})
    if mels.size > 0:
        print('mel mean/std:', float(np.mean(mels)), float(np.std(mels)))
        print('min/max:', float(np.min(mels)), float(np.max(mels)))

# Check index overlap if index files exist
train_idx_path = Path(config.MEL_TRAIN_INDEX_PATH_STR)
val_idx_path = Path(config.MEL_VAL_INDEX_PATH_STR)
if train_idx_path.exists() and val_idx_path.exists():
    try:
        with open(train_idx_path, 'r') as f:
            t_idx = json.load(f)
        with open(val_idx_path, 'r') as f:
            v_idx = json.load(f)
        # assume index lists of file ids or paths
        set_t = set(t_idx)
        set_v = set(v_idx)
        overlap = set_t.intersection(set_v)
        print('train index count:', len(set_t), 'val index count:', len(set_v), 'overlap:', len(overlap))
    except Exception as e:
        print('Could not read index files:', e)
else:
    print('Index files missing for overlap check')

# Quick check: are labels balanced per distance if included
if train is not None and 'distance' in train:
    print('Train has distance metadata (unexpected top-level key)')

print('\nDone diagnostics')
