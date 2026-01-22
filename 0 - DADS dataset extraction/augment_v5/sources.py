import json
import os
import random
from pathlib import Path
from .utils import stable_int_seed, distribute_counts


def scan_sources(dads_root, labels):
    base = Path(dads_root)
    out = {}
    for role, sub in labels.items():
        p = base / sub
        files = sorted([str(x) for x in p.glob("*.wav")]) if p.exists() else []
        out[role] = files
    return out


def _hash_path_stat(path):
    st = Path(path).stat()
    import hashlib as _hash

    h = _hash.sha256()
    h.update(str(path).encode())
    h.update(str(st.st_size).encode())
    return "sha256:" + h.hexdigest()


def build_source_index(cfg, out_path):
    dads_root = Path(cfg["sources"]["dads_root"])
    labels = cfg["sources"]["labels"]
    scanned = scan_sources(dads_root, labels)
    index = {}
    for role, files in scanned.items():
        index[role] = []
        for p in files:
            index[role].append({"path": p, "hash": _hash_path_stat(p)})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2))
    return index


def build_source_split(cfg, index, out_path):
    # Use output.split_ratio for source splitting (anti-leakage)
    ratios = cfg.get("output", {}).get("split_ratio")
    if not ratios:
        raise ValueError("output.split_ratio missing in config")
    master = str(cfg.get("master_seed", "0"))
    split = {k: {"train": [], "val": [], "test": []} for k in index}
    for role, items in index.items():
        paths = [i["path"] for i in items]
        rng = random.Random(stable_int_seed(f"{master}_{role}"))
        rng.shuffle(paths)
        n = len(paths)
        counts = distribute_counts(n, ratios)
        split[role]["train"] = paths[: counts[0]]
        split[role]["val"] = paths[counts[0] : counts[0] + counts[1]]
        split[role]["test"] = paths[counts[0] + counts[1] :]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2))
    return split
