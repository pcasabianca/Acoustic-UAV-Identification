import json
import os
from pathlib import Path
import multiprocessing as mp

from . import sources, allocator, worker, merge_metadata, validate


def run(config_path, mode, jobs, clean):
    cfg = json.loads(Path(config_path).read_text())
    base = Path(config_path).resolve().parent
    tmp = base / "tmp"
    tmp.mkdir(exist_ok=True)

    # ensure output dirs
    for s in ("dataset_train", "dataset_val", "dataset_test"):
        (base / s / "0").mkdir(parents=True, exist_ok=True)
        (base / s / "1").mkdir(parents=True, exist_ok=True)

    # Clean outputs if requested (CLI --clean or config)
    # NOTE: don't perform automatic cleaning when running in `validate` mode
    should_clean = False
    if mode in ("full", "tasks", "gen"):
        should_clean = bool(clean) or bool(cfg.get("output", {}).get("clean_before_build", False))
    if should_clean:
        for s in ("dataset_train", "dataset_val", "dataset_test"):
            sd = base / s
            if sd.exists():
                for wav in sd.rglob("*.wav"):
                    try:
                        wav.unlink()
                    except Exception:
                        pass
                m = sd / "metadata.jsonl"
                if m.exists():
                    try:
                        m.unlink()
                    except Exception:
                        pass

    if mode in ("full", "tasks"):
        idx = sources.build_source_index(cfg, tmp / "source_index.json")
        ssplit = sources.build_source_split(cfg, idx, tmp / "source_split.json")
        allocator.build_tasks(cfg, ssplit, tmp / "tasks")

    if mode in ("full", "gen"):
        out_cfg = cfg.get("output", {})
        if "splits" in out_cfg:
            split_names = list(out_cfg["splits"].keys())
        else:
            # default to train/val/test when using total_samples + split_ratio
            split_names = ["train", "val", "test"]
        shards_dir = tmp / "shards"
        # ensure shards dir is clean to avoid merging stale shards
        if shards_dir.exists():
            for f in shards_dir.iterdir():
                try:
                    f.unlink()
                except Exception:
                    pass
        shards_dir.mkdir(parents=True, exist_ok=True)

        for split in split_names:
            tasks_path = tmp / f"tasks_{split}.jsonl"
            if not tasks_path.exists():
                print("No tasks for", split)
                continue
            tasks = [json.loads(l) for l in tasks_path.open()]
            # partition tasks into job chunks
            n = max(1, jobs)
            chunks = [[] for _ in range(n)]
            for i, t in enumerate(tasks):
                chunks[i % n].append(t)

            shards_dir = tmp / "shards"
            shards_dir.mkdir(parents=True, exist_ok=True)
            procs = []
            for i, chunk in enumerate(chunks):
                shard_path = shards_dir / f"{split}_meta_part_{i:03d}.jsonl"
                p = mp.Process(target=worker.process_shard, args=(chunk, base / f"dataset_{split}", shard_path, cfg["audio"], cfg.get("mix", {})))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            # fail-fast: if any worker exited with non-zero, abort before merge
            bad = [p for p in procs if getattr(p, "exitcode", 0) != 0]
            if bad:
                raise RuntimeError(f"One or more worker processes failed for split {split}")

            merge_metadata.merge_split(split, shards_dir, base / f"dataset_{split}" / "metadata.jsonl")

    if mode in ("full", "validate"):
        validate.run_all_splits(base, cfg)
