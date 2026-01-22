import json
from pathlib import Path


def merge_split(split, shards_dir, out_metadata_path):
    shards_dir = Path(shards_dir)
    out_metadata_path = Path(out_metadata_path)
    lines = []
    for p in sorted(shards_dir.glob(f"{split}_meta_part_*.jsonl")):
        with p.open() as fh:
            for l in fh:
                try:
                    lines.append(json.loads(l))
                except Exception as e:
                    raise ValueError(f"Invalid JSON in shard {p}: {e}")

    lines = sorted(lines, key=lambda x: x.get("path", ""))
    out_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with out_metadata_path.open("w") as out:
        for item in lines:
            out.write(json.dumps(item) + "\n")
