import json
import random
from pathlib import Path
from .utils import stable_int_seed
from .template_resolver import instantiate_chain


def _distribute_counts(n_total, weights):
    total_w = sum(weights)
    floats = [n_total * (w / total_w) for w in weights]
    floors = [int(x) for x in floats]
    rem = n_total - sum(floors)
    # distribute remainder by largest fractional part
    fracs = [(i, floats[i] - floors[i]) for i in range(len(floors))]
    fracs.sort(key=lambda x: x[1], reverse=True)
    for i in range(rem):
        idx = fracs[i][0]
        floors[idx] += 1
    return floors


def pick_sources_for_scenario(rng, pools, components):
    out = []
    for c in components:
        role = c["role"]
        count = c.get("count", 1)
        pool = list(pools.get(role, []))
        if len(pool) < count:
            raise ValueError(f"Not enough sources for role {role}")
        # sample without replacement inside a single task
        out_paths = rng.sample(pool, count)
        for p in out_paths:
            out.append({"role": role, "source_path": p})
    return out


def build_tasks(cfg, source_split, out_dir):
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    master = int(cfg.get("master_seed", 0))
    scenarios = cfg["scenarios"]

    # Determine per-split sample counts
    out_cfg = cfg.get("output", {})
    if "total_samples" in out_cfg and "split_ratio" in out_cfg:
        total = int(out_cfg["total_samples"])
        ratios = out_cfg["split_ratio"]
        # expect ratios in order [train, val, test]
        split_names = ["train", "val", "test"]
        counts = [int(total * r) for r in ratios]
        # adjust any rounding difference
        diff = total - sum(counts)
        for i in range(diff):
            counts[i % len(counts)] += 1
        splits = dict(zip(split_names, counts))
    else:
        # fall back to explicit per-split n_samples
        splits = {}
        for k, v in out_cfg.get("splits", {}).items():
            splits[k] = int(v.get("n_samples", 0))

    allocations = {k: {"n_samples": v} for k, v in splits.items()}

    # Precompute scenario weights separated by label (support targets quotas)
    label0_scenarios = [s for s in scenarios if int(s.get("label", 0)) == 0]
    label1_scenarios = [s for s in scenarios if int(s.get("label", 0)) == 1]
    w0 = [s.get("weight", 1.0) for s in label0_scenarios]
    w1 = [s.get("weight", 1.0) for s in label1_scenarios]

    # load source_index to map path->hash if available
    src_index_path = out_dir.parent / "source_index.json"
    path2hash = {}
    if src_index_path.exists():
        idx = json.loads(src_index_path.read_text())
        for role, items in idx.items():
            for it in items:
                path2hash[it["path"]] = it.get("hash")

    # prepare audio target samples if available
    audio_cfg = cfg.get("audio", {})
    sr = int(audio_cfg.get("sample_rate", 22050))
    dur_s = float(audio_cfg.get("duration_s", 4.0))
    target_samples = int(sr * dur_s)

    for split_name, n in splits.items():
        tasks_path = out_dir.parent / f"tasks_{split_name}.jsonl"
        rng = random.Random(stable_int_seed(f"{master}_{split_name}"))
        pools = {k: v.get(split_name, []) for k, v in source_split.items()}

        # Determine targets quotas if present
        targets = cfg.get("targets", {})
        label0_ratio = float(targets.get("label0_ratio", 0.0))
        # compute n0,n1 with rounding
        n0 = int(round(n * label0_ratio)) if label0_ratio > 0 else 0
        n1 = n - n0

        # allocate per-scenario counts for label0 and label1 separately
        per_scen = []
        per_scen += _distribute_counts(n0, w0) if label0_scenarios else []
        per_scen += _distribute_counts(n1, w1) if label1_scenarios else []
        # per_scen now corresponds to label0_scenarios followed by label1_scenarios

        with tasks_path.open("w") as fh:
            idx = 0
            # iterate scenarios in same order as original list but pick counts mapped from label groups
            # build mapping from scenario name to count
            scounts = {}
            i = 0
            for s in label0_scenarios:
                scounts[s.get("name")] = per_scen[i] if i < len(per_scen) else 0
                i += 1
            for s in label1_scenarios:
                scounts[s.get("name")] = per_scen[i] if i < len(per_scen) else 0
                i += 1

            for si, sc in enumerate(scenarios):
                count = scounts.get(sc.get("name"), 0)
                for _ in range(count):
                    sample_id = f"{split_name}_{idx:08d}"
                    # per-sample deterministic RNG for source picking and template resolution
                    pick_seed = stable_int_seed(f"{master}:{split_name}:{sample_id}:sources")
                    per_pick_rng = random.Random(pick_seed)
                    components = pick_sources_for_scenario(per_pick_rng, pools, sc["components"])
                    # instantiate per-sample RNG for template resolution (keeps chain deterministic)
                    per_seed = stable_int_seed(f"{master}:{split_name}:{sample_id}:{sc.get('name')}")
                    per_rng = random.Random(per_seed)

                    comps = []
                    for c in components:
                        # component may have a chain_template (preferred) or legacy 'chain'
                        templ = None
                        for ct in sc.get("components", []):
                            if ct.get("role") == c["role"]:
                                templ = ct.get("chain_template") if "chain_template" in ct else ct.get("chain", [])
                                break

                        inst_chain = instantiate_chain(templ or [], per_rng)
                        # ensure trim_or_pad is first
                        trim_op = {"op": "trim_or_pad", "params": {"target_samples": target_samples}}
                        comp_chain = [trim_op] + inst_chain

                        comps.append({
                            "role": c["role"],
                            "source_path": c["source_path"],
                            "source_hash": path2hash.get(c["source_path"]),
                            "chain": comp_chain,
                        })

                    # instantiate global_chain
                    # global chain template support
                    global_templ = sc.get("global_chain_template") if "global_chain_template" in sc else sc.get("global_chain", [])
                    global_chain = instantiate_chain(global_templ or [], per_rng)
                    # defaults: if empty, add normalize_peak and stats
                    if not global_chain:
                        global_chain = [
                            {"op": "normalize_peak", "params": {"target_dbfs": cfg.get("mix", {}).get("normalize_peak_dbfs", -1.0)}},
                            {"op": "stats", "params": {"sr": sr}},
                        ]

                    task = {
                        "sample_id": sample_id,
                        "split": split_name,
                        "label": sc["label"],
                        "scenario": sc["name"],
                        "seed": per_seed,
                        "components": comps,
                        "global_chain": global_chain,
                        "output_relpath": f"{sc['label']}/{sample_id}.wav",
                    }
                    fh.write(json.dumps(task) + "\n")
                    idx += 1

    (out_dir.parent / "allocations.json").write_text(json.dumps(allocations, indent=2))
    return allocations
