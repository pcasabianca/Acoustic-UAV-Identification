#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from augment_v5 import builder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="build_config.json")
    p.add_argument("--mode", choices=("full", "tasks", "gen", "validate"), default="full")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        raise SystemExit(2)

    builder.run(str(config_path), args.mode, args.jobs, args.clean)


if __name__ == "__main__":
    main()
