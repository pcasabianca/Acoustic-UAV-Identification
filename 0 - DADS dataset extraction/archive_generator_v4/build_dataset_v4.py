#!/usr/bin/env python3
"""Minimal wrapper for v4 dataset build (canonical entrypoint).

This wrapper treats `build_config.json` in the same directory as authoritative
and refuses CLI overrides. It delegates to `augment.generator.run_build`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    base = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(description="Wrapper for v4 augmentation builder")
    parser.add_argument('--config', '-c', type=str, default=str(base / 'build_config.json'),
                        help='Path to build_config.json (default: local build_config.json)')
    parser.add_argument('--out-dir', type=str, default=str(base), help='Output directory for datasets')
    parser.add_argument('--total-samples', type=int, default=None, help='Total number of samples to generate')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker processes to use')
    parser.add_argument('--dry-run', action='store_true', help='Do not write files; run validation only')
    parser.add_argument('--show-progress', action='store_true', help='Show progress during build')
    parser.add_argument('--quiet', action='store_true', help='Suppress informational output')

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}")
        return 2

    # Ensure local extraction-root augment package is importable
    sys.path.insert(0, str(base))

    try:
        from augment.generator import run_build
    except Exception as e:
        print("ERROR: failed to import augment.generator:", e)
        return 3

    rc = run_build(str(cfg_path), out_dir=str(args.out_dir), dry_run=args.dry_run,
                   show_progress=args.show_progress, total=args.total_samples,
                   num_workers=args.num_workers, quiet=bool(args.quiet))
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
