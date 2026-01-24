#!/usr/bin/env python3
"""
Safe archiving utility for models/results and optional cleanup.

Usage:
  python tools/archive.py --out ./archives --no-delete  # just archive
  python tools/archive.py --out ./archives --confirm     # archive and delete targets

This is a safer replacement for `0 - DADS dataset extraction/archive_and_cleanup.py`.
"""
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as project_config


def gather_items(script_dir: Path):
    items = {
        'saved_models': script_dir / 'saved_models',
        'results': script_dir / 'results',
        'visualization_outputs': script_dir.parent / '6 - Visualization' / 'outputs',
        'config': script_dir / 'augment_config_v2.json'
    }
    return items


def create_archive(out_dir: Path, script_dir: Path, dry_run=False):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f'archive_{timestamp}'
    archive_dir = out_dir / archive_name
    if dry_run:
        print(f"DRY: would create archive at {archive_dir}")
        return archive_dir

    archive_dir.mkdir(parents=True, exist_ok=True)
    items = gather_items(script_dir)
    for key, path in items.items():
        if path.exists():
            dst = archive_dir / path.name
            try:
                if path.is_dir():
                    shutil.copytree(path, dst)
                else:
                    shutil.copy2(path, dst)
                print(f"  ✓ Copied {path} -> {dst}")
            except Exception as e:
                print(f"  ✗ Failed to copy {path}: {e}")
        else:
            print(f"  - Not found (skip): {path}")

    # Save metadata
    metadata = {'archive_date': timestamp, 'source_dir': str(script_dir)}
    with open(archive_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Archive created: {archive_dir}")
    return archive_dir


def cleanup_targets(script_dir: Path, dry_run=False):
    targets = [
        script_dir / 'dataset_test',
        script_dir / 'dataset_train',
        script_dir / 'dataset_val',
        script_dir / 'dataset_augmented',
        script_dir / 'dataset_combined',
        script_dir / 'extracted_features',
        script_dir / 'saved_models',
        script_dir / 'results'
    ]

    for t in targets:
        if t.exists():
            if dry_run:
                print(f"DRY: would delete {t}")
            else:
                try:
                    if t.is_dir():
                        shutil.rmtree(t)
                    else:
                        t.unlink()
                    print(f"  ✓ Deleted {t}")
                except Exception as e:
                    print(f"  ✗ Failed to delete {t}: {e}")
        else:
            print(f"  - Not found (skip): {t}")


def main():
    parser = argparse.ArgumentParser(description='Safe archiving and optional cleanup')
    parser.add_argument('--out', type=str, default='archives', help='Output archives folder')
    parser.add_argument('--confirm', action='store_true', help='Confirm destructive cleanup after archiving')
    parser.add_argument('--dry-run', action='store_true', help='Show actions without performing them')
    args = parser.parse_args()

    script_dir = Path(project_config.PROJECT_ROOT) / '0 - DADS dataset extraction'
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_dir = create_archive(out_dir, script_dir, dry_run=args.dry_run)

    if args.confirm:
        cleanup_targets(script_dir, dry_run=args.dry_run)
    else:
        print('Cleanup not confirmed; archive-only mode.')


if __name__ == '__main__':
    main()
