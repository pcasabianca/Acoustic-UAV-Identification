#!/usr/bin/env python3
"""Re-extract Mel features for a specific split (small subset) for verification.

This helper runs the Mel extractor script for a given split (train/val/test)
using the centralized config and writes to `extracted_features/mel_{split}.json`.

Usage:
  python tools/reextract_mel_subset.py --split train --num-segments 1
"""
import argparse
import subprocess
from pathlib import Path
import os

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "1 - Preprocessing and Features Extraction" / "Mel_Preprocess_and_Feature_Extract.py"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train','val','test'], default='train')
    parser.add_argument('--num-segments', type=int, default=1, help='Number of segments per file (small=1)')
    parser.add_argument('--spec-augment', choices=['auto','on','off'], default='off')
    args = parser.parse_args()

    if not SCRIPT.exists():
        print(f"Mel extractor not found at: {SCRIPT}")
        raise SystemExit(1)

    cmd = [
        'python3',
        str(SCRIPT),
        '--split', args.split,
        '--num_segments', str(args.num_segments),
        '--spec_augment', args.spec_augment
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("Done.")
    except subprocess.CalledProcessError as e:
        print(f"Extractor failed with code {e.returncode}")
        raise
