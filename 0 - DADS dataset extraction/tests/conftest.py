import sys
from pathlib import Path


# Ensure repository root (two levels up) is on sys.path so tests can import augment_v5
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
