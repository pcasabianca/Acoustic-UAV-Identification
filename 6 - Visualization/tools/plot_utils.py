from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def set_style():
    """Apply consistent plotting style across scripts."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def get_output_dir(script_path: Optional[Path] = None) -> Path:
    """Return the outputs directory adjacent to the calling script.

    If script_path is None the outputs folder next to the utils module is returned.
    """
    if script_path is None:
        base = Path(__file__).parent
    else:
        base = Path(script_path).parent

    out = base / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig, name: str, script_path: Optional[Path] = None, dpi: int = 300):
    out = get_output_dir(script_path)
    path = out / name
    try:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"[OK] Saved: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to save figure {path}: {e}")
