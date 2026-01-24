from pathlib import Path
import os
import matplotlib.pyplot as plt
from typing import Optional
import config


def resolve_output_dir(out_dir: Optional[str] = None) -> Path:
    if out_dir:
        out = Path(out_dir)
    else:
        out = Path(config.VIZ_OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out


def savefig(fig, out_path: Path, dpi: int = 200):
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
