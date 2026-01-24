import pandas as pd
from pathlib import Path
from typing import Optional


def load_history_csv(path: str):
    p = Path(path)
    df = pd.read_csv(p)
    if 'epoch' not in df.columns:
        df.insert(0, 'epoch', range(1, len(df) + 1))
    return df


def load_training_metadata(path: str) -> Optional[dict]:
    import json
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())
