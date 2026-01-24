from tools.io.training_loader import load_history_csv
import pandas as pd
import tempfile


def test_load_history_csv_minimal():
    df = pd.DataFrame({'accuracy': [0.5, 0.6], 'val_accuracy': [0.55, 0.65]})
    with tempfile.NamedTemporaryFile('w+', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        out = load_history_csv(f.name)
        assert 'epoch' in out.columns
        assert len(out) == 2
