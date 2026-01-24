import json
import tempfile
from pathlib import Path


def _ensure_local_tools_on_path():
    import sys
    base = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(base))


def test_thresholds_loader_new_schema():
    _ensure_local_tools_on_path()
    from tools.io.thresholds_loader import load_calibrated_thresholds, get_threshold

    sample = {'models': {'CNN': {'threshold': 0.55}, 'RNN': {'threshold': 0.42}}}
    with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as f:
        f.write(json.dumps(sample))
        f.flush()
        data = load_calibrated_thresholds(f.name)
        assert get_threshold(data, 'cnn') == 0.55
        assert get_threshold(data, 'RNN') == 0.42
