import json
import tempfile
from tools.io.perf_loader import load_perf


def test_load_perf_minimal():
    sample = {
        'metadata': {'model': 'CNN', 'split': 'val', 'threshold': 0.5, 'total_samples': 2},
        'global_metrics': {'total': 2},
        'predictions': [
            {'filename': 'a.wav', 'true_label': 1, 'prob_class1': 0.9},
            {'filename': 'b.wav', 'true_label': 0, 'prob_class1': 0.1}
        ]
    }
    with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as f:
        f.write(json.dumps(sample))
        f.flush()
        out = load_perf(f.name, strict=True)
        assert out['metadata']['model'] == 'CNN'
        assert len(out['predictions']) == 2


def test_infer_model_with_underscore():
    sample = {
        'metadata': {'threshold': 0.5, 'total_samples': 1},
        'global_metrics': {'total': 1},
        'predictions': [{'filename': 'a.wav', 'true_label': 1, 'prob_class1': 0.9}]
    }
    with tempfile.NamedTemporaryFile('w+', prefix='attention_crnn_val_', suffix='.json', delete=False) as f:
        f.write(json.dumps(sample))
        f.flush()
        out = load_perf(f.name, strict=True)
        assert out['metadata']['model'] == 'ATTENTION_CRNN'


def test_reject_non_perf_in_strict_mode():
    sample = {'notpreds': []}
    with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as f:
        f.write(json.dumps(sample))
        f.flush()
        try:
            load_perf(f.name, strict=True)
            assert False, 'should have raised'
        except ValueError:
            pass
