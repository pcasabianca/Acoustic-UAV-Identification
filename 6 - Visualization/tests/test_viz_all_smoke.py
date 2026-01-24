import json
from pathlib import Path
import importlib.util


def _import_viz_all():
    base = Path(__file__).resolve().parents[1]
    script_path = base / 'scripts' / 'viz_all.py'
    spec = importlib.util.spec_from_file_location('viz_all', str(script_path))
    mod = importlib.util.module_from_spec(spec)
    # ensure the local '6 - Visualization' directory is on sys.path so imports like `tools` resolve
    import sys
    sys.path.insert(0, str(base))
    try:
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.path.pop(0)
        except Exception:
            pass
    return mod


def test_viz_all_smoke(tmp_path):
    perf_dir = tmp_path / 'perfs'
    perf_dir.mkdir()
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    # minimal perf JSON
    perf = {
        'metadata': {'model': 'TESTMODEL', 'split': 'val', 'threshold': 0.5},
        'global_metrics': {'tp': 1, 'tn': 0, 'fp': 0, 'fn': 0, 'accuracy_drone': 1.0},
        'predictions': [
            {'filename': 'a.wav', 'true_label': 1, 'prob_class1': 0.95, 'snr_db': -5}
        ]
    }
    pfile = perf_dir / 'testmodel_val.json'
    pfile.write_text(json.dumps(perf))

    # minimal history csv
    hist = results_dir / 'testmodel_history.csv'
    hist.write_text('epoch,loss,val_loss,accuracy\n0,1.0,1.0,0.5\n1,0.5,0.6,0.8\n')

    mod = _import_viz_all()
    res = mod.run(models=['TESTMODEL'], splits=('val',), perf_dir=str(perf_dir), results_dir=str(results_dir), out_dir=str(out_dir), snr_bins=[-30, -10, 0, 10], strict=False)

    # verify artifacts
    assert res['selected_count'] >= 1
    outp = Path(res['out_dir'])
    assert (outp / 'perf_vs_snr').exists()
    assert (outp / 'training').exists()
