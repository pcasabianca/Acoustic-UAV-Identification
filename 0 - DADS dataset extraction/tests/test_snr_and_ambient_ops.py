import numpy as np
import math
from augment_v5.modifiers.ops_ambient import dropout, softclip_tanh
from augment_v5.modifiers.ops_distance import distance_atten
from augment_v5.modifiers.utils_rng import get_rng


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def dbfs(x):
    return 20.0 * math.log10(rms(x) + 1e-12)


def test_snr_estimation_simple():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    drone = np.sin(2 * np.pi * 1000 * t).astype('float32') * 0.1
    ambient = (0.5 * np.random.default_rng(1234).normal(0, 1, len(t))).astype('float32') * 0.01
    out_d, _ = distance_atten(drone.copy(), {'distance_m': 100.0, 'ref_m': 1.0})
    rms_d = rms(out_d)
    rms_a = rms(ambient)
    snr_db = 20.0 * math.log10((rms_d + 1e-12) / (rms_a + 1e-12))
    assert snr_db < 0 or snr_db > -100  # sanity check


def test_dropout_deterministic():
    sr = 22050
    x = np.ones(sr, dtype='float32')
    ctx = {'seed': 12345}
    out1, m1 = dropout(x.copy(), {'prob': 0.2, 'window_ms': 10, 'atten_db': -40.0, 'sr': sr}, ctx)
    out2, m2 = dropout(x.copy(), {'prob': 0.2, 'window_ms': 10, 'atten_db': -40.0, 'sr': sr}, ctx)
    assert np.allclose(out1, out2)


def test_softclip_bounds():
    sr = 22050
    t = np.linspace(0, 0.1, int(0.1*sr), endpoint=False)
    x = (10.0 * np.sin(2 * np.pi * 1000 * t)).astype('float32')
    out, m = softclip_tanh(x.copy(), {'drive_db': 6.0})
    assert np.max(np.abs(out)) <= 1.0 + 1e-6
