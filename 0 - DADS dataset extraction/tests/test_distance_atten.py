import numpy as np
import math
from augment_v5.modifiers.ops_distance import distance_atten, trajectory_atten


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def dbfs(x):
    r = rms(x)
    return 20.0 * math.log10(r + 1e-12)


def test_distance_attenuation_values():
    sr = 22050
    dur = 4.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sine = np.sin(2 * np.pi * 1000 * t).astype("float32")

    out1, m1 = distance_atten(sine.copy(), {"distance_m": 1.0, "ref_m": 1.0})
    out200, m200 = distance_atten(sine.copy(), {"distance_m": 200.0, "ref_m": 1.0})
    out500, m500 = distance_atten(sine.copy(), {"distance_m": 500.0, "ref_m": 1.0})

    assert "distance_m" in m200 and "distance_m" in m500

    db1 = dbfs(out1)
    db200 = dbfs(out200)
    db500 = dbfs(out500)

    # expected attenuation (dB) ~ -20*log10(distance)
    expect200 = -20.0 * math.log10(200.0)
    expect500 = -20.0 * math.log10(500.0)

    assert abs((db200 - db1) - expect200) < 1.0
    assert abs((db500 - db1) - expect500) < 1.0


def test_trajectory_attenuation_and_mix():
    rng = np.random.default_rng(12345)
    sr = 22050
    dur = 4.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sine = np.sin(2 * np.pi * 1000 * t).astype("float32")
    ambient = (0.5 * rng.normal(0, 1, len(t))).astype("float32")

    out_traj, m_traj = trajectory_atten(
        sine.copy(), {"distance_start_m": 500.0, "distance_end_m": 200.0, "ref_m": 1.0}
    )
    assert not np.isnan(out_traj).any()

    db_traj = dbfs(out_traj)
    db200 = dbfs(distance_atten(sine.copy(), {"distance_m": 200.0})[0])
    db500 = dbfs(distance_atten(sine.copy(), {"distance_m": 500.0})[0])

    # trajectory output should be between 500m and 200m end values
    assert db500 - 1.0 <= db_traj <= db200 + 1.0

    def snr(sig, noise):
        p_signal = np.mean(sig ** 2)
        p_noise = np.mean(noise ** 2)
        return 10.0 * math.log10((p_signal + 1e-18) / (p_noise + 1e-18))

    snr200 = snr(distance_atten(sine.copy(), {"distance_m": 200.0})[0], ambient)
    snr500 = snr(distance_atten(sine.copy(), {"distance_m": 500.0})[0], ambient)

    assert snr500 < snr200
