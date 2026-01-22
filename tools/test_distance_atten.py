import numpy as np
from pathlib import Path
import math

# import the ops directly
from __future__ import annotations

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '0 - DADS dataset extraction'))
from augment_v5.modifiers.ops_distance import distance_atten, trajectory_atten


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))


def dbfs(x):
    r = rms(x)
    return 20.0 * math.log10(r + 1e-12)


def synth_sine(freq, sr, dur, amp=1.0):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    return (amp * np.sin(2*np.pi*freq*t)).astype('float32')


def synth_ambient_noise(sr, dur, amp=0.1):
    return (amp * np.random.default_rng(1234).normal(0, 1, int(sr*dur))).astype('float32')


def report():
    sr = 22050
    dur = 4.0
    drone = synth_sine(1000, sr, dur, amp=1.0)
    ambient_close = synth_ambient_noise(sr, dur, amp=0.5)

    # reference at 1m
    out_ref, meta_ref = distance_atten(drone.copy(), {'distance_m': 1.0, 'ref_m': 1.0})
    # far distances
    out_200, meta_200 = distance_atten(drone.copy(), {'distance_m': 200.0, 'ref_m': 1.0})
    out_500, meta_500 = distance_atten(drone.copy(), {'distance_m': 500.0, 'ref_m': 1.0})

    # trajectory 500->200 approaching
    out_traj, meta_traj = trajectory_atten(drone.copy(), {'distance_start_m': 500.0, 'distance_end_m':200.0, 'ref_m':1.0})

    # mix with ambient close (simulating ambient within 1-5m using gain 1/ref)
    # ambient already amplitude set; compute RMS
    ref_db = dbfs(out_ref)
    d200_db = dbfs(out_200)
    d500_db = dbfs(out_500)
    traj_db = dbfs(out_traj)

    # compute mixed signals
    mix_200 = out_200 + ambient_close
    mix_500 = out_500 + ambient_close
    mix_ref = out_ref + ambient_close

    # compute drone vs ambient contributions (we have separate arrays)
    drone_rms_ref = rms(out_ref)
    drone_rms_200 = rms(out_200)
    drone_rms_500 = rms(out_500)
    ambient_rms = rms(ambient_close)

    print('Reference (1m):', meta_ref)
    print('200m meta:', meta_200)
    print('500m meta:', meta_500)
    print('trajectory meta:', meta_traj)
    print('RMS (drone) dBFS: ref {:.2f} dB, 200m {:.2f} dB, 500m {:.2f} dB, traj {:.2f} dB'.format(ref_db, d200_db, d500_db, traj_db))
    print('RMS (ambient close) dBFS: {:.2f} dB'.format(dbfs(ambient_close)))
    print('\nMixed levels (drone + ambient):')
    print(' Mix ref total {:.2f} dB, drone {:.2f} dB, ambient {:.2f} dB'.format(dbfs(mix_ref), ref_db, dbfs(ambient_close)))
    print(' Mix 200m total {:.2f} dB, drone {:.2f} dB, ambient {:.2f} dB'.format(dbfs(mix_200), d200_db, dbfs(ambient_close)))
    print(' Mix 500m total {:.2f} dB, drone {:.2f} dB, ambient {:.2f} dB'.format(dbfs(mix_500), d500_db, dbfs(ambient_close)))

    # compute SNR drone vs ambient in mixes (approx via RMS ratios)
    def snr(drone_arr, ambient_arr):
        p_signal = np.mean(drone_arr**2)
        p_noise = np.mean(ambient_arr**2)
        return 10.0 * math.log10((p_signal + 1e-18) / (p_noise + 1e-18))

    print('\nSNR (drone vs ambient):')
    print(' SNR ref (1m): {:.2f} dB'.format(snr(out_ref, ambient_close)))
    print(' SNR 200m: {:.2f} dB'.format(snr(out_200, ambient_close)))
    print(' SNR 500m: {:.2f} dB'.format(snr(out_500, ambient_close)))
    print(' SNR traj (500->200): {:.2f} dB'.format(snr(out_traj, ambient_close)))


if __name__ == '__main__':
    report()
