"""Test recorder that stores audio in RAM, analyses it, then writes WAV to disk.

This mirrors `test_recorder_pi.py` but routes the WAV bytes through
`utils.ram_utils.RamStore` to exercise in-memory handling and a
bit-exactness check when saving to disk.
"""
from __future__ import annotations
import os
import argparse
import io
import numpy as np

from utils.raspberry_recorder import RaspberryRecorder
from utils.raspberry_recorder import RaspberryRecorder as RR
from utils.wav_utils import wav_bytes
import utils.raspberry_recorder as rrmod


def analyze_wav_bytes(b: bytes, samplerate: int):
    import wave
    import numpy as np

    buf = io.BytesIO(b)
    with wave.open(buf, 'rb') as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        data = wf.readframes(nframes)

    if sampwidth != 2:
        raise NotImplementedError('Only 16-bit WAV supported')

    arr = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        arr = arr.reshape((-1, channels))

    float_arr = arr.astype(np.float32) / 32768.0

    N = len(float_arr)
    if N <= 0:
        return {'main_freq': float('nan'), 'nframes': 0}

    # Reuse analyze_capture-style FFT peak detection (without real timestamps)
    from numpy.fft import rfft, rfftfreq
    lf = rfftfreq(N, d=1.0 / float(samplerate))
    data0 = float_arr - float_arr.mean()
    fft = rfft(data0)
    FILTER = (lf > 200)
    main_freq = lf[(FILTER * np.abs(fft)).argmax()]
    return {'main_freq': float(main_freq), 'nframes': N, 'samplerate': samplerate}


def save_capture_to_ram(out_path: str, duration: float = 4.0, f: float = 3000.0):
    # Use RaspberryRecorder exclusively
    rec = RaspberryRecorder(config_path='deployment_config.json', use_ram=True, use_disk=False)
    rec.duration = duration
    rec.sample_rate = int(f)

    meta = rec.record_and_store(filename=None)

    # Get RAM key and buffer
    key = meta.get('ram_key')
    if key is None:
        raise RuntimeError('No RAM key returned from recorder')

    buf = rec.get_ram_buffer(key)
    assert buf is not None
    bdata = buf.getvalue()

    print(f"Stored buffer in RAM under key: {key} (size={len(bdata)} bytes)")

    # Analyze from RAM using same helper
    analysis = analyze_wav_bytes(bdata, int(f))
    print('Analysis from RAM:', analysis)

    # Save to disk and check bit-exactness
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as fobj:
        fobj.write(bdata)

    with open(out_path, 'rb') as fobj:
        disk_bytes = fobj.read()
    print(f"Saved WAV to disk at: {out_path} (size={len(disk_bytes)} bytes)"
          )
    if disk_bytes == bdata:
        print('Bit-exact save: OK')
    else:
        print('Bit-exact save: FAILED')
        import itertools
        print('RAM bytes:', len(bdata), 'disk bytes:', len(disk_bytes))
        for i, (a, b) in enumerate(itertools.zip_longest(bdata, disk_bytes, fillvalue=None)):
            if a != b:
                print('First differing byte index:', i, 'ram=', a, 'disk=', b)
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='recordings/test_recorder_pi_via_ram.wav')
    parser.add_argument('--duration', type=float, default=4.0)
    parser.add_argument('--samplerate', type=float, default=24000.0)
    args = parser.parse_args()

    save_capture_to_ram(args.out, duration=args.duration, f=args.samplerate)


if __name__ == '__main__':
    main()
