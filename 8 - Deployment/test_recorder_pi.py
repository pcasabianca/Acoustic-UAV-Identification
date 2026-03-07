"""Test recorder for Raspberry Pi that uses `utils.spi_utils` and `utils.wav_utils`.

Behavior:
- Try to capture real ADC samples via SPI; on failure, generate a synthetic sine.
- Save resulting audio to `recordings/test_recorder_pi.wav`.
"""
from __future__ import annotations
import os
import argparse
import numpy as np

from utils.wav_utils import write_wav
from utils.spi_utils import SPIReader, timed_capture, analyze_capture
from utils.spi_utils import highpass_filter

def plot_fft(times, data, f):
    import matplotlib.pyplot as plt
    from scipy.fft import rfft, rfftfreq

    N = len(data)
    lf = rfftfreq(N, d=1/f)
    data = data - np.mean(data)
    fft = rfft(data)
    print(f"FFT peak frequency: {lf[np.abs(fft).argmax()]:.1f} Hz")
    plt.figure(figsize=(10, 6))
    plt.plot(lf, np.abs(fft))
    plt.title('FFT of Captured Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, f/2)
    plt.grid()
    plt.savefig('recordings/test_recorder_pi_fft.png')
    plt.close()
    print('Saved FFT plot to recordings/test_recorder_pi_fft.png')

def save_capture_from_spi(out_path: str, duration: float = 4.0, f: float = 3000.0):
    N = int(duration * f)
    reader = SPIReader(bus=0, device=0, max_speed_hz=1000000)
    reader.open()
    try:
        times, timesAlt, data = timed_capture(N, f, reader, busy_wait=False, wait_before_read=True)
    finally:
        reader.close()
        
    biais = np.mean(data)
    print(f"Captured {N} samples at nominal {f} Hz with mean value { biais:.2f}")
    res = analyze_capture(times, timesAlt, data, f)
    print('Capture analysis:', res)
    
    # Get biais as mean of captured data, assuming it represents the ADC's zero level
    #biais = np.mean(data)
    print(f"Estimated bias (zero level) from captured data: {biais:.2f}")
    
    # Correct for bias 
    audio = data - biais
    print(f"Audio range after bias correction: [{audio.min():.2f}, {audio.max():.2f}]")
    
    # High-pass filter to remove low-frequency noise and DC offset
    #cutoff_hz = 100.0  # Adjust as needed to remove low-frequency noise
    #audio = highpass_filter(audio, f, cutoff_hz)
    
    # Normalize to [-1, 1]
    min_audio, max_audio = np.min(audio), np.max(audio)
    print(f"Audio range before normalization: [{min_audio:.2f}, {max_audio:.2f}]")
    factor = max(abs(min_audio), abs(max_audio))
    print(f"Normalization factor: {factor:.2f}")
    if factor > 0:
        audio = audio / factor
    else:
        print("Warning: captured audio is silent")
    # FFT
    plot_fft(times, data, f)

    

    # Write WAV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_wav(out_path, audio, int(f), channels=1)
    print('Saved WAV to', out_path)


def save_synthetic(out_path: str, duration: float = 4.0, samplerate: int = 3000, tone_freq: float = 440.0):
    t = np.linspace(0, duration, int(duration * samplerate), endpoint=False)
    audio = 0.6 * np.sin(2 * np.pi * tone_freq * t)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_wav(out_path, audio.astype(np.float32), samplerate, channels=1)
    print('Saved synthetic WAV to', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='recordings/test_recorder_pi.wav')
    parser.add_argument('--duration', type=float, default=20.0)
    parser.add_argument('--samplerate', type=float, default=24000.0)
    args = parser.parse_args()

    out = args.out
    try:
        save_capture_from_spi(out, duration=args.duration, f=args.samplerate)
    except Exception as e:
        print('SPI capture failed, falling back to synthetic signal:', e)
        save_synthetic(out, duration=args.duration, samplerate=int(args.samplerate), tone_freq=440.0)

if __name__ == '__main__':
    main()
