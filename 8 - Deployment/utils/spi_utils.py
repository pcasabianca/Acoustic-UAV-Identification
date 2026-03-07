"""SPI utilities inspired by `test_rapide.py`.

Provides a small `SPIReader` wrapper and timed capture helpers.
"""
from __future__ import annotations
import time
try:
    import spidev
except Exception:  # allow import on non-RPI for linting/tests
    spidev = None
import numpy as np
try:
    from scipy.fft import rfft, rfftfreq, irfft
except Exception:
    from numpy.fft import rfft, rfftfreq, irfft


def precise_sleep(dt: float) -> None:
    end = time.perf_counter() + dt
    while time.perf_counter() < end:
        pass


class SPIReader:
    """Simple SPI helper for MCP3001-like reads.

    Example:
        r = SPIReader(); r.open(); rx, val = r.read_mcp3001(); r.close()
    """

    def __init__(self, bus: int = 0, device: int = 0, max_speed_hz: int = 1000000, mode: int = 0, no_cs: bool = False):
        self.bus = bus
        self.device = device
        self.max_speed_hz = int(max_speed_hz)
        self.mode = int(mode)
        self.no_cs = bool(no_cs)
        self._spi = None

    def open(self):
        if spidev is None:
            raise RuntimeError('spidev module not available')
        self._spi = spidev.SpiDev()
        self._spi.open(self.bus, self.device)
        self._spi.max_speed_hz = self.max_speed_hz
        self._spi.mode = self.mode
        self._spi.no_cs = self.no_cs

    def close(self):
        if self._spi is not None:
            try:
                self._spi.close()
            finally:
                self._spi = None

    def read_mcp3001(self):
        """Perform a 16-clock read and decode 10-bit MCP3001 response.

        Returns (rx_bytes, value)
        """
        if self._spi is None:
            raise RuntimeError('SPI device not open')
        rx = self._spi.xfer2([0x00, 0x00])
        val = ((rx[0] & 0x1F) << 5) | (rx[1] >> 3)
        return rx, val


def timed_capture(N: int, f: float, reader: SPIReader, busy_wait: bool = True, wait_before_read: bool = True):
    """Capture `N` samples at nominal frequency `f` using `reader`.

    - If `wait_before_read` is True the function waits until the target
      timestamp then reads (gives more regular sampling times).
    - If `busy_wait` is False, the function uses `time.sleep()` with a
      final short busy-wait to improve CPU usage.

    Returns (times, timesAlt, data) where:
      - `times` are actual timestamps of each read
      - `timesAlt` are the nominal scheduled timestamps (start + i*dt)
      - `data` are the integer samples
    """
    dt = 1.0 / float(f)
    times = np.zeros(N)
    timesAlt = np.zeros(N)
    data = np.zeros(N)

    start = time.perf_counter()
    for i in range(N):
        target = start + i * dt
        timesAlt[i] = target
        if wait_before_read:
            if busy_wait:
                while time.perf_counter() < target:
                    pass
            else:
                # sleep most of the remaining time, then busy-wait
                now = time.perf_counter()
                rem = target - now
                if rem > 0.0006:
                    time.sleep(rem - 0.0005)
                while time.perf_counter() < target:
                    pass

        rx, v = reader.read_mcp3001()
        data[i] = v
        times[i] = time.perf_counter()

        if not wait_before_read:
            # if read-first, then wait until the scheduled next slot
            if busy_wait:
                while time.perf_counter() < target + dt:
                    pass
            else:
                now = time.perf_counter()
                rem = target + dt - now
                if rem > 0.0006:
                    time.sleep(rem - 0.0005)
                while time.perf_counter() < target + dt:
                    pass

    return times, timesAlt, data


def analyze_capture(times, timesAlt, data, f: float):
    N = len(data)
    lf = rfftfreq(N, d=1.0 / float(f))
    data = data - np.mean(data)
    fft = rfft(data)
    FILTER = (lf > 200)
    main_freq = lf[(FILTER * np.abs(fft)).argmax()]
    real_freq = float('inf')
    alt_freq = float('inf')
    duration = None
    if (times[-1] - times[0]) > 0:
        real_freq = N / (times[-1] - times[0])
        duration = times[-1] - times[0]
    if (timesAlt[-1] - timesAlt[0]) > 0:
        alt_freq = N / (timesAlt[-1] - timesAlt[0])
    return {
        'main_freq': float(main_freq),
        'real_freq': float(real_freq),
        'alt_freq': float(alt_freq),
        'duration': duration,
    }


def highpass_filter(data, samplerate: float, cutoff_hz: float):
    """Apply a simple high-pass filter via FFT.

    This zeros FFT bins below `cutoff_hz` and returns the time-domain
    filtered signal (float32). Works on 1D arrays; if `data` is 2D the
    first column is used.
    """
    import numpy as np

    arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr[:, 0]

    N = arr.shape[0]
    if N <= 0:
        return arr.astype(np.float32)

    fft = rfft(arr)
    freqs = rfftfreq(N, d=1.0 / float(samplerate))
    mask = freqs >= float(cutoff_hz)
    # Zero out low-frequency bins
    fft_filtered = fft * mask
    filtered = irfft(fft_filtered, n=N)
    return filtered.astype(np.float32)


if __name__ == '__main__':
    # quick demonstration (small N to avoid long hardware runs)
    import sys

    N = 2000
    f = 30000
    r = SPIReader()
    try:
        r.open()
    except Exception as e:
        print('Could not open SPI device:', e)
        sys.exit(1)
    try:
        times, timesAlt, data = timed_capture(N, f, r, busy_wait=False)
    finally:
        r.close()

    res = analyze_capture(times, timesAlt, data, f)
    print(f"Capture done: real_freq={res['real_freq']:.2f} Hz alt_freq={res['alt_freq']:.2f} Hz main_freq={res['main_freq']:.2f} Hz")
