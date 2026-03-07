"""Simple WAV utilities: read, write, bytes.

Functions operate on numpy arrays with float32 in [-1, 1] or int16.
"""
from __future__ import annotations
import wave
import io
import numpy as np


def _ensure_mono_array(samples, channels: int):
    arr = np.asarray(samples)
    if channels == 1:
        if arr.ndim == 2:
            # collapse to mono by taking first channel
            return arr[:, 0]
        return arr
    else:
        # expect shape (N, channels)
        if arr.ndim == 1:
            raise ValueError("Expected multi-channel array with shape (N, channels)")
        return arr


def write_wav(path: str, samples: np.ndarray, samplerate: int, channels: int = 1) -> None:
    """Write `samples` to `path` as 16-bit WAV.

    `samples` may be float32 in [-1,1] or int16.
    """
    arr = np.asarray(samples)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # convert float [-1,1] to int16
        clipped = np.clip(arr, -1.0, 1.0)
        int_data = (clipped * 32767).astype(np.int16)
    elif arr.dtype == np.int16:
        int_data = arr
    else:
        int_data = arr.astype(np.int16)

    # handle channels
    if channels == 1:
        frames = int_data.tobytes()
    else:
        # expect shape (N, channels)
        if int_data.ndim != 2 or int_data.shape[1] != channels:
            raise ValueError('samples must have shape (N, channels) for multi-channel')
        frames = int_data.tobytes()

    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        wf.writeframes(frames)


def read_wav(path: str) -> tuple[np.ndarray, int, int]:
    """Read WAV file and return (samples_float32, samplerate, channels).

    Samples returned as float32 in [-1, 1].
    """
    with wave.open(str(path), 'rb') as wf:
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
    return float_arr, sr, channels


def wav_bytes(samples: np.ndarray, samplerate: int, channels: int = 1) -> bytes:
    """Return WAV file bytes for given samples.

    Useful for in-memory storage.
    """
    buf = io.BytesIO()
    arr = np.asarray(samples)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        clipped = np.clip(arr, -1.0, 1.0)
        int_data = (clipped * 32767).astype(np.int16)
    else:
        int_data = arr.astype(np.int16)

    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        wf.writeframes(int_data.tobytes())

    return buf.getvalue()


if __name__ == '__main__':
    # quick smoke test: write then read back
    import numpy as np
    samples = 0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 44100, False))
    write_wav('test_write.wav', samples, 44100, channels=1)
    s, sr, ch = read_wav('test_write.wav')
    print('written/read', s.shape, sr, ch)
