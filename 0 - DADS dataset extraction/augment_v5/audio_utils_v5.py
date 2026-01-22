import math
import numpy as np
try:
    import soundfile as sf
except Exception as e:
    sf = None


def load_audio(path, target_sr=None):
    if sf is None:
        raise RuntimeError("soundfile (pysoundfile) is required: pip install soundfile")
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        raise RuntimeError("Resampling not implemented; provide files at target sample rate")
    return data.astype("float32"), sr


def trim_or_pad(x, target_samples):
    if len(x) > target_samples:
        return x[:target_samples]
    if len(x) < target_samples:
        pad = np.zeros(target_samples - len(x), dtype=x.dtype)
        return np.concatenate([x, pad])
    return x


def audio_stats(x, sr):
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    peak_dbfs = -300.0 if peak == 0 else 20.0 * math.log10(peak)
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    rms_dbfs = -300.0 if rms == 0 else 20.0 * math.log10(rms)
    clip_count = int(((np.abs(x) >= 1.0)).sum())
    duration_s = len(x) / float(sr) if sr else 0.0
    return {"peak_dbfs": peak_dbfs, "rms_dbfs": rms_dbfs, "clip_count": clip_count, "duration_s": duration_s}


def normalize_peak(x, target_dbfs=-1.0):
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return x
    current_db = 20.0 * math.log10(peak)
    gain_db = target_dbfs - current_db
    gain = 10 ** (gain_db / 20.0)
    return (x * gain).astype("float32")
