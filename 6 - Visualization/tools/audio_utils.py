import numpy as np
import librosa


def crossfade(signal1, signal2, fade_duration_samples):
    if fade_duration_samples == 0 or len(signal1) < fade_duration_samples:
        return np.concatenate([signal1, signal2])
    fade_out = np.linspace(1, 0, fade_duration_samples)
    fade_in = np.linspace(0, 1, fade_duration_samples)
    end_part = signal1[-fade_duration_samples:] * fade_out
    start_part = signal2[:fade_duration_samples] * fade_in
    crossfaded_part = end_part + start_part
    result = np.concatenate([
        signal1[:-fade_duration_samples],
        crossfaded_part,
        signal2[fade_duration_samples:]
    ])
    return result


def loop_audio(signal, target_duration, sr, crossfade_duration=0.1):
    target_samples = int(target_duration * sr)
    signal_length = len(signal)
    if signal_length >= target_samples:
        return signal[:target_samples]
    crossfade_samples = int(crossfade_duration * sr)
    crossfade_samples = min(crossfade_samples, max(0, signal_length // 4))
    result = signal.copy()
    while len(result) < target_samples:
        if len(result) + signal_length - crossfade_samples <= target_samples:
            result = crossfade(result, signal, crossfade_samples)
        else:
            remaining = target_samples - len(result)
            if remaining > crossfade_samples:
                result = crossfade(result, signal[:remaining], crossfade_samples)
            else:
                result = np.concatenate([result, signal[:remaining]])
    return result[:target_samples]


def ensure_duration(signal, sr, target_duration, crossfade_duration=0.1):
    target_samples = int(target_duration * sr)
    if len(signal) == target_samples:
        return signal
    elif len(signal) > target_samples:
        return signal[:target_samples]
    else:
        return loop_audio(signal, target_duration, sr, crossfade_duration)


def load_audio_file(file_path, sr, duration=None):
    """Load WAV file using soundfile when possible for speed and robustness.
    Falls back to librosa.load on failure.
    """
    import soundfile as sf
    try:
        # soundfile can read WAVs quickly; if file is longer than duration, read full and slice
        data, fs = sf.read(str(file_path), dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if fs != sr:
            # resample if necessary using librosa (small cost)
            data = librosa.resample(data.astype('float32'), orig_sr=fs, target_sr=sr)
        if duration is not None:
            max_samples = int(duration * sr)
            if len(data) > max_samples:
                data = data[:max_samples]
        return data
    except Exception:
        try:
            signal, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
            return signal
        except Exception:
            return None
