from pathlib import Path
import os
import json
import tempfile
import wave
import numpy as np

try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    _HAS_SF = False


def atomic_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    try:
        os.replace(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def write_wav_atomic(path: Path, audio_array, sr: int, duration_s: float = 4.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')

    # If audio_array is None, generate silence
    if audio_array is None:
        try:
            from .. import config as project_config
            duration_s = float(getattr(project_config, 'AUDIO_DURATION_S', duration_s))
        except Exception:
            pass
        samples = int(sr * duration_s)
        audio_array = np.zeros(samples, dtype=np.float32)

    # Try to write using soundfile if available
    if _HAS_SF:
        try:
            sf.write(str(tmp), audio_array, sr, subtype='FLOAT')
        except Exception:
            # fallback to wave below
            _HAS_SF_FALLBACK = True
        else:
            _HAS_SF_FALLBACK = False
    else:
        _HAS_SF_FALLBACK = True

    if _HAS_SF_FALLBACK:
        # Write 16-bit PCM using stdlib wave
        if np.issubdtype(audio_array.dtype, np.floating):
            # scale to int16
            scaled = (audio_array * 32767.0).astype(np.int16)
        else:
            scaled = audio_array.astype(np.int16)
        with wave.open(str(tmp), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(scaled.tobytes())

    try:
        # ensure data flushed to disk
        with open(str(tmp), 'rb'):
            pass
        os.replace(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def append_jsonl_line(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
