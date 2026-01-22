import os
import tempfile
from pathlib import Path
try:
    import soundfile as sf
except Exception:
    sf = None


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def atomic_write_wav(path, audio, sr):
    if sf is None:
        raise RuntimeError("soundfile required to write wav files")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=str(path.parent))
    os.close(fd)
    try:
        sf.write(tmp, audio, sr)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
