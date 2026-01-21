from pathlib import Path
import json
import os


class SplitWriter:
    def __init__(self, split_dir: Path, meta_name: str = 'augmentation_samples.jsonl'):
        self.split_dir = Path(split_dir)
        self.meta_name = meta_name
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.split_dir / self.meta_name
        if not self._meta_path.exists():
            self._meta_path.write_text('', encoding='utf-8')

    def write_wav(self, label: str, filename: str, audio, sr: int):
        d = self.split_dir / str(label)
        d.mkdir(parents=True, exist_ok=True)
        p = d / filename
        # write using augment.io.write_wav_atomic for atomic and real audio output
        try:
            from .io import write_wav_atomic
        except Exception:
            # fallback: touch file
            p.write_bytes(b'')
            return p

        write_wav_atomic(p, audio, sr)
        return p

    def append_jsonl(self, train_meta: dict, debug_meta: dict = None):
        line_obj = {'train_meta': train_meta}
        if debug_meta is not None:
            line_obj['debug_meta'] = debug_meta
        with self._meta_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(line_obj, ensure_ascii=False) + '\n')
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
        return self._meta_path
