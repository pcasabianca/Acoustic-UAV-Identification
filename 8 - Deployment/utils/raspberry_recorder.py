"""Raspberry Recorder component using SPI and in-RAM storage.

Loads `deployment_config.json` at init and respects its settings.
Provides simple API to record audio, store in RAM (via `utils.ram_utils.RamStore`),
optionally save to disk, and return metadata including RAM key and disk path.
"""
from __future__ import annotations
import json
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

import numpy as np

from .ram_utils import RamStore
from .wav_utils import wav_bytes, write_wav
from .spi_utils import SPIReader, timed_capture, analyze_capture


class RaspberryRecorder:
    """Recorder that follows `deployment_config.json` settings.

    Methods:
        record_and_store() -> dict: record audio, store in RAM/disk per flags,
            and return metadata including `ram_key` and/or `disk_path`.
    """

    def __init__(self, config_path: str = 'deployment_config.json', use_ram: bool = True, use_disk: bool = False, logger: Optional[logging.Logger] = None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        audio_conf = self.config.get('audio', {})
        rec_conf = self.config.get('recording', {})

        self.sample_rate = int(audio_conf.get('sample_rate', 22050))
        self.duration = float(audio_conf.get('duration_seconds', 4.0))
        self.channels = int(audio_conf.get('channels', 1))

        self.use_ram = bool(use_ram)
        self.use_disk = bool(use_disk)
        # Logger to use for warnings/info; fall back to module logger
        self.logger = logger or logging.getLogger(__name__)

        self.recordings_dir = Path(rec_conf.get('recordings_directory', './recordings'))
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        # Optionally a fixed filename may be specified in the config under recording.unique_filename
        self.unique_filename = rec_conf.get('unique_filename')

        # Whether to also write debug copies even when using RAM
        self.save_debug = bool(rec_conf.get('save_debug_recordings', False))

        # Device index may be present (for PyAudio in other contexts)
        self.device_index = rec_conf.get('device_index', None)

        # Ram store used when use_ram True
        self._ram_store = RamStore(max_items=rec_conf.get('ram_max_items', 8)) if self.use_ram else None

    def _default_filename(self) -> str:
        if self.unique_filename:
            return self.unique_filename
        return datetime.now().strftime('recording_%Y%m%d_%H%M%S.wav')

    def record_audio(self) -> np.ndarray:
        """Record audio and return numpy array float32 in [-1, 1].

        On Raspberry Pi this uses SPIReader; on systems without spidev it
        falls back to a synthetic sine wave for testing.
        """
        N = int(self.duration * self.sample_rate)
        try:
            reader = SPIReader(bus=0, device=0, max_speed_hz=1000000)
            reader.open()
            try:
                times, timesAlt, data = timed_capture(N, self.sample_rate, reader, busy_wait=False, wait_before_read=True)
            finally:
                reader.close()

            biais = float(np.mean(data))
            audio = data - biais
            
            # Detect if there's a problem (almost no variation) and log warning
            low_var = False
            stdv = float(np.std(audio))
            if stdv < 0.2: # empirical threshold for "no signal"; adjust as needed based on testing
                low_var = True
                try:
                    # Log at DEBUG here to avoid duplicating the visible WARNING
                    # which the orchestrator will emit when it observes low_variation.
                    self.logger.debug(f"low audio variation detected (std={stdv})")
                except Exception:
                    # best-effort logging; avoid raising from recorder
                    pass
            
            # normalize peak to 1.0 if non-zero
            peak = max(abs(audio.min()), abs(audio.max()))
            if peak > 0:
                audio = audio / peak
            return audio.astype(np.float32)

        except Exception:
            # Fallback synthetic tone for non-RPi/testing
            t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
            tone = 0.6 * np.sin(2 * np.pi * 440.0 * t)
            return tone.astype(np.float32)

    def get_audio_bytes(self, audio: np.ndarray) -> bytes:
        return wav_bytes(audio, int(self.sample_rate), channels=self.channels)

    def record_and_store(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Record audio, optionally store in RAM and/or disk.

        Returns metadata dict with keys: `ram_key`, `disk_path`, `samplerate`, `nframes`, `bytes`.
        """
        audio = self.record_audio()
        b = self.get_audio_bytes(audio)

        # Determine low-variation flag (recompute here to ensure defined)
        try:
            stdv_meta = float(np.std(audio))
            low_var_meta = stdv_meta < 2
        except Exception:
            low_var_meta = False

        meta: Dict[str, Any] = {
            'samplerate': int(self.sample_rate),
            'nframes': int(len(audio)),
            'channels': int(self.channels),
            'bytes': len(b),
            'low_variation': low_var_meta,
        }

        # Store in RAM
        if self.use_ram and self._ram_store is not None:
            key = self._ram_store.store(None, b)
            meta['ram_key'] = key

        # Save to disk if requested
        target_name = filename or self._default_filename()
        disk_path = None
        if self.use_disk or self.save_debug:
            disk_path = str(self.recordings_dir / target_name)
            # write exact bytes to disk
            with open(disk_path, 'wb') as f:
                f.write(b)
            meta['disk_path'] = disk_path

        # include raw audio array in meta for callers that need it
        try:
            meta['audio'] = audio
        except Exception:
            pass

        return meta

    def get_ram_buffer(self, key: str) -> Optional[io.BytesIO]:
        if not self._ram_store:
            return None
        return self._ram_store.get_buffer(key)

    def delete_ram_key(self, key: str) -> bool:
        if not self._ram_store:
            return False
        return self._ram_store.delete(key)

    def clear_ram(self) -> None:
        if not self._ram_store:
            return
        self._ram_store.clear()

    def cleanup(self):
        # placeholder for symmetry with other recorders
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Raspberry Recorder test')
    parser.add_argument('--config', type=str, default='deployment_config.json')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--ram', action='store_true')
    parser.add_argument('--disk', action='store_true')
    args = parser.parse_args()

    rec = RaspberryRecorder(config_path=args.config, use_ram=args.ram, use_disk=args.disk)
    if args.duration is not None:
        rec.duration = args.duration

    meta = rec.record_and_store(filename=args.out)
    print('Recorded:', meta)