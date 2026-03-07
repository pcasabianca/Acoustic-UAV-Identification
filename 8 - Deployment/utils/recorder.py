"""Unified recorder interface for Laptop and Raspberry recorders.

Provides `RecorderFactory.get_recorder()` which returns an object with
methods: `record_audio()`, `get_audio_bytes(audio)`, `save_audio(audio, filename)`, `cleanup()`.
"""
from __future__ import annotations
from typing import Optional
import json
from pathlib import Path

from .raspberry_recorder import RaspberryRecorder
from .laptop_recorder import LaptopRecorder


class RecorderFactory:
    @staticmethod
    def get_recorder(config_path: str = 'deployment_config.json', prefer: str = 'auto', use_ram: bool = True, use_disk: bool = False, logger=None):
        """Return recorder instance per `prefer` ('raspberry'|'laptop'|'auto').

        'auto' will attempt Raspberry first (SPI) then laptop fallback.
        """
        # Load config to check recording.device preference
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

        mode = prefer
        if prefer == 'auto':
            # Use recording.enabled + platform heuristics could be added; default to raspberry then laptop
            mode = 'raspberry'

        if mode == 'raspberry':
            try:
                rec = RaspberryRecorder(config_path=config_path, use_ram=use_ram, use_disk=use_disk, logger=logger)
                return rec
            except Exception:
                # fallback to laptop
                return LaptopRecorder(config_path=config_path, use_ram=use_ram, logger=logger)
        elif mode == 'laptop':
            return LaptopRecorder(config_path=config_path, use_ram=use_ram, logger=logger)
        else:
            return LaptopRecorder(config_path=config_path, use_ram=use_ram, logger=logger)


if __name__ == '__main__':
    rec = RecorderFactory.get_recorder()
    a = rec.record_audio()
    print('Recorded:', getattr(a, 'shape', None))