"""In-memory RAM storage utilities for audio buffers.

Provides a simple thread-safe FIFO-backed store (`RamStore`) for bytes
buffers (e.g. WAV bytes). Designed to mirror the simple, focused style
of `wav_utils.py` but for ephemeral RAM storage.

Usage:
    from utils.ram_utils import RamStore
    store = RamStore(max_items=3)
    key = store.store(None, wav_bytes)
    buf = store.get_buffer(key)  # BytesIO positioned at 0
"""
from __future__ import annotations
import io
import threading
from collections import OrderedDict
from typing import Optional, Iterable


class RamStore:
    """Thread-safe in-memory store for bytes objects with optional FIFO eviction.

    - Keys are strings.
    - If `max_items` is set, oldest entries are evicted when limit exceeded.
    """

    def __init__(self, max_items: Optional[int] = None):
        self._lock = threading.Lock()
        self._store: OrderedDict[str, bytes] = OrderedDict()
        self.max_items = int(max_items) if max_items is not None else None

    def store(self, key: Optional[str], data: bytes) -> str:
        """Store `data` under `key`. If `key` is None a numeric key is generated.

        Returns the key used.
        """
        with self._lock:
            if key is None:
                # generate simple incremental key
                idx = 0
                while True:
                    k = f"buf_{idx}"
                    if k not in self._store:
                        key = k
                        break
                    idx += 1

            # If key exists, replace it and keep insertion order by moving to end
            if key in self._store:
                del self._store[key]

            # Evict oldest if needed
            if self.max_items is not None:
                while len(self._store) >= self.max_items:
                    self._store.popitem(last=False)

            self._store[key] = data
            return key

    def get(self, key: str) -> Optional[bytes]:
        """Return bytes for `key` or None if missing."""
        with self._lock:
            return self._store.get(key)

    def get_buffer(self, key: str) -> Optional[io.BytesIO]:
        """Return a fresh `BytesIO` for the stored bytes (seek at 0).

        Caller may read/seek without affecting the internal store.
        """
        data = self.get(key)
        if data is None:
            return None
        return io.BytesIO(data)

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if deleted."""
        with self._lock:
            return self._store.pop(key, None) is not None

    def keys(self) -> Iterable[str]:
        """Return current keys in insertion order."""
        with self._lock:
            return list(self._store.keys())

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def size_bytes(self, key: str) -> Optional[int]:
        """Return size in bytes of stored item or None if missing."""
        with self._lock:
            v = self._store.get(key)
            return len(v) if v is not None else None

    def total_bytes(self) -> int:
        """Return total bytes used by stored buffers (approx)."""
        with self._lock:
            return sum(len(v) for v in self._store.values())


__all__ = ["RamStore"]


if __name__ == '__main__':
    # Quick smoke test
    store = RamStore(max_items=2)
    k1 = store.store(None, b'hello')
    k2 = store.store(None, b'world')
    print('keys after two stores:', store.keys())
    # add third -> evict oldest (k1)
    k3 = store.store('third', b'!')
    print('keys after third store:', store.keys())
    print('sizes:', {k: store.size_bytes(k) for k in store.keys()})