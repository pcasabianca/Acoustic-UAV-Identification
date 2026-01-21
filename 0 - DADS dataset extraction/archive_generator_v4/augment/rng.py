import zlib
import numpy as np


def seed_from_key(master_seed: int, seed_key: str) -> int:
    return (int(master_seed) + zlib.crc32(seed_key.encode('utf-8'))) & 0xFFFFFFFF


def rng_for_key(master_seed: int, seed_key: str) -> np.random.Generator:
    seed = seed_from_key(master_seed, seed_key)
    return np.random.default_rng(int(seed))
