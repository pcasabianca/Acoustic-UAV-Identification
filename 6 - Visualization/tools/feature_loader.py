"""
Universal Feature Loader for MEL and MFCC
Auto-detects NPZ (fast) or JSON (legacy) formats
Provides unified API for all trainers and scripts
"""

import json
import numpy as np
from pathlib import Path
import sys

# Import centralized config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_mel_features(split='train', prefer_npz=True):
    """
    Load MEL spectrogram features with auto-detection of NPZ or JSON format.
    
    Args:
        split: 'train', 'val', or 'test'
        prefer_npz: If True, try NPZ first (10x faster)
    
    Returns:
        tuple: (mels, labels, mapping) where:
            - mels: numpy array of shape (n_samples, n_mels, time_frames)
            - labels: numpy array of shape (n_samples,)
            - mapping: list of class names ['0', '1']
    
    Examples:
        >>> mels, labels, mapping = load_mel_features('train')
        >>> print(f"Loaded {len(mels)} samples, shape: {mels[0].shape}")
    """
    features_dir = Path(config.EXTRACTED_FEATURES_DIR)
    
    # Try NPZ first if preferred
    if prefer_npz:
        npz_path = features_dir / f"mel_{split}.npz"
        if npz_path.exists():
            return _load_mel_npz(npz_path)
    
    # Fallback to JSON
    json_path = features_dir / f"mel_{split}.json"
    if json_path.exists():
        return _load_mel_json(json_path)
    
    # Try NPZ as last resort if not preferred initially
    if not prefer_npz:
        npz_path = features_dir / f"mel_{split}.npz"
        if npz_path.exists():
            return _load_mel_npz(npz_path)
    
    raise FileNotFoundError(
        f"No MEL features found for split '{split}'. "
        f"Tried: {npz_path}, {json_path}"
    )


def load_mfcc_features(split='train', prefer_npz=True):
    """
    Load MFCC features with auto-detection of NPZ or JSON format.
    
    Args:
        split: 'train', 'val', or 'test'
        prefer_npz: If True, try NPZ first (10x faster)
    
    Returns:
        tuple: (mfccs, labels, mapping) where:
            - mfccs: numpy array of shape (n_samples, n_mfcc, time_frames)
            - labels: numpy array of shape (n_samples,)
            - mapping: list of class names ['0', '1']
    
    Examples:
        >>> mfccs, labels, mapping = load_mfcc_features('train')
        >>> print(f"Loaded {len(mfccs)} samples, shape: {mfccs[0].shape}")
    """
    features_dir = Path(config.EXTRACTED_FEATURES_DIR)
    
    # Try NPZ first if preferred
    if prefer_npz:
        npz_path = features_dir / f"mfcc_{split}.npz"
        if npz_path.exists():
            return _load_mfcc_npz(npz_path)
    
    # Fallback to JSON
    json_path = features_dir / f"mfcc_{split}.json"
    if json_path.exists():
        return _load_mfcc_json(json_path)
    
    # Try NPZ as last resort
    if not prefer_npz:
        npz_path = features_dir / f"mfcc_{split}.npz"
        if npz_path.exists():
            return _load_mfcc_npz(npz_path)
    
    raise FileNotFoundError(
        f"No MFCC features found for split '{split}'. "
        f"Tried: {npz_path}, {json_path}"
    )


def _load_mel_npz(path):
    """Load MEL features from compressed NPZ format (fast)."""
    data = np.load(str(path), allow_pickle=False)
    mels = data['mels']
    labels = data['labels']
    mapping = data['mapping'].tolist() if 'mapping' in data else ['0', '1']
    return mels, labels, mapping


def _load_mel_json(path):
    """Load MEL features from JSON format (legacy, slow)."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    mels = np.array(data['mel'], dtype=np.float32)
    labels = np.array(data['labels'], dtype=np.int32)
    mapping = data.get('mapping', ['0', '1'])
    
    return mels, labels, mapping


def _load_mfcc_npz(path):
    """Load MFCC features from compressed NPZ format (fast)."""
    data = np.load(str(path), allow_pickle=False)
    mfccs = data['mfccs']
    labels = data['labels']
    mapping = data['mapping'].tolist() if 'mapping' in data else ['0', '1']
    return mfccs, labels, mapping


def _load_mfcc_json(path):
    """Load MFCC features from JSON format (legacy, slow)."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    mfccs = np.array(data['mfcc'], dtype=np.float32)
    labels = np.array(data['labels'], dtype=np.int32)
    mapping = data.get('mapping', ['0', '1'])
    
    return mfccs, labels, mapping


def get_feature_stats(split='train', feature_type='mel'):
    """
    Get statistics about extracted features without loading full data.
    
    Args:
        split: 'train', 'val', or 'test'
        feature_type: 'mel' or 'mfcc'
    
    Returns:
        dict: Statistics including file size, format, sample count
    """
    features_dir = Path(config.EXTRACTED_FEATURES_DIR)
    
    stats = {
        'split': split,
        'feature_type': feature_type,
        'format': None,
        'file_size_mb': 0,
        'num_samples': 0,
        'shape': None
    }
    
    # Check NPZ
    npz_path = features_dir / f"{feature_type}_{split}.npz"
    if npz_path.exists():
        stats['format'] = 'npz'
        stats['file_size_mb'] = npz_path.stat().st_size / (1024 * 1024)
        
        # Quick peek without loading all data
        data = np.load(str(npz_path), allow_pickle=False)
        key = 'mels' if feature_type == 'mel' else 'mfccs'
        stats['num_samples'] = len(data[key])
        stats['shape'] = data[key].shape
        return stats
    
    # Check JSON
    json_path = features_dir / f"{feature_type}_{split}.json"
    if json_path.exists():
        stats['format'] = 'json'
        stats['file_size_mb'] = json_path.stat().st_size / (1024 * 1024)
        
        # For JSON, we need to load to get count (slow)
        with open(json_path, 'r') as f:
            data = json.load(f)
        key = 'mel' if feature_type == 'mel' else 'mfcc'
        stats['num_samples'] = len(data[key])
        if stats['num_samples'] > 0:
            stats['shape'] = np.array(data[key][0]).shape
        return stats
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test feature loader')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--type', choices=['mel', 'mfcc'], default='mel')
    parser.add_argument('--stats-only', action='store_true', help='Show stats without loading')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"FEATURE LOADER TEST - {args.type.upper()} {args.split}")
    print(f"{'='*60}\n")
    
    if args.stats_only:
        stats = get_feature_stats(args.split, args.type)
        print(f"Format: {stats['format']}")
        print(f"File size: {stats['file_size_mb']:.2f} MB")
        print(f"Samples: {stats['num_samples']}")
        print(f"Shape: {stats['shape']}")
    else:
        import time
        start = time.time()
        
        if args.type == 'mel':
            features, labels, mapping = load_mel_features(args.split)
        else:
            features, labels, mapping = load_mfcc_features(args.split)
        
        elapsed = time.time() - start
        
        print(f"Loaded in {elapsed:.2f}s")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Class mapping: {mapping}")
        print(f"Memory usage: {features.nbytes / (1024**2):.2f} MB")
    
    print(f"\n{'='*60}\n")