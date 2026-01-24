#!/usr/bin/env python3
"""
Quick test of NPZ feature extraction and loading
Tests the new optimized pipeline on a small subset
"""

import sys
import numpy as np
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

import config
from feature_loader import load_mel_features, load_mfcc_features, get_feature_stats

def test_npz_features():
    """Test if NPZ features exist and can be loaded."""
    
    print("\n" + "="*80)
    print("🧪 TESTING NPZ FEATURE EXTRACTION & LOADING")
    print("="*80 + "\n")
    
    features_dir = Path(config.EXTRACTED_FEATURES_DIR)
    
    # Check what features exist
    for split in ['train', 'val', 'test']:
        print(f"\n📁 Checking {split.upper()} split:")
        print("-" * 80)
        
        for feature_type in ['mel', 'mfcc']:
            # Check NPZ
            npz_path = features_dir / f"{feature_type}_{split}.npz"
            json_path = features_dir / f"{feature_type}_{split}.json"
            
            if npz_path.exists():
                size_mb = npz_path.stat().st_size / (1024**2)
                print(f"  ✅ {feature_type.upper()} NPZ: {npz_path.name} ({size_mb:.2f} MB)")
                
                # Test loading
                try:
                    if feature_type == 'mel':
                        features, labels, mapping = load_mel_features(split)
                    else:
                        features, labels, mapping = load_mfcc_features(split)
                    
                    print(f"     Shape: {features.shape}, Labels: {len(labels)}, Classes: {mapping}")
                except Exception as e:
                    print(f"     ❌ Load failed: {e}")
                    
            elif json_path.exists():
                size_mb = json_path.stat().st_size / (1024**2)
                print(f"  📄 {feature_type.upper()} JSON (legacy): {json_path.name} ({size_mb:.2f} MB)")
            else:
                print(f"  ⚠️  {feature_type.upper()}: No features found")
    
    print("\n" + "="*80)
    print("✓ Feature test complete")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_npz_features()
