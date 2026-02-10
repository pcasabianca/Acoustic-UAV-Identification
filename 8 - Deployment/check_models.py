#!/usr/bin/env python3
"""Quick diagnostic to check which models are present and loadable."""

import sys
from pathlib import Path

print("=" * 70)
print("MODEL FILES DIAGNOSTIC")
print("=" * 70)

models_dir = Path("models")

# List all .keras files
print("\n1. Files in models/:")
keras_files = sorted(models_dir.glob("*.keras"))
for f in keras_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  - {f.name} ({size_mb:.1f} MB)")

print(f"\nTotal: {len(keras_files)} files")

# Check archives
archives = models_dir / "archives"
if archives.exists():
    print("\n2. Files in models/archives/:")
    archive_files = sorted(archives.glob("*.keras"))
    for f in archive_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    print(f"\nTotal: {len(archive_files)} files")

# Try loading each
print("\n3. Testing model loading:")
try:
    import tensorflow as tf
    
    all_files = list(models_dir.glob("*.keras")) + list(models_dir.glob("archives/*.keras"))
    
    for model_file in all_files:
        try:
            print(f"\n  Testing: {model_file.name}")
            model = tf.keras.models.load_model(model_file, compile=False)
            print(f"    ✓ Loaded successfully")
            print(f"    Input shape: {model.input_shape}")
            print(f"    Output shape: {model.output_shape}")
        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            
except ImportError:
    print("  [SKIP] TensorFlow not installed, cannot test loading")

print("\n" + "=" * 70)
