#!/usr/bin/env python3
"""
Test Deployment System
Validates the deployment setup before using on Raspberry Pi.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def test_config():
    """Test configuration file."""
    print("=" * 70)
    print("TEST 1: Configuration Validation")
    print("=" * 70)
    
    config_path = Path("deployment_config.json")
    
    if not config_path.exists():
        print("  ✗ Config file not found!")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required = ['audio', 'detection', 'models', 'feature_extraction', 'output']
        all_ok = True
        
        for section in required:
            if section in config:
                print(f"  [OK] Section '{section}' present")
            else:
                print(f"  [MISSING] Section '{section}' MISSING!")
                all_ok = False
        
        return all_ok
        
    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON: {e}")
        return False


def test_directories():
    """Test directory structure."""
    print("\n" + "=" * 70)
    print("TEST 2: Directory Structure")
    print("=" * 70)
    
    required_dirs = ['models', 'audio_input', 'logs']
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  [OK] Directory '{dir_name}' exists")
        else:
            print(f"  [MISSING] Directory '{dir_name}' missing!")
            all_ok = False
    
    return all_ok


def test_models():
    """Test model files."""
    print("\n" + "=" * 70)
    print("TEST 3: Model Files")
    print("=" * 70)
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("  [MISSING] Models directory not found!")
        return False
    
    model_files = list(models_dir.glob("*.keras"))
    
    if not model_files:
        print("  [MISSING] No model files found!")
        print("  -> Run ./setup_deployment.sh to copy models")
        return False
    
    print(f"  Found {len(model_files)} model(s):")
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"    - {model_file.name} ({size_mb:.2f} MB)")
    
    return True


def test_dependencies():
    """Test Python dependencies."""
    print("\n" + "=" * 70)
    print("TEST 4: Python Dependencies")
    print("=" * 70)
    
    dependencies = {
        'tensorflow': 'TensorFlow',
        'librosa': 'Librosa',
        'numpy': 'NumPy',
    }
    
    all_ok = True
    
    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"  [OK] {display_name} installed")
        except ImportError:
            print(f"  [MISSING] {display_name} NOT installed!")
            all_ok = False
    
    # Optional dependencies
    optional = {
        'pyaudio': 'PyAudio (for recording)',
    }
    
    print("\n  Optional dependencies:")
    for module_name, display_name in optional.items():
        try:
            __import__(module_name)
            print(f"    [OK] {display_name}")
        except ImportError:
            print(f"    [MISSING] {display_name} (not needed for file monitoring)")
    
    return all_ok


def test_detector_import():
    """Test detector script import."""
    print("\n" + "=" * 70)
    print("TEST 5: Detector Script")
    print("=" * 70)
    
    detector_path = Path("drone_detector.py")
    
    if not detector_path.exists():
        print("  [MISSING] drone_detector.py not found!")
        return False

    print(f"  [OK] drone_detector.py exists")
    
    # Try to import (syntax check)
    try:
        import drone_detector
        print(f"  [OK] Script imports successfully")
        return True
    except Exception as e:
        print(f"  [ERROR] Import failed: {e}")
        return False


def test_sample_audio():
    """Test with sample audio if available."""
    print("\n" + "=" * 70)
    print("TEST 6: Sample Audio Processing")
    print("=" * 70)
    
    # Look for sample audio in project
    sample_paths = [
        "../0 - DADS dataset extraction/dataset_test/1/aug_drone_500m_00001.wav",
        "../0 - DADS dataset extraction/dataset_test/0/orig_dads_0_00001.wav",
    ]
    
    sample_found = None
    for path in sample_paths:
        if Path(path).exists():
            sample_found = path
            break
    
    if not sample_found:
        print("  [MISSING] No sample audio found (skipped)")
        print("    -> Copy a WAV file to audio_input/ to test manually")
        return True  # Not a failure
    
    print(f"  Found sample: {Path(sample_found).name}")
    
    # Try to process with detector
    try:
        import drone_detector
        detector = drone_detector.DroneDetector()
        
        # Extract features
        features = detector.extract_mel_features(Path(sample_found))
        
        if features is not None:
            print(f"  ✓ Feature extraction works (shape: {features.shape})")
            
            # Try prediction if models loaded
            if detector.models:
                predictions = detector.predict(features)
                print(f"  ✓ Prediction works")
                for model_name, prob in predictions.items():
                    print(f"    • {model_name}: {prob:.2%}")
            else:
                print("  ⊗ No models loaded (can't test prediction)")
            
            return True
        else:
            print("  ✗ Feature extraction failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DEPLOYMENT SYSTEM TEST SUITE")
    print("=" * 70)
    
    # Change to deployment directory
    os.chdir(Path(__file__).parent)
    
    results = {
        'Configuration': test_config(),
        'Directories': test_directories(),
        'Models': test_models(),
        'Dependencies': test_dependencies(),
        'Detector Script': test_detector_import(),
        'Sample Processing': test_sample_audio(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    if passed == total:
        print("\nAll tests passed! Ready for deployment.")
        print("\nNext steps:")
        print("  1. Transfer to Raspberry Pi:")
        print("     scp -r . pi@raspberrypi:/home/pi/drone-detection/")
        print("\n  2. On Raspberry Pi, run:")
        print("     python3 drone_detector.py --continuous")
    else:
        print("\nSome tests failed. Please fix issues before deployment.")
        
        if not results['Models']:
            print("\nTip: Run ./setup_deployment.sh to copy trained models")
        
        if not results['Dependencies']:
            print("\nTip: Install dependencies with:")
            print("     pip3 install tensorflow librosa numpy")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
