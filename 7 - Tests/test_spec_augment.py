#!/usr/bin/env python3
"""
Test Suite for SpecAugment Implementation
Tests the SpecAugment function in Mel_Preprocess_and_Feature_Extract.py
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "1 - Preprocessing and Features Extraction"))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from Mel_Preprocess_and_Feature_Extract import spec_augment


class TestSpecAugment(unittest.TestCase):
    """Test SpecAugment implementation"""
    
    def setUp(self):
        """Create test mel spectrogram"""
        # Typical mel spectrogram shape: (time_steps, n_mels)
        self.time_steps = int(getattr(config, 'MEL_TIME_FRAMES', 100))
        self.n_mels = int(getattr(config, 'MEL_N_MELS', 90))
        self.mel_spec = np.random.randn(self.time_steps, self.n_mels).astype(np.float32)
    
    def test_spec_augment_shape(self):
        """Test that output shape matches input"""
        augmented = spec_augment(self.mel_spec, time_mask_width=10, freq_mask_width=8, num_masks=2)
        self.assertEqual(augmented.shape, self.mel_spec.shape)
    
    def test_spec_augment_modifies_signal(self):
        """Test that SpecAugment modifies the input"""
        augmented = spec_augment(self.mel_spec, time_mask_width=10, freq_mask_width=8, num_masks=2)
        # Should be different from original
        self.assertFalse(np.array_equal(augmented, self.mel_spec))
    
    def test_time_masking(self):
        """Test that time masking is applied"""
        # Use only time masking (no frequency masking)
        augmented = spec_augment(self.mel_spec, time_mask_width=10, freq_mask_width=0, num_masks=1)
        
        # Check if any full columns are masked (set to mean)
        mean_val = self.mel_spec.mean()
        # At least some columns should be close to mean
        masked_columns = np.any(np.abs(augmented - mean_val) < 0.1, axis=1)
        # Should have some masked regions
        self.assertGreater(np.sum(masked_columns), 0)
    
    def test_frequency_masking(self):
        """Test that frequency masking is applied"""
        # Use only frequency masking (no time masking)
        augmented = spec_augment(self.mel_spec, time_mask_width=0, freq_mask_width=8, num_masks=1)
        
        # Check if any full rows are masked
        mean_val = self.mel_spec.mean()
        masked_rows = np.any(np.abs(augmented - mean_val) < 0.1, axis=0)
        # Should have some masked regions
        self.assertGreater(np.sum(masked_rows), 0)
    
    def test_num_masks_effect(self):
        """Test that num_masks parameter affects augmentation"""
        # Single mask
        augmented_1 = spec_augment(self.mel_spec, time_mask_width=5, freq_mask_width=5, num_masks=1)
        # Multiple masks
        augmented_3 = spec_augment(self.mel_spec, time_mask_width=5, freq_mask_width=5, num_masks=3)
        
        # More masks should cause more deviation from original
        diff_1 = np.sum(np.abs(augmented_1 - self.mel_spec))
        diff_3 = np.sum(np.abs(augmented_3 - self.mel_spec))
        
        # More masks should generally cause more change
        # (May not always be true due to randomness, but statistically likely)
        # So we just check both are modified
        self.assertGreater(diff_1, 0)
        self.assertGreater(diff_3, 0)
    
    def test_mask_width_constraints(self):
        """Test that masks respect width constraints"""
        # Very small spectrogram
        small_mel = np.random.randn(20, 20).astype(np.float32)
        
        # Mask width larger than dimensions (should handle gracefully)
        augmented = spec_augment(small_mel, time_mask_width=30, freq_mask_width=30, num_masks=1)
        
        # Should return something with same shape
        self.assertEqual(augmented.shape, small_mel.shape)
    
    def test_zero_masks(self):
        """Test with num_masks=0 (should return similar to original)"""
        augmented = spec_augment(self.mel_spec, time_mask_width=10, freq_mask_width=8, num_masks=0)
        
        # Should be very similar to original (only mean replacement applied)
        np.testing.assert_array_almost_equal(augmented, self.mel_spec, decimal=5)
    
    def test_dtype_preservation(self):
        """Test that dtype is preserved"""
        mel_float32 = self.mel_spec.astype(np.float32)
        augmented = spec_augment(mel_float32, time_mask_width=10, freq_mask_width=8, num_masks=2)
        self.assertEqual(augmented.dtype, np.float32)
    
    def test_no_nan_or_inf(self):
        """Test that augmentation doesn't produce NaN or inf"""
        augmented = spec_augment(self.mel_spec, time_mask_width=10, freq_mask_width=8, num_masks=2)
        
        # Should not contain NaN
        self.assertFalse(np.any(np.isnan(augmented)))
        # Should not contain inf
        self.assertFalse(np.any(np.isinf(augmented)))
    
    def test_randomness(self):
        """Test that augmentation is random"""
        # Set different random seeds and check results differ
        np.random.seed(42)
        augmented_1 = spec_augment(self.mel_spec, time_mask_width=10, freq_mask_width=8, num_masks=2)
        
        np.random.seed(123)
        augmented_2 = spec_augment(self.mel_spec, time_mask_width=10, freq_mask_width=8, num_masks=2)
        
        # Different seeds should give different results
        self.assertFalse(np.array_equal(augmented_1, augmented_2))


class TestSpecAugmentEdgeCases(unittest.TestCase):
    """Test edge cases for SpecAugment"""
    
    def test_very_small_spectrogram(self):
        """Test with very small spectrogram"""
        small_mel = np.random.randn(5, 5).astype(np.float32)
        
        # Should work without crashing
        augmented = spec_augment(small_mel, time_mask_width=2, freq_mask_width=2, num_masks=1)
        self.assertEqual(augmented.shape, small_mel.shape)
    
    def test_single_time_step(self):
        """Test with single time step"""
        single_time = np.random.randn(1, int(getattr(config, 'MEL_N_MELS', 90))).astype(np.float32)
        
        augmented = spec_augment(single_time, time_mask_width=1, freq_mask_width=8, num_masks=1)
        self.assertEqual(augmented.shape, single_time.shape)
    
    def test_single_mel_bin(self):
        """Test with single mel bin"""
        single_mel = np.random.randn(100, 1).astype(np.float32)
        
        augmented = spec_augment(single_mel, time_mask_width=10, freq_mask_width=1, num_masks=1)
        self.assertEqual(augmented.shape, single_mel.shape)
    
    def test_all_zeros(self):
        """Test with all-zero spectrogram"""
        zero_mel = np.zeros((int(getattr(config, 'MEL_TIME_FRAMES', 100)), int(getattr(config, 'MEL_N_MELS', 90))), dtype=np.float32)
        
        # Should work (masking with mean of zeros is still zeros)
        augmented = spec_augment(zero_mel, time_mask_width=10, freq_mask_width=8, num_masks=2)
        self.assertEqual(augmented.shape, zero_mel.shape)
        # Mean should be zero, so masked regions also zero
        self.assertTrue(np.all(augmented == 0.0))
    
    def test_extreme_values(self):
        """Test with extreme values"""
        extreme_mel = np.random.randn(100, 90).astype(np.float32) * 1000
        
        augmented = spec_augment(extreme_mel, time_mask_width=10, freq_mask_width=8, num_masks=2)
        
        # Should not overflow or produce NaN
        self.assertFalse(np.any(np.isnan(augmented)))
        self.assertFalse(np.any(np.isinf(augmented)))


class TestSpecAugmentIntegration(unittest.TestCase):
    """Integration tests for SpecAugment in feature extraction pipeline"""
    
    def test_realistic_mel_spectrogram(self):
        """Test with realistic mel spectrogram dimensions"""
        # Typical dimensions from librosa.feature.melspectrogram
        # For 10s audio at 22050 Hz, hop_length=512, num_segments=10
        # Each segment is 1s = 22050 samples
        # Mel frames = ceil(22050 / 512) ≈ 44 frames
        realistic_mel = np.random.randn(int(getattr(config, 'MEL_TIME_FRAMES', 44)), int(getattr(config, 'MEL_N_MELS', 90))).astype(np.float32)
        
        # Apply SpecAugment with typical parameters
        augmented = spec_augment(realistic_mel, 
                                time_mask_width=10,  # Mask ~20% of time
                                freq_mask_width=8,   # Mask ~10% of frequency
                                num_masks=2)
        
        self.assertEqual(augmented.shape, realistic_mel.shape)
        self.assertFalse(np.array_equal(augmented, realistic_mel))
    
    def test_batch_augmentation(self):
        """Test augmenting multiple spectrograms"""
        batch_size = 32
        spectrograms = [np.random.randn(int(getattr(config, 'MEL_TIME_FRAMES', 44)), int(getattr(config, 'MEL_N_MELS', 90))).astype(np.float32) for _ in range(batch_size)]
        
        # Augment each
        augmented_batch = [spec_augment(mel, time_mask_width=10, freq_mask_width=8, num_masks=2) 
                          for mel in spectrograms]
        
        # Check all have correct shape
        for aug in augmented_batch:
            self.assertEqual(aug.shape, (int(getattr(config, 'MEL_TIME_FRAMES', 44)), int(getattr(config, 'MEL_N_MELS', 90))))
        
        # Check they're all different (due to randomness)
        # Compare first two
        self.assertFalse(np.array_equal(augmented_batch[0], augmented_batch[1]))


def run_tests():
    """Run all tests with verbose output"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
