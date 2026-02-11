"""
Tests for difficulty scoring module.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from augment_v5.difficulty import (
    compute_difficulty_score,
    _drone_difficulty,
    _ambient_difficulty,
    _compute_rms_variability,
    _compute_spectral_simplicity,
    _compute_temporal_stability
)


class TestDroneDifficulty:
    """Test difficulty scoring for drone samples."""
    
    def test_low_snr_harder_than_high_snr(self):
        """Lower SNR should produce higher difficulty."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        meta_low = {'snr_est_db': 0.0, 'distance_m': 50.0}
        meta_high = {'snr_est_db': 30.0, 'distance_m': 50.0}
        
        diff_low = compute_difficulty_score(audio, meta_low, label=1)
        diff_high = compute_difficulty_score(audio, meta_high, label=1)
        
        assert diff_low > diff_high, "Low SNR should be harder"
        
    def test_far_distance_harder_than_close(self):
        """Greater distance should produce higher difficulty."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        meta_close = {'snr_est_db': 15.0, 'distance_m': 10.0}
        meta_far = {'snr_est_db': 15.0, 'distance_m': 200.0}
        
        diff_close = compute_difficulty_score(audio, meta_close, label=1)
        diff_far = compute_difficulty_score(audio, meta_far, label=1)
        
        assert diff_far > diff_close, "Far distance should be harder"
    
    def test_augmentation_increases_difficulty(self):
        """Augmented samples should be harder."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        meta_plain = {'snr_est_db': 15.0, 'sources': []}
        meta_aug = {
            'snr_est_db': 15.0,
            'sources': [
                {'doppler_shift_hz': 100},
                {'reverb_rt60': 0.5},
                {'adc_bits': 12}
            ]
        }
        
        diff_plain = compute_difficulty_score(audio, meta_plain, label=1)
        diff_aug = compute_difficulty_score(audio, meta_aug, label=1)
        
        assert diff_aug > diff_plain, "Augmented should be harder"
    
    def test_score_in_valid_range(self):
        """Score should be in [0, 1]."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        meta = {'snr_est_db': 10.0, 'distance_m': 100.0}
        
        score = compute_difficulty_score(audio, meta, label=1)
        
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"


class TestAmbientDifficulty:
    """Test difficulty scoring for non-drone samples."""
    
    def test_tonal_harder_than_noise(self):
        """Tonal/simple signals should be harder (confuse with drone)."""
        sr = 22050
        t = np.linspace(0, 1, sr)
        
        # Pure tone (simple)
        audio_tonal = (np.sin(2 * np.pi * 440 * t) * 0.3).astype('float32')
        # White noise (complex)
        audio_noise = (np.random.randn(sr) * 0.3).astype('float32')
        
        meta = {}
        
        diff_tonal = compute_difficulty_score(audio_tonal, meta, label=0)
        diff_noise = compute_difficulty_score(audio_noise, meta, label=0)
        
        assert diff_tonal > diff_noise, "Tonal should be harder than noise"
    
    def test_stable_harder_than_variable(self):
        """Stable amplitude should be harder (resembles constant drone)."""
        sr = 22050
        
        # Stable amplitude
        audio_stable = (np.ones(sr) * 0.2).astype('float32')
        # Variable amplitude (envelope modulation)
        t = np.linspace(0, 1, sr)
        envelope = 0.1 + 0.15 * np.sin(2 * np.pi * 3 * t)
        audio_variable = (envelope * np.random.randn(sr)).astype('float32')
        
        meta = {}
        
        diff_stable = compute_difficulty_score(audio_stable, meta, label=0)
        diff_variable = compute_difficulty_score(audio_variable, meta, label=0)
        
        assert diff_stable > diff_variable, "Stable should be harder"
    
    def test_hard_negative_increases_difficulty(self):
        """Hard negative flag should increase difficulty."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        meta_regular = {'hard_negative_used': False}
        meta_hard = {'hard_negative_used': True}
        
        diff_regular = compute_difficulty_score(audio, meta_regular, label=0)
        diff_hard = compute_difficulty_score(audio, meta_hard, label=0)
        
        assert diff_hard > diff_regular, "Hard negative should be harder"
    
    def test_score_in_valid_range(self):
        """Score should be in [0, 1]."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        meta = {}
        
        score = compute_difficulty_score(audio, meta, label=0)
        
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"


class TestHelperFunctions:
    """Test individual helper functions."""
    
    def test_rms_variability_constant_signal(self):
        """Constant signal should have low variability."""
        audio = np.ones(22050).astype('float32') * 0.5
        
        var_score = _compute_rms_variability(audio, {})
        
        assert var_score < 0.1, "Constant signal should have low variability"
    
    def test_rms_variability_varying_signal(self):
        """Varying signal should have higher variability."""
        # Create signal with amplitude modulation
        sr = 22050
        t = np.linspace(0, 1, sr)
        envelope = 0.1 + 0.4 * np.abs(np.sin(2 * np.pi * 2 * t))
        audio = (envelope * np.random.randn(sr)).astype('float32')
        
        var_score = _compute_rms_variability(audio, {})
        
        assert var_score > 0.2, "Varying signal should have higher variability"
    
    def test_spectral_simplicity_pure_tone(self):
        """Pure tone should have high simplicity score."""
        sr = 22050
        t = np.linspace(0, 1, sr)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype('float32')
        
        simple_score = _compute_spectral_simplicity(audio, {})
        
        assert simple_score > 0.6, "Pure tone should be spectrally simple"
    
    def test_spectral_simplicity_white_noise(self):
        """White noise should have low simplicity score."""
        audio = np.random.randn(22050).astype('float32') * 0.5
        
        simple_score = _compute_spectral_simplicity(audio, {})
        
        assert simple_score < 0.5, "White noise should be spectrally complex"
    
    def test_temporal_stability_constant(self):
        """Constant amplitude should be stable."""
        audio = np.ones(22050).astype('float32') * 0.3
        
        stab_score = _compute_temporal_stability(audio, {})
        
        assert stab_score > 0.7, "Constant signal should be temporally stable"
    
    def test_temporal_stability_modulated(self):
        """Amplitude-modulated signal should be less stable."""
        sr = 22050
        t = np.linspace(0, 1, sr)
        envelope = 0.1 + 0.4 * np.abs(np.sin(2 * np.pi * 5 * t))
        audio = (envelope * np.random.randn(sr)).astype('float32')
        
        stab_score = _compute_temporal_stability(audio, {})
        
        assert stab_score < 0.5, "Modulated signal should be less stable"


class TestDistanceExtraction:
    """Test distance extraction from sources metadata."""
    
    def test_distance_extracted_from_sources(self):
        """Distance should be extracted from sources[].distance_m."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        meta = {
            'snr_est_db': 10.0,
            'sources': [
                {'role': 'drone', 'distance_m': 250.0},
                {'role': 'ambient'}
            ]
        }
        
        config = {'w_snr': 0.0, 'w_distance': 1.0, 'w_rms_var': 0.0, 'w_augmentation': 0.0, 'distance_max_m': 450.0}
        score = compute_difficulty_score(audio, meta, label=1, config=config)
        
        # With distance=250m, distance_max=450m, score should be ~0.56
        assert 0.45 <= score <= 0.65, f"Distance 250/450 should give score ~0.56, got {score}"
    
    def test_distance_fallback_when_missing(self):
        """Should use distance=0 when not in sources."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        meta = {
            'snr_est_db': 10.0,
            'sources': [
                {'role': 'ambient'}  # No drone source with distance
            ]
        }
        
        score = compute_difficulty_score(audio, meta, label=1)
        
        # Should not crash, use fallback distance=0
        assert 0.0 <= score <= 1.0
    
    def test_top_level_distance_still_works(self):
        """Top-level distance_m should still work if present."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        meta = {
            'snr_est_db': 10.0,
            'distance_m': 300.0  # Top-level
        }
        
        config = {'w_snr': 0.0, 'w_distance': 1.0, 'w_rms_var': 0.0, 'w_augmentation': 0.0, 'distance_max_m': 450.0}
        score = compute_difficulty_score(audio, meta, label=1, config=config)
        
        # With distance=300m, score should be ~0.67
        assert 0.55 <= score <= 0.75, f"Distance 300/450 should give score ~0.67, got {score}"


class TestPureDroneHandling:
    """Test special handling for pure drone samples (no ambient)."""
    
    def test_pure_drone_detection(self):
        """Pure drones (SNR > 100 dB) should have very low difficulty."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        meta = {
            'snr_est_db': 210.0,  # Pure drone signature
            'sources': [
                {'role': 'drone', 'distance_m': 5.0}
            ]
        }
        
        score = compute_difficulty_score(audio, meta, label=1)
        
        # Pure drone at 5m should be very easy
        assert score < 0.15, f"Pure drone should have low difficulty, got {score}"
    
    def test_pure_drone_distance_still_matters(self):
        """Even for pure drones, distance should affect difficulty."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        close_meta = {
            'snr_est_db': 215.0,
            'sources': [{'role': 'drone', 'distance_m': 5.0}]
        }
        
        far_meta = {
            'snr_est_db': 212.0,
            'sources': [{'role': 'drone', 'distance_m': 100.0}]
        }
        
        close_score = compute_difficulty_score(audio, close_meta, label=1)
        far_score = compute_difficulty_score(audio, far_meta, label=1)
        
        # Far pure drone should be harder
        assert far_score > close_score, f"Far pure drone ({far_score}) should be harder than close ({close_score})"
    
    def test_non_pure_drone_uses_snr(self):
        """Non-pure drones should use SNR in calculation."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        
        # Both at same distance but different SNR
        good_snr_meta = {
            'snr_est_db': 20.0,
            'sources': [
                {'role': 'drone', 'distance_m': 50.0},
                {'role': 'ambient'}
            ]
        }
        
        bad_snr_meta = {
            'snr_est_db': -20.0,
            'sources': [
                {'role': 'drone', 'distance_m': 50.0},
                {'role': 'ambient'}
            ]
        }
        
        good_score = compute_difficulty_score(audio, good_snr_meta, label=1)
        bad_score = compute_difficulty_score(audio, bad_snr_meta, label=1)
        
        # Bad SNR should be harder
        assert bad_score > good_score, f"Bad SNR ({bad_score}) should be harder than good SNR ({good_score})"


class TestConfigurability:
    """Test configuration options."""
    
    def test_custom_weights(self):
        """Custom weights should affect score."""
        audio = np.random.randn(22050).astype('float32') * 0.1
        meta = {'snr_est_db': 10.0, 'distance_m': 100.0}
        
        config_default = {}
        config_snr_heavy = {'w_snr': 0.9, 'w_distance': 0.05, 'w_rms_var': 0.025, 'w_augmentation': 0.025}
        
        score_default = compute_difficulty_score(audio, meta, label=1, config=config_default)
        score_snr_heavy = compute_difficulty_score(audio, meta, label=1, config=config_snr_heavy)
        
        # Scores should differ with different weights
        assert score_default != score_snr_heavy
    
    def test_custom_snr_range(self):
        """Custom SNR range should affect scaling."""
        # Use deterministic audio to avoid RMS variability interference
        np.random.seed(42)
        audio = (np.ones(22050) * 0.1).astype('float32')  # constant signal
        
        # Test with two different SNR values in different ranges
        meta_low = {'snr_est_db': 5.0, 'distance_m': 0.0}
        meta_high = {'snr_est_db': 15.0, 'distance_m': 0.0}
        
        # Wide range: both 5dB and 15dB are relatively close
        config_wide = {'snr_min_db': -20.0, 'snr_max_db': 40.0, 'w_snr': 1.0, 'w_distance': 0.0, 'w_rms_var': 0.0, 'w_augmentation': 0.0}
        
        score_low_wide = compute_difficulty_score(audio, meta_low, label=1, config=config_wide)
        score_high_wide = compute_difficulty_score(audio, meta_high, label=1, config=config_wide)
        
        # With wide range, difference should be small
        diff_wide = abs(score_low_wide - score_high_wide)
        
        # Narrow range: 5dB vs 15dB spans the whole range
        config_narrow = {'snr_min_db': 5.0, 'snr_max_db': 15.0, 'w_snr': 1.0, 'w_distance': 0.0, 'w_rms_var': 0.0, 'w_augmentation': 0.0}
        
        score_low_narrow = compute_difficulty_score(audio, meta_low, label=1, config=config_narrow)
        score_high_narrow = compute_difficulty_score(audio, meta_high, label=1, config=config_narrow)
        
        # With narrow range, difference should be large
        diff_narrow = abs(score_low_narrow - score_high_narrow)
        
        assert diff_narrow > diff_wide, f"Narrow range should magnify differences: {diff_narrow} vs {diff_wide}"
        assert diff_narrow > 0.5, f"Narrow range difference too small: {diff_narrow}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
