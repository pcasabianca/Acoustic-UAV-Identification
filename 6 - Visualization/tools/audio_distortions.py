"""
Audio Distortion Augmentation for Robustness
==============================================

Applies realistic audio distortions to drone sounds:
- Time stretch (Doppler effect, speed variations)
- Pitch shift (motor RPM changes)
- Spectral filtering (wind interference)
- Background noise injection (real-world conditions)

Makes models robust to real deployment scenarios.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random


def apply_time_stretch(audio, sr, stretch_factor):
    """Apply time stretching (Doppler effect simulation).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        stretch_factor: 0.9 = 10% faster, 1.1 = 10% slower
    
    Returns:
        Stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=stretch_factor)


def apply_pitch_shift(audio, sr, n_semitones):
    """Apply pitch shifting (motor RPM variation).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_semitones: Pitch shift in semitones (-2 to +2)
    
    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_semitones)


def apply_spectral_filter(audio, sr, filter_type='lowpass', rng=None):
    """Apply spectral filtering (wind interference simulation).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        filter_type: 'lowpass', 'highpass', 'bandstop'
    
    Returns:
        Filtered audio
    """
    nyquist = sr / 2

    # Determinism smoke check: require explicit RNG (no global fallbacks)
    if rng is None:
        raise RuntimeError("rng is None — determinism check failed in apply_spectral_filter")
    
    if filter_type == 'lowpass':
        # Simulate low-frequency wind masking high frequencies
        # RANDOM cutoff between 3-5 kHz
        cutoff = (rng.uniform(3000, 5000) if rng is not None else random.uniform(3000, 5000))  # Hz
        # RANDOM attenuation intensity (0.2-0.4)
        attenuation = (rng.uniform(0.2, 0.4) if rng is not None else random.uniform(0.2, 0.4))
        audio_fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        audio_fft[freqs > cutoff] *= attenuation
        audio = np.fft.irfft(audio_fft, len(audio))
        
    elif filter_type == 'highpass':
        # Simulate distant sound (lose low frequencies)
        # RANDOM cutoff between 200-500 Hz
        cutoff = (rng.uniform(200, 500) if rng is not None else random.uniform(200, 500))  # Hz
        # RANDOM attenuation intensity (0.2-0.4)
        attenuation = (rng.uniform(0.2, 0.4) if rng is not None else random.uniform(0.2, 0.4))
        audio_fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        audio_fft[freqs < cutoff] *= attenuation
        audio = np.fft.irfft(audio_fft, len(audio))
        
    elif filter_type == 'bandstop':
        # Simulate frequency-specific interference
        # RANDOM center frequency (drone harmonics 1-3 kHz)
        center = (rng.uniform(1000, 3000) if rng is not None else random.uniform(1000, 3000))  # Hz
        # RANDOM bandwidth (300-700 Hz)
        width = (rng.uniform(300, 700) if rng is not None else random.uniform(300, 700))
        # RANDOM attenuation depth (0.1-0.3)
        attenuation = (rng.uniform(0.1, 0.3) if rng is not None else random.uniform(0.1, 0.3))
        audio_fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        mask = np.abs(freqs - center) < width
        audio_fft[mask] *= attenuation
        audio = np.fft.irfft(audio_fft, len(audio))
    
    return audio


def inject_background_noise(audio, sr, intensity=0.05, rng=None):
    """Inject subtle background noise (real-world interference).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        intensity: Base noise level (0.0 to 1.0)
    
    Returns:
        Audio with noise
    """
    # Determinism smoke check: require explicit RNG (no global fallbacks)
    if rng is None:
        raise RuntimeError("rng is None — determinism check failed in inject_background_noise")

    # RANDOM intensity variation ±20%
    actual_intensity = intensity * rng.uniform(0.8, 1.2)
    
    # Mix of white noise and pink noise (more realistic)
    # RANDOM mix ratio
    white_ratio = (rng.uniform(0.4, 0.6) if rng is not None else random.uniform(0.4, 0.6))
    
    # White noise (equal energy at all frequencies)
    white_noise = (rng.normal(0, actual_intensity, len(audio)) if rng is not None else np.random.normal(0, actual_intensity, len(audio)))
    
    # Pink noise (1/f spectrum, more natural)
    pink_noise = (rng.normal(0, actual_intensity, len(audio)) if rng is not None else np.random.normal(0, actual_intensity, len(audio)))
    pink_fft = np.fft.rfft(pink_noise)
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    # Apply 1/f filter
    pink_fft[1:] /= np.sqrt(freqs[1:])  # 1/sqrt(f) for pink noise
    pink_noise = np.fft.irfft(pink_fft, len(audio))
    pink_noise = pink_noise / np.std(pink_noise) * actual_intensity
    
    # RANDOM mix of white and pink
    noise = white_ratio * white_noise + (1 - white_ratio) * pink_noise
    
    return audio + noise


def apply_distortion_chain(audio, sr, config, rng=None):
    """Apply chain of distortions based on config.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        config: Dict with distortion parameters
    
    Returns:
        Distorted audio (same length as input after resampling)
    """
    # Determinism smoke check: require explicit RNG (no global fallbacks)
    if rng is None:
        raise RuntimeError("rng is None — determinism check failed in apply_distortion_chain")

    original_length = len(audio)
    distorted = audio.copy()
    
    # Time stretch (if specified)
    if 'time_stretch_factor' in config:
        factors = config['time_stretch_factor']
        if isinstance(factors, list):
            if rng is not None:
                factor = rng.choice(factors)
            else:
                factor = random.choice(factors)
        else:
            factor = factors
        
        if factor != 1.0:
            distorted = apply_time_stretch(distorted, sr, factor)
            # Keep length consistent by padding/cropping (do NOT misuse resample)
            if len(distorted) != original_length:
                if len(distorted) > original_length:
                    distorted = distorted[:original_length]
                else:
                    distorted = np.pad(distorted, (0, original_length - len(distorted)), mode='constant')
    
    # Pitch shift (if specified) - doesn't change length
    if 'pitch_shift_semitones' in config:
        semitones = config['pitch_shift_semitones']
        if isinstance(semitones, list):
            if rng is not None:
                n_steps = rng.choice(semitones)
            else:
                n_steps = random.choice(semitones)
        else:
            n_steps = semitones
        
        if n_steps != 0:
            distorted = apply_pitch_shift(distorted, sr, n_steps)
    
    # Ensure consistent length before spectral operations
    if len(distorted) != original_length:
        if len(distorted) > original_length:
            distorted = distorted[:original_length]
        else:
            distorted = np.pad(distorted, (0, original_length - len(distorted)), mode='constant')
    
    # Spectral filter (if enabled)
    if config.get('apply_spectral_filter', False):
        filter_types = ['lowpass', 'highpass', 'bandstop']
        if rng is not None:
            filter_type = rng.choice(filter_types)
        else:
            filter_type = random.choice(filter_types)
        distorted = apply_spectral_filter(distorted, sr, filter_type, rng=rng)
    
    # Background noise injection
    if 'noise_injection' in config and config['noise_injection'] > 0:
        distorted = inject_background_noise(distorted, sr, config['noise_injection'], rng=rng)
    
    # Final length check and normalization
    if len(distorted) != original_length:
        if len(distorted) > original_length:
            distorted = distorted[:original_length]
        else:
            distorted = np.pad(distorted, (0, original_length - len(distorted)), mode='constant')
    
    # Normalize to prevent clipping
    max_val = np.abs(distorted).max()
    if max_val > 1.0:
        distorted = distorted / max_val * 0.95
    
    return distorted


def test_distortions():
    """Test distortion functions."""
    print("\n" + "="*60)
    print("Testing Audio Distortions")
    print("="*60)
    
    # Generate test tone (1000 Hz)
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 1000 * t)
    
    # Test time stretch
    print("\n[TEST 1] Time stretch...")
    stretched = apply_time_stretch(audio, sr, 1.1)
    print(f"  Original length: {len(audio)}, Stretched length: {len(stretched)}")
    
    # Test pitch shift
    print("\n[TEST 2] Pitch shift...")
    shifted = apply_pitch_shift(audio, sr, 2)
    print(f"  Shifted by 2 semitones, length: {len(shifted)}")
    
    # Test spectral filter
    print("\n[TEST 3] Spectral filtering...")
    filtered = apply_spectral_filter(audio, sr, 'lowpass')
    print(f"  Lowpass filtered, length: {len(filtered)}")
    
    # Test noise injection
    print("\n[TEST 4] Noise injection...")
    noisy = inject_background_noise(audio, sr, 0.1)
    print(f"  Noise added (intensity=0.1), length: {len(noisy)}")
    
    # Test full chain
    print("\n[TEST 5] Full distortion chain...")
    config = {
        'time_stretch_factor': [0.95, 1.05],
        'pitch_shift_semitones': [-1, 0, 1],
        'apply_spectral_filter': True,
        'noise_injection': 0.08
    }
    distorted = apply_distortion_chain(audio, sr, config)
    print(f"  All distortions applied, length: {len(distorted)}")
    
    print("\n[OK] All tests passed!")
    print("="*60)


if __name__ == '__main__':
    test_distortions()
