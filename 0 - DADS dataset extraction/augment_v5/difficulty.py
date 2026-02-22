"""
Difficulty scoring for acoustic samples.
Computes a normalized difficulty score [0.0, 1.0] for both drone and non-drone samples.
"""
import numpy as np


def compute_difficulty_score(audio, metadata, label, config=None):
    """
    Compute difficulty score for a sample.
    
    Args:
        audio: numpy array of audio signal
        metadata: dict with sample metadata (SNR, RMS, transforms, etc.)
        label: int, class label (0=no-drone, 1=drone)
        config: optional dict with difficulty scoring parameters
        
    Returns:
        float: difficulty score in [0.0, 1.0] where higher = more difficult
    """
    cfg = config or {}
    
    if label == 1:
        # Drone samples: based on acoustic degradation
        return _drone_difficulty(audio, metadata, cfg)
    else:
        # Non-drone samples: based on confusion potential
        return _ambient_difficulty(audio, metadata, cfg)


def _drone_difficulty(audio, metadata, cfg):
    """
    Drone difficulty based on SNR, distance, RMS variability, and augmentations.
    
    Special handling for pure drone samples (no ambient):
    - Detected when SNR > 100 dB (indicates no ambient noise)
    - SNR component excluded from difficulty calculation
    - Difficulty based only on distance, RMS variability, and augmentations
    """
    # Extract metadata fields with fallbacks
    snr_db = metadata.get('snr_est_db')
    if snr_db is None:
        snr_db = metadata.get('resulting_snr_db', 20.0)
    
    # Extract distance from sources metadata (where it's actually stored)
    distance_m = metadata.get('distance_m', None)
    if distance_m is None:
        # Distance is in sources[i].distance_m for drone sources
        sources = metadata.get('sources', [])
        for src in sources:
            if src.get('role') == 'drone' and 'distance_m' in src:
                distance_m = src['distance_m']
                break
        if distance_m is None:
            distance_m = 0.0  # Fallback if no distance found
    
    # Check if this is a pure drone sample (no ambient)
    # Pure drones have very high SNR (>100 dB) because ambient RMS ≈ 0
    is_pure_drone = float(snr_db) > 100.0
    
    # SNR component: lower SNR = harder (invert scale)
    # Skip SNR for pure drones as it's meaningless (division by near-zero)
    if is_pure_drone:
        snr_score = 0.0  # Pure drone = very easy (no noise to compete with)
        use_snr = False
    else:
        snr_min = float(cfg.get('snr_min_db', -10.0))
        snr_max = float(cfg.get('snr_max_db', 40.0))
        snr_score = 1.0 - np.clip((float(snr_db) - snr_min) / (snr_max - snr_min), 0.0, 1.0)
        use_snr = True
    
    # Distance component: farther = harder
    dist_max = float(cfg.get('distance_max_m', 450.0))  # Updated to match hard scenarios
    dist_score = np.clip(float(distance_m) / dist_max, 0.0, 1.0)
    
    # RMS temporal variability (instability makes it harder)
    rms_var_score = _compute_rms_variability(audio, cfg)
    
    # Augmentation complexity score
    aug_score = _compute_augmentation_score(metadata, cfg)
    
    # Weighted combination
    w_snr = float(cfg.get('w_snr', 0.50))
    w_dist = float(cfg.get('w_distance', 0.20))
    w_rms = float(cfg.get('w_rms_var', 0.15))
    w_aug = float(cfg.get('w_augmentation', 0.15))
    
    # For pure drones: redistribute SNR weight to distance (the primary difficulty factor)
    if not use_snr:
        # Renormalize weights without SNR
        total_w = w_dist + w_rms + w_aug
        if total_w > 0:
            w_dist_norm = w_dist / total_w
            w_rms_norm = w_rms / total_w
            w_aug_norm = w_aug / total_w
        else:
            w_dist_norm = 1.0
            w_rms_norm = 0.0
            w_aug_norm = 0.0
        difficulty = w_dist_norm * dist_score + w_rms_norm * rms_var_score + w_aug_norm * aug_score
    else:
        difficulty = w_snr * snr_score + w_dist * dist_score + w_rms * rms_var_score + w_aug * aug_score
    
    return float(np.clip(difficulty, 0.0, 1.0))


def _ambient_difficulty(audio, metadata, cfg):
    """
    Ambient (non-drone) difficulty based on similarity to drone characteristics.
    
    Hard ambients = those that confuse with drones:
    - Tonal/harmonic (like propeller hum)
    - Low frequency dominant (100-500 Hz)
    - Stable amplitude (constant sound)
    - Periodic modulation (like rotating machinery)
    """
    # 1. Tonality: tonal sounds (motors, machines) confuse with drones
    tonality_score = _compute_tonality(audio, cfg)
    
    # 2. Low-frequency energy: drones have strong low-freq content
    lowfreq_score = _compute_low_frequency_ratio(audio, cfg)
    
    # 3. Temporal stability: constant amplitude resembles drone hum
    stability_score = _compute_temporal_stability(audio, cfg)
    
    # 4. Periodic modulation: rotating machinery sounds like propellers
    modulation_score = _compute_periodic_modulation(audio, cfg)
    
    # 5. Hard negative indicator (manual tagging)
    hard_neg = 1.0 if metadata.get('hard_negative_used', False) else 0.0
    
    # Weighted combination
    w_tonal = float(cfg.get('w_tonality', 0.30))
    w_lowfreq = float(cfg.get('w_low_freq', 0.25))
    w_stab = float(cfg.get('w_stability', 0.20))
    w_mod = float(cfg.get('w_modulation', 0.15))
    w_hard = float(cfg.get('w_hard_neg', 0.10))
    
    difficulty = (w_tonal * tonality_score + 
                  w_lowfreq * lowfreq_score + 
                  w_stab * stability_score + 
                  w_mod * modulation_score + 
                  w_hard * hard_neg)
    
    return float(np.clip(difficulty, 0.0, 1.0))


def _compute_rms_variability(audio, cfg):
    """Compute RMS variability over sliding windows."""
    win_size = cfg.get('rms_window_samples', 2048)
    hop = win_size // 2
    
    if len(audio) < win_size:
        return 0.0
    
    rms_vals = []
    for i in range(0, len(audio) - win_size, hop):
        segment = audio[i:i + win_size]
        rms_vals.append(np.sqrt(np.mean(segment.astype('float64') ** 2)))
    
    if len(rms_vals) < 2:
        return 0.0
    
    # Coefficient of variation (normalized std)
    mean_rms = np.mean(rms_vals)
    std_rms = np.std(rms_vals)
    cv = std_rms / (mean_rms + 1e-9)
    
    return float(np.clip(cv * 2.0, 0.0, 1.0))  # scale to [0, 1]


def _compute_augmentation_score(metadata, cfg):
    """Score based on applied augmentations."""
    score = 0.0
    sources = metadata.get('sources', [])
    
    # Check for various augmentations in source chains
    for src in sources:
        # Doppler
        if 'doppler_shift_hz' in src or 'doppler_max_shift_hz' in src:
            score += 0.15
        # Reverberation
        if 'reverb_rt60' in src or 'rir_path' in src:
            score += 0.15
        # Time stretch
        if 'time_stretch_rate' in src:
            score += 0.10
        # Intensity modulation / flutter
        if 'flutter_freq_hz' in src or 'intensity_mod' in src:
            score += 0.10
        # MEMS/hardware effects
        if 'adc_bits' in src or 'thermal_noise_dbfs' in src:
            score += 0.20
    
    return float(np.clip(score, 0.0, 1.0))


def _compute_spectral_simplicity(audio, cfg):
    """
    Simple = tonal/harmonic = high difficulty (confuses with drone).
    Complex = noisy/chaotic = low difficulty (easy to distinguish).
    """
    # Compute power spectrum
    n_fft = cfg.get('n_fft', 2048)
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    power = spectrum ** 2
    power_norm = power / (np.sum(power) + 1e-9)
    
    # Spectral flatness: low flatness = tonal = harder
    geometric_mean = np.exp(np.mean(np.log(power_norm + 1e-9)))
    arithmetic_mean = np.mean(power_norm)
    flatness = geometric_mean / (arithmetic_mean + 1e-9)
    
    # Invert: low flatness (tonal) -> high score
    simplicity = 1.0 - flatness
    return float(np.clip(simplicity, 0.0, 1.0))


def _compute_temporal_stability(audio, cfg):
    """
    Stable envelope = constant amplitude = resembles drone = harder.
    Variable envelope = changing amplitude = easier to distinguish.
    """
    # Compute envelope via RMS over windows
    win_size = cfg.get('temp_window_samples', 1024)
    hop = win_size // 2
    
    if len(audio) < win_size:
        return 0.0
    
    envelope = []
    for i in range(0, len(audio) - win_size, hop):
        segment = audio[i:i + win_size]
        envelope.append(np.sqrt(np.mean(segment.astype('float64') ** 2)))
    
    if len(envelope) < 2:
        return 0.0
    
    # Low variance = stable = harder
    variance = np.var(envelope)
    mean_env = np.mean(envelope)
    cv = np.sqrt(variance) / (mean_env + 1e-9)
    
    # Invert: low CV (stable) -> high score
    stability = np.exp(-cv * 5.0)  # decay function
    return float(np.clip(stability, 0.0, 1.0))


def _compute_tonality(audio, cfg):
    """
    Compute tonality score using spectral flatness.
    Tonal sounds (harmonic, like motors) have HIGH tonality = HIGH difficulty.
    Noisy sounds (white noise, wind) have LOW tonality = LOW difficulty.
    """
    n_fft = cfg.get('n_fft', 2048)
    
    if len(audio) < n_fft:
        return 0.5  # Neutral score for short signals
    
    # Compute power spectrum
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    power = spectrum ** 2
    power_norm = power / (np.sum(power) + 1e-9)
    
    # Spectral flatness: 0 = tonal, 1 = white noise
    geometric_mean = np.exp(np.mean(np.log(power_norm + 1e-12)))
    arithmetic_mean = np.mean(power_norm)
    flatness = geometric_mean / (arithmetic_mean + 1e-12)
    
    # Tonality = inverse of flatness
    # Tonal (motor-like) = hard to distinguish from drone
    tonality = 1.0 - flatness
    return float(np.clip(tonality, 0.0, 1.0))


def _compute_low_frequency_ratio(audio, cfg):
    """
    Compute ratio of energy in low frequencies (100-500 Hz).
    Drones have dominant low-frequency content.
    Ambient with strong low-freq = confuses with drone = hard.
    """
    n_fft = cfg.get('n_fft', 2048)
    sr = cfg.get('sr', 22050)
    
    if len(audio) < n_fft:
        return 0.5
    
    # Compute power spectrum
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    power = spectrum ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    
    # Define drone frequency range (typical propeller hum)
    low_freq_min = 100.0  # Hz
    low_freq_max = 500.0  # Hz
    
    # Find indices in drone frequency range
    low_mask = (freqs >= low_freq_min) & (freqs <= low_freq_max)
    
    # Compute energy ratio
    low_energy = np.sum(power[low_mask])
    total_energy = np.sum(power) + 1e-9
    
    ratio = low_energy / total_energy
    
    # Scale to [0, 1]: high ratio = high difficulty
    # Empirically, drones have 20-60% energy in this range
    score = np.clip(ratio * 2.0, 0.0, 1.0)
    
    return float(score)


def _compute_periodic_modulation(audio, cfg):
    """
    Detect periodic amplitude modulation (like rotating propellers).
    Compute autocorrelation to find periodic patterns.
    High periodicity = resembles drone = hard.
    """
    # Compute envelope
    win_size = cfg.get('mod_window_samples', 512)
    hop = cfg.get('mod_hop_samples', 256)
    sr = cfg.get('sr', 22050)
    
    if len(audio) < win_size * 2:
        return 0.0
    
    envelope = []
    for i in range(0, len(audio) - win_size, hop):
        segment = audio[i:i + win_size]
        rms = np.sqrt(np.mean(segment.astype('float64') ** 2))
        envelope.append(rms)
    
    envelope = np.array(envelope)
    
    if len(envelope) < 10:
        return 0.0
    
    # Normalize envelope
    envelope = envelope - np.mean(envelope)
    envelope = envelope / (np.std(envelope) + 1e-9)
    
    # Compute autocorrelation (exclude zero lag)
    autocorr = np.correlate(envelope, envelope, mode='full')
    autocorr = autocorr[len(autocorr)//2 + 1:]  # Keep positive lags only
    
    # Find maximum autocorrelation in relevant lag range
    # Drone modulation typically 1-10 Hz
    sr_envelope = sr / hop  # Envelope sample rate
    min_period = int(sr_envelope * 0.1)  # 10 Hz max
    max_period = int(sr_envelope * 1.0)  # 1 Hz min
    
    if len(autocorr) < max_period:
        return 0.0
    
    relevant_autocorr = autocorr[min_period:min_period + max_period]
    
    # Peak autocorrelation indicates periodicity
    max_autocorr = np.max(relevant_autocorr) if len(relevant_autocorr) > 0 else 0.0
    norm_autocorr = max_autocorr / (autocorr[0] + 1e-9)
    
    # High autocorrelation = periodic = hard
    score = np.clip(norm_autocorr, 0.0, 1.0)
    
    return float(score)
