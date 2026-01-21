from pathlib import Path


REQUIRED_KEYS = ['advanced', 'audio_parameters', 'augmentation_scenarios', 'class_proportions', 'output']


def validate_build_config(path: Path, cfg: dict):
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"build_config.json missing required keys: {missing}")
    # basic checks
    out = cfg.get('output', {})
    if 'total_samples' not in out:
        raise ValueError('output.total_samples is required')
    return True
