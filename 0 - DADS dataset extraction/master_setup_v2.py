"""
Master Setup Script v2.0 - Enhanced Dataset Pipeline
Automates the complete dataset preparation workflow with zero data leakage.

This script orchestrates dataset creation only:
1. Download DADS dataset from HuggingFace
2. Augment BOTH drone and no-drone classes (optional)
3. Combine original + augmented samples
4. Split into train/validation/test sets (zero overlap)
5. Validate file properties against `config.py` and write a validation report

Usage:
    # Quick setup with defaults (100 samples per class)
    python master_setup_v2.py
    
    # Custom sample counts
    python master_setup_v2.py --drone-samples 200 --no-drone-samples 200
    
    # Skip specific steps
    python master_setup_v2.py --skip-download --skip-augmentation
    
    # Dry run (see what would happen)
    python master_setup_v2.py --dry-run
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import librosa
import soundfile as sf
import numpy as np
import random

# Add project root to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
try:
    from tools.audio_utils import ensure_duration
except Exception:
    def ensure_duration(signal, sr, target_duration, crossfade_duration=0.1):
        target_samples = int(target_duration * sr)
        if len(signal) >= target_samples:
            return signal[:target_samples]
        return np.pad(signal, (0, target_samples - len(signal)), mode='constant')


# Total number of visible steps for printed progress
TOTAL_STEPS = 3


class colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{colors.HEADER}{colors.BOLD}{'='*80}")
    print(f"{text}")
    print(f"{'='*80}{colors.ENDC}\n")


def print_step(step_num, total_steps, description):
    """Print a formatted step indicator."""
    print(f"\n{colors.CYAN}{colors.BOLD}[STEP {step_num}/{total_steps}] {description}{colors.ENDC}")
    print(f"{colors.CYAN}{'─'*80}{colors.ENDC}\n")


def print_success(message):
    """Print a success message."""
    print(f"{colors.GREEN}✓ {message}{colors.ENDC}")


def print_warning(message):
    """Print a warning message."""
    print(f"{colors.YELLOW}⚠ {message}{colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{colors.RED}✗ {message}{colors.ENDC}")


def print_info(message):
    """Print an info message."""
    print(f"{colors.BLUE}ℹ {message}{colors.ENDC}")


def cleanup_directories(script_dir, dry_run=False):
    """
    Clean up existing dataset directories to prevent contamination.
    
    Args:
        script_dir: Script directory path
        dry_run: If True, only show what would be deleted
    
    Returns:
        bool: Always True
    """
    print_info("Cleaning up existing dataset directories...")
    
    dirs_to_clean = [
        'dataset_test',
        'dataset_augmented', 
        'dataset_combined',
        'dataset_train',
        'dataset_val',
        'dataset_dads',
        'extracted_features'
    ]
    
    for dir_name in dirs_to_clean:
        dir_path = script_dir / dir_name
        if dir_path.exists():
            if dry_run:
                print(f"  Would delete: {dir_path}")
            else:
                print(f"  Deleting: {dir_path}")
                # Use a robust removal with an onerror handler to attempt
                # to clear stubborn files/dirs (permissions, transient mounts,
                # or race conditions). If removal ultimately fails, warn
                # and continue so the pipeline can report and recover.
                def _on_rm_error(func, path, exc_info):
                    try:
                        import stat
                        os.chmod(path, stat.S_IWRITE)
                    except Exception:
                        pass
                    try:
                        if os.path.isdir(path) and not os.path.islink(path):
                            # Try to remove contents manually
                            for root, dirs, files in os.walk(path, topdown=False):
                                for name in files:
                                    try:
                                        os.remove(os.path.join(root, name))
                                    except Exception:
                                        pass
                                for name in dirs:
                                    try:
                                        os.rmdir(os.path.join(root, name))
                                    except Exception:
                                        pass
                            try:
                                os.rmdir(path)
                            except Exception:
                                pass
                        else:
                            try:
                                os.remove(path)
                            except Exception:
                                pass
                    except Exception:
                        pass

                try:
                    shutil.rmtree(dir_path, onerror=_on_rm_error)
                    # Final check
                    if dir_path.exists():
                        print_warning(f"  Unexpected error: could not fully remove {dir_path}")
                    else:
                        print_success(f"Cleaned {dir_name}/")
                except Exception as e:
                    print_warning(f"  Unexpected error: {e}")
        else:
            print(f"  {dir_name}/ not found (skip)")
    
    print()
    return True


def run_command(cmd, cwd=None, dry_run=False):
    """
    Execute a shell command and return success status.
    
    Args:
        cmd: Command to execute (list or string)
        cwd: Working directory for command
        dry_run: If True, only print command without executing
    
    Returns:
        bool: True if successful (or dry run), False otherwise
    """
    if isinstance(cmd, list):
        cmd_str = ' '.join(cmd)
    else:
        cmd_str = cmd
    
    print(f"  Running: {cmd_str}")
    
    if dry_run:
        print_warning("  (Dry run - command not executed)")
        return True
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=isinstance(cmd, str),
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"  Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"  Error: {e}")
        return False


def check_requirements(script_dir):
    """Check if required packages are installed."""
    print_info("Checking requirements...")
    
    required_packages = [
        'numpy',
        'librosa',
        'soundfile',
        'datasets'  # HuggingFace datasets
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print_error(f"\nMissing packages: {', '.join(missing)}")
        print_info("Install with: pip install " + ' '.join(missing))
        return False
    
    print_success("All required packages are installed")
    return True


def step_1_download(script_dir, drone_samples, no_drone_samples, dry_run):
    """Step 1: Download DADS dataset or use offline dataset if available."""
    print_step(1, TOTAL_STEPS, "Prepare DADS Dataset")
    
    if dry_run:
        print_warning("Dry run - skipping dataset preparation")
        return True

    # Check if offline dataset exists
    offline_dir = config.DATASET_DADS_OFFLINE_DIR
    if offline_dir.exists() and (offline_dir / "0").exists() and (offline_dir / "1").exists():
        print(f"{colors.GREEN}✓ Found offline DADS dataset: {offline_dir}{colors.ENDC}")
        print(f"{colors.CYAN}  Using offline dataset directly (no download needed){colors.ENDC}\n")
        
        # Simply use the offline dataset as source - no need to copy
        # The augmentation and splitting steps will read from this directly
        # Update the config to point to offline dataset
        import config as cfg_module
        cfg_module.DATASET_DADS_DIR = offline_dir
        config.DATASET_DADS_DIR = offline_dir
        
        print_success(f"Using offline dataset with {len(list((offline_dir / '0').glob('*.wav'))):,} ambient + {len(list((offline_dir / '1').glob('*.wav'))):,} drone samples")
        return True
    
    # No offline dataset - download from HuggingFace
    print(f"{colors.YELLOW}⚠ No offline dataset found at: {offline_dir}{colors.ENDC}")
    print(f"{colors.CYAN}  Downloading from Hugging Face...{colors.ENDC}")
    print(f"{colors.CYAN}  (Run: ./run_full_pipeline.sh --download-offline-dads to create offline dataset){colors.ENDC}\n")
    
    try:
        sys.path.insert(0, str(script_dir))
        from download_and_prepare_dads import download_and_prepare_dads
        max_per_class = max(drone_samples, no_drone_samples)
        # Use centralized dataset path from config
        download_and_prepare_dads(output_dir=str(config.DATASET_DADS_DIR), max_samples=None, max_per_class=max_per_class, verbose=True)
        return True
    except Exception as e:
        print_error(f"Download step failed: {e}")
        return False


def step_2_augment(script_dir, config_file, dry_run, total_samples=None, num_workers=None, augment_dry_run=False):
    """Step 2: Augment both drone and no-drone classes."""
    print_step(2, TOTAL_STEPS, "Augment Both Classes")
    
    if dry_run:
        print_warning("Dry run - skipping augmentation execution")
        return True
    # The function now delegates to the v4 builder wrapper so the augment
    # package within this extraction root is used. The wrapper supports
    # CLI overrides for total samples and worker count.
    # If dry_run is requested, skip execution.
    if dry_run:
        print_warning("Dry run - skipping augmentation execution")
        return True

    try:
        # Resolve config path
        cfg_candidate = Path(config_file)
        if cfg_candidate.is_absolute():
            cfg_path = cfg_candidate
        else:
            cfg_path = script_dir / config_file
        if not cfg_path.exists():
            print_error(f"Augmentation config not found: {cfg_path}")
            return False

        # Call v4 wrapper via the local script so it uses the local augment package
        builder = script_dir / 'build_dataset_v4.py'
        if not builder.exists():
            print_error(f"Augmentation wrapper not found: {builder}")
            return False

        cmd = [sys.executable, str(builder), '--config', str(cfg_path), '--out-dir', str(script_dir)]
        if total_samples is not None:
            cmd += ['--total-samples', str(int(total_samples))]
        if num_workers is not None:
            cmd += ['--num-workers', str(int(num_workers))]
        print_info(f"Running augmentation builder: {' '.join(cmd)}")
        # Ensure master prints its header before builder output
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # Run builder with unbuffered python and stream output so master
        # controls ordering and can prefix builder logs for clarity.
        env = os.environ.copy()
        env.setdefault('PYTHONUNBUFFERED', '1')
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            # Stream lines, prefixing to indicate origin
            if proc.stdout is not None:
                for line in proc.stdout:
                    # strip trailing newlines and print with flush to keep order
                    print(f"  [BUILDER] {line.rstrip()}", flush=True)
            ret = proc.wait()
            if ret != 0:
                print_error(f"Augmentation builder failed with exit code {ret}")
                return False
        except Exception as e:
            print_error(f"Augmentation builder execution error: {e}")
            return False

        print_success("Augmentation completed (builder path)")
        return True
    except Exception as e:
        print_error(f"Augmentation failed: {e}")
        return False

def step_3_validate(script_dir, dry_run=False):
    """Step 3: Validate dataset files and extracted features.

    Runs `tools/validate_dataset.py` which inspects audio files and
    `extracted_features/` JSONs and writes `tools/validate_report.json`.
    """
    print_step(3, TOTAL_STEPS, "Validate Dataset Files and Features")

    # If dry_run, still run validation on what's present (no files written),
    # but allow caller to skip running it by passing dry_run=True.
    if dry_run:
        print_warning("Dry run - running validation in read-only mode")

    try:
        validator = script_dir.parent / 'tools' / 'validate_dataset.py'
        if not validator.exists():
            # Validator script is optional in some environments; warn but continue
            print_warning(f"Validator not found: {validator}")
            print_warning("Skipping dataset-level validation (validator script missing)")
            return True

        cmd = [sys.executable, str(validator)]
        print_info(f"Running dataset validator: {cmd}")
        result = subprocess.run(cmd, check=True)
        print_success("Validation completed; see tools/validate_report.json")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Dataset validator failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"Validation error: {e}")
        return False
    
def step_4_summary(script_dir):
    """Final summary: concise dataset report and location of validation output."""
    print_step(4, TOTAL_STEPS, "Setup Complete - Dataset Created and Validated")

    # Count files in each dataset folder under script_dir
    datasets = [
        'dataset_dads', 'dataset_augmented',
        'dataset_train', 'dataset_val', 'dataset_test'
    ]

    print(f"\n{colors.BOLD}Dataset Summary:{colors.ENDC}")
    print(f"{'─'*80}")
    for name in datasets:
        p = script_dir / name
        if p.exists():
            class_0 = len(list((p / '0').glob('*.wav'))) if (p / '0').exists() else 0
            class_1 = len(list((p / '1').glob('*.wav'))) if (p / '1').exists() else 0
            total = class_0 + class_1
            print(f"  {name:20s} | Class 0: {class_0:4d} | Class 1: {class_1:4d} | Total: {total:4d}")
        else:
            print(f"  {name:20s} (not present)")

    print(f"\nValidation report: tools/validate_report.json")
    print(f"{colors.GREEN}{colors.BOLD}✓ Dataset creation complete. Use training scripts separately when ready.{colors.ENDC}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Master setup script for enhanced dataset pipeline v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick setup with defaults
  python master_setup_v2.py
  
  # Custom sample counts
  python master_setup_v2.py --drone-samples 200 --no-drone-samples 200
  
  # Custom train/val/test split
  python master_setup_v2.py --train 0.8 --val 0.1 --test 0.1
  
  # Skip certain steps
  python master_setup_v2.py --skip-download --skip-augmentation
  
  # Dry run (preview without executing)
  python master_setup_v2.py --dry-run
        """
    )
    
    # Dataset parameters
    parser.add_argument('--drone-samples', type=int, default=1000,
                        help='Number of drone samples to download per class (default: 1000)')
    parser.add_argument('--no-drone-samples', type=int, default=1000,
                        help='Number of no-drone samples to download per class (default: 1000)')
    
    # Split ratios
    parser.add_argument('--train', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    
    # Configuration
    # Default augmentation config is taken from centralized `config.py`
    parser.add_argument('--config', type=str, default=str(config.AUGMENTATION_CONFIG.name),
                        help=f'Augmentation config file (default: {config.AUGMENTATION_CONFIG.name})')
    # Augmentation overrides
    parser.add_argument('--augment-samples', type=int, default=None,
                        help='Override total augmentation samples (default from config)')
    parser.add_argument('--augment-workers', type=int, default=None,
                        help='Override augmentation worker count (default from config)')
    parser.add_argument('--augment-dry-run', action='store_true', help='Run augmentation in dry-run mode')
    parser.add_argument('--mid-scale', action='store_true', help='Run a mid-scale augmentation (faster validation)')
    
    # Step control
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download (use existing dataset_test)')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Skip augmentation (use existing dataset_augmented)')
    parser.add_argument('--merge-originals', action='store_true',
                        help='When distributing into splits, include original DADS files alongside augmentations')
    # Note: feature extraction and training are intentionally out of scope
    # for this master setup script. Use separate preprocessing/training scripts.
    
    # Execution mode
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform dry run without creating files')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Skip cleanup (keep existing datasets - may cause contamination)')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Print header
    print_header("ENHANCED DATASET PIPELINE v2.0")
    print(f"Working directory: {script_dir}\n")
    
    if args.dry_run:
        print_warning("DRY RUN MODE - No files will be created\n")
    
    # Check requirements
    if not check_requirements(script_dir):
        print_error("\nSetup aborted due to missing requirements")
        return 1
    
    # Step 0: Cleanup (unless disabled)
    if not args.no_cleanup:
        print_step(0, TOTAL_STEPS, "Cleanup Existing Datasets")
        cleanup_directories(script_dir, args.dry_run)
    else:
        print_warning("Skipping cleanup - Existing datasets will be REUSED (risk of contamination!)")
        print()
    
    # Validate ratios
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.01:
        print_error(f"Split ratios must sum to 1.0 (got {total_ratio})")
        return 1
    
    # Execute pipeline steps
    steps_success = []
    
    # Step 1: Download
    if not args.skip_download:
        success = step_1_download(script_dir, args.drone_samples, args.no_drone_samples, args.dry_run)
        steps_success.append(('Download', success))
        if not success and not args.dry_run:
            print_error("Setup failed at download step")
            return 1
    else:
        print_info("Skipping download step (using existing dataset_test)")
    
    # Step 2: Augmentation
    if not args.skip_augmentation:
        # decide total samples (priority order):
        # 1) CLI `--augment-samples` if provided
        # 2) `output.total_samples` in augmentation config (build_config.json)
        # 3) `config.DATASET_TARGET_SIZE` as a final fallback
        # For `--mid-scale` mode we keep the previous behaviour of capping at 2000.
        try:
            cfg_candidate = Path(args.config)
            cfg_path = cfg_candidate if cfg_candidate.is_absolute() else script_dir / args.config
            total_from_cfg = None
            if cfg_path.exists():
                try:
                    cfg_json = json.loads(cfg_path.read_text(encoding='utf-8'))
                    total_from_cfg = cfg_json.get('output', {}).get('total_samples')
                    if total_from_cfg is not None:
                        total_from_cfg = int(total_from_cfg)
                except Exception:
                    total_from_cfg = None
        except Exception:
            total_from_cfg = None

        if args.augment_samples is not None:
            augment_total = int(args.augment_samples)
        else:
            base_total = total_from_cfg if total_from_cfg is not None else int(config.DATASET_TARGET_SIZE)
            if args.mid_scale:
                augment_total = min(2000, int(base_total))
            else:
                augment_total = int(base_total)

        success = step_2_augment(script_dir, args.config, args.dry_run, total_samples=augment_total, num_workers=args.augment_workers, augment_dry_run=args.augment_dry_run)
        steps_success.append(('Augmentation', success))
        if not success and not args.dry_run:
            print_error("Setup failed at augmentation step")
            return 1
        # After builder completes, run augmentation validator and reporter (master orchestrates reporting)
        info_name = 'augmentation_samples.jsonl'
        meta_path = script_dir / 'dataset_train' / info_name
        if meta_path.exists():
            vcmd = [sys.executable, str(script_dir.parent / 'tools' / 'validate_augmentation.py'), str(meta_path)]
            print_info(f"Running augmentation validator: {' '.join(vcmd)}")
            vret = subprocess.run(vcmd, check=False)
            if vret.returncode != 0:
                print_error('Augmentation validation failed')
                if not args.dry_run:
                    return 1

            rcpt = [sys.executable, str(script_dir.parent / 'tools' / 'report_augmentation.py'), str(meta_path), str(script_dir.parent / 'tools' / 'augmentation_report.json')]
            print_info(f"Running augmentation reporter: {' '.join(rcpt)}")
            rret = subprocess.run(rcpt, check=False)
            if rret.returncode != 0:
                print_warning('Augmentation reporting encountered issues')
        else:
            print_warning(f"Augmentation meta file not found for validation/report: {meta_path}")
    else:
        print_info("Skipping augmentation step (using existing dataset_augmented)")
    
    # Step 3: Validate
    success = step_3_validate(script_dir, args.dry_run)
    steps_success.append(('Validation', success))
    if not success and not args.dry_run:
        print_error("Setup failed at validation step")
        return 1
    
    # Final summary
    if not args.dry_run:
        step_4_summary(script_dir)
    else:
        print_warning("\nDry run complete - no files were created")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{colors.YELLOW}Setup interrupted by user{colors.ENDC}")
        exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n{colors.RED}Unexpected error: {e}{colors.ENDC}")
        exit(1)
