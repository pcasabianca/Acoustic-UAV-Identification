#!/usr/bin/env bash
################################################################################
# start_detection.sh
# Modern launcher for Drone Detection System
# - Activates / uses the `drone` conda env if available
# - Validates dependencies and model files
# - Provides simple CLI to run detection, recording, tests and device listing
################################################################################

set -euo pipefail

# Colors (fall back if not a terminal)
GREEN='\033[0;32m' || true
YELLOW='\033[1;33m' || true
RED='\033[0;31m' || true
BLUE='\033[0;34m' || true
NC='\033[0m' || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

command_exists() { command -v "$1" >/dev/null 2>&1; }

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --with-recording      Start continuous detection with automatic recording
  --laptop              Force laptop recording backend (PyAudio)
  --once                Run a single record->feature->predict cycle and exit
  --test FILE.wav       Run detection on a single WAV file and exit
  --list-devices        List audio devices (PyAudio)
  --save-to-disk        Allow saving recordings to disk (default: RAM-only)
  --help                Show this help

Examples:
  $0 --with-recording           # Continuous monitoring with RAM-only recording
  $0 --with-recording --laptop  # Force PyAudio backend
  $0 --once                    # Run once and exit (convenient for cron)
  $0 --test recordings/sample.wav

EOF
}

# Evaluate Conda environment (prefer activation but fallback to `conda run`)
USE_CONDA_RUN=false
if command_exists conda; then
    CONDA_BASE=$(conda info --base 2>/dev/null || true)
    if [ -n "$CONDA_BASE" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_BASE/etc/profile.d/conda.sh" || true
        if conda env list | grep -Eq '^[[:space:]]*drone[[:space:]]'; then
            if [ "${CONDA_DEFAULT_ENV:-}" != "drone" ]; then
                if ! conda activate drone 2>/dev/null; then
                    USE_CONDA_RUN=true
                fi
            fi
        fi
    fi
fi

# Build python command
if [ "$USE_CONDA_RUN" = true ]; then
    PYTHON_CMD=(conda run -n drone python)
elif command_exists python3; then
    PYTHON_CMD=(python3)
else
    PYTHON_CMD=(python)
fi

# Quick function to run python inline
py() { "${PYTHON_CMD[@]}" - <<PY
$1
PY
}

# dependency checks
echo -e "${BLUE}[1/3]${NC} Validating runtime and dependencies..."
MISSING=()
if ! "${PYTHON_CMD[@]}" -c "import tensorflow" 2>/dev/null; then MISSING+=(tensorflow); fi
if ! "${PYTHON_CMD[@]}" -c "import librosa" 2>/dev/null; then MISSING+=(librosa); fi
if ! "${PYTHON_CMD[@]}" -c "import numpy" 2>/dev/null; then MISSING+=(numpy); fi
if ! "${PYTHON_CMD[@]}" -c "import soundfile" 2>/dev/null; then MISSING+=(soundfile); fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo -e "${RED}Missing Python packages:${NC} ${MISSING[*]}"
    echo "Install into the chosen interpreter, for example:"
    echo "  ${PYTHON_CMD[*]} -m pip install librosa numpy soundfile"
    echo "Or create the conda env from environment.yml"
    exit 1
fi

# Check models
echo -e "${BLUE}[2/3]${NC} Checking models/"
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo -e "${RED}No models found in models/ directory${NC}"
    echo "Place your .keras model files in the models/ directory."
    exit 1
fi

# Check config
echo -e "${BLUE}[3/3]${NC} Checking deployment_config.json"
if [ ! -f "deployment_config.json" ]; then
    echo -e "${RED}Missing deployment_config.json${NC}"; exit 1
fi
if ! "${PYTHON_CMD[@]}" -c "import json,sys; json.load(open('deployment_config.json'))" 2>/dev/null; then
    echo -e "${RED}Invalid JSON in deployment_config.json${NC}"; exit 1
fi

mkdir -p audio_input logs

# Parse args
MODE='detection'
FORCE_LAPTOP=false
TEST_FILE=''
ONCE=false
SAVE_TO_DISK=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-recording) MODE='with_recording'; shift ;;
        --laptop) FORCE_LAPTOP=true; shift ;;
        --test) TEST_FILE="$2"; MODE='test'; shift 2 ;;
        --list-devices) MODE='list_devices'; shift ;;
        --once) ONCE=true; shift ;;
        --save-to-disk) SAVE_TO_DISK=true; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    if [ -n "${DETECTOR_PID:-}" ]; then kill "$DETECTOR_PID" 2>/dev/null || true; fi
    exit 0
}
trap cleanup INT TERM

run_detector() {
    ARGS=(drone_detector.py)
    if [ "$MODE" = "with_recording" ]; then
        ARGS+=(--continuous --with-recording)
    elif [ "$MODE" = "detection" ]; then
        ARGS+=(--continuous)
    fi
    if [ "$FORCE_LAPTOP" = true ]; then ARGS+=(--laptop); fi
    if [ "$SAVE_TO_DISK" = true ]; then ARGS+=(--save-to-disk); fi

    if [ "$ONCE" = true ]; then
        # run a single cycle
        ARGS=(drone_detector.py --once)
    fi

    echo -e "${GREEN}Running:${NC} ${PYTHON_CMD[*]} ${ARGS[*]}"
    "${PYTHON_CMD[@]}" "${ARGS[@]}"
}

case "$MODE" in
    list_devices)
        echo "Listing audio devices via LaptopRecorder (if available)..."
        "${PYTHON_CMD[@]}" - <<PY
try:
    from utils.laptop_recorder import LaptopRecorder
    lr = LaptopRecorder()
    if hasattr(lr, 'list_devices'):
        lr.list_devices()
    else:
        print('No device-listing available')
except Exception as e:
    print('Error listing devices:', e)
PY
        ;;

    test)
        if [ -z "$TEST_FILE" ]; then echo "Missing test file"; usage; exit 1; fi
        if [ ! -f "$TEST_FILE" ]; then echo "Test file not found: $TEST_FILE"; exit 1; fi
        echo "Running single-file test: $TEST_FILE"
        "${PYTHON_CMD[@]}" drone_detector.py --file "$TEST_FILE"
        ;;

    with_recording|detection)
        if [ "$MODE" = "with_recording" ]; then
            # If laptop forced, ensure pyaudio exists
            if [ "$FORCE_LAPTOP" = true ]; then
                if ! "${PYTHON_CMD[@]}" -c "import pyaudio" 2>/dev/null; then
                    echo -e "${YELLOW}PyAudio not available in interpreter; falling back to detection-only${NC}"
                    MODE='detection'
                fi
            fi
            echo "Starting detector (mode=$MODE)"
        else
            echo "Starting detector (mode=detection-only)"
        fi

        if [ "$ONCE" = true ]; then
            # single cycle and exit
            ARGS=(drone_detector.py --once)
            if [ "$MODE" = "with_recording" ]; then ARGS+=(--with-recording); fi
            if [ "$FORCE_LAPTOP" = true ]; then ARGS+=(--laptop); fi
            if [ "$SAVE_TO_DISK" = true ]; then ARGS+=(--save-to-disk); fi
            echo "${PYTHON_CMD[*]} ${ARGS[*]}"
            "${PYTHON_CMD[@]}" "${ARGS[@]}"
            exit 0
        fi

        # Continuous run
        ARGS=(drone_detector.py)
        if [ "$MODE" = "with_recording" ]; then ARGS+=(--continuous --with-recording); else ARGS+=(--continuous); fi
        if [ "$FORCE_LAPTOP" = true ]; then ARGS+=(--laptop); fi
        if [ "$SAVE_TO_DISK" = true ]; then ARGS+=(--save-to-disk); fi

        "${PYTHON_CMD[@]}" "${ARGS[@]}" &
        DETECTOR_PID=$!
        echo "Detector started (PID: $DETECTOR_PID)"
        wait $DETECTOR_PID
        ;;
    *)
        usage; exit 1 ;;
esac

# end