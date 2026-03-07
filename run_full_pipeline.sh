#!/bin/bash
################################################################################
# Acoustic UAV Detection - Complete Pipeline (v3.2)
# 
# Updated pipeline reflecting the new architecture:
# 1. Dataset preparation (master_setup_v3.py)
# 2. MEL feature extraction (Mel_Preprocess_and_Feature_Extract.py)
# 3. Model training (CNN, RNN, CRNN, ATTENTION_CRNN, EFFICIENTNET, MOBILENET, CONFORMER, TCN, AST, MULTIRES)
# 4. Threshold calibration (calibrate_thresholds.py - multi-criteria)
# 5. Performance evaluation (Universal_Perf_Tester.py)
# 6. TFLite conversion and validation (tflite_conv_orchestrator.py)
# 7. Visualizations (run_visualizations.py)
#
# Usage:
#   ./run_full_pipeline.sh [OPTIONS]
#
# Options:
#   --download-offline-dads  Download full DADS dataset for offline use (run once)
#   --skip-dataset        Skip dataset preparation (reuse existing)
#   --skip-features       Skip feature extraction (reuse existing MEL files)
#   --skip-training       Skip model training (reuse existing models)
#   --skip-calibration    Skip threshold calibration (use existing thresholds)
#   --skip-testing        Skip performance testing (reuse existing results)
#   --models MODEL1,MODEL2  Train only specific models (CNN,RNN,CRNN,ATTENTION_CRNN,EFFICIENTNET,MOBILENET,CONFORMER,TCN,AST,MULTIRES)
#   --ssl-pretrain        Use Universal_SSL_Trainer instead of per-model trainer wrappers
#   --ssl-epochs N        Number of SSL pretraining epochs (default: 5)
#   --parallel            Train models in parallel (faster but more CPU)
#   --skip-viz            Skip visualizations
#   --skip-tflite-conversion  Skip TFLite conversion and validation
#   --use-class-aware-calibration  Use legacy class-aware calibration (deprecated)
#   --skip-pre-calib-eval  Skip baseline evaluation (legacy mode)
#   --skip-post-calib-eval Skip post-calibration evaluation (legacy mode)
#   --help                Show this help message
#
# Examples:
#   ./run_full_pipeline.sh --parallel                        # Full pipeline
#   ./run_full_pipeline.sh --skip-dataset --models CNN,RNN   # Train specific models
#   ./run_full_pipeline.sh --skip-training --skip-calibration  # Reuse models/thresholds
#   ./run_full_pipeline.sh --download-offline-dads           # Download DADS once
################################################################################

set -e  # Exit on error

# Ensure a sane PATH in non-interactive/nohup/sandbox contexts
DEFAULT_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/run/host/usr/bin:/run/host/usr/local/bin"
if [ -z "${PATH:-}" ]; then
    export PATH="$DEFAULT_PATH"
else
    export PATH="$PATH:$DEFAULT_PATH"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default settings
SKIP_DATASET=false
SKIP_FEATURES=false
SKIP_TRAINING=false
SKIP_TESTING=false
MODELS="CNN,RNN,CRNN,ATTENTION_CRNN,EFFICIENTNET,MOBILENET,CONFORMER,TCN,AST,MULTIRES"
PARALLEL=false
THRESHOLDS="0.5"
SKIP_VIZ=false
SKIP_TFLITE_CONVERSION=false
USE_CLASS_AWARE_CAL=false
SKIP_PRE_CALIB_EVAL=false
SKIP_CALIBRATION=false
SKIP_POST_CALIB_EVAL=false
TRAINER_FLAGS=""
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv/bin/python"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"
# Determine default number of workers for evaluation (N-1 cores)
CPU_COUNT=$(nproc 2>/dev/null || echo 1)
DEFAULT_WORKERS=$(( CPU_COUNT > 1 ? CPU_COUNT - 1 : 1 ))
# Cap to avoid oversubscription
if [ $DEFAULT_WORKERS -gt 12 ]; then
    DEFAULT_WORKERS=12
fi
WORKERS="$DEFAULT_WORKERS"
AUGMENT_SAMPLES=""
AUGMENT_WORKERS="$WORKERS"
MID_SCALE=false
SSL_PRETRAIN=false
SSL_EPOCHS=5

# Check for --no-nohup flag first (before parsing other args)
USE_NOHUP=true
for arg in "$@"; do
    if [ "$arg" = "--no-nohup" ]; then
        USE_NOHUP=false
        break
    fi
done

# Preserve the original arguments so we can re-launch with the same flags
ORIGINAL_ARGS=("$@")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --download-offline-dads)
            echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${BLUE}║      DOWNLOADING FULL DADS DATASET (OFFLINE MODE)         ║${NC}"
            echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            cd "$PROJECT_DIR/0 - DADS dataset extraction"
            bash download_full_dads_offline.sh
            echo ""
            echo -e "${GREEN}✓ Offline dataset downloaded successfully${NC}"
            echo -e "${GREEN}  Future runs will automatically use the offline dataset${NC}"
            echo ""
            exit 0
            ;;
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-features)
            SKIP_FEATURES=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-testing)
            SKIP_TESTING=true
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --ssl-pretrain)
            SSL_PRETRAIN=true
            shift
            ;;
        --ssl-epochs)
            SSL_EPOCHS="$2"
            shift 2
            ;;
        --thresholds)
            THRESHOLDS="$2"
            shift 2
            ;;
        --skip-viz)
            SKIP_VIZ=true
            shift
            ;;
        --skip-tflite-conversion)
            SKIP_TFLITE_CONVERSION=true
            shift
            ;;
        --skip-pre-calib-eval)
            SKIP_PRE_CALIB_EVAL=true
            shift
            ;;
        --skip-calibration)
            SKIP_CALIBRATION=true
            shift
            ;;
        --skip-post-calib-eval)
            SKIP_POST_CALIB_EVAL=true
            shift
            ;;
        --trainer-flags)
            TRAINER_FLAGS="$2"
            shift 2
            ;;
        --use-class-aware-calibration)
            USE_CLASS_AWARE_CAL=true
            shift
            ;;
        --no-nohup)
            # Internal flag, already processed
            shift
            ;;
        --augment-samples)
            AUGMENT_SAMPLES="$2"
            shift 2
            ;;
        --augment-workers)
            AUGMENT_WORKERS="$2"
            shift 2
            ;;
        --mid-scale)
            MID_SCALE=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# PID file for tracking
PID_FILE="$PROJECT_DIR/run_pipeline.pid"

# Relaunch under nohup if not already
if [ "$USE_NOHUP" = true ] && [ -z "${NOHUP_LAUNCHED:-}" ]; then
    echo "Re-launching under nohup → log: $LOG_FILE"
    # Use the preserved original args so flags like --skip-* are forwarded
    # Use absolute path to this script to ensure nohup can execute it from any cwd
    SCRIPT_ABS="$PROJECT_DIR/$(basename "$0")"
    env NOHUP_LAUNCHED=1 nohup "$SCRIPT_ABS" "${ORIGINAL_ARGS[@]}" --no-nohup > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started background process PID=$(cat $PID_FILE)"
    exit 0
fi

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    case $level in
        INFO)
            echo -e "${GREEN}[$timestamp][INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[$timestamp][WARN]${NC} $message"
            ;;
        ERROR)
            echo -e "${RED}[$timestamp][ERROR]${NC} $message"
            ;;
        STEP)
            echo -e "${CYAN}[$timestamp][STEP]${NC} $message"
            ;;
        SUCCESS)
            echo -e "${GREEN}[$timestamp][✓]${NC} $message"
            ;;
        *)
            echo -e "[$timestamp] $message"
            ;;
    esac
}

# Print header
print_header() {
    echo -e "${MAGENTA}"
    echo "================================================================================"
    echo "  ACOUSTIC UAV DETECTION - FULL PIPELINE V3.0"
    echo "================================================================================"
    echo -e "${NC}"
    log INFO "Timestamp: $TIMESTAMP"
    log INFO "Log File: $LOG_FILE"
    log INFO "Skip Dataset: $SKIP_DATASET"
    log INFO "Skip Features: $SKIP_FEATURES"
    log INFO "Skip Training: $SKIP_TRAINING"
    log INFO "Skip Testing: $SKIP_TESTING"
    log INFO "Models: $MODELS"
    log INFO "Parallel: $PARALLEL"
    log INFO "Thresholds: $THRESHOLDS"
    log INFO "Skip Viz: $SKIP_VIZ"
    log INFO "Trainer Flags: $TRAINER_FLAGS"
    log INFO "SSL Pretrain: $SSL_PRETRAIN"
    log INFO "SSL Epochs: $SSL_EPOCHS"
    echo ""
}

# Check GPU support
check_gpu_support() {
    GPU_AVAILABLE=false
    ROCM_IMAGE="acoustic-uav-rocm"
    GFX_VERSION="11.0.2"

    # Resolve binaries robustly (PATH may differ under nohup/non-interactive/sandbox shells)
    PODMAN_BIN="$(command -v podman 2>/dev/null || true)"
    ROCM_SMI_BIN="$(command -v rocm-smi 2>/dev/null || true)"

    # Fallback to common host and system locations
    if [ -z "$PODMAN_BIN" ]; then
        for candidate in \
            "/usr/bin/podman" \
            "/bin/podman" \
            "/usr/local/bin/podman" \
            "/run/host/usr/bin/podman" \
            "/run/host/usr/local/bin/podman"; do
            if [ -x "$candidate" ]; then
                PODMAN_BIN="$candidate"
                break
            fi
        done
    fi

    if [ -z "$ROCM_SMI_BIN" ]; then
        for candidate in \
            "/usr/bin/rocm-smi" \
            "/opt/rocm/bin/rocm-smi" \
            "/run/host/usr/bin/rocm-smi" \
            "/run/host/opt/rocm/bin/rocm-smi"; do
            if [ -x "$candidate" ]; then
                ROCM_SMI_BIN="$candidate"
                break
            fi
        done
    fi

    # Check if podman and rocm-smi are available
    if [ -n "$PODMAN_BIN" ] && [ -n "$ROCM_SMI_BIN" ]; then
        # Check if custom image is present
        if "$PODMAN_BIN" images | grep -q "acoustic-uav-rocm"; then
            # Check if GPU is detected
            if "$ROCM_SMI_BIN" --showproductname &> /dev/null; then
                GPU_AVAILABLE=true
                log SUCCESS "AMD GPU detected - Training with GPU acceleration enabled"
            else
                log WARN "rocm-smi detected but GPU not accessible from this context"
            fi
        elif "$PODMAN_BIN" images | grep -q "rocm/tensorflow"; then
            log INFO "Base image found but custom image is missing"
            log INFO "Building custom image..."
            if "$PODMAN_BIN" build -t acoustic-uav-rocm -f "$PROJECT_DIR/Dockerfile.rocm" "$PROJECT_DIR" &>> "$LOG_FILE"; then
                ROCM_IMAGE="acoustic-uav-rocm"
                if "$ROCM_SMI_BIN" --showproductname &> /dev/null; then
                    GPU_AVAILABLE=true
                    log SUCCESS "AMD GPU detected - Training with GPU acceleration enabled"
                else
                    log WARN "ROCm image ready but GPU not accessible from this context"
                fi
            else
                log WARN "Image build failed - training on CPU"
            fi
        else
            log INFO "TensorFlow-ROCm image not found - training on CPU only"
            log INFO "To enable GPU: ./gpu.sh --test"
        fi
    else
        log WARN "GPU tools not found (podman='$PODMAN_BIN', rocm-smi='$ROCM_SMI_BIN') - Training on CPU only"
    fi
    
    if [ "$GPU_AVAILABLE" = false ]; then
        log INFO "Training on CPU only"
    fi
}

# Run python script with GPU if available
run_python() {
    local script="$1"
    shift  # Remove first argument, keep the rest
    local args="$@"
    
    # RNN/LSTM models cannot use AMD GPU due to ROCm MIOpen limitations - force CPU
    local force_cpu=false
    if [[ "$script" =~ (RNN_Trainer|CRNN_Trainer|Attention_CRNN_Trainer) ]]; then
        force_cpu=true
        log INFO "LSTM model - CPU only (ROCm MIOpen limitation)"
    fi
    
    if [ "$GPU_AVAILABLE" = true ] && [ "$force_cpu" = false ]; then
        log INFO "Running with GPU: $script"
        "$PODMAN_BIN" run --rm \
            --device=/dev/kfd \
            --device=/dev/dri \
            --group-add video \
            --security-opt seccomp=unconfined \
            --security-opt label=disable \
            -v "$PROJECT_DIR:/workspace:rw" \
            -w "/workspace" \
            -e HSA_OVERRIDE_GFX_VERSION="$GFX_VERSION" \
            -e PYTHONUNBUFFERED=1 \
            "$ROCM_IMAGE" \
            python3 -u "/workspace/$script" $args
    else
        PYTHONUNBUFFERED=1 "$VENV_PATH" -u "$script" $args
    fi
}

# Print header
print_header() {
    echo -e "${MAGENTA}"
    echo "================================================================================"
    echo "  ACOUSTIC UAV DETECTION - FULL PIPELINE V3.0"
    echo "================================================================================"
    echo -e "${NC}"
    log INFO "Timestamp: $TIMESTAMP"
    log INFO "Log File: $LOG_FILE"
    log INFO "Skip Dataset: $SKIP_DATASET"
    log INFO "Skip Features: $SKIP_FEATURES"
    log INFO "Skip Training: $SKIP_TRAINING"
    log INFO "Skip Testing: $SKIP_TESTING"
    log INFO "Models: $MODELS"
    log INFO "Parallel: $PARALLEL"
    log INFO "Thresholds: $THRESHOLDS"
    log INFO "Skip Viz: $SKIP_VIZ"
    log INFO "Trainer Flags: $TRAINER_FLAGS"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log STEP "Checking prerequisites..."
    
    # Check for GPU support
    check_gpu_support
    
    if [ ! -x "$VENV_PATH" ]; then
        log ERROR "Python venv executable not found or not executable: $VENV_PATH"
        exit 1
    fi
    log SUCCESS "Python venv OK"
    
    # Verify critical scripts exist
    local required_scripts=(
        "0 - DADS dataset extraction/master_setup_v3.py"
        "1 - Preprocessing and Features Extraction/Mel_Preprocess_and_Feature_Extract.py"
        "2 - Model Training/CNN_Trainer.py"
        "2 - Model Training/RNN_Trainer.py"
        "2 - Model Training/CRNN_Trainer.py"
        "2 - Model Training/Attention_CRNN_Trainer.py"
        "2 - Model Training/EfficientNet_Trainer.py"
        "2 - Model Training/MobileNet_Trainer.py"
        "2 - Model Training/Conformer_Trainer.py"
        "2 - Model Training/TCN_Trainer.py"
        "2 - Model Training/AST_Trainer.py"
        "2 - Model Training/MultiRes_Trainer.py"
        "2 - Model Training/Universal_SSL_Trainer.py"
        "3 - Single Model Performance Calculation/Universal_Perf_Tester.py"
        "3 - Single Model Performance Calculation/calibrate_thresholds.py"
        "6 - Visualization/run_visualizations.py"
    )
    
    for script in "${required_scripts[@]}"; do
        if [ ! -f "$PROJECT_DIR/$script" ]; then
            log ERROR "Missing critical script: $script"
            exit 1
        fi
    done
    log SUCCESS "All scripts found"
}

# Step 1: Dataset preparation
prepare_dataset() {
    if [ "$SKIP_DATASET" = true ]; then
        log STEP "Skipping dataset preparation (--skip-dataset)"
        return
    fi
    
    log STEP "STEP 1/5: Dataset Preparation"
    log INFO "Running master_setup_v3.py..."
    
    cd "$PROJECT_DIR/0 - DADS dataset extraction"

    # Build augmentation args
    AUGMENT_ARGS=( )
    if [ -n "$AUGMENT_SAMPLES" ]; then
        AUGMENT_ARGS+=("--augment-samples" "$AUGMENT_SAMPLES")
    fi
    if [ -n "$AUGMENT_WORKERS" ]; then
        AUGMENT_ARGS+=("--augment-workers" "$AUGMENT_WORKERS")
    fi
    if [ "$MID_SCALE" = true ]; then
        AUGMENT_ARGS+=("--mid-scale")
    fi

    # Ensure the venv python is present and executable
    MASTER_SCRIPT="$PROJECT_DIR/0 - DADS dataset extraction/master_setup_v3.py"
    if [ ! -x "$VENV_PATH" ]; then
        log ERROR "Python venv executable not found or not executable: $VENV_PATH"
        log ERROR "Set VENV_PATH to a valid python executable or create the virtualenv at $PROJECT_DIR/.venv"
        exit 1
    fi
    if [ ! -f "$MASTER_SCRIPT" ]; then
        log ERROR "Master setup script not found: $MASTER_SCRIPT"
        exit 1
    fi

    if ! "$VENV_PATH" "$MASTER_SCRIPT" "${AUGMENT_ARGS[@]}"; then
        log ERROR "Dataset preparation failed"
        exit 1
    fi
    
    cd "$PROJECT_DIR"
    log SUCCESS "Dataset prepared successfully"
}

# Step 2: Feature extraction
extract_features() {
    if [ "$SKIP_FEATURES" = true ]; then
        log STEP "Skipping feature extraction (--skip-features)"
        return
    fi
    
    log STEP "STEP 2/5: MEL Feature Extraction"
    log INFO "Extracting features for train/val/test splits..."
    
    cd "$PROJECT_DIR/1 - Preprocessing and Features Extraction"
    
    for split in train val test; do
        log INFO "Extracting MEL features for $split..."
        if ! "$VENV_PATH" Mel_Preprocess_and_Feature_Extract.py --split "$split"; then
            log ERROR "Feature extraction failed for $split"
            exit 1
        fi
    done
    
    cd "$PROJECT_DIR"
    log SUCCESS "All features extracted"
}

# Step 3: Model training
train_models() {
    if [ "$SKIP_TRAINING" = true ]; then
        log STEP "Skipping model training (--skip-training)"
        return
    fi
    
    log STEP "STEP 3/5: Model Training"
    
    # Convert comma-separated list to array
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    
    if [ "$PARALLEL" = true ]; then
        log INFO "Training models in PARALLEL: ${MODEL_ARRAY[*]}"
        
        local pids=()
        for model in "${MODEL_ARRAY[@]}"; do
            train_single_model "$model" &
            pids+=($!)
        done
        
        # Wait for all to complete
        local failed=0
        for i in "${!pids[@]}"; do
            if ! wait "${pids[$i]}"; then
                log ERROR "Model ${MODEL_ARRAY[$i]} training failed"
                failed=1
            fi
        done
        
        if [ $failed -eq 1 ]; then
            log ERROR "Some models failed to train"
            exit 1
        fi
    else
        log INFO "Training models SEQUENTIALLY: ${MODEL_ARRAY[*]}"
        
        for model in "${MODEL_ARRAY[@]}"; do
            train_single_model "$model"
        done
    fi
    
    log SUCCESS "All models trained"
}

# Train a single model
train_single_model() {
    local model=$1
    local model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
    
    log INFO "Training $model..."
    
    case $model in
        CNN)
            trainer="CNN_Trainer.py"
            ;;
        RNN)
            trainer="RNN_Trainer.py"
            ;;
        CRNN)
            trainer="CRNN_Trainer.py"
            ;;
        ATTENTION_CRNN)
            trainer="Attention_CRNN_Trainer.py"
            ;;
        EFFICIENTNET)
            trainer="EfficientNet_Trainer.py"
            ;;
        MOBILENET)
            trainer="MobileNet_Trainer.py"
            ;;
        CONFORMER)
            trainer="Conformer_Trainer.py"
            ;;
        TCN)
            trainer="TCN_Trainer.py"
            ;;
        AST)
            trainer="AST_Trainer.py"
            ;;
        MULTIRES)
            trainer="MultiRes_Trainer.py"
            ;;
        *)
            log ERROR "Unknown model: $model"
            return 1
            ;;
    esac
    
    if [ "$SSL_PRETRAIN" = true ]; then
        # SSL route: universal self-supervised pretraining + supervised fine-tuning
        if ! run_python "2 - Model Training/Universal_SSL_Trainer.py" --model "$model_lower" --ssl_epochs "$SSL_EPOCHS" $TRAINER_FLAGS; then
            log ERROR "$model SSL training failed"
            return 1
        fi
    else
        # Forward any trainer-specific flags (e.g. --min_epochs, --max_epochs, --use_dynamic_weight, --dyn_min_w, --dyn_max_w, --batch_size)
        if ! run_python "2 - Model Training/$trainer" $TRAINER_FLAGS; then
            log ERROR "$model training failed"
            return 1
        fi
    fi
    
    log SUCCESS "$model trained successfully"
}

# Step 4: Threshold Calibration (after training, before testing)
calibrate_thresholds() {
    if [ "$SKIP_CALIBRATION" = true ]; then
        log STEP "Skipping threshold calibration (--skip-calibration)"
        return
    fi
    
    log STEP "STEP 4/6: Threshold Calibration (Multi-Criteria Hierarchical)"
    
    # Convert comma-separated list to array
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    
    local calibrator="$PROJECT_DIR/3 - Single Model Performance Calculation/calibrate_thresholds.py"
    
    if [ ! -f "$calibrator" ]; then
        log ERROR "Calibration script not found: $calibrator"
        return 1
    fi
    
    local failed_calibrations=0
    
    log INFO "Running multi-criteria threshold calibration..."
    log INFO "Criteria: Tier 1 (Constraints) → Tier 2 (Balanced Precision) → Tier 3 (F1/Recall)"
    echo ""
    
    # Build model list for calibrator
    CALIB_MODELS=$(echo "$MODELS" | tr ',' ' ')
    
    for model in $CALIB_MODELS; do
        log INFO "  • Calibrating $model thresholds..."
        # Run unbuffered Python and use line-buffered grep so log lines are flushed live
        if PYTHONUNBUFFERED=1 "$VENV_PATH" -u "$calibrator" --model "$model" 2>&1 | grep --line-buffered -v "cuda\|CUDA\|GPU"; then
            log SUCCESS "  ✓ $model calibration complete"
        else
            log WARN "  ✗ $model calibration failed"
            ((failed_calibrations++))
        fi
        echo ""
    done
    
    if [ $failed_calibrations -eq 0 ]; then
        log SUCCESS "All threshold calibrations completed successfully"
    else
        log WARN "Threshold calibration completed with $failed_calibrations issue(s)"
    fi
    
    # Show calibrated thresholds summary
    local calib_file="$PROJECT_DIR/0 - DADS dataset extraction/results/calibrated_thresholds.json"
    if [ -f "$calib_file" ]; then
        log INFO "Calibrated thresholds saved to: $calib_file"
    fi
}

# Step 5: Performance calculation with calibrated thresholds
calculate_performance() {
    if [ "$SKIP_TESTING" = true ]; then
        log STEP "Skipping performance testing (--skip-testing)"
        return
    fi
    
    log STEP "STEP 5/6: Performance Evaluation (Using Calibrated Thresholds)"
    
    # Convert comma-separated list to array
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    
    local calibrator="$PROJECT_DIR/3 - Single Model Performance Calculation/calibrate_thresholds.py"
    local tester="$PROJECT_DIR/3 - Single Model Performance Calculation/Universal_Perf_Tester.py"
    
    if [ ! -f "$calibrator" ] || [ ! -f "$tester" ]; then
        log ERROR "Required scripts not found"
        return 1
    fi
    
    local failed_tests=0

    log INFO "Running tests with calibrated thresholds from config"
    for model in "${MODEL_ARRAY[@]}"; do
        for split in train val test; do
            log INFO "  • Calibrated test: $model on $split"
            # Run tester unbuffered; stream through line-buffered grep to see live output
            if ! PYTHONUNBUFFERED=1 "$VENV_PATH" -u "$tester" --model "$model" --split "$split" --workers "$WORKERS" 2>&1 | grep --line-buffered -v "cuda\|CUDA\|GPU"; then
                log WARN "    ✗ Calibrated test failed: $model $split"
                ((failed_tests++))
            fi
        done
    done

    if [ $failed_tests -eq 0 ]; then
        log SUCCESS "Class-aware calibration workflow completed successfully"
    else
        log WARN "Class-aware calibration workflow completed with $failed_tests issues"
    fi
    return

    # Test with calibrated thresholds on all splits
    log INFO "Testing models with calibrated thresholds on all splits..."
    for model in "${MODEL_ARRAY[@]}"; do
        for split in train val test; do
            log INFO "  • Testing $model on $split..."
            
            if PYTHONUNBUFFERED=1 "$VENV_PATH" -u "$tester" \
                --model "$model" \
                --split "$split" \
                --workers "$WORKERS" 2>&1 | \
                grep --line-buffered -E "(Dataset|Running test|Loaded calibrated|Accuracy|Results saved)" | \
                grep --line-buffered -v "cuda\|CUDA\|GPU" > /dev/null; then
                log SUCCESS "  ✓ $model $split"
            else
                log WARN "  ✗ Test failed: $model $split"
                ((failed_tests++))
            fi
        done
        echo ""
    done

    if [ $failed_tests -eq 0 ]; then
        log SUCCESS "All performance tests completed successfully"
    else
        log WARN "Performance calculation completed with $failed_tests issues"
    fi
}

# Step 6: TFLite conversion and validation
run_tflite_conversion() {
    if [ "$SKIP_TFLITE_CONVERSION" = true ]; then
        log STEP "Skipping TFLite conversion (--skip-tflite-conversion)"
        return
    fi

    # Also respect the Python-side flag (config.TFLITE_ENABLED)
    TFLITE_ENABLED=$(PYTHONUNBUFFERED=1 "$VENV_PATH" -c \
        "import sys; sys.path.insert(0,'$PROJECT_DIR'); import config; print(str(getattr(config,'TFLITE_ENABLED',True)).lower())" \
        2>/dev/null || echo "true")
    if [ "$TFLITE_ENABLED" = "false" ]; then
        log STEP "Skipping TFLite conversion (config.TFLITE_ENABLED=False)"
        return
    fi

    log STEP "STEP 6/7: TFLite Conversion and Validation"

    local orchestrator="$PROJECT_DIR/5 - TFLite Conversion/tflite_conv_orchestrator.py"
    if [ ! -f "$orchestrator" ]; then
        log ERROR "Orchestrator not found: $orchestrator"
        return 1
    fi

    # Convert comma-separated MODELS list to space-separated for Python
    local model_args
    model_args=$(echo "$MODELS" | tr ',' ' ')

    log INFO "Converting models: $model_args"
    if ! PYTHONUNBUFFERED=1 "$VENV_PATH" -u "$orchestrator" \
            --models $model_args \
            --split test \
            --force 2>&1 | grep --line-buffered -v "cuda\|CUDA\|GPU"; then
        log WARN "TFLite conversion completed with warnings (non-fatal)"
    else
        log SUCCESS "TFLite conversion and validation complete"
    fi
}

# Step 7: Visualizations (Enhanced Pipeline)
generate_visualizations() {
    if [ "$SKIP_VIZ" = true ]; then
        log STEP "Skipping visualizations (--skip-viz)"
        return
    fi
    
    log STEP "STEP 7/7: Generating Visualizations"
    
    local viz_script="$PROJECT_DIR/6 - Visualization/run_visualizations.py"
    
    if [ ! -f "$viz_script" ]; then
        log ERROR "Visualization script not found: $viz_script"
        return 1
    fi
    
    log INFO "Running unified visualization pipeline..."
    
    # Run visualization unbuffered and stream output live
    if ! PYTHONUNBUFFERED=1 "$VENV_PATH" -u "$viz_script" 2>&1 | grep --line-buffered -v "cuda\|CUDA\|GPU"; then
        log WARN "Visualization generation had warnings (non-fatal)"
    else
        log SUCCESS "Enhanced visualizations generated successfully"
    fi
}

# Generate final report
generate_report() {
    log STEP "Generating Pipeline Report..."
    
    local report_file="$LOG_DIR/pipeline_report_${TIMESTAMP}.txt"
    
    {
        echo "================================================================================"
        echo "  ACOUSTIC UAV DETECTION - PIPELINE EXECUTION REPORT"
        echo "================================================================================"
        echo ""
        echo "Execution Details:"
        echo "  Timestamp: $TIMESTAMP"
        echo "  Duration: $(date -ud "@$(($(date +%s) - $(date -d "$(echo $TIMESTAMP | sed 's/_/ /;s/\(..\)\(..\)\(..\)$/\1:\2:\3/')" +%s)))" +%H:%M:%S)"
        echo "  Log File: $LOG_FILE"
        echo ""
        echo "Configuration:"
        echo "  Dataset Prepared: $([ "$SKIP_DATASET" = false ] && echo "Yes" || echo "No (skipped)")"
        echo "  Features Extracted: $([ "$SKIP_FEATURES" = false ] && echo "Yes" || echo "No (skipped)")"
        echo "  Models Trained: $MODELS"
        echo "  Training Mode: $([ "$PARALLEL" = true ] && echo "Parallel" || echo "Sequential")"
        echo "  Thresholds Tested: $THRESHOLDS"
        echo "  Visualizations: $([ "$SKIP_VIZ" = false ] && echo "Generated" || echo "Skipped")"
        echo ""
        echo "Output Locations:"
        echo "  Models: 0 - DADS dataset extraction/saved_models/"
        echo "  Performance: 0 - DADS dataset extraction/results/performance/"
        echo "  Visualizations: 6 - Visualization/outputs/"
        echo "  Training History: 0 - DADS dataset extraction/results/*_history.csv"
        echo ""
        echo "Next Steps:"
        echo "  1. Review performance results: 6 - Visualization/outputs/performance_summary.txt"
        echo "  2. Check visualizations: 6 - Visualization/outputs/"
        echo "  3. Analyze threshold impact: 6 - Visualization/outputs/threshold_calibration_*.png"
        echo "  4. Use quick_viz.py for custom visualizations"
        echo ""
    } > "$report_file"
    
    cat "$report_file"
    log SUCCESS "Report saved: $report_file"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    print_header
    check_prerequisites
    
    prepare_dataset
    extract_features
    train_models
    calibrate_thresholds
    calculate_performance
    run_tflite_conversion
    generate_visualizations
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo -e "${MAGENTA}"
    echo "================================================================================"
    echo "  ✓ PIPELINE COMPLETED SUCCESSFULLY"
    echo "================================================================================"
    echo -e "${NC}"
    log SUCCESS "Total duration: $(date -ud "@$duration" +%H:%M:%S)"
    
    generate_report
    
    # Cleanup PID file
    rm -f "$PID_FILE"
}

# Run main
main "$@"
