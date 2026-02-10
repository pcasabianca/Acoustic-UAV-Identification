#!/bin/bash
################################################################################
# Automated Setup Script for Raspberry Pi Drone Detection System
# 
# This script:
# 1. Installs Miniconda if not present
# 2. Creates conda environment from environment.yml
# 3. Installs system dependencies
# 4. Verifies installation
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo -e "${BLUE}  RASPBERRY PI DRONE DETECTION - AUTOMATED SETUP${NC}"
echo "========================================================================"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Install system dependencies
echo -e "${BLUE}[1/5]${NC} Installing system dependencies..."
echo ""

if command_exists apt-get; then
    echo "  Updating package list..."
    sudo apt-get update -qq
    
    echo "  Installing required packages..."
    sudo apt-get install -y \
        portaudio19-dev \
        libsndfile1 \
        python3-full \
        python3-pip \
        wget \
        || { echo -e "${RED}✗ Failed to install system dependencies${NC}"; exit 1; }
    
    echo -e "${GREEN}✓ System dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ apt-get not found, skipping system dependencies${NC}"
fi

echo ""

# Step 2: Install Miniconda if not present
echo -e "${BLUE}[2/5]${NC} Checking for Conda installation..."
echo ""

if command_exists conda; then
    echo -e "${GREEN}✓ Conda already installed${NC}"
    CONDA_PATH=$(which conda)
else
    echo "  Conda not found, installing Miniconda..."
    
    MINICONDA_DIR="$HOME/miniconda3"
    
    if [ -d "$MINICONDA_DIR" ]; then
        echo "  Miniconda directory exists but conda not in PATH"
        echo "  Initializing conda..."
        source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    else
        # Detect architecture
        ARCH=$(uname -m)
        
        if [ "$ARCH" = "aarch64" ]; then
            INSTALLER="Mambaforge-Linux-aarch64.sh"
            URL="https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER"
        elif [ "$ARCH" = "x86_64" ]; then
            INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
            URL="https://repo.anaconda.com/miniconda/$INSTALLER"
        else
            echo -e "${RED}✗ Unsupported architecture: $ARCH${NC}"
            exit 1
        fi
        
        echo "  Downloading Miniconda for $ARCH..."
        wget -q --show-progress "$URL" -O /tmp/miniconda.sh
        
        echo "  Installing Miniconda..."
        bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
        rm /tmp/miniconda.sh
        
        echo "  Initializing conda..."
        "$MINICONDA_DIR/bin/conda" init bash
        source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    fi
    
    echo -e "${GREEN}✓ Conda installed${NC}"
fi

echo ""

# Step 3: Create conda environment
echo -e "${BLUE}[3/5]${NC} Creating conda environment from environment.yml..."
echo ""

if [ ! -f "environment.yml" ]; then
    echo -e "${RED}✗ environment.yml not found!${NC}"
    exit 1
fi

# Remove existing environment if present
if conda env list | grep -q "^drone "; then
    echo "  Removing existing 'drone' environment..."
    conda env remove -n drone -y -q
fi

echo "  Creating environment (this may take several minutes)..."
conda env create -f environment.yml -q || {
    echo -e "${RED}✗ Failed to create conda environment${NC}"
    echo ""
    echo "Trying alternative installation method..."
    conda create -n drone python=3.11 -y -q
    conda activate drone
    conda install -c conda-forge tensorflow librosa numpy -y -q
    pip install soundfile pyaudio
}

echo -e "${GREEN}✓ Conda environment created${NC}"

echo ""

# Step 4: Verify installation
echo -e "${BLUE}[4/5]${NC} Verifying installation..."
echo ""

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drone

echo "  Testing Python packages..."
python -c "
import sys
errors = []

try:
    import tensorflow as tf
    print('  ✓ TensorFlow:', tf.__version__)
except Exception as e:
    errors.append(f'TensorFlow: {e}')

try:
    import librosa
    print('  ✓ librosa:', librosa.__version__)
except Exception as e:
    errors.append(f'librosa: {e}')

try:
    import soundfile
    print('  ✓ soundfile:', soundfile.__version__)
except Exception as e:
    errors.append(f'soundfile: {e}')

try:
    import numpy
    print('  ✓ numpy:', numpy.__version__)
except Exception as e:
    errors.append(f'numpy: {e}')

try:
    import pyaudio
    print('  ✓ pyaudio: OK')
except Exception as e:
    errors.append(f'pyaudio: {e}')

if errors:
    print('\n  Errors:')
    for error in errors:
        print(f'    ✗ {error}')
    sys.exit(1)
" || { echo -e "${RED}✗ Package verification failed${NC}"; exit 1; }

echo ""
echo -e "${GREEN}✓ All packages installed correctly${NC}"

echo ""

# Step 5: Final setup
echo -e "${BLUE}[5/5]${NC} Final setup..."
echo ""

# Make scripts executable
chmod +x start_detection.sh quickstart_pi.sh 2>/dev/null || true

# Create directories
mkdir -p audio_input logs models

echo -e "${GREEN}✓ Setup complete${NC}"

echo ""
echo "========================================================================"
echo -e "${GREEN}  INSTALLATION SUCCESSFUL!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     conda activate drone"
echo ""
echo "  2. Start the detection system:"
echo "     ./start_detection.sh --with-recording"
echo ""
echo "  Or simply run:"
echo "     ./start_detection.sh"
echo "     (it will auto-activate the environment)"
echo ""
echo "========================================================================"
