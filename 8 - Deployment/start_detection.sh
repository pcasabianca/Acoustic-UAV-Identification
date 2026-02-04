#!/bin/bash
################################################################################
# Drone Detection System - Unified Launcher
# 
# Démarre le système complet de détection de drones :
# - Enregistrement audio depuis le microphone (optionnel)
# - Détection en temps réel
#
# Usage:
#   ./start_detection.sh                  # Détection seule (fichiers audio manuels)
#   ./start_detection.sh --with-recording # Avec enregistrement automatique
#   ./start_detection.sh --test FILE.wav  # Test avec un fichier
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detect script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Try to activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
    echo -e "${GREEN}✓${NC} Virtual environment activated"
fi

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Arrêt du système...${NC}"
    
    # Kill background processes
    if [ ! -z "$RECORDER_PID" ]; then
        kill $RECORDER_PID 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Enregistreur arrêté"
    fi
    
    if [ ! -z "$DETECTOR_PID" ]; then
        kill $DETECTOR_PID 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Détecteur arrêté"
    fi
    
    echo -e "${GREEN}Système arrêté proprement${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Header
echo "========================================================================"
echo "  🚁 SYSTÈME DE DÉTECTION DE DRONES"
echo "========================================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python 3 n'est pas installé!"
    exit 1
fi

# Check dependencies
echo -e "${BLUE}[1/4]${NC} Vérification des dépendances..."

MISSING_DEPS=()

python3 -c "import tensorflow" 2>/dev/null || MISSING_DEPS+=("tensorflow")
python3 -c "import librosa" 2>/dev/null || MISSING_DEPS+=("librosa")
python3 -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")
python3 -c "import soundfile" 2>/dev/null || MISSING_DEPS+=("soundfile")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${RED}[ERROR]${NC} Dépendances manquantes: ${MISSING_DEPS[@]}"
    echo ""
    echo "Installation requise:"
    echo "  pip3 install tensorflow librosa numpy soundfile"
    echo ""
    echo "Sur Raspberry Pi, utiliser TensorFlow Lite:"
    echo "  pip3 install tensorflow-lite librosa numpy soundfile"
    exit 1
fi

echo -e "${GREEN}✓${NC} Toutes les dépendances sont installées"

# Check models
echo -e "${BLUE}[2/4]${NC} Vérification des modèles..."

if [ ! -d "models" ] || [ -z "$(ls -A models/*.keras 2>/dev/null)" ]; then
    echo -e "${RED}[ERROR]${NC} Aucun modèle trouvé dans models/"
    echo ""
    echo "Les modèles doivent être présents dans le dossier 'models/'"
    echo "Si vous transférez ce dossier depuis la machine d'entraînement,"
    echo "assurez-vous d'inclure le dossier 'models/' avec les fichiers .keras"
    exit 1
fi

MODEL_COUNT=$(ls -1 models/*.keras 2>/dev/null | wc -l)
echo -e "${GREEN}✓${NC} $MODEL_COUNT modèle(s) trouvé(s)"

# Check config
echo -e "${BLUE}[3/4]${NC} Vérification de la configuration..."

if [ ! -f "deployment_config.json" ]; then
    echo -e "${RED}[ERROR]${NC} Fichier deployment_config.json manquant!"
    exit 1
fi

# Validate JSON
if ! python3 -c "import json; json.load(open('deployment_config.json'))" 2>/dev/null; then
    echo -e "${RED}[ERROR]${NC} deployment_config.json invalide!"
    exit 1
fi

echo -e "${GREEN}✓${NC} Configuration valide"

# Create directories
mkdir -p audio_input logs

# Parse arguments
MODE="detection_only"
TEST_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-recording)
            MODE="with_recording"
            shift
            ;;
        --test)
            MODE="test"
            TEST_FILE="$2"
            shift 2
            ;;
        --list-devices)
            python3 audio_recorder.py --list-devices
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --with-recording      Démarrer avec enregistrement audio automatique"
            echo "  --test FILE.wav       Tester avec un fichier audio spécifique"
            echo "  --list-devices        Lister les périphériques audio disponibles"
            echo "  --help                Afficher cette aide"
            echo ""
            echo "Exemples:"
            echo "  $0                              # Détection seule"
            echo "  $0 --with-recording             # Avec enregistrement"
            echo "  $0 --test ../dataset/drone.wav  # Test unitaire"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Option inconnue: $1"
            echo "Utilisez --help pour voir les options disponibles"
            exit 1
            ;;
    esac
done

# Launch based on mode
echo -e "${BLUE}[4/4]${NC} Démarrage du système..."
echo ""

case $MODE in
    test)
        # Test mode
        if [ ! -f "$TEST_FILE" ]; then
            echo -e "${RED}[ERROR]${NC} Fichier non trouvé: $TEST_FILE"
            exit 1
        fi
        
        echo "========================================================================"
        echo "  MODE TEST - Analyse d'un fichier unique"
        echo "========================================================================"
        echo ""
        
        python3 drone_detector.py --file "$TEST_FILE"
        ;;
        
    with_recording)
        # Recording + Detection mode
        echo "========================================================================"
        echo "  MODE ENREGISTREMENT + DÉTECTION"
        echo "========================================================================"
        echo ""
        
        # Check PyAudio
        if ! python3 -c "import pyaudio" 2>/dev/null; then
            echo -e "${YELLOW}[WARN]${NC} PyAudio non installé, enregistrement impossible"
            echo ""
            echo "Installation:"
            echo "  pip3 install pyaudio"
            echo "  Sur Raspberry Pi: sudo apt-get install python3-pyaudio"
            echo ""
            echo "Basculement en mode détection seule..."
            MODE="detection_only"
        else
            echo -e "${GREEN}[Démarrage]${NC} Enregistreur audio..."
            python3 audio_recorder.py --interval 5 --duration 4 > logs/recorder.log 2>&1 &
            RECORDER_PID=$!
            
            echo -e "${GREEN}✓${NC} Enregistreur démarré (PID: $RECORDER_PID)"
            echo ""
            sleep 2
        fi
        
        if [ "$MODE" = "with_recording" ]; then
            echo -e "${GREEN}[Démarrage]${NC} Détecteur de drones..."
            python3 drone_detector.py --continuous &
            DETECTOR_PID=$!
            
            echo -e "${GREEN}✓${NC} Détecteur démarré (PID: $DETECTOR_PID)"
            echo ""
            echo "Appuyez sur Ctrl+C pour arrêter"
            echo ""
            
            # Wait for processes (will be interrupted by cleanup)
            wait
        fi
        ;;
        
    detection_only)
        # Detection only mode
        echo "========================================================================"
        echo "  MODE DÉTECTION SEULE"
        echo "========================================================================"
        echo ""
        echo "Le système surveille le dossier: audio_input/"
        echo ""
        echo "Pour ajouter des fichiers audio à analyser:"
        echo "  1. Copier des fichiers WAV (4s, 22050Hz, mono) dans audio_input/"
        echo "  2. Le système les analysera automatiquement toutes les 5 secondes"
        echo ""
        echo "Appuyez sur Ctrl+C pour arrêter"
        echo ""
        
        python3 drone_detector.py --continuous &
        DETECTOR_PID=$!
        
        # Wait for process (will be interrupted by cleanup)
        wait
        ;;
esac

# Note: cleanup function is called automatically via trap on exit
