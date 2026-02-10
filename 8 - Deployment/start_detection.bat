@echo off
REM ============================================================================
REM Drone Detection System - Windows Launcher
REM 
REM Usage:
REM   start_detection.bat                  - Detection seule (fichiers manuels)
REM   start_detection.bat --with-recording - Avec enregistrement automatique
REM   start_detection.bat --test FILE.wav  - Test avec un fichier
REM ============================================================================

setlocal enabledelayedexpansion

REM Header
echo ========================================================================
echo   SYSTEME DE DETECTION DE DRONES
echo ========================================================================
echo.

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python n'est pas installe ou pas dans PATH!
    echo.
    echo Telecharger depuis: https://www.python.org/downloads/
    echo N'oubliez pas de cocher "Add Python to PATH" lors de l'installation
    pause
    exit /b 1
)

REM Check dependencies
echo [1/4] Verification des dependances...

python -c "import tensorflow" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] tensorflow manquant
    goto :missing_deps
)

python -c "import librosa" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] librosa manquant
    goto :missing_deps
)

python -c "import numpy" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] numpy manquant
    goto :missing_deps
)

python -c "import soundfile" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] soundfile manquant
    goto :missing_deps
)

echo OK - Toutes les dependances sont installees
goto :check_models

:missing_deps
echo.
echo Installation requise:
echo   pip install tensorflow librosa numpy soundfile
echo.
echo Pour l'enregistrement audio (optionnel):
echo   pip install pyaudio
pause
exit /b 1

:check_models
REM Check models
echo [2/4] Verification des modeles...

if not exist "models" (
    echo [ERROR] Dossier models/ manquant!
    pause
    exit /b 1
)

dir /b models\*.keras >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Aucun modele trouve dans models/
    echo.
    echo Les modeles doivent etre presents dans le dossier 'models/'
    pause
    exit /b 1
)

for /f %%i in ('dir /b models\*.keras 2^>nul ^| find /c /v ""') do set MODEL_COUNT=%%i
echo OK - %MODEL_COUNT% modele(s) trouve(s)

REM Check config
echo [3/4] Verification de la configuration...

if not exist "deployment_config.json" (
    echo [ERROR] Fichier deployment_config.json manquant!
    pause
    exit /b 1
)

python -c "import json; json.load(open('deployment_config.json'))" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] deployment_config.json invalide!
    pause
    exit /b 1
)

echo OK - Configuration valide

REM Create directories
if not exist "audio_input" mkdir audio_input
if not exist "logs" mkdir logs

REM Parse arguments
set MODE=detection_only
set TEST_FILE=

:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--with-recording" (
    set MODE=with_recording
    shift
    goto :parse_args
)
if "%~1"=="--test" (
    set MODE=test
    set TEST_FILE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--list-devices" (
    python audio_recorder.py --list-devices
    pause
    exit /b 0
)
if "%~1"=="--help" (
    echo Usage: %~nx0 [OPTIONS]
    echo.
    echo Options:
    echo   --with-recording      Demarrer avec enregistrement audio automatique
    echo   --test FILE.wav       Tester avec un fichier audio specifique
    echo   --list-devices        Lister les peripheriques audio disponibles
    echo   --help                Afficher cette aide
    echo.
    echo Exemples:
    echo   %~nx0                              # Detection seule
    echo   %~nx0 --with-recording             # Avec enregistrement
    echo   %~nx0 --test ..\dataset\drone.wav  # Test unitaire
    pause
    exit /b 0
)
echo [ERROR] Option inconnue: %~1
echo Utilisez --help pour voir les options disponibles
pause
exit /b 1

:end_parse

REM Launch based on mode
echo [4/4] Demarrage du systeme...
echo.

if "%MODE%"=="test" (
    if not exist "%TEST_FILE%" (
        echo [ERROR] Fichier non trouve: %TEST_FILE%
        pause
        exit /b 1
    )
    
    echo ========================================================================
    echo   MODE TEST - Analyse d'un fichier unique
    echo ========================================================================
    echo.
    
    python drone_detector.py --file "%TEST_FILE%"
    echo.
    pause
    exit /b 0
)

if "%MODE%"=="with_recording" (
    echo ========================================================================
    echo   MODE ENREGISTREMENT + DETECTION
    echo ========================================================================
    echo.
    
    REM Check PyAudio
    python -c "import pyaudio" 2>nul
    if !ERRORLEVEL! NEQ 0 (
        echo [WARN] PyAudio non installe, enregistrement impossible
        echo.
        echo Installation: pip install pyaudio
        echo.
        echo Si installation echoue, telecharger wheel depuis:
        echo   https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
        echo.
        echo Basculement en mode detection seule...
        set MODE=detection_only
        timeout /t 3 >nul
    )
)

if "%MODE%"=="detection_only" (
    echo ========================================================================
    echo   MODE DETECTION SEULE
    echo ========================================================================
    echo.
    echo Le systeme surveille le dossier: audio_input\
    echo.
    echo Pour ajouter des fichiers audio a analyser:
    echo   1. Copier fichiers WAV vers audio_input\
    echo   2. Analyse automatique toutes les 5 secondes
    echo.
    echo Appuyez sur Ctrl+C pour arreter
    echo.
)

echo [Demarrage] Detecteur de drones...

if "%MODE%"=="with_recording" (
    python drone_detector.py --continuous --with-recording
) else (
    python drone_detector.py --continuous
)

endlocal
