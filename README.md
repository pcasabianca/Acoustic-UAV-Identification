# Détection acoustique de drones (Acoustic UAV Identification)

Pipeline d'apprentissage profond complet pour la détection de drones (UAV) par analyse acoustique, de la préparation du jeu de données au déploiement temps réel sur Raspberry Pi.

Basé sur l'article : [MDPI Drones 5(3):54](https://www.mdpi.com/2504-446X/5/3/54)

---

## Prérequis

- **Python 3.12**
- Dépendances : `pip install -r container_requirements.txt`

Paquets principaux : TensorFlow 2.20, NumPy, scikit-learn, librosa, pandas, audiomentations, soundfile, tqdm, termcolor.

---

## Installation

```bash
git clone <url-du-depot>
cd Acoustic-UAV-Identification
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r container_requirements.txt
```

---

## Démarrage rapide

```bash
# Pipeline complet (toutes les étapes, tous les modèles)
./run_full_pipeline.sh

# Reprise depuis l'entraînement (dataset et features déjà prêts)
./run_full_pipeline.sh --skip-dataset --skip-features

# Entraîner seulement deux modèles
./run_full_pipeline.sh --skip-dataset --skip-features --models CNN,ATTENTION_CRNN

# Afficher l'aide complète
./run_full_pipeline.sh --help
```

Le pipeline s'exécute en arrière-plan via `nohup` et journalise dans `logs/pipeline_<timestamp>.log`. Pour l'exécuter au premier plan :

```bash
./run_full_pipeline.sh --no-nohup
```

---

## Options du pipeline (run_full_pipeline.sh)

| Option | Description |
|---|---|
| `--download-offline-dads` | Télécharge le dataset DADS complet (180 k fichiers, a exécuter une seule fois) |
| `--skip-dataset` | Réutilise le dataset existant, saute la préparation |
| `--skip-features` | Réutilise les features MEL existantes, saute l'extraction |
| `--skip-training` | Réutilise les modèles existants, saute l'entraînement |
| `--skip-calibration` | Saute la calibration des seuils |
| `--skip-testing` | Saute l'évaluation des performances |
| `--skip-tflite-conversion` | Saute la conversion TFLite |
| `--skip-viz` | Saute la génération des visualisations |
| `--models M1,M2,...` | Entraîne uniquement les modèles spécifiés (ex. `CNN,RNN,ATTENTION_CRNN`) |
| `--parallel` | Entraîne les modèles en parallèle (plus rapide, plus de RAM) |
| `--ssl-pretrain` | Pré-entraîne en auto-supervisé avant le fine-tuning |
| `--ssl-epochs N` | Nombre d'époques de pré-entraînement SSL (défaut : 5) |
| `--trainer-flags "..."` | Arguments supplémentaires transmis directement à Universal_Curriculum_Trainer.py |
| `--no-nohup` | Exécute le pipeline au premier plan sans nohup |

### Exemples avancés

```bash
# Télécharger le dataset DADS une fois pour toutes
./run_full_pipeline.sh --download-offline-dads

# Retrainer uniquement sur les modèles légers
./run_full_pipeline.sh --skip-dataset --skip-features --models CNN,RNN,CRNN

# Pipeline avec pré-entraînement SSL et entraînement parallèle
./run_full_pipeline.sh --skip-dataset --ssl-pretrain --ssl-epochs 10 --parallel

# Passer des flags au trainer (ex. : focal loss personnalisée)
./run_full_pipeline.sh --skip-dataset --skip-features \
  --trainer-flags "--loss focal --focal_alpha 0.65 --focal_gamma 2.5"
```

---

## Structure du projet

```
Acoustic-UAV-Identification/
  config.py                          Source de vérité pour tous les chemins et hyperparamètres
  run_full_pipeline.sh               Point d'entrée — orchestre l'ensemble du pipeline
  container_requirements.txt         Dépendances Python
  GPU_SETUP.md                       Guide GPU (AMD/ROCm)
  |
  0 - DADS dataset extraction/       Téléchargement et préparation du dataset DADS
  1 - Preprocessing and Features/    Extraction des features MEL et MFCC
  2 - Model Training/                Architectures + Universal_Curriculum_Trainer
  3 - Single Model Performance/      Évaluation, calibration des seuils, conversion pour viz
  4 - Late Fusion Networks/          Vote majoritaire et vote pondéré entre modèles
  5 - TFLite Conversion/             Export .tflite pour déploiement embarqué
  6 - Visualization/                 Courbes de performance par distance et par modèle
  7 - Tests/                         Suite de tests Pytest
  8 - Deployment/                    Détecteur temps réel sur Raspberry Pi
```

Chaque dossier contient un README décrivant son fonctionnement et ses commandes.

---

## Modèles disponibles

| Identifiant | Architecture | Sortie | Notes |
|---|---|---|---|
| `CNN` | CNN 2D (2 blocs conv + dense) | softmax | Léger, rapide |
| `RNN` | Réseau récurrent | softmax | |
| `CRNN` | CNN + LSTM | softmax | Bon à longue distance |
| `ATTENTION_CRNN` | CRNN + attention temporelle + attention de canal + connexions résiduelles | softmax | Meilleure robustesse fréquentielle |
| `EFFICIENTNET` | EfficientNet-B0 (transfer learning) | sigmoid | |
| `MOBILENET` | MobileNet (transfer learning) | sigmoid | Optimal embarqué |
| `CONFORMER` | Convolution + self-attention (Conformer) | sigmoid | GPU recommandé, batch réduit à 8 |
| `TCN` | Temporal Convolutional Network | sigmoid | |
| `AST` | Audio Spectrogram Transformer | sigmoid | Requiert bfloat16, batch réduit à 8 |
| `MULTIRES` | CNN multi-résolution parallèle | sigmoid | Meilleure sensibilité multi-échelle |

Tous les modèles sont entraînés via `2 - Model Training/Universal_Curriculum_Trainer.py` avec apprentissage par curriculum.

---

## Configuration (config.py)

`config.py` est la **source de vérité unique** pour tous les chemins, hyperparamètres et constantes du projet. Les scripts ne définissent pas leurs propres chemins : ils importent `config`.

### Paramètres clés

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `SAMPLE_RATE` | 22050 | Fréquence d'échantillonnage audio (Hz) |
| `AUDIO_DURATION_S` | 4.0 | Durée de chaque fenêtre audio (secondes) |
| `MEL_N_MELS` | 128 | Nombre de bandes Mel |
| `MEL_N_FFT` | 2048 | Taille de la fenêtre FFT |
| `MEL_HOP_LENGTH` | 512 | Pas de la STFT en échantillons |
| `BATCH_SIZE` | 16 | Taille de batch (AST/Conformer : 8 automatiquement) |
| `LEARNING_RATE` | 0.0001 | Taux d'apprentissage |
| `MIN_EPOCHS` | 25 | Époques minimum avant early stopping |
| `MAX_EPOCHS` | 250 | Époques maximum |
| `TFLITE_QUANTIZATION` | `"float16"` | Mode de quantification TFLite (`"none"`, `"float16"`) |
| `LOSS_DEFAULTS['loss_type']` | `"focal"` | Fonction de perte par défaut |
| `LOSS_DEFAULTS['focal_alpha']` | 0.60 | Alpha de la focal loss |
| `LOSS_DEFAULTS['focal_gamma']` | 2.0 | Gamma de la focal loss |

### Variables d'environnement

| Variable | Description |
|---|---|
| `DATASET_ROOT_OVERRIDE` | Nom du sous-dossier dataset actif (`dataset_combined`, `dataset_train`, etc.) |
| `V4_ACTIVE_DATASET_ID` | ID d'un dataset v4 dans `datasets_v4/` |
| `TF_INTRA_OP_THREADS` | Threads TensorFlow pour opérations lourdes (défaut : 10) |
| `TF_INTER_OP_THREADS` | Threads TensorFlow pour coordination (défaut : 2) |
| `AUGMENTATION_WORKERS` | Workers parallèles pour l'augmentation de données |
| `DRONE_INFERENCE_BACKEND` | Backend d'inférence déploiement : `keras` ou `tflite` |

---

## GPU

- **AMD (ROCm)** : conteneurisé via Podman — voir [GPU_SETUP.md](GPU_SETUP.md)
- **NVIDIA (CUDA) NON TESTE** : `pip install tensorflow[and-cuda]`, aucun conteneur requis a priori
- **CPU uniquement** : fonctionne sans configuration GPU (plus lent)

---

## Jeu de données (DADS)

**DADS** (Drone Audio Detection Samples) — 180 320 fichiers audio, ~6,81 Go.

- Classe `0` : ambiance sans drone (16 729 samples)
- Classe `1` : drone (163 591 samples)
- Source : [Hugging Face — geronimobasso/drone-audio-detection-samples](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples)
- Téléchargement alternatif : [Dropbox](https://www.dropbox.com/sh/tpmsj5cac5btkf3/AABDmpZZwKw0lnHEviTey5y3a?dl=0)

Le pipeline télécharge automatiquement le dataset à la première exécution. Pour un téléchargement unique préalable :

```bash
./run_full_pipeline.sh --download-offline-dads
```

---

## Déploiement (Raspberry Pi)

Le dossier `8 - Deployment/` contient un système de détection temps réel autonome.

```bash
# Copier le projet sur le Raspberry Pi
scp -r Acoustic-UAV-Identification/ pi@raspberrypi.local:~/

# Sur le Pi : démarrer le détecteur
cd Acoustic-UAV-Identification/8\ -\ Deployment/
python drone_detector.py
```

Le détecteur charge les 9 modèles (`.keras` ou `.tflite`), enregistre des fenêtres de 4 secondes en continu et journalise les probabilités par modèle. Voir [8 - Deployment/README.md](8%20-%20Deployment/README.md).

---

## Tests

```bash
cd "7 - Tests"
pytest
# ou depuis la racine
pytest --rootdir=. -c "7 - Tests/pytest.ini"
```

---

**Etat** : développement actif — mars 2026
