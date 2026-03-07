# Déploiement — Détection de drones en temps réel sur Raspberry Pi

Système de détection acoustique de drones en temps réel pour Raspberry Pi. Ce dossier est autonome : copiez-le sur la machine cible et lancez `./start_detection.sh`.

---

## Installation

Nécessite Conda (recommandé pour Raspberry Pi) :

```bash
conda env create -f environment.yml
conda activate drone
```

Installe Python 3.11, TensorFlow 2.16.2, librosa, numpy, soundfile et spidev.

---

## Utilisation

```bash
# Détection continue avec enregistrement automatique
./start_detection.sh --with-recording

# Forcer le backend PyAudio (ordinateur portable / micro USB)
./start_detection.sh --with-recording --laptop

# Effectuer un seul cycle enregistrement -> features -> prédiction puis quitter
./start_detection.sh --once

# Tester sur un fichier WAV existant
./start_detection.sh --test recordings/sample.wav

# Lister les périphériques audio
./start_detection.sh --list-devices
```

---

## Structure du dossier

```
start_detection.sh          Script de lancement
drone_detector.py           Détecteur principal
audio_recorder.py           Enregistreur continu (optionnel)
deployment_config.json      Tous les paramètres de configuration
thresholds.json             Seuils de décision (rechargés à chaud)
environment.yml             Définition de l'environnement Conda
models/                     Fichiers de modèles .keras pré-entraînés
audio_input/                Dossier surveillé pour les fichiers WAV
recordings/                 Enregistrements sauvegardés (si activé)
logs/                       Journaux de prédictions et d'exécution
utils/                      Scripts utilitaires
```

---

## Format audio requis

Les fichiers WAV doivent être : **22050 Hz, mono, 4 secondes**.

L'enregistreur produit automatiquement des fichiers dans ce format. Pour les fichiers manuels, rééchantillonner si nécessaire :

```bash
ffmpeg -i input.wav -ar 22050 -ac 1 -t 4 output.wav
```

---

## Configuration

Tous les paramètres sont dans `deployment_config.json`. Les réglages les plus fréquemment modifiés :

```json
"detection": {
    "enabled_models": ["CNN"],
    "voting_strategy": "majority"
}
```

**Stratégies de vote** :
- `majority` — plus de la moitié des modèles activés doivent voter drone (défaut)
- `any` — un seul modèle suffit (rappel plus élevé, plus de faux positifs)
- `unanimous` — tous les modèles doivent être d'accord (taux de faux positifs le plus bas)

Les **seuils** dans `thresholds.json` sont rechargés à chaque prédiction — modifiables sans redémarrer le détecteur.

---

## Modèles

Neuf modèles pré-entraînés sont disponibles : CNN, RNN, CRNN, Attention-CRNN, TCN, EfficientNet, MobileNet, AST, MultiRes.

Sur Raspberry Pi, utiliser un ou deux modèles légers pour maintenir la latence en dessous de 500 ms :

```json
"enabled_models": ["CNN"]
```

Le CNN est le plus rapide et suffit pour la plupart des déploiements. Ajouter Attention-CRNN pour une précision supérieure si le budget CPU le permet.

---

## Mise à jour des seuils après recalibration

```bash
# Prévisualiser les changements
python utils/extract_threshold_updates.py

# Appliquer
python utils/extract_threshold_updates.py --write
```

---

## Démarrage automatique avec systemd

Créer `/etc/systemd/system/drone-detector.service` :

```ini
[Unit]
Description=Drone Detection
After=sound.target

[Service]
User=pi
WorkingDirectory=/home/pi/8 - Deployment
ExecStart=/bin/bash start_detection.sh --with-recording
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable drone-detector
sudo systemctl start drone-detector
```

---

**Dernière mise à jour** : mars 2026
