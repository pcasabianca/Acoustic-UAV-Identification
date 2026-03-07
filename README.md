# Détection acoustique de drones

Pipeline d'entraînement complet pour la détection acoustique de drones (UAV) par apprentissage profond sur le jeu de données DADS.

Basé sur l'article : [MDPI Drones 5(3):54](https://www.mdpi.com/2504-446X/5/3/54)

---

## Prérequis

**Python 3.12**, `pip install -r container_requirements.txt`

Paquets principaux : TensorFlow 2.20, NumPy, scikit-learn, librosa, pandas, audiomentations.

---

## Démarrage rapide

```bash
# Pipeline complet
./run_full_pipeline.sh

# Ignorer des étapes déjà effectuées, modèles spécifiques
./run_full_pipeline.sh --skip-dataset --skip-features --models CNN,CONFORMER

./run_full_pipeline.sh --help
```

---

## Structure du projet

```
0 - DADS dataset extraction/     Téléchargement et préparation du jeu de données
1 - Preprocessing and Features/  Extraction de caractéristiques MEL / MFCC
2 - Model Training/              Architectures et entraîneur par curriculum
3 - Single Model Performance/    Évaluation et calibration des seuils
4 - Late Fusion Networks/        Ensembles par vote majoritaire et pondéré
5 - TFLite Conversion/           Export TFLite pour déploiement embarqué
6 - Visualization/               Courbes de performance par distance
7 - Tests/                       Suite de tests Pytest
8 - Deployment/                  Déploiement temps réel sur Raspberry Pi
```

---

## Modèles

CNN, RNN, CRNN, Attention-CRNN, EfficientNet, MobileNet, Conformer, TCN, AST, MultiRes.  
Tous entraînés via `2 - Model Training/Universal_Curriculum_Trainer.py` avec apprentissage par curriculum.

---

## GPU

- **AMD** (RX 7600, ROCm) : conteneurisé via Podman — voir [GPU_SETUP.md](GPU_SETUP.md)
- **NVIDIA** (CUDA) : `pip install tensorflow[and-cuda]`, aucun conteneur requis
- **CPU** : fonctionne sans aucune configuration GPU

---

## Jeu de données

Jeu de données DADS — [téléchargement depuis Dropbox](https://www.dropbox.com/sh/tpmsj5cac5btkf3/AABDmpZZwKw0lnHEviTey5y3a?dl=0).  
Téléchargement automatique : `./run_full_pipeline.sh` (étape 1).

---

**État** : développement actif — mars 2026
