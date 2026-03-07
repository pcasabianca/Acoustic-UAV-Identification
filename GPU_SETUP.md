# Configuration GPU

## AMD — RX 7600 (ROCm, Bazzite Linux)

TensorFlow-ROCm n'est plus disponible via pip. La seule méthode supportée est l'image officielle Docker/Podman.

### État

- GPU : AMD Radeon RX 7600 (gfx1102)
- ROCm : 6.17.7 (pré-installé sur Bazzite)
- TensorFlow-ROCm : 2.20.0 (conteneur Podman `acoustic-uav-rocm`)
- Utilisation GPU : 100 % pendant l'entraînement

### Architecture

```
Hôte (Bazzite)
├── ROCm 6.17.7
├── Podman 5.x
└── Conteneur : acoustic-uav-rocm
    ├── Base : docker.io/rocm/tensorflow:latest
    ├── TensorFlow 2.20 + ROCm
    └── Dépendances Python (sklearn, librosa, pandas, ...)
```

### Installation

Construire l'image une fois :
```bash
podman build -t acoustic-uav-rocm -f Dockerfile.rocm .
```

Tester le GPU :
```bash
./gpu.sh --test
```

### Utilisation

Le pipeline détecte le GPU automatiquement :
```bash
./run_full_pipeline.sh --models CNN
# [✓] AMD GPU detected - Training with GPU acceleration enabled
```

### Notes techniques

**HSA_OVERRIDE_GFX_VERSION=11.0.2** — force la compatibilité pour gfx1102 (RDNA 3) avec l'image ROCm compilée pour des architectures plus anciennes.

**LSTM/GRU sur ROCm** : `use_cudnn=False` est défini dans les entraîneurs RNN, CRNN et Attention-CRNN pour éviter l'erreur MIOpen « packed input ». Légèrement plus lent que MIOpen natif, mais entièrement fonctionnel.

**Compilation XLA** : le premier lancement prend du temps supplémentaire pendant la compilation des noyaux GPU par XLA. Les lancements suivants sont rapides.

### Périphériques du conteneur

```bash
--device=/dev/kfd
--device=/dev/dri
--group-add video
--security-opt label=disable   # requis sur Fedora/Bazzite
-v $(pwd):/workspace:rw
```

### Surveillance

```bash
watch -n 1 rocm-smi --showuse
rocm-smi --showtemp
```

### Dépannage

**GPU non détecté** :
```bash
rocm-smi
podman images | grep acoustic-uav-rocm
podman build -t acoustic-uav-rocm -f Dockerfile.rocm .
```

**Module Python manquant** : utiliser `acoustic-uav-rocm`, et non l'image de base `rocm/tensorflow`.

**Mettre à jour l'image ROCm** :
```bash
podman pull docker.io/rocm/tensorflow:latest
podman build -t acoustic-uav-rocm -f Dockerfile.rocm .
```

---

## NVIDIA — CUDA

Installation pip standard, aucun conteneur requis.

### Prérequis

- Pilote NVIDIA >= 525 (`nvidia-smi` doit fonctionner)

### Installation

```bash
# TF 2.16+ — télécharge automatiquement les wheels CUDA/cuDNN
pip install tensorflow[and-cuda]
```

Pour TF 2.15 avec CUDA installé sur le système :
```bash
pip install tensorflow==2.15.1    # CUDA 11.8
pip install tensorflow==2.20.0    # CUDA 12.x
```

### Vérification

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Croissance mémoire

Déjà activée dans `Universal_Curriculum_Trainer.py`. Pour l'activer manuellement :
```python
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Surveillance

```bash
nvidia-smi -l 1
```

---

## Références

- ROCm : https://rocmdocs.amd.com/
- TensorFlow ROCm : https://github.com/ROCm/tensorflow-upstream
- TensorFlow CUDA : https://www.tensorflow.org/install/pip

---

**Dernière mise à jour** : mars 2026
