# 2 - Entrainement des modeles

Ce dossier contient les architectures de tous les modeles de detection de drone ainsi que le trainer universel qui centralise la logique d'entrainement.

## Principe general

Chaque fichier `*_Trainer.py` ne contient que la **definition de l'architecture** (`build_model` ou `build_*`). L'entrainement lui-meme est entierement delegue a `Universal_Curriculum_Trainer.py`, qui est le seul point d'entree pour lancer un entrainement. Executer directement l'un des trainers specifiques revient a appeler l'UCT avec l'argument `--model` adequat.

## Script principal : Universal_Curriculum_Trainer.py

Implementte le curriculum learning : les donnees les plus faciles sont presentees en premier, puis la difficulte augmente progressivement au fil des epochs selon la strategie choisie.

### Lancement

```bash
python "2 - Model Training/Universal_Curriculum_Trainer.py" --model <architecture>
```

### Arguments

| Argument | Valeurs possibles | Defaut | Description |
|---|---|---|---|
| `--model` | `cnn`, `rnn`, `crnn`, `attention_crnn`, `efficientnet`, `mobilenet`, `conformer`, `tcn`, `ast`, `multires` | obligatoire | Architecture a entrainer |
| `--curriculum_strategy` | `linear`, `step`, `exponential`, `none` | `linear` | Strategie de curriculum |
| `--loss` | `bce`, `weighted_bce`, `focal` | `focal` | Fonction de perte |
| `--focal_alpha` | float | `0.60` | Parametre alpha de la focal loss |
| `--focal_gamma` | float | `2.0` | Parametre gamma de la focal loss |
| `--batch_size` | int | depuis config.py | Taille du batch |
| `--learning_rate` | float | depuis config.py | Taux d'apprentissage |
| `--min_epochs` | int | depuis config.py | Epochs minimum avant early stopping |
| `--max_epochs` | int | depuis config.py | Epochs maximum |
| `--blocks` | int | `6` | Nombre de blocs Conformer (Conformer uniquement) |
| `--use_class_weight` | flag | desactive | Poids de classe equilibres |
| `--mixed_precision` | flag | auto pour AST | Force bfloat16 (reduit la memoire) |
| `--verbose` | `0`, `1`, `2` | `2` | Niveau de verbosity |

### Exemples

```bash
# CNN avec strategie lineaire
python "2 - Model Training/Universal_Curriculum_Trainer.py" --model cnn --curriculum_strategy linear

# Attention-CRNN avec focal loss
python "2 - Model Training/Universal_Curriculum_Trainer.py" --model attention_crnn --loss focal --focal_alpha 0.65

# EfficientNet par steps, batch de 16
python "2 - Model Training/Universal_Curriculum_Trainer.py" --model efficientnet --curriculum_strategy step --batch_size 16

# Conformer, 4 blocs, precision mixte
python "2 - Model Training/Universal_Curriculum_Trainer.py" --model conformer --blocks 4 --mixed_precision
```

## Architectures disponibles

| Fichier | Fonction exposee | Architecture |
|---|---|---|
| `CNN_Trainer.py` | `build_model` | CNN 2D (2 couches conv + dense) |
| `RNN_Trainer.py` | `build_model` | RNN recurrent |
| `CRNN_Trainer.py` | `build_model` | CNN suivi d'un LSTM |
| `Attention_CRNN_Trainer.py` | `build_attention_crnn` | CRNN + attention temporelle et de canal, connexions residuelles |
| `EfficientNet_Trainer.py` | `build_model` | EfficientNet (transfer learning) |
| `MobileNet_Trainer.py` | `build_model` | MobileNet (leger, adapte embarque) |
| `Conformer_Trainer.py` | `build_model` | Conformer (convolution + self-attention) |
| `TCN_Trainer.py` | `build_model` | Temporal Convolutional Network |
| `AST_Trainer.py` | `build_model` | Audio Spectrogram Transformer |
| `MultiRes_Trainer.py` | `build_model` | Multi-resolution CNN parallele |

## Fichiers utilitaires

- **`loss_functions.py`** : Fonctions de perte personnalisees (focal loss, weighted BCE, recall-focused).
- **`distance_stratified_callback.py`** : Callback Keras qui evalue les performances par tranche de distance a chaque epoch.
- **`dynamic_class_weight.py`** : Callback qui ajuste dynamiquement le poids de la classe drone en fonction du rappel courant.
- **`Universal_SSL_Trainer.py`** : Trainer experimental pour l'apprentissage auto-supervise (non utilise en production).

## Prerequis

Les features Mel doivent avoir ete extraites au prealable (dossier `1 - Preprocessing and Features Extraction`). Les chemins de sauvegarde des modeles sont definis dans `config.py` a la racine du projet.
