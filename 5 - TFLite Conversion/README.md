# 5 - Conversion TFLite

Ce dossier convertit les modeles Keras (`.keras`) au format TFLite (`.tflite`) pour le deploiement sur Raspberry Pi, puis verifie que la fidelite de l'inference est preservee apres conversion.

## Principe

La conversion peut appliquer differents modes de quantification pour reduire la taille du modele et accelerer l'inference sur CPU embarque. Un testeur de fidelite compare ensuite les predictions du modele Keras original et du modele TFLite converti sur un sous-ensemble d'echantillons.

## Ordre d'execution recommande

```
tflite_conv_orchestrator.py  (appelle en interne les deux scripts ci-dessous)
    --> keras_to_tflite_converter.py   (conversion)
    --> tflite_conv_tester.py          (validation fidelite)
```

---

## 1. tflite_conv_orchestrator.py

Point d'entree principal. Lit `8 - Deployment/deployment_config.json` pour connaitre les fichiers Keras de chaque modele, tente la conversion et la validation pour chacun, et renvoie un code de retour non nul si au moins un modele a echoue.

### Lancement

```bash
# Convertir plusieurs modeles
python "5 - TFLite Conversion/tflite_conv_orchestrator.py" --models CNN RNN CRNN

# Convertir un seul modele en forcant la re-conversion meme si le .tflite existe deja
python "5 - TFLite Conversion/tflite_conv_orchestrator.py" --models CNN --force

# Evaluer la fidelite sur le split test
python "5 - TFLite Conversion/tflite_conv_orchestrator.py" --models ATTENTION_CRNN --split test
```

### Arguments

| Argument | Valeurs | Defaut | Description |
|---|---|---|---|
| `--models` | liste de noms | obligatoire | Modeles a convertir (ex. `CNN RNN ATTENTION_CRNN`) |
| `--split` | `train`, `val`, `test` | `val` | Split utilise pour le test de fidelite |
| `--force` | flag | desactive | Reconvertir meme si le `.tflite` existe deja |

La strategie de quantification est lue depuis `config.TFLITE_QUANTIZATION` (`float32`, `float16` ou `int8`). Si la quantification choisie echoue, l'orchestrateur tente automatiquement les modes de secours dans l'ordre `float16 -> float32`.

---

## 2. keras_to_tflite_converter.py

Convertit un seul fichier `.keras` en `.tflite`. Peut etre utilise directement si on souhaite convertir un modele sans passer par l'orchestrateur.

```bash
python "5 - TFLite Conversion/keras_to_tflite_converter.py" --model-path chemin/vers/modele.keras
```

Les fichiers `.tflite` sont ecrits dans le repertoire configure par `config.TFLITE_DIR`.

---

## 3. tflite_conv_tester.py

Compare les predictions du modele Keras original et du modele TFLite sur un ensemble d'echantillons. Produit un JSON de comparaison par echantillon (probabilite Keras, probabilite TFLite, ecart absolu) et affiche la correlation globale, l'accuracy et le MAE.

```bash
python "5 - TFLite Conversion/tflite_conv_tester.py" --model CNN --split val
```

Les JSON de comparaison sont ecrits dans `config.TFLITE_COMPARISON_DIR`.

---

## Prerequis

- Les modeles `.keras` doivent etre sauvegardes dans `config.MODELS_DIR` (produits par le dossier `2 - Model Training`).
- Les features Mel du split choisi doivent exister (dossier `1 - Preprocessing`).
- `8 - Deployment/deployment_config.json` doit lister les modeles et leurs noms de fichiers.
- La configuration de quantification (`config.TFLITE_QUANTIZATION`) est definie dans `config.py`.

## Sortie

Les fichiers `.tflite` produits sont destines a etre copies dans `8 - Deployment/models/` pour le deploiement.
