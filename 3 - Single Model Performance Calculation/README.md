# 3 - Calcul de performance des modeles individuels

Ce dossier contient les scripts permettant d'evaluer chaque modele entraine, de calibrer ses seuils de decision et de convertir les resultats pour la visualisation.

## Ordre d'execution

```
Universal_Perf_Tester.py  -->  calibrate_thresholds.py  -->  convert_results_for_viz.py
```

---

## 1. Universal_Perf_Tester.py

Charge un modele `.keras` et l'evalue sur un split du dataset (train / val / test). Produit un fichier JSON de resultats par echantillon contenant : probabilite predite, label reel, nom du fichier, distance estimee.

### Lancement

```bash
python "3 - Single Model Performance Calculation/Universal_Perf_Tester.py" --model <NOM> --split <SPLIT>
```

### Arguments

| Argument | Valeurs | Defaut | Description |
|---|---|---|---|
| `--model` | `CNN`, `RNN`, `CRNN`, `ATTENTION_CRNN`, `EFFICIENTNET`, `MOBILENET`, `CONFORMER`, `TCN`, `AST`, `MULTIRES` | obligatoire | Modele a evaluer |
| `--split` | `train`, `val`, `test` | obligatoire | Partition du dataset |
| `--threshold` | float | depuis config.py | Seuil de decision (par defaut : seuil calibre du modele) |
| `--workers` | int | cpu_count() - 1 | Nombre de workers multiprocessing |

### Exemples

```bash
# Evaluation du CNN sur la validation
python "3 - Single Model Performance Calculation/Universal_Perf_Tester.py" --model CNN --split val

# Evaluation d'Attention-CRNN sur le test avec seuil personnalise
python "3 - Single Model Performance Calculation/Universal_Perf_Tester.py" --model ATTENTION_CRNN --split test --threshold 0.55
```

### Sorties

Les resultats JSON sont ecrits dans le dossier configure par `config.RESULTS_DIR` sous la forme `<modele>_<split>_results.json`.

---

## 2. calibrate_thresholds.py

Calibre le seuil de decision de chaque modele en minimisant l'ecart entre la precision sur la classe drone et la precision sur la classe ambient. Objectif : garder un biais leger en faveur de la detection drone tout en evitant un taux de fausses alarmes excessif.

La calibration utilise les resultats JSON produits par `Universal_Perf_Tester.py` (split `val`).

### Lancement

```bash
# Calibrer un seul modele
python "3 - Single Model Performance Calculation/calibrate_thresholds.py" --model CNN

# Calibrer tous les modeles d'un coup
python "3 - Single Model Performance Calculation/calibrate_thresholds.py" --all-models
```

### Arguments avances

| Argument | Defaut | Description |
|---|---|---|
| `--max-balance-gap` | `0.10` | Ecart maximal tolere entre les precisions des deux classes |
| `--balance-target-delta` | `0.03` | Biais cible en faveur du drone (delta = acc_drone - acc_ambient) |
| `--min-recall` | depuis config | Rappel minimum impose pour la detection drone |

### Sortie

Fichier `calibrated_thresholds.json` dans `config.RESULTS_DIR`, puis copie automatique vers `8 - Deployment/thresholds.json` pour le deploiement.

---

## 3. convert_results_for_viz.py

Transforme les fichiers de resultats bruts (`*_results.json`) au format attendu par les scripts de visualisation du dossier `6 - Visualization`.

### Lancement

```bash
python "3 - Single Model Performance Calculation/convert_results_for_viz.py"
```

Aucun argument : le script lit tous les `*_results.json` presents dans `config.RESULTS_DIR` et ecrit les fichiers transformes au meme endroit.

---

## Prerequis

- Les modeles `.keras` doivent etre entraine et sauvegardes (dossier `2 - Model Training`).
- Les features Mel du split cible doivent exister (dossier `1 - Preprocessing`).
- Les chemins sont definis dans `config.py` a la racine.
