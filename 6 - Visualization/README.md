# 6 - Visualisation

Ce dossier genere toutes les figures d'analyse de performance des modeles : courbes ROC, matrices de confusion, performance par distance, comparaison inter-modeles, etc.

## Structure

```
6 - Visualization/
  run_visualizations.py      <- point d'entree CLI
  performance_by_distance.py <- analyse par tranche de distance (script autonome)
  scripts/
    viz_all.py               <- orchestrateur interne appele par run_visualizations.py
    ...                      <- scripts de visualisation individuels
  outputs/                   <- figures PNG et JSON produits (gitignore)
  tools/                     <- utilitaires partages (loaders, formatters)
  tests/                     <- tests des scripts de visualisation
```

---

## Lancement principal : run_visualizations.py

Appelle `scripts/viz_all.py` qui orchestre la generation de toutes les figures en une seule passe.

```bash
python "6 - Visualization/run_visualizations.py"
```

### Arguments optionnels

| Argument | Description |
|---|---|
| `--perf-dir` | Repertoire contenant les fichiers de performance (resulats de l'etape 3). Defaut : depuis config.py |
| `--results-dir` | Repertoire des resultats bruts par echantillon. Defaut : depuis config.py |
| `--out-dir` | Repertoire de sortie pour les figures. Defaut : `6 - Visualization/outputs/` |
| `--models` | Liste de modeles a inclure (ex. `CNN CRNN ATTENTION_CRNN`). Defaut : tous |
| `--snr-bins` | Definition des tranches SNR pour les courbes par rapport au bruit |
| `--strict` | Echec si un modele est manquant plutot que de l'ignorer |
| `--use-test-outputs` | Ecrire les sorties dans `outputs_test/` (pour tests CI) |

### Exemples

```bash
# Generer toutes les figures pour tous les modeles
python "6 - Visualization/run_visualizations.py"

# Generer uniquement pour CNN et Attention-CRNN
python "6 - Visualization/run_visualizations.py" --models CNN ATTENTION_CRNN

# Sortie dans un dossier specifique
python "6 - Visualization/run_visualizations.py" --out-dir /tmp/figures
```

Un fichier manifest JSON horodate (liste tous les artefacts produits) est ecrit automatiquement dans `outputs/manifests/`.

---

## Script autonome : performance_by_distance.py

Genere des graphiques de performance (F1, recall, precision) en fonction de la distance entre le microphone et le drone, en utilisant les metadonnees de distance des resultats de l'etape 3.

```bash
python "6 - Visualization/performance_by_distance.py"
```

---

## Prerequis

- Les fichiers de resultats JSON produits par `3 - Single Model Performance Calculation` doivent etre presents.
- Les chemins d'entree (`config.RESULTS_DIR`, `config.VIZ_OUTPUT_DIR`) sont definis dans `config.py`.
- Les dependances graphiques (`matplotlib`, `seaborn`) doivent etre installees.

## Sorties

Tous les fichiers produits (PNG, JSON) sont ecrits dans `outputs/` qui est dans le `.gitignore`. Le manifest permet de reconstruire la liste des artefacts generes lors d'une execution.
