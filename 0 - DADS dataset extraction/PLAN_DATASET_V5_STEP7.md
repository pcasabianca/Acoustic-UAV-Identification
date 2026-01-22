# Master Setup v5 (dataset) — download DADS + build dataset v5 + compat run_full_pipeline.sh

## Objectif
Mettre à jour `0 - DADS dataset extraction/master_setup_v2.py` pour :
1) **Télécharger DADS automatiquement si nécessaire**
2) **Générer dataset_train/val/test via build_dataset_v5.py**
3) Être **raccord avec config.py** (SR, durée, paths)
4) Fonctionner tel quel via `run_full_pipeline.sh` (qui appelle master_setup_v2.py)

## Contraintes
- Layout dataset inchangé : dans `0 - DADS dataset extraction/` → `dataset_train/`, `dataset_val/`, `dataset_test/` (+ `metadata.jsonl`)
- Regénération = overwrite (clean avant build)
- Builder v5 piloté par un JSON (config), mais master_setup doit patcher les paramètres critiques depuis config.py

---

## 1) API CLI (compat run_full_pipeline.sh)
`run_full_pipeline.sh` appelle :
- `python master_setup_v2.py --augment-samples N --augment-workers J --mid-scale`

Donc master_setup_v2.py doit accepter au minimum :
- `--config` (default: `config.AUGMENTATION_CONFIG.name`)
- `--augment-samples N` (override output.total_samples)
- `--augment-workers J` (override jobs)
- `--mid-scale` (cap total_samples à 2000)
- `--skip-download` (optionnel mais utile)
- `--no-cleanup` (optionnel)
- `--dry-run` (optionnel)

---

## 2) Step A — Préflight + download DADS (si nécessaire)

### 2.1 Cible dataset
On utilise **l’offline dataset** : `config.DATASET_DADS_OFFLINE_DIR` (ex: `0 - DADS dataset extraction/dataset_DADS_offline`)

Condition “OK” :
- `offline_dir/0` existe
- `offline_dir/1` existe
- contient des `.wav`

### 2.2 Si offline absent → télécharger
Si offline absent et `--skip-download` n’est pas activé :
- méthode 1 (préférée) : si `download_full_dads_offline.sh` existe dans le même dossier, exécuter :
  - `bash download_full_dads_offline.sh`
- méthode 2 (fallback python) : si `download_and_prepare_dads.py` existe et expose `download_and_prepare_dads(...)`, l’utiliser.

Puis revérifier la structure `offline_dir/0` et `offline_dir/1`.
Si toujours absent → fail-fast avec message.

### 2.3 Path config.py
Importer `config.py` depuis la racine du repo :
- `PROJECT_ROOT = Path(__file__).resolve().parents[1]`
- `sys.path.insert(0, str(PROJECT_ROOT))`
- `import config`

---

## 3) Step B — Cleanup (si pas --no-cleanup)
Supprimer :
- `dataset_train/`, `dataset_val/`, `dataset_test/`
- `tmp/` (shards/tasks/config effective)
- `config.EXTRACTED_FEATURES_DIR` (pour éviter mismatch feature extraction après regen dataset)

---

## 4) Step C — Générer un build_config effectif (patch JSON)

Charger `args.config` (dans le dossier d’extraction).
Puis patcher **toujours** :
- `cfg.audio.sample_rate = config.SAMPLE_RATE`
- `cfg.audio.duration_s = config.AUDIO_DURATION_S`
- `cfg.sources.dads_root = config.DATASET_DADS_OFFLINE_DIR.name` (chemin relatif depuis extraction dir)
- `cfg.output.clean_before_build = true`

Overrides CLI :
- `--augment-samples N` → `cfg.output.total_samples = N`
- `--mid-scale` → `cfg.output.total_samples = min(cfg.output.total_samples, 2000)`

Ecrire :
- `tmp/build_config_effective.json`

⚠️ Important : `build_dataset_v5.py` interprète `--config` comme un chemin relatif à `0 - DADS dataset extraction/`, donc lui passer :
- `--config tmp/build_config_effective.json`

---

## 5) Step D — Lancer la génération v5
Exécuter (cwd = extraction dir) :
- `python build_dataset_v5.py --config tmp/build_config_effective.json --mode full --jobs J --clean`

où `J` = `--augment-workers` sinon `config.AUGMENTATION_MAX_WORKERS` (ou cpu_count).

Le master_setup doit :
- exit nonzero si subprocess échoue

---

## 6) Step E — Résumé
Afficher :
- nb wav par split
- nb lignes metadata.jsonl par split
- ratio OK (wav==metadata lines) sinon exit error

---

## 7) Mise à jour config.py
Mettre à jour :
- `AUGMENTATION_WRAPPER = EXTRACTION_DIR / 'build_dataset_v5.py'`
(au lieu de v4)

Note: `AUGMENTATION_CONFIG` peut rester `build_config.json`.
→ si on veut que run_full_pipeline utilise ta config parfaite sans passer `--config`,
alors tu mets simplement ton JSON parfait dans `build_config.json`.
