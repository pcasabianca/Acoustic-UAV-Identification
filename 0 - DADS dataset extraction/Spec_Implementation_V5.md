# Dataset Builder v5 — Architecture & Spécification d’Implémentation (DADS Drone Audio)

## 0) Objectif (v5) et contraintes non négociables

**Objectif v5 :** générer un dataset robuste (train/val/test) à partir de DADS (classe 0 = sans drone, classe 1 = drone), en mélangeant des sources et en écrivant WAV + `metadata.jsonl` par split.

**Contraintes :**
- **Layout imposé (inchangé) :**
  - tout est dans `0 - DADS dataset extraction/`
  - dossiers de sortie obligatoires et fixes :
    - `dataset_train/`
    - `dataset_val/`
    - `dataset_test/`
- **Pas d’ID de dataset.** À chaque exécution, on **overwrite** (rebuild complet).
- `metadata.jsonl` doit être **dans chaque dossier split** (`dataset_train/metadata.jsonl`, etc.). Pas de sidecar JSON.
- Autorisé : un dossier `tmp/` **dans le même dossier que les scripts**, pour y mettre fichiers intermédiaires (split, tasks, shards…).
- Le code doit être **simple, direct**, sans fallback “magique”.
- **Aucun script ne doit dépasser ~100 lignes** si possible. On factorise en modules très courts.

**Démarrage v0 (première version fonctionnelle) :**
- **Pas de modifiers** (pas de distance, pas de MEMS, pas de RIR, pas de pitch, etc.).
- On fait uniquement :
  1) split anti-leakage des sources DADS
  2) génération de tasks déterministes (sources déjà choisies)
  3) worker : load → trim/pad → mix simple → normalize peak → write WAV
  4) metadata shard → merge → `metadata.jsonl`
  5) validation stricte (WAV count == JSONL lines, lisibilité, durée/SR)

> Les modifiers seront ajoutés ensuite via une `chain` d’opérations. La structure v5 est pensée pour ça, mais v0 n’en utilise aucun.


---

## 1) Layout des fichiers (final)

Dans `0 - DADS dataset extraction/` :

build_config.json # config v5
build_dataset_v5.py # entrypoint CLI

tmp/ # fichiers intermédiaires (overwrite)
source_index.json # index des sources DADS (paths + hash + stats)
source_split.json # pools disjoints train/val/test (anti-leakage)
allocations.json # résumé allocation (#samples/scenario/split/label)
tasks_train.jsonl # task list instanciée (train)
tasks_val.jsonl
tasks_test.jsonl
shards/
train_meta_part_000.jsonl # shards metadata générés par workers
...
val_meta_part_...
test_meta_part_...
logs/ # optionnel (debug workers)

dataset_train/
0/ # WAV label 0
1/ # WAV label 1
metadata.jsonl

dataset_val/
0/
1/
metadata.jsonl

dataset_test/
0/
1/
metadata.jsonl

augment_v5/ # modules v5 (tous courts)
init.py
builder.py # pipeline global (full/tasks/gen/validate)
sources.py # scan + hash + split pools
allocator.py # allocation + instantiation des tasks (sources fixées)
worker.py # exécution d'une task (v0: mix simple)
io_atomic.py # write wav atomique + helpers FS
merge_metadata.py # merge shards -> metadata.jsonl
validate.py # validations strictes
task_format.py # validation minimale des structures JSON (types/keys)
audio_utils_v5.py # load/resample/trim/pad/normalize/stats (minimal)


**Règle “<100 lignes” :**
- Chaque module fait 1 job, 1–3 fonctions, pas de classes lourdes.
- Si un module dépasse 100 lignes, scinder (ex: `audio_io.py`, `audio_ops.py`).


---

## 2) Pipeline global (v5)

Modes CLI :
- `--mode full` : run complet (clean -> index -> split -> tasks -> gen -> merge -> validate)
- `--mode tasks` : (re)génère `tmp/source_index.json`, `tmp/source_split.json`, `tmp/tasks_*.jsonl`
- `--mode gen` : exécute les tasks existantes et produit les WAV + shards + metadata merge
- `--mode validate` : validation seule sur `dataset_train|val|test`

### Étapes (mode `full`)
1) **Clean** sorties : vider `dataset_train/0`, `dataset_train/1`, `dataset_train/metadata.jsonl` (idem val/test) si `clean_before_build=true`.
2) **Index sources** DADS : lister `dads_root/0/*.wav` et `dads_root/1/*.wav`, calculer hash et stats minimales (durée, sr si facile).
3) **Split anti-leakage** : créer `source_split.json` avec pools disjoints par label et split.
4) **Allocation + tasks** : produire `tasks_train.jsonl`, `tasks_val.jsonl`, `tasks_test.jsonl` :
   - chaque task choisit *exactement* ses sources (paths + hash)
   - tirage déterministe des paramètres v0 (principalement noms/sources)
5) **Génération** (workers) :
   - chaque worker lit des lignes tasks
   - génère WAV
   - écrit metadata shard `tmp/shards/<split>_meta_part_<id>.jsonl`
6) **Merge** : concat + tri des shards -> `dataset_<split>/metadata.jsonl`
7) **Validation stricte** : cohérence WAV/JSONL, lisibilité, sr/durée, etc.


---

## 3) Déterminisme et anti-leakage

### Déterminisme
- `build_config.json` contient `master_seed`.
- Pour chaque sample, dériver un `seed` reproductible :
  - `seed = hash(master_seed + split + i_sample + scenario_name)` (hash stable, ex. sha256->int32)
- **Important :**
  - Le sampling des sources et tous les tirages aléatoires se font **uniquement lors de la génération des tasks**.
  - Les workers ne font **aucun** “choix aléatoire” (ils exécutent la recette).

### Anti-leakage
- On splitte les sources DADS au début :
  - `by_label["ambient"]["train|val|test"]` disjoints
  - `by_label["drone"]["train|val|test"]` disjoints
- Les tasks du split X ne peuvent choisir que dans le pool X.
- Validation : vérifier qu’aucun `source_hash` n’apparaît dans plusieurs splits (optionnel v0, recommandé).


---

## 4) Format `build_config.json` (v5 minimal, v0 sans modifiers)

**Fichier :** `0 - DADS dataset extraction/build_config.json`

Exemple minimal :

```json
{
  "version": 5,
  "master_seed": 12345,

  "audio": {
    "sample_rate": 22050,
    "duration_s": 4.0
  },

  "output": {
    "clean_before_build": true,
    "splits": {
      "train": { "n_samples": 2000 },
      "val":   { "n_samples": 200 },
      "test":  { "n_samples": 200 }
    }
  },

  "sources": {
    "dads_root": "dataset_DADS_offline",
    "labels": {
      "ambient_dir": "0",
      "drone_dir": "1"
    },
    "split_policy": {
      "ratios": { "train": 0.85, "val": 0.10, "test": 0.05 }
    }
  },

  "scenarios": [
    {
      "name": "ambient_only",
      "weight": 1.0,
      "label": 0,
      "components": [
        { "role": "ambient", "count": 1 }
      ]
    },
    {
      "name": "drone_plus_ambient",
      "weight": 1.0,
      "label": 1,
      "components": [
        { "role": "drone", "count": 1 },
        { "role": "ambient", "count": 1 }
      ]
    }
  ],

  "mix": {
    "mode": "sum",
    "normalize_peak_dbfs": -1.0
  }
}


Notes :

v0 : pas de chain_template.

mix.mode="sum" : mix = somme des composantes (drone + ambient).

Normalisation finale : peak à -1 dBFS.

5) Format des Tasks (v5) — tmp/tasks_<split>.jsonl

Principe : 1 ligne JSON = 1 sample entièrement déterminé.

Format minimal v0 (pas de modifiers) :

{
  "sample_id": "train_00001234",
  "split": "train",
  "label": 1,
  "scenario": "drone_plus_ambient",
  "seed": 984312341,

  "components": [
    {
      "role": "drone",
      "source_path": "dataset_DADS_offline/1/drone_abc.wav",
      "source_hash": "sha256:..."
    },
    {
      "role": "ambient",
      "source_path": "dataset_DADS_offline/0/amb_xyz.wav",
      "source_hash": "sha256:..."
    }
  ],

  "output_relpath": "1/train_00001234.wav"
}


Règles :

output_relpath est relatif à dataset_<split>/.

Le dossier label (0/ ou 1/) est inclus dans output_relpath (ex: 1/xxx.wav).

6) Metadata (v5) — dataset_<split>/metadata.jsonl

Règle : 1 ligne JSON par WAV généré.

Format minimal v0 (recommandé) :

{
  "split": "train",
  "label": 1,
  "scenario": "drone_plus_ambient",
  "path": "1/train_00001234.wav",

  "sources": [
    { "role": "drone", "path": "dataset_DADS_offline/1/drone_abc.wav", "hash": "sha256:..." },
    { "role": "ambient", "path": "dataset_DADS_offline/0/amb_xyz.wav", "hash": "sha256:..." }
  ],

  "audio": {
    "sample_rate": 22050,
    "duration_s": 4.0,
    "peak_dbfs": -1.0,
    "rms_dbfs": -23.4,
    "clip_count": 0
  }
}

Important (concurrence) :

Les workers n’écrivent pas dans metadata.jsonl directement.

Ils écrivent dans tmp/shards/<split>_meta_part_XXX.jsonl.

Le merge est séquentiel pour produire dataset_<split>/metadata.jsonl final.

7) Responsabilités par module (contrat, <100 lignes)
7.1 build_dataset_v5.py (entrypoint)

Parse args : --config, --mode, --jobs, --clean.

Appelle augment_v5.builder.run(...).

7.2 augment_v5/builder.py

Fonction principale : run(config_path, mode, jobs, clean)

Charge config JSON

Définit chemins :

datasets : dataset_train|val|test

tmp : tmp/

Si mode in {full,tasks} :

sources.build_source_index(...) -> tmp/source_index.json

sources.build_source_split(...) -> tmp/source_split.json

allocator.build_tasks(...) -> tmp/tasks_*.jsonl + tmp/allocations.json

Si mode in {full,gen} :

lance workers (multiprocessing) sur tasks de chaque split

merge_metadata.merge_split(...) -> dataset_<split>/metadata.jsonl

Si mode in {full,validate} :

validate.run_all_splits(...)

7.3 augment_v5/sources.py

scan_sources(dads_root) -> dict(paths_by_label)

build_source_index(...) -> write tmp/source_index.json

hash sha256 des fichiers (ou hash du path + taille si perf) — simple et stable

build_source_split(...) -> write tmp/source_split.json

split disjoint par label, basé sur master_seed + shuffle

7.4 augment_v5/allocator.py

pick_scenario(rng, scenarios, weights)

pick_sources_for_task(rng, pools, scenario_components) -> components[]

build_tasks(split, n_samples, pools, config) -> write tmp/tasks_<split>.jsonl

sample_id = ${split}_${i:08d}

output_relpath = ${label}/${sample_id}.wav

allocations.json : juste résumé (counts par scénario, label)

7.5 augment_v5/worker.py

run_worker(split, tasks_path, out_dir, shard_path, audio_cfg, mix_cfg, worker_id, jobs)

Lire tasks (stream)

Pour chaque task :

load audio sources

trim_or_pad pour durée/sr

mix sum

normalize peak

write wav atomique

compute stats

append 1 ligne JSON metadata au shard

7.6 augment_v5/audio_utils_v5.py

Fonctions minimales :

load_audio(path, target_sr) -> np.ndarray, sr

trim_or_pad(x, target_len) -> x2

normalize_peak(x, peak_dbfs) -> x2

audio_stats(x, sr) -> dict(peak_dbfs, rms_dbfs, clip_count, duration_s)

(Option) resample simple si sr != target_sr (librairie existante si dispo, sinon refuser/raise pour v0)

7.7 augment_v5/io_atomic.py

ensure_dir(path)

atomic_write_wav(path, audio, sr) :

write temp file then os.replace(temp, final)

7.8 augment_v5/merge_metadata.py

merge_split(split, tmp_shards_dir, out_metadata_path)

concat shards du split

tri stable par path

write metadata.jsonl

7.9 augment_v5/validate.py

validate_split(dataset_dir, audio_cfg) -> raise on error

compte WAV dans 0/ et 1/

compte lignes JSONL

vérifie wav lisible, sr, durée, pas de NaN/Inf

run_all_splits(...)

7.10 augment_v5/task_format.py

validation minimale (présence des clés obligatoires)

pas de schéma lourd, juste des asserts/raises propres

8) Concurrence (important)

Problème à éviter : plusieurs workers qui écrivent en même temps dans metadata.jsonl.
Solution v5 :

chaque worker écrit dans son shard tmp/shards/<split>_meta_part_<id>.jsonl

merge final séquentiel

Écriture WAV :

écriture atomique (temp + replace) pour éviter fichiers partiels

9) Validation stricte (v0)

Échouer (raise) si :

#wav_total != #lines(metadata.jsonl) pour un split

audio illisible

sr != config sr

durée != config durée (tolérance très petite, ex ±1 sample)

NaN/Inf présent

(option) clip_count trop élevé

10) CLI (exemples)

Build complet :

python build_dataset_v5.py --config build_config.json --mode full --jobs 8
Générer seulement les tasks (debug) :

bash
Copier le code
python build_dataset_v5.py --config build_config.json --mode tasks
Génération à partir des tasks existantes :

bash
Copier le code
python build_dataset_v5.py --config build_config.json --mode gen --jobs 8
Validation seule :

bash
Copier le code
python build_dataset_v5.py --config build_config.json --mode validate
11) Règles de simplicité (à respecter strictement)
Pas de classes sauf besoin évident.

Pas de frameworks.

Chaque fichier : idéalement <100 lignes.

Des fonctions courtes, des noms explicites, et des raise ValueError clairs.

Tout ce qui est aléatoire est confiné à allocator.build_tasks.

Les workers n’inventent rien : ils exécutent la recette (tasks).

Aucune dépendance “exotique” : utiliser soit les utilitaires audio déjà présents dans le repo, soit une solution minimale (soundfile + numpy).

12) Extensions futures (hors v0)
Quand v0 fonctionne, ajouter :

components[].chain[] et global_chain[]

apply_chain(audio, chain) + orchestrateurs env_orch, hardware_orch, etc.

Mais sans casser le format v0 : v0 = chain vide.

yaml
Copier le code