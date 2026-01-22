# Dataset Builder v5 — Plan d’implémentation en 3 étapes (DADS → dataset_train/val/test)

> **Objectif global :** pipeline simple, déterministe, sans fuite de données entre splits, compatible avec le layout imposé :
>
> - `0 - DADS dataset extraction/dataset_train|dataset_val|dataset_test`
> - `metadata.jsonl` **dans chaque split**
> - overwrite complet à chaque run (pas d’ID de dataset)
> - fichiers intermédiaires dans `0 - DADS dataset extraction/tmp/`
> - **aucun script > 100 lignes** si possible (sinon scinder en modules)
> - **pas de fallback / pas de comportement implicite** : erreurs claires, ordre explicite

---

## 0) Layout final (inchangé côté datasets)

Dans `0 - DADS dataset extraction/` :

build_config.json
build_dataset_v5.py

tmp/
source_index.json
source_split.json
allocations.json
tasks_train.jsonl
tasks_val.jsonl
tasks_test.jsonl
shards/
train_meta_part_000.jsonl
...
val_meta_part_...
test_meta_part_...

dataset_train/
0/
1/
metadata.jsonl

dataset_val/
0/
1/
metadata.jsonl

dataset_test/
0/
1/
metadata.jsonl

augment_v5/
init.py
builder.py
sources.py
allocator.py
worker.py
io_atomic.py
merge_metadata.py
validate.py
task_format.py
audio_utils_v5.py

modifiers/ # étape 2 puis 3
init.py
registry.py
chain.py
ops_basic.py
orchestrators.py # étape 3
ops_distance.py # étape 3
ops_mems.py # étape 3
ops_filters.py # étape 3
ops_rir.py # étape 3
ops_mix.py # étape 3

pgsql
Copier le code

---

## 1) Config v5 (v0, sans modifiers) — **celle-ci est la référence**

- IMPORTANT : **on utilise `output.split_ratio` pour TOUT** :
  - répartition des *samples générés*
  - répartition des *sources DADS* dans `source_split.json` (anti-leakage)

```json
{
  "version": 5,
  "master_seed": 12345,

  "audio": { "sample_rate": 22050, "duration_s": 4.0 },

  "output": {
    "clean_before_build": true,
    "total_samples": 1000,
    "split_ratio": [0.7, 0.15, 0.15]
  },

  "sources": {
    "dads_root": "dataset_DADS_offline",
    "labels": { "ambient": "0", "drone": "1" }
  },

  "scenarios": [
    { "name": "ambient_only", "weight": 1.0, "label": 0,
      "components": [ { "role": "ambient", "count": 1 } ] },

    { "name": "drone_plus_ambient", "weight": 1.0, "label": 1,
      "components": [ { "role": "drone", "count": 1 },
                      { "role": "ambient", "count": 1 } ] }
  ],

  "mix": { "mode": "sum", "normalize_peak_dbfs": -1.0 }
}
2) Définition des invariants (à respecter dans les 3 étapes)
2.1 Déterminisme (fondamental)
Toute variabilité est confinée à allocator.build_tasks(...).

Les workers n’échantillonnent rien : ils exécutent les tasks.

Interdiction d’utiliser hash() Python pour les seeds (non stable).

Utiliser une fonction stable : stable_int_seed("...") via sha256 → int32.

2.2 Anti-leakage (fondamental)
Construire tmp/source_split.json à partir des sources DADS :

pools disjoints par label et split

Les tasks train choisissent uniquement dans le pool train, etc.

Validation recommandée : vérifier que source_hash n’apparaît pas dans plusieurs splits.

2.3 Écriture robuste
Pas d’écriture concurrente dans dataset_<split>/metadata.jsonl.

Les workers écrivent des shards :

tmp/shards/<split>_meta_part_<worker_id>.jsonl

Merge séquentiel vers dataset_<split>/metadata.jsonl.

WAV écrits atomiquement : temp puis os.replace.

2.4 Simplicité (contrainte de style)
Pas de classes lourdes.

1 module = 1 responsabilité.

Si un module dépasse ~100 lignes, scinder.

ÉTAPE 1 — Stabiliser v0 (pipeline simple “sources → mix → wav + metadata + validate”)
1.1 Fonctionnalités attendues (v0)
Lire build_config.json

Clean output si demandé

Indexer les sources DADS : liste des wav + sha256 par fichier (min requis)

Split anti-leakage des sources en utilisant output.split_ratio

Créer des tasks instanciées tmp/tasks_train|val|test.jsonl :

scénario tiré selon weight

sources choisies dans le pool correspondant

source_hash inclus dans la task

Générer WAV :

load source(s)

trim/pad à duration_s

mix = somme

normalize peak à mix.normalize_peak_dbfs

write wav dans dataset_<split>/<label>/<sample_id>.wav

écrire 1 ligne metadata dans shard

Merge shards → dataset_<split>/metadata.jsonl

Validation stricte :

#wav == #jsonl lines par split

wav lisible, sr OK, durée OK (±1 sample)

pas de NaN/Inf

clip_count (optionnel seuil)

1.2 Format des tasks v0 (instanciées)
1 ligne JSON = 1 sample déterminé :

json
Copier le code
{
  "sample_id": "train_00000042",
  "split": "train",
  "label": 1,
  "scenario": "drone_plus_ambient",
  "seed": 123456789,

  "components": [
    { "role": "drone",
      "source_path": "dataset_DADS_offline/1/xxx.wav",
      "source_hash": "sha256:..." },

    { "role": "ambient",
      "source_path": "dataset_DADS_offline/0/yyy.wav",
      "source_hash": "sha256:..." }
  ],

  "output_relpath": "1/train_00000042.wav",

  "chain": [],                 // optionnel étape 2 (vide en v0)
  "global_chain": []           // optionnel étape 2 (vide en v0)
}
1.3 Format metadata v0
dataset_<split>/metadata.jsonl : 1 ligne par wav :

json
Copier le code
{
  "split": "train",
  "label": 1,
  "scenario": "drone_plus_ambient",
  "path": "1/train_00000042.wav",
  "sources": [
    {"role":"drone","path":"...","hash":"sha256:..."},
    {"role":"ambient","path":"...","hash":"sha256:..."}
  ],
  "audio": {
    "sample_rate": 22050,
    "duration_s": 4.0,
    "peak_dbfs": -1.0,
    "rms_dbfs": -23.4,
    "clip_count": 0
  }
}
1.4 Points “no compromise” à implémenter en étape 1
Remplacer toute utilisation de hash() par stable_int_seed().

merge_metadata ne doit pas ignorer silencieusement des lignes invalides :

si parse JSON fail → raise.

builder doit fonctionner avec output.total_samples + output.split_ratio.

source_split.json doit utiliser output.split_ratio (mêmes ratios que génération).

ÉTAPE 2 — Ajouter le moteur “modifiers” (sans activer d’effets)
But : préparer l’architecture pour les modifiers, sans changer le comportement v0.
Les chain existent mais restent vides.

2.1 Nouveaux modules (courts)
Créer augment_v5/modifiers/ :

registry.py
Un dict global OP_REGISTRY: dict[str, callable]

Un décorateur @register("op_name")

Aucune logique métier, juste un mapping.

chain.py
apply_chain(audio, chain, ctx) -> (audio, meta_updates)

parcourt la liste dans l’ordre

appelle l’op via OP_REGISTRY[op]

agrège les meta updates

validate_chain(chain) : checks minimaux (list, dict, clé op)

Si op inconnue → raise

ops_basic.py
Ops minimales réutilisées immédiatement (ou wrappers autour d’existant) :

trim_or_pad

normalize_peak

stats

(optionnel) gain_db (utile plus tard)

2.2 Modification du worker (sans changer v0)
Pour chaque composante :

load + trim/pad

apply_chain(component_audio, component.chain) (chain vide en v0)

Mix

apply_chain(mix_audio, task.global_chain) (vide en v0)

Puis write + stats + metadata

2.3 Modification de l’allocator (sans changer v0)
Ecrire dans chaque task :

components[].chain = []

global_chain = []

Ne pas modifier build_config.json encore (v0 inchangé).

ÉTAPE 3 — Spécifier et activer les modifiers (architecture complète)
But : rendre le pipeline “recette déterministe” :

build_config contient des *_chain_template avec distributions

allocator instancie (tirage RNG) → tasks avec valeurs concrètes

worker applique strictement les ops dans l’ordre

3.1 Nouvelle structure config (extension, compatible)
On ajoute optionnellement :

scenarios[].components[].chain_template: []

scenarios[].global_chain_template: []

Exemple minimal (activation progressive) :

json
Copier le code
{
  "scenarios": [
    {
      "name": "drone_plus_ambient",
      "weight": 1.0,
      "label": 1,
      "components": [
        {
          "role":"drone","count":1,
          "chain_template": [
            {"op":"normalize_rms","target_dbfs":{"dist":"uniform","min":-32,"max":-26}},
            {"op":"distance","distance_m":{"dist":"uniform","min":80,"max":250}}
          ]
        },
        {
          "role":"ambient","count":1,
          "chain_template": [
            {"op":"normalize_rms","target_dbfs":{"dist":"uniform","min":-38,"max":-30}}
          ]
        }
      ],
      "global_chain_template": [
        {"op":"mix_snr","snr_db":{"dist":"uniform","min":2,"max":14}},
        {"op":"hardware_orch","cfg":{"mems":{"enabled":true}}},
        {"op":"normalize_peak","peak_dbfs":-1.0},
        {"op":"stats"}
      ]
    }
  ]
}
Règle : les distributions n’existent QUE dans build_config.
Les tasks doivent contenir uniquement des valeurs concrètes.

3.2 Instanciation des templates (dans allocator)
resolve_value(spec, rng) :

number → passthrough

{dist:"uniform",min,max} → sample

{dist:"choice",values:[...]} → choice

{dist:"normal",mu,sigma} → normal

instantiate_chain(chain_template, rng) :

transforme chain_template en chain (valeurs concrètes)

3.3 Orchestrateurs (macro-ops) — option recommandé
Créer orchestrators.py :

env_orch : applique distance→doppler→pitch→rir dans un ordre figé

filters_orch : hp→lp→notch

hardware_orch : mems→adc→noise

Ces orchestrateurs sont eux-mêmes des ops du registry, mais “déplient” un cfg en sous-ops.

3.4 Ops par fichier (pour rester <100 lignes)
ops_distance.py : distance / air absorption

ops_filters.py : HP/LP/notch

ops_mems.py : bruit + coloration MEMS

ops_rir.py : convolution RIR

ops_mix.py : mix_snr / mix_sum

Chaque fichier :

1–3 ops max

tests rapides (au moins assertions et shapes)

3.5 Validation renforcée (étape 3)
Vérifier que l’ordre des ops appliquées correspond aux chain (loggable).

Anti-leakage stricte via source_hash sur metadata :

aucun hash source présent dans plusieurs splits

Statistiques : distribution distance, snr, etc. (optionnel, utile)

4) CLI attendue
python build_dataset_v5.py --config build_config.json --mode full --jobs 8

--mode tasks|gen|validate

--clean force un clean outputs (si on veut override config)

5) Checklist d’acceptation (à cocher)
Étape 1 OK si :
mode full produit WAV + metadata dans dataset_train|val|test

metadata.jsonl a exactement autant de lignes que de wav

seed stable : deux runs consécutifs donnent les mêmes tasks (identiques) et les mêmes outputs (à tolérance audio près si floating)

pas de hash() Python

merge strict (pas d’ignore silencieux)

Étape 2 OK si :
mêmes outputs que l’étape 1 (chains vides)

apply_chain fonctionne (avec ops_basic)

format tasks inclut chain/global_chain vides

Étape 3 OK si :
config templates → tasks instanciées → audio modifié selon ops

ordre strict garanti

orchestrateurs testés au moins sur un scénario

yaml
Copier le code
