# Dataset Builder v5 — Step 3 (Activation des modifiers)  
**Contexte :** Steps 1+2 sont en place (pipeline v0 fonctionnel + squelette modifiers).  
**Objectif Step 3 :** rendre les modifiers réellement configurables via **templates instanciés en tasks**, tout en gardant le code **simple, déterministe, modulable** et en respectant le layout imposé.

---

## 0) Contraintes non négociables (rappel)

- Output dirs fixes dans `0 - DADS dataset extraction/` :
  - `dataset_train/`, `dataset_val/`, `dataset_test/`
  - `metadata.jsonl` **dans chaque split**
- **Overwrite complet** à chaque run (pas d’ID de dataset).
- Intermédiaires uniquement dans `tmp/` (tasks, split, shards…).
- **Aucun fichier > 100 lignes** si possible (sinon scinder).
- **Aucune magie / aucun fallback** : erreurs explicites, ordre explicite.
- **Déterminisme** :
  - INTERDIT d’utiliser `hash()` Python.
  - Toute aléa doit être résolu **uniquement au moment de créer les tasks**.
- **Anti-leakage** : pools sources disjoints train/val/test (par rôle), et tasks d’un split ne piochent que dans son pool.
- **Ratio unique** : `output.split_ratio` sert à
  1) répartir les **samples générés** (train/val/test)
  2) répartir les **sources DADS** dans `source_split.json`

---

## 1) Ce que Step 3 change (en une phrase)

On passe de **tasks “sans effets”** à **tasks “recette complète”** :  
`build_config.json` contient des `*_chain_template` (avec distributions) → `allocator` instancie → `tasks_*.jsonl` contiennent des chaînes **concrètes** (scalars only) → `worker` exécute la recette strictement dans l’ordre.

> Important : **le mix reste un simple SUM**.  
> Les drones lointains/forts sont obtenus en amont via des modifiers (gain/distance) appliqués aux composantes, pas via un mix “intelligent”.

---

## 2) Format exact attendu des Chains (aligné sur votre `apply_chain`)

Votre moteur actuel applique une `chain` qui est une liste d’objets :

```json
{"op": "op_name", "params": { ... }}
```

Règles Step 3 :
- `chain` et `global_chain` sont des **list** (peuvent être vides).
- Chaque item :
  - doit être un dict contenant `op`
  - peut contenir `params` (dict, défaut `{}`)
- **Les tasks ne doivent contenir aucune distribution** (pas de `{ "dist": ... }` dans une task).
  - si on détecte `dist` dans une task → `raise`.

---

## 3) Évolutions build_config.json (compatibles v0)

### 3.1 Champs optionnels ajoutés (templates)
On ajoute (optionnel) :

- `scenarios[].components[].chain_template: [ {"op":..., "params":...}, ... ]`
- `scenarios[].global_chain_template: [ ... ]`

Si absent :
- `components[].chain_template` = `[]`
- `global_chain_template` = `[]` (puis on ajoute un default global en tasks, voir §5.3)

### 3.2 Distributions autorisées dans les templates
Dans les templates **uniquement**, `params` peuvent contenir des “specs” :

- Uniforme :
```json
{"dist":"uniform","min":-20.0,"max":0.0}
```

- Choice :
```json
{"dist":"choice","values":[40,60,80]}
```

- Normal :
```json
{"dist":"normal","mu":0.0,"sigma":0.4}
```

Règles :
- le resolver retourne une valeur scalaire (int/float/bool/str).
- la résolution se fait dans `allocator` avec un RNG **par sample** (recommandé).

### 3.3 Exemple de config Step 3 (mix SUM + distance/gain)
Exemple (minimal) :

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

  "mix": { "mode": "sum", "normalize_peak_dbfs": -1.0 },

  "scenarios": [
    {
      "name": "ambient_only",
      "weight": 1.0,
      "label": 0,
      "components": [
        {
          "role": "ambient",
          "count": 1,
          "chain_template": [
            { "op":"gain_db", "params": { "db": {"dist":"uniform","min":-6,"max":6} } }
          ]
        }
      ],
      "global_chain_template": []
    },

    {
      "name": "drone_plus_ambient",
      "weight": 1.0,
      "label": 1,
      "components": [
        {
          "role": "drone",
          "count": 1,
          "chain_template": [
            { "op":"distance_gain", "params": { "distance_m": {"dist":"uniform","min":30,"max":300}, "ref_m": 30.0 } }
          ]
        },
        {
          "role": "ambient",
          "count": 1,
          "chain_template": []
        }
      ],
      "global_chain_template": []
    }
  ]
}
```

---

## 4) Format tasks_*.jsonl (instanciées, scalars only)

Chaque task doit contenir des chains **concrètes** :

```json
{
  "sample_id": "train_00000042",
  "split": "train",
  "label": 1,
  "scenario": "drone_plus_ambient",
  "seed": 123456789,

  "components": [
    {
      "role": "drone",
      "source_path": "dataset_DADS_offline/1/xxx.wav",
      "source_hash": "sha256:....",
      "chain": [
        {"op":"trim_or_pad","params":{"target_samples":88200}},
        {"op":"distance_gain","params":{"distance_m":183.2,"ref_m":30.0}}
      ]
    },
    {
      "role": "ambient",
      "source_path": "dataset_DADS_offline/0/yyy.wav",
      "source_hash": "sha256:....",
      "chain": [
        {"op":"trim_or_pad","params":{"target_samples":88200}}
      ]
    }
  ],

  "global_chain": [
    {"op":"normalize_peak","params":{"target_dbfs":-1.0}},
    {"op":"stats","params":{"sr":22050}}
  ],

  "output_relpath": "1/train_00000042.wav"
}
```

---

## 5) Changements de code à faire (Step 3) — précis et minimal

### 5.1 S’assurer que les ops sont enregistrées (IMPORTANT)
Actuellement, les décorateurs `@register(...)` ne fonctionnent que si les modules d’ops sont importés.

**Action :** modifier `augment_v5/modifiers/__init__.py` pour importer explicitement les ops (side effects) :

- `from . import ops_basic`
- `from . import ops_gain`
- `from . import ops_distance`
- (plus tard : ops_filters, ops_mems, ops_rir, orchestrators, etc.)

But : `OP_REGISTRY` est prêt dès qu’on importe `augment_v5.modifiers`.

### 5.2 Ajouter un resolver de templates → scalars
Créer un module court (ex : `augment_v5/template_resolver.py`) ou l’implémenter dans `allocator.py` si ça tient <100 lignes.

Fonctions requises :
- `resolve_value(v, rng)` :
  - si `v` est dict avec `"dist"` : sample (uniform/choice/normal)
  - si `v` est list/dict : résoudre récursivement
  - sinon : retourner v
- `instantiate_chain(chain_template, rng)` :
  - pour chaque item:
    - copier `op`
    - `params = resolve_value(item.get("params", {}), rng)`
  - retourner une liste `chain` avec scalars only

**Règle :** dans la task finale, aucun `dist` ne doit rester.

### 5.3 Modifier `allocator` : instancier les chains + RNG par sample
**Actions :**
1) Construire `n_train,n_val,n_test` depuis `output.total_samples` et `output.split_ratio` (déjà en place).
2) Split sources DADS avec `output.split_ratio` (déjà).
3) Lors de `build_tasks(split)` :
   - choisir scénario via weights
   - choisir sources dans pool du split
   - construire un RNG **par sample** :
     - `seed = stable_int_seed(f"{master_seed}:{split}:{sample_id}:{scenario_name}")`
4) Construire `components[].chain` :
   - commencer par `trim_or_pad(target_samples)`
   - ajouter `instantiate_chain(chain_template, rng)`
5) Construire `global_chain` :
   - `instantiate_chain(global_chain_template, rng)`
   - si vide, ajouter par défaut :
     - `normalize_peak(target_dbfs=mix.normalize_peak_dbfs)`
     - `stats(sr=audio.sample_rate)`
6) Écrire la task (JSONL).

### 5.4 Modifier `worker` : supprimer toute opération implicite (worker = exécuteur)
Le worker doit faire uniquement :
1) Charger les sources.
2) `apply_chain` sur chaque composante (dans l’ordre).
3) Mixer = **SUM** (sans autre logique).
4) `apply_chain` sur `global_chain`.
5) Écrire WAV atomiquement.
6) Écrire 1 ligne metadata dans son shard.
7) **Fail-fast** : si une task échoue → raise (ou accumulate puis raise à la fin). Pas de “print et continue”.

**Important :**
- Ne plus faire `trim_or_pad` et `normalize_peak` en “dur” dans le worker : c’est dans les chains.
- Le worker doit importer `apply_chain` depuis `augment_v5.modifiers` et importer `augment_v5.modifiers` (pour register ops via __init__).

### 5.5 Modifier `task_format.validate_task_structure`
Étendre la validation minimale :
- `components[].chain` si présent → list
- `global_chain` si présent → list
- items chain : dict contenant `op` (et params dict optionnel)

### 5.6 Validation renforcée (recommandé mais simple)
Dans `validate.py`, ajouter :
- check NaN/Inf sur audio
- anti-leakage via `source_hash` dans metadata :
  - collecter hashes par split et vérifier intersections vides
- option : check clip_count max

---

## 6) Nouveaux modifiers à implémenter maintenant (minimal & utiles)

### 6.1 `ops_gain.py` (NOUVEAU)
- `gain_db` : multiplie l’audio par `10**(db/20)`

Signature (alignée registry) :
- `@register("gain_db")`
- `def op_gain_db(audio, params, ctx=None) -> (audio, meta)`
  - params: `{"db": float}`

### 6.2 `ops_distance.py` (remplacer placeholder)
- `distance_gain` : applique une atténuation simple en fonction de la distance
  - params: `{"distance_m": float, "ref_m": float}`
  - gain = `ref_m / max(distance_m, 1e-6)` (ou une formule en dB équivalente)
  - option : retourner meta `{ "distance_m":..., "distance_gain":... }`

> On reste volontairement simple : la distance sert ici à atténuer (drones lointains). Les effets plus réalistes (air absorption / RIR) seront ajoutés plus tard via d’autres ops.

---

## 7) Tests d’acceptation Step 3

### Test 1 — Compatibilité v0
- Mettre `chain_template=[]` partout + global_chain_template absent.
- Run : `mode full`
- Résultat : dataset généré identique (mêmes counts, valide, pas d’erreur).

### Test 2 — Activation gain_db
- Ajouter `gain_db` sur une composante ambient (±6 dB)
- Vérifier :
  - metadata contient toujours 1 ligne par wav
  - pipeline valide
  - stats RMS varient (visible dans metadata)

### Test 3 — Activation distance_gain
- Ajouter `distance_gain` sur drone
- Vérifier :
  - metadata contient `distance_m` (si vous l’enregistrez)
  - amplitude drone varie comme attendu

### Test 4 — Déterminisme
- Deux runs successifs même config :
  - `tmp/tasks_*.jsonl` identiques
  - counts identiques
  - validation OK

---

## 8) Commandes

Build complet :
```bash
python build_dataset_v5.py --config build_config.json --mode full --jobs 4 --clean
```

Générer uniquement tasks :
```bash
python build_dataset_v5.py --config build_config.json --mode tasks
```

Valider :
```bash
python build_dataset_v5.py --config build_config.json --mode validate
```
