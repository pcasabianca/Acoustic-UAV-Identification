# Dataset Builder v5 — Step 3 Hardening + “Modifiers Minimal Pack” (vers un système complet)

> **But de cette itération :** consolider Step 3 (templates → tasks → exécution), supprimer les derniers implicites, et ajouter un **pack minimal de modifiers** cohérent avec ta philosophie :  
> **on atténue/amplifie des composantes (ex: drone lointain), puis on mixe en somme**, puis on applique une post‐chaîne globale (normalize + stats, puis plus tard hardware/env/rir).
>
> **Contraintes inchangées :**
> - Dossiers de sortie fixes : `0 - DADS dataset extraction/dataset_train|dataset_val|dataset_test`
> - `metadata.jsonl` dans chaque split
> - overwrite complet à chaque run
> - `tmp/` pour les artefacts (`source_split`, `tasks`, `shards`, etc.)
> - code **simple**, **direct**, **<100 lignes par fichier** si possible
> - déterminisme : **interdit** d’utiliser `hash()` Python pour les seeds

---

## 1) Ce qu’on veut obtenir après cette passe

### 1.1 Pipeline “recette only”
- **Toutes** les opérations audio doivent être **explicites** dans les tasks :
  - `components[].chain` (exécutée par composante)
  - `global_chain` (exécutée après mix)
- Le worker ne doit plus avoir de logique implicite de normalisation/trim : il exécute la recette.

### 1.2 Templates propres dans build_config
- Supporter dans `build_config.json` :
  - `components[].chain_template` (optionnel)
  - `scenario.global_chain_template` (optionnel)
- Les templates peuvent contenir des distributions (`uniform/choice/normal`), **mais les tasks instanciées ne doivent contenir que des valeurs concrètes**.  
  → **Zéro** clé `dist` dans `tmp/tasks_*.jsonl`.

### 1.3 Fail-fast réel end-to-end
- Si un worker plante, **le build doit échouer** avant merge.
- Pas de “print et continue” masquant des erreurs.

### 1.4 Modifiers minimal pack (cohérent avec le mix sum)
- Ajouter des ops simples et utiles :
  - `gain_db` (déjà ajouté)
  - `distance_atten` (op unique et claire)
- Le mix reste **sum**. Pas de `mix_snr`, pas de `normalize_rms`.

---

## 2) Changements recommandés (concrets, module par module)

### 2.1 `augment_v5/allocator.py` — templates, instanciation, robustesse
**Objectifs :**
1) Lire **template OU legacy** (rétro-compatible) :
   - composantes : `chain_template` **ou** `chain`
   - global : `global_chain_template` **ou** `global_chain`
2) Instancier les templates en chaînes **concrètes** via `instantiate_chain(...)` :
   - aucune distribution ne doit survivre dans les tasks
3) RNG **par sample** (stabilité) :
   - `seed = stable_int_seed(f"{master_seed}:{split}:{sample_id}:{scenario_name}")`
4) Préfixer systématiquement la chaîne composante avec :
   - `trim_or_pad(target_samples=...)`
5) Garantir `global_chain` minimale :
   - contient `normalize_peak(target_dbfs=...)` si absent
   - contient `stats(sr=...)` si absent

> **Pourquoi RNG par sample ?**  
> Cela évite l’“effet papillon” (si tu changes les poids des scénarios, ça ne décale pas tout l’aléatoire du split).

---

### 2.2 `augment_v5/worker.py` — exécution pure
**Objectifs :**
- Importer `augment_v5.modifiers` (pour enregistrer les ops via side effects).
- Supprimer tout traitement hardcodé (trim/normalize) :  
  **tout doit passer par `components[].chain` et `global_chain`**.
- Mix uniquement :
  - `mix = sum(components_audio)` (avec alignement de longueur déjà assuré par `trim_or_pad`)
- Fail-fast :
  - si une task échoue → lever l’exception (et donc exitcode non nul).

---

### 2.3 `augment_v5/builder.py` — fail-fast sur workers
**Objectifs :**
- Après `join()` de chaque process, vérifier `exitcode`.
- Si `exitcode != 0` pour au moins un worker :
  - `raise RuntimeError(...)`
  - **ne pas** lancer le merge metadata

> Ça transforme les crashs workers en échec visible (au lieu d’un dataset partiel).

---

### 2.4 `augment_v5/task_format.py` — validation de chain/global_chain
**Objectifs :**
- Étendre la validation :
  - `components[].chain` optionnel mais si présent : `list[dict]` avec clé `"op": str`, `"params": dict` optionnel
  - `global_chain` idem
- (Option) Dans `allocator`, appeler `validate_chain(chain)` une fois les ops importées, pour attraper les erreurs d’op tôt.

---

### 2.5 `augment_v5/modifiers/ops_distance.py` — simplifier “distance” en une seule op
**Objectif :** ne garder qu’une op claire correspondant à ton besoin : **atténuer/amplifier avant mix**.

✅ Créer/standardiser :
- `op: "distance_atten"`
- params :
  - `distance_m: float`
  - `ref_m: float = 1.0`
- facteur amplitude :
  - `gain = ref_m / max(distance_m, 1e-6)`  (amplitude ~ 1/r)
- meta retournée :
  - `distance_m`, `distance_gain`, `distance_db` (20log10(gain))

> **Important :** éviter d’avoir plusieurs variantes “distance_*” qui créent de la confusion.  
> Une op simple = moins de bugs.

---

### 2.6 `augment_v5/modifiers/__init__.py` — registry OK
Ton init est parfait : il importe explicitement les ops pour peupler `OP_REGISTRY` via side effects.

---

### 2.7 `augment_v5/validate.py` — vers une validation plus “complète”
Recommandé (sans complexifier) :
- ajouter check NaN/Inf
- ajouter anti-leakage simple via `source_hash` (déjà dans metadata) :
  - aucun `source_hash` ne doit apparaître dans plusieurs splits

---

## 3) Exemple de config “modifiers quick” (recommandée pour tester Step 3)

Créer `build_config_modifiers_quick.json` (ex: 20 samples) :

- drone : `distance_atten` (distance random)
- ambient : `gain_db` (petite variation)
- global : normalize_peak + stats

> Le mix reste `sum`.

---

## 4) Tests attendus

### 4.1 Génération complète rapide
```bash
python build_dataset_v5.py --config build_config_modifiers_quick.json --mode full --jobs 2 --clean
```

### 4.2 Vérifications minimales
- `dataset_train/metadata.jsonl` a autant de lignes que de WAV dans `dataset_train/0` + `dataset_train/1`
- idem pour `val` et `test`
- tasks dans `tmp/tasks_*.jsonl` :
  - ne contiennent **aucun** objet `{ "dist": ... }`
- si un worker plante → `builder` doit échouer avant merge

---

## 5) Roadmap “vers un système complet” (sans casser le worker)

Le principe : **le worker ne bouge plus** (recette only). On enrichit uniquement :
- le resolver (instanciation),
- la registry,
- des ops unitaires (petits fichiers),
- des orchestrateurs (macro‐ops).

### Prochaine marche (après cette passe)
1) **Filtres simples** : `highpass`, `lowpass` (scipy obligatoire, sinon raise clair)
2) **RIR** : convolution (si dépendances OK), sinon off par défaut
3) **Hardware** : mems/noise/adc via un orchestrateur `hardware_orch`
4) **Env** : doppler / pitch / absorption via `env_orch`

Toujours :
- 1 op = 1 fonction pure
- 1 fichier = 1–3 ops
- config = templates → tasks concrètes

---

# Copilot Prompt — appliquer cette passe (hardening + distance_atten)

Colle ce prompt dans Copilot Chat :

```text
Implémente les changements décrits dans "Dataset Builder v5 — Step 3 Hardening + Modifiers Minimal Pack".

Contraintes:
- Layout sortie inchangé: dataset_train/val/test + metadata.jsonl
- tmp/ pour tasks/shards
- <100 lignes par fichier si possible
- Déterminisme: aucun hash() Python pour seeds
- Mix = sum uniquement (pas de mix_snr, pas de normalize_rms)
- Fail-fast: si un worker plante, builder doit échouer avant merge

A FAIRE:
1) allocator.py:
- Supporter keys template OU legacy:
  component chain: chain_template OR chain
  global chain: global_chain_template OR global_chain
- Instancier via instantiate_chain (template_resolver.py)
- RNG par sample: seed = stable_int_seed(f"{master_seed}:{split}:{sample_id}:{scenario_name}")
- Prepend trim_or_pad op to each component chain with target_samples = int(sr*duration_s)
- Ensure global_chain contains normalize_peak and stats if missing (append if absent)
- Write tasks with concrete "chain" and "global_chain" only (no dist objects)

2) worker.py:
- Import augment_v5.modifiers to ensure OP_REGISTRY is populated
- Remove any hardcoded trim/normalize/stats; rely only on chains
- For each component: load -> apply_chain(component.chain)
- Mix: sum of arrays
- Apply global_chain
- Fail-fast: if any task fails, raise (no print+continue)

3) builder.py:
- After joining each worker process, check exitcode.
- If any exitcode != 0: raise RuntimeError and do NOT merge shards.

4) task_format.py:
- Extend validate_task_structure to validate:
  components[].chain is list of dicts with "op" str, optional "params" dict
  global_chain is list of dicts with same format

5) ops_distance.py:
- Simplify to a single op "distance_atten" (amplitude 1/r):
  params: distance_m, ref_m=1.0
  gain = ref_m / max(distance_m, 1e-6)
  meta returns distance_m, distance_gain, distance_db
- Remove or deprecate extra distance ops to avoid confusion

6) validate.py (optional but recommended):
- Add NaN/Inf checks
- Add anti-leakage check using source_hash across splits (read metadata.jsonl)

7) Add config example:
- Create build_config_modifiers_quick.json with:
  drone component chain_template includes distance_atten with distance_m uniform [50,250]
  ambient chain_template includes gain_db with db uniform [-6,0]
  global_chain_template includes normalize_peak -1.0 and stats(sr)
- Verify:
  python build_dataset_v5.py --config build_config_modifiers_quick.json --mode full --jobs 2 --clean
```
