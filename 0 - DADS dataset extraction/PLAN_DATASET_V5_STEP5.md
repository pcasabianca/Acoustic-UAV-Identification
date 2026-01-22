# Dataset Builder v5 — Step 5: Motion (trajectory + doppler), orchestrators standards, checks de distribution

> **Objectif :** rapprocher le générateur d’un usage terrain (boîtier porté, MEMS, milieu très bruité, détection jusqu’à ~300 m) en ajoutant :
> 1) **mouvement** : `trajectory_atten` (gain variable dans le temps) + **doppler obligatoire**
> 2) **scénarios standards** via orchestrateurs (env/drone/hardware) pour config lisible et cohérente
> 3) **checks de distribution** sur `metadata.jsonl` (distance, mouvement, ratio label, bruit/hardware, clipping, etc.)
>
> **Contraintes inchangées :**
> - Worker “recette-only”, pas de logique de sampling dans le worker
> - Mix = **SUM uniquement**
> - Déterminisme : tout stochastique via `ctx` seed/rng, pas de `hash()` Python
> - Code simple, modulaire, idéalement <100 lignes/fichier
> - Layout sortie inchangé + `metadata.jsonl` par split

---

## 0) Conventions essentielles

### 0.1 `ctx` pour déterminisme
Les ops qui utilisent du hasard doivent obtenir un RNG via `ctx` :
- le worker appelle `apply_chain(audio, chain, ctx={"seed": task["seed"]})`.

Si vous avez `utils_rng.get_rng(ctx)`, il doit :
- utiliser `ctx["rng"]` si fourni
- sinon construire `np.random.default_rng(ctx["seed"])`
- sinon raise (pas de fallback)

### 0.2 Doppler “pragmatique” mais robuste
Le doppler physique complet est lourd. Pour la robustesse d’un détecteur, une approximation contrôlée suffit :
- **resampling variable dans le temps**, implémenté **par segments** :
  - découper l’audio en K segments (ex 32)
  - resample chaque segment avec un ratio légèrement différent `ratio_i`
  - concat + trim/pad pour retrouver la longueur d’origine

Dépendance : `scipy.signal.resample_poly` (sinon **raise ImportError**).

---

## 1) Modifiers à ajouter / compléter

### 1.1 `trajectory_atten` (distance time-varying)
**Fichier :** `augment_v5/modifiers/ops_distance.py`

Op : `trajectory_atten`  
Params :
- `distance_start_m: float`
- `distance_end_m: float`
- `ref_m: float = 1.0`
- `curve: "linear"|"ease_in_out" = "linear"`
- `min_distance_m: float = 1e-3`

Algorithme :
- `N = len(audio)`
- distance(t) linéaire ou smoothstep
- `gain(t) = ref_m / max(distance(t), min_distance_m)`
- `audio_out = audio * gain(t)`

Meta minimum :
- `distance_start_m`, `distance_end_m`, `ref_m`
- `gain_min`, `gain_max`
- `db_min`, `db_max`
- `distance_min`, `distance_max`

Compat :
- conserver `distance_atten` (gain constant)
- test: `trajectory_atten` avec start=end ≈ `distance_atten`

---

### 1.2 Doppler obligatoire : `doppler_segmented`
**Nouveau fichier :** `augment_v5/modifiers/ops_doppler.py` (court)

Op : `doppler_segmented`  
Params :
- `v_mps: float`
- `direction: int` (+1 approaching, -1 receding)
- `c_mps: float = 343.0`
- `segments: int = 32` (>= 8)
- `max_ratio: float = 1.05` (clamp)
- `sr: int` (requis)
- `profile: "linear"|"ease_in_out" = "linear"` (évolution du ratio)

Modèle simple :
- ratio doppler “cible” : `r_target = c / (c - direction*v)` (approche)
- clamp dans `[1/max_ratio, max_ratio]`
- construire un ratio temps `r(t)` entre 1 et r_target selon profile
- segmenter en `segments` segments
- pour segment i :
  - `ratio_i` = moyenne de r(t) sur ce segment
  - resample segment par `ratio_i` via `resample_poly`
- concat + trim/pad à longueur d’origine

Meta :
- `doppler_v_mps`, `doppler_direction`, `doppler_ratio_min`, `doppler_ratio_max`, `segments`

**Pas de fallback :** si scipy absent → raise ImportError.

---

## 2) Orchestrateurs standards (scénarios réutilisables)

> Objectif : rendre les scénarios lisibles, éviter les configs incohérentes, imposer un ordre fixe.

### 2.1 `env_orch` (standard environnement)
**Fichier :** `augment_v5/modifiers/orchestrators.py`

Params (instanciés par allocator) :
- `distance_mode: "static"|"trajectory"`
- si static : `distance_m`
- si trajectory : `distance_start_m`, `distance_end_m`
- `ref_m`
- `air_cutoff_hz` optionnel (lowpass)
- `doppler` optionnel :
  - `enabled: bool`
  - `v_mps`, `direction`, `segments`, `max_ratio`, `profile`
- `sr`

Sous-chain (ordre figé) :
1) distance_atten OU trajectory_atten
2) (option) lowpass(air_cutoff_hz)
3) (option) doppler_segmented

> Documenter l’ordre et le conserver.

### 2.2 `hardware_orch` (standard MEMS)
Params :
- `noise_rms_dbfs`
- `adc_bits`
- `hp_hz` optionnel
- `sr`

Sous-chain :
1) (option) highpass(hp_hz)
2) add_white_noise(noise_rms_dbfs)
3) adc_quantize(adc_bits)

### 2.3 `drone_orch` (standard drone) — recommandé
But : pack “signature drone” sans confondre avec l’environnement/hardware.
Implémenter au minimum :
- `amp_flutter` : enveloppe lente ±X dB, déterministe via ctx seed
Params :
- `flutter_db: float` (ex 0–2)
- `flutter_hz: float` (ex 0.3–1.5)
- `sr`

Sous-chain :
1) amp_flutter
(Option plus tard : jitter pitch, blade-passing tones, etc.)

---

## 3) Scénarios terrain : config JSON “field v1”

Créer `build_config_field_v1.json` (première version utile) :

- **30% label 0** (noise only)
- label 1 : distances jusqu’à 300m, **mouvement fréquent**, doppler fréquent
- **toujours** au moins 1 ambient
- **global_chain toujours** :
  - hardware_orch (noise + adc + hp option)
  - normalize_peak(-1 dBFS)
  - stats(sr)

Exemples de scénarios (recommandés) :

### Label 0 (30%)
- `noise_only_1amb` (10%)
- `noise_only_2amb` (15%)
- `noise_only_harsh` (5%) : plus de noise_rms_dbfs, filtres, etc.

### Label 1 (70%)
- `drone_far_static` (20%) : static distance 150–300
- `drone_mid_static` (10%) : 60–150
- `drone_moving_in` (15%) : trajectory 250→50 + doppler approaching
- `drone_moving_out` (15%) : 50→250 + doppler receding
- `drone_multi` (10%) : 2 drones distances différentes + éventuellement 2 trajectoires

Pour chaque scénario label 1 :
- drone components : env_orch + drone_orch + (option) gain_db
- ambient components : gain_db léger (variabilité) + (option) lowpass/hp

---

## 4) Checks de distribution (post-build)

Étendre `validate.py` avec `validate_distribution(...)` et l’appeler dans `--mode validate` et dans `full` :

### 4.1 Ratio label 0
- global ratio label0 ~ 0.30 (tolérance ±0.02)
- par split (tolérance plus large)

### 4.2 Couverture distances (jusqu’à 300 m)
Extraire depuis metadata :
- static: `distance_m`
- trajectory: `distance_start_m`, `distance_end_m`
Construire bins :
- [0–60], [60–150], [150–300]
Exiger :
- présence dans chaque bin
- pour label 1, bin [150–300] >= 25% (reco) pour coller au terrain

### 4.3 Mouvement (label 1)
moving si `distance_start_m != distance_end_m` (ou flag)
Exiger : moving_ratio(label1) >= 0.50

### 4.4 Bruit/hardware
Exiger que metadata contienne des champs hardware :
- `noise_rms_dbfs` / `adc_bits` (selon vos metas)
Exiger qu’une fraction ait “bruit fort” (ex noise_rms_dbfs > -55 dBFS)

### 4.5 Doppler
Exiger qu’une fraction de label1 utilise doppler (ex >= 40%) via meta `doppler_*`

### 4.6 Clipping
Exiger `clip_count` raisonnable (ex <= 1% des samples, ou seuil total)

---

## 5) Tests unitaires à ajouter (focus motion/doppler)

### 5.1 `test_ops_distance.py`
- `trajectory_atten` start=end ≈ `distance_atten`
- endpoints corrects
- no NaN/Inf

### 5.2 `test_ops_doppler.py` (nouveau)
- v=0 → quasi identité
- direction +1 vs -1 → shift fréquentiel opposé sur un sinus
- déterminisme : seed fixe → même sortie
- si scipy absent : skip (mais l’op doit raise ImportError)

### 5.3 `test_orchestrators.py`
- `env_orch` == sous-chain explicite (distance/trajectory + lowpass + doppler)
- `hardware_orch` == sous-chain explicite
- `drone_orch` == sous-chain explicite (amp_flutter)

---

## 6) Imports / registry
Mettre à jour `augment_v5/modifiers/__init__.py` pour importer :
- `ops_doppler`
- `orchestrators` (si pas déjà)
afin de remplir `OP_REGISTRY`.

---

# Prompt Copilot — Step 5 (motion + orch standards + distribution checks)

Colle dans Copilot Chat :

```text
Implement Step 5: motion (trajectory_atten + doppler), standard orchestrators (env/hardware/drone), and distribution checks.

Constraints:
- Worker recipe-only, no sampling logic in worker
- Mix stays SUM only
- Deterministic: all stochastic ops use ctx seed/rng, no Python hash()
- Keep files <100 lines if possible
- Layout output unchanged; metadata.jsonl per split
- SciPy required for doppler and filters; if missing raise ImportError; tests may skip when scipy missing

A) trajectory_atten
- In augment_v5/modifiers/ops_distance.py add @register("trajectory_atten")
- Apply time-varying gain envelope based on distance_start/end and curve linear/ease_in_out
- Meta includes start/end, gain/db min/max, distance min/max
- Add strict tests in tests/test_ops_distance.py: start=end matches distance_atten, endpoints correct, no NaN/Inf

B) doppler_segmented (mandatory)
- New file augment_v5/modifiers/ops_doppler.py with @register("doppler_segmented")
- Implement segmented resampling with scipy.signal.resample_poly
- ratio target r=c/(c-direction*v), clamp to max_ratio, vary over time profile, apply per segment
- Trim/pad to original length
- Meta includes v, direction, ratio_min/max, segments
- Add tests tests/test_ops_doppler.py:
  v=0 near identity; deterministic with fixed seed; sine frequency shift differs for direction +1 vs -1
  skip tests if scipy missing

C) Orchestrators standards
- In augment_v5/modifiers/orchestrators.py implement/extend:
  @register("env_orch"): distance_atten or trajectory_atten + optional lowpass(air_cutoff_hz) + optional doppler_segmented, fixed order
  @register("hardware_orch"): optional highpass + add_white_noise + adc_quantize, fixed order
  @register("drone_orch"): amp_flutter deterministic via ctx seed (slow envelope), fixed order
- Update tests/test_orchestrators.py to compare each orch output with explicit subchain output.

D) Registry imports
- Update augment_v5/modifiers/__init__.py to import ops_doppler and orchestrators (and any new ops) to populate OP_REGISTRY.

E) Distribution checks
- Extend validate.py with validate_distribution reading metadata.jsonl for each split and checking:
  label0 ratio ~ 0.30 (+/- tol)
  distance bins coverage up to 300m; ensure far bin [150,300] >= 25% among label1
  moving ratio among label1 >= 0.50
  hardware/noise meta present; fraction with strong noise (e.g. noise_rms_dbfs > -55)
  doppler usage fraction among label1 >= 0.40 (via meta doppler_* fields)
  clipping rate reasonable
- Hook validate_distribution into --mode validate and full.

F) Field config v1
- Create build_config_field_v1.json:
  30% label 0 noise-only scenarios; label1 scenarios include far static, mid static, moving in/out with doppler, multi-drone
  always at least 1 ambient
  component chains use env_orch + drone_orch (+ optional gain_db)
  global_chain uses hardware_orch + normalize_peak + stats
- Confirm:
  pytest -q
  python build_dataset_v5.py --config build_config_field_v1.json --mode full --jobs 4 --clean
