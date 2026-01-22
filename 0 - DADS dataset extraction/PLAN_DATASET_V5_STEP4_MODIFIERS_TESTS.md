# Dataset Builder v5 — Step 4: Compléter tous les modifiers + orchestrators + tests

> **Objectif de cette étape :** enlever tous les `TODO_*` restants dans `augment_v5/modifiers/`, fournir un set “complet” d’ops et d’orchestrateurs utilisables en production, et ajouter une suite de **tests unitaires** (avec un focus sur la **distance attenuation**).
>
> **Philosophie conservée :**
> - Les **composantes** (drone / ambient / etc.) sont **atténuées / amplifiées** via leurs `components[].chain`.
> - Le **mix** (dans le worker) reste un **SUM** (somme des composantes déjà modifiées).
> - Le **post-traitement** se fait dans `global_chain` (normalize_peak + stats, puis env/hardware/rir).
> - Code **simple** / **modulaire** / **<100 lignes par fichier** si possible (sinon scinder).
> - Déterminisme : tout aléatoire doit être seedé via `ctx` (`seed`/`rng`), jamais via `hash()` Python.

---

## 0) Conventions v5 (important)

### 0.1 Format d’un op (aligné avec `apply_chain`)
Chaque op est un dict :
```json
{"op": "op_name", "params": {...}}
```

Signature des fonctions op (registry) :
```py
def op(audio: np.ndarray, params: dict, ctx: dict | None) -> tuple[np.ndarray, dict]:
    ...
    return audio_out, meta_dict
```

- `audio` : mono float32/float64 en [-1,1] (ou proche)
- `params` : valeurs **concrètes** (pas de `dist`)
- `ctx` : optionnel. Si l’op a besoin de randomness → utiliser `ctx`.

### 0.2 Randomness (si nécessaire)
**Règle :** si une op ajoute du bruit ou un effet stochastique, elle doit être déterministe via `ctx`.
- Convention recommandée : `ctx` contient **soit** `rng` (np.random.Generator), **soit** `seed` (int).
- Helper simple à créer (dans `modifiers/utils_rng.py`, <40 lignes) :
  - `get_rng(ctx) -> np.random.Generator` :
    - si ctx["rng"] existe → return
    - elif ctx["seed"] existe → return np.random.default_rng(ctx["seed"])
    - else → raise (pas de fallback)

**Conséquence :** le worker doit appeler `apply_chain(audio, chain, ctx={"seed": task["seed"]})`.

---

## 1) Ce qu’il faut implémenter (modifiers)

> Tous les fichiers placeholder `ops_*.py` et `orchestrators.py` doivent devenir **fonctionnels** (plus de TODO).

### 1.1 `ops_distance.py` (distance attenuation + air absorption simple)
Implémenter **au minimum** :

#### Op A — `distance_atten`
But : atténuation amplitude ~ 1/r (suffisant pour “drone lointain”).
Params :
- `distance_m: float`
- `ref_m: float = 1.0` (distance de référence)
- `min_distance_m: float = 1e-3` (sécurité)
Formule :
- `gain = ref_m / max(distance_m, min_distance_m)`
- `audio_out = audio * gain`
Meta :
- `distance_m`
- `distance_gain`
- `distance_db = 20*log10(gain)`

#### Op B (optionnel mais conseillé) — `air_absorb_lp`
But : simuler “perte d’aigus avec la distance” sans modèle physique lourd.
Params :
- `cutoff_hz: float` (valeur déjà instanciée dans task) **OU**
- `distance_m: float` + `cutoff_hz_at_1m` + `slope` (mais garder simple)
Recommandation : pour rester déterministe et simple :
- Instancier directement `cutoff_hz` dans les tasks (via template_resolver).
- `air_absorb_lp` applique un lowpass à `cutoff_hz`.

> **Note** : `air_absorb_lp` dépend de `ops_filters.lowpass`. Si scipy absent → raise clair.

---

### 1.2 `ops_filters.py` (HP / LP / Notch)
Implémenter (scipy requis, sinon raise explicite) :
- `highpass`: params `hz`, `sr`, `order=4`
- `lowpass`: params `hz`, `sr`, `order=4`
- `notch`: params `f0`, `q`, `sr`

Implementation recommandée :
- `scipy.signal.butter` + `lfilter` pour HP/LP
- `scipy.signal.iirnotch` + `lfilter` pour notch

Meta :
- retourner les params appliqués, ex : `{"highpass_hz":..., "lowpass_hz":...}`

---

### 1.3 `ops_rir.py` (convolution RIR)
Implémenter une op simple :

#### Op — `convolve_rir`
Params :
- `rir: list[float]` **ou** `rir_path: str` (au début, préférer `rir` pour tests)
- `mode: "same"|"full"` (défaut "same")
- `normalize_rir: bool` (défaut true : normaliser l’énergie)
Implementation :
- convolution numpy (FFT optionnel mais pas nécessaire au début)
- output "same" (même longueur que input) recommandé pour la pipeline

Meta :
- longueur RIR, normalize flag

> Plus tard, on branchera un pool RIR et une sélection via allocator.

---

### 1.4 `ops_mems.py` (hardware / MEMS / ADC) — version simple mais utile
Implémenter **au minimum** :

#### Op A — `add_white_noise`
Params :
- `noise_rms_dbfs: float` (ex: -60)
- `seed: optional` (si non fourni → utiliser ctx)
Implémentation :
- `rng = get_rng(ctx)`
- générer bruit gaussien, scaler pour atteindre RMS cible
- `audio_out = audio + noise`
Meta :
- `noise_rms_dbfs`

#### Op B — `adc_quantize`
Params :
- `bits: int` (ex: 12/14/16)
- `clip: bool = true`
Implémentation :
- si clip : clamp [-1,1]
- quantifier sur `2^bits` niveaux (simple rounding)
Meta :
- `adc_bits`

#### Op C (optionnel) — `mems_hp`
Params :
- `hp_hz: float`, `sr: int`
Implémentation :
- appeler `ops_filters.highpass`

---

### 1.5 `ops_mix.py` (compléter sans casser le worker)
Même si le worker mixe déjà, ce fichier ne doit plus être placeholder.
Implémenter **au minimum** :
- `mix_sum(parts: list[np.ndarray]) -> np.ndarray` (helper pur, pas op registry)
- (optionnel) `@register("mix_sum_ctx")` : op qui attend `ctx["parts"]` et retourne la somme  
  *Ne pas l’utiliser tant que le worker mixe en dur.*

---

## 2) Orchestrators (macro-ops) — `orchestrators.py`

But : garder la config lisible, et appliquer un ordre figé.

Implémenter 3 orchestrateurs enregistrés :

### 2.1 `filters_orch`
Params possibles :
- `hp_hz`, `lp_hz`, `notches` (liste)
- `sr` requis si filtres
Déplie vers :
- highpass → lowpass → notches (dans cet ordre)

### 2.2 `env_orch`
Params possibles :
- `distance_m`, `ref_m` → `distance_atten`
- `air_cutoff_hz` → `lowpass`
Déplie vers :
- distance_atten → (air_absorb_lp / lowpass)

### 2.3 `hardware_orch`
Params possibles :
- `noise_rms_dbfs` → add_white_noise
- `adc_bits` → adc_quantize
Déplie vers :
- add_white_noise → adc_quantize

**Implémentation recommandée :**
- construire une `subchain` (liste op dict) puis appeler `apply_chain(audio, subchain, ctx)`.

---

## 3) Tests — ajouter une suite pytest (focus distance)

### 3.1 Structure tests
Créer un dossier :
```
tests/
  test_registry.py
  test_ops_distance.py
  test_ops_gain.py
  test_ops_filters.py
  test_ops_rir.py
  test_ops_mems.py
  test_orchestrators.py
```

Exécution :
```bash
pytest -q
```

### 3.2 Test distance (prioritaire, précis)
`test_ops_distance.py` doit vérifier :

#### Cas 1 — scaling exact
- entrée : sinus ou constant `audio = np.ones(N)*0.1`
- appliquer `distance_atten(distance_m=10, ref_m=1)` → gain = 0.1
- assert : `max(abs(out - audio*0.1)) < 1e-8` (tolérance stricte)

#### Cas 2 — meta correct
- vérifier `distance_db ≈ 20*log10(gain)`

#### Cas 3 — robustesse min_distance
- distance_m très petit (0) → doit utiliser min_distance_m et ne pas exploser (pas NaN/Inf)

### 3.3 Test filters
`test_ops_filters.py` :
- si scipy absent : `pytest.skip(...)` **OU** faire échouer clairement (à décider)
- HP : DC rejeté (signal constant -> sortie ~0)
- LP : composante haute fréquence fortement réduite (sine à 8kHz vs cutoff 2kHz)

### 3.4 Test RIR
`test_ops_rir.py` :
- si `audio` est une impulsion (1 au début, reste 0), convolve_rir(mode="same") doit retourner la RIR (alignement à définir, documenter)

### 3.5 Test MEMS/ADC
`test_ops_mems.py` :
- add_white_noise : RMS augmente (grossier)
- adc_quantize : nombre de niveaux réduit / valeurs quantifiées

### 3.6 Test orchestrateurs
`test_orchestrators.py` :
- `filters_orch` doit produire même résultat que appliquer highpass puis lowpass
- `env_orch` idem distance_atten puis lowpass
- `hardware_orch` idem add_white_noise puis adc_quantize (seed fixe via ctx)

---

## 4) Mise à jour worker (ctx seed)
Modifier `worker.py` :
- Lors de `apply_chain` sur composantes et global_chain :
  - passer `ctx={"seed": task["seed"]}`

> Cela garantit que les ops stochastiques (bruit) sont déterministes.

---

## 5) Dépendances
- `pytest` pour tests
- `scipy` requis pour filtres (HP/LP/notch)
- `numpy` (déjà là)

**Pas de fallback :**
- Si un op requiert scipy et que scipy n’est pas installé :
  - `raise ImportError("scipy required for ...")`
- Les tests peuvent `skip` si scipy absent, mais l’op doit raise clairement.

---

# Prompt Copilot — Step 4 (implémentation + tests)

Colle dans Copilot Chat :

```text
Implémente Step 4: compléter tous les modifiers et orchestrators + ajouter des tests pytest.

Contrainte: code simple, modulaire, <100 lignes/fichier si possible. Pas de TODO placeholders.

1) modifiers:
- ops_distance.py:
  implement @register("distance_atten") (amplitude ~ 1/r) with meta distance_m, distance_gain, distance_db
  optionally implement air_absorb_lp or keep it as a thin wrapper around lowpass (cutoff_hz param)
- ops_filters.py:
  implement @register("highpass"), @register("lowpass"), @register("notch") using scipy.signal (raise ImportError if scipy missing)
- ops_rir.py:
  implement @register("convolve_rir") supporting params rir (list/np array) and mode="same"
- ops_mems.py:
  implement @register("add_white_noise") deterministic via ctx seed/rng
  implement @register("adc_quantize") (bits)
  optionally implement @register("mems_hp") calling highpass
- ops_mix.py:
  remove placeholder; implement helper mix_sum(parts) and optionally an op mix_sum_ctx expecting ctx["parts"] (not used by worker yet)
- orchestrators.py:
  implement @register("filters_orch"), @register("env_orch"), @register("hardware_orch") that build subchains then call apply_chain(audio, subchain, ctx)

2) modifiers/__init__.py:
- Ensure it imports all ops modules and orchestrators to populate OP_REGISTRY:
  ops_basic, ops_gain, ops_distance, ops_filters, ops_rir, ops_mems, ops_mix, orchestrators

3) worker.py:
- Pass ctx={"seed": task["seed"]} into apply_chain for component and global chains.

4) tests (pytest):
Create tests/ with:
- test_registry.py: importing augment_v5.modifiers registers key ops
- test_ops_distance.py: precise scaling tests for distance_atten (focus), meta correctness, min_distance
- test_ops_gain.py: gain_db scaling test
- test_ops_filters.py: HP removes DC, LP attenuates high freq (skip if scipy missing)
- test_ops_rir.py: impulse convolved gives expected output
- test_ops_mems.py: noise increases RMS; adc_quantize reduces unique values; deterministic noise with fixed seed
- test_orchestrators.py: filters_orch/env_orch/hardware_orch match applying equivalent subchain

Run:
pytest -q

Do not implement dataset "complete config" yet (later). Focus on completing modifiers + orchestrators + tests. Especially distance attenuation tests must be strict and deterministic.
```

