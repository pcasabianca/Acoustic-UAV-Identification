# Dataset Builder v5 — Step 6: “Field-Perfect” (contrôle des ratios + SNR + variété maximale)

## Objectif
Construire un dataset robuste pour détecter des drones **jusqu’à 300 m**, dans du **bruit fort MEMS**, avec **effets multiples**, y compris des drones **quasi inaudibles à l’oreille**.

Contraintes maintenues :
- worker “recette-only”
- mix = SUM
- déterminisme via ctx/seed, zéro hash() python
- metadata.jsonl par split
- config JSON pilotée (templates -> tasks concrètes)

---

## 1) Problème à corriger: ratios & distributions non garantis
Aujourd’hui, on échantillonne par poids de scénarios -> le ratio label0 dérive (ex 0.3/1.05 ≈ 28.6% au lieu de 30%).  
=> On doit **forcer** les quotas (label0, bins distance, moving, doppler, bruit fort).

---

## 2) Modification clé: allocation “à quotas” (allocator)
### 2.1 Quotas à introduire dans la config (nouvelle section `targets`)
Ajouter dans `build_config_field_v2.json` :

```json
"targets": {
  "label0_ratio": 0.30,

  "distance_bins_m": [0, 60, 150, 300],
  "distance_bin_weights_label1": [0.25, 0.35, 0.40],

  "moving_ratio_label1": 0.70,
  "doppler_ratio_label1": 0.70,

  "strong_noise_ratio": 0.25,
  "strong_noise_dbfs": -55.0,

  "snr_bins_db": [-30, -20, -10, 0],
  "snr_bin_weights_label1": [0.25, 0.40, 0.25, 0.10]
}
But:

distances : couvrir proche/mid/far, sans être 90% far

moving/doppler : majoritaires (terrain)

bruit fort : une vraie fraction “terrible”

SNR : forcer une partie très dure (ex -30 à -20 dB)

2.2 Réécrire l’allocation scenario → tasks
Dans allocator.build_tasks :

calculer n_train/n_val/n_test

pour chaque split :

n0 = round(n_split * label0_ratio)

n1 = n_split - n0

répartir n0 sur scénarios label0 via weights

répartir n1 sur scénarios label1 via weights

pour label1, forcer “groupes” (bins distance, moving/static, doppler on/off) :

soit via scénarios distincts (“drone_far_static”, “drone_far_moving”, …)

soit via un champ scenario.tags + quotas par tags

Important : source picking doit être per-sample, pas dépendant d’un RNG global du split
=> seed = stable_int_seed(f"{master}:{split}:{sample_id}:sources")
Cela rend la génération stable même si on change les weights.

2.3 Maximiser la variété des sources DADS
Au lieu de rng.sample(pool) qui peut réutiliser trop souvent :

construire une permutation déterministe du pool par split/role

consommer en round-robin (wrap-around)
=> usage plus uniforme, plus de diversité, toujours déterministe.

3) Mesure SNR et métadonnées “utiles terrain”
3.1 Ajouter un SNR estimé dans la metadata (worker)
Après application des chains composantes et AVANT mix :

calculer rms_drone = RMS de la somme des composantes role=drone (après env_orch/drone_orch)

calculer rms_ambient = RMS de la somme des composantes role=ambient

snr_est_db = 20*log10((rms_drone+eps)/(rms_ambient+eps))
Stocker :

au top-level metadata: snr_est_db, rms_drone, rms_ambient

garder aussi dans components[].meta si déjà présent

But : pouvoir forcer des distributions SNR (drones quasi inaudibles).

3.2 Validation distribution SNR
Dans validate_distribution :

vérifier présence snr_est_db sur label1

vérifier couverture des bins SNR selon targets.snr_bin_weights_label1

alerter/raise si la fraction “très basse SNR” est trop faible

4) “Complex non-drone” : enrichir label0 avec un orchestrateur ambient
Créer un orchestrateur ambient_orch (ou étendre hardware/env) pour rendre les négatifs difficiles :

filtres (HP/LP/Notch aléatoires)

dropouts (micro-coupures)

softclip/saturation (boîtier saturé)

transient clicks (manipulation / choc)

hum 50/100 Hz léger (option)

Implémenter 2 ops simples (déterministes via ctx):

dropout: zéro/atténue quelques fenêtres temporelles

softclip_tanh: saturation douce

Puis ambient_orch = [filters_orch?, dropout?, softclip?] (ordre fixé)

But : négatifs “complexes” = moins de faux positifs en conditions réelles.

5) Durcir validate_distribution (et le rendre “strict”)
Actuellement ça print des WARNING. Pour du “parfait” :

ajouter targets.strict=true (ou validation.strict=true)

si strict : lever ValueError si contraintes non satisfaites

Tolérances N-aware :

tol_ratio_effective = max(config_tol, 2/total_samples_split)

Et corriger l’usage de strong_noise_dbfs partout (pas de -55 codé en dur).

6) Nouveau fichier config: build_config_field_v2.json
Créer une config plus riche que v1 :
Label 0 (30%) :

noise_only_1amb

noise_only_2amb

noise_only_3amb

noise_only_harsh_hw

noise_only_complex (ambient_orch + hardware_orch)

Label 1 (70%) :

near_static (0–60)

mid_static (60–150)

far_static (150–300)

near_moving, mid_moving, far_moving (trajectory + doppler)

multi_drone (2 drones) + ambient

Global chain TOUJOURS :

hardware_orch (avec noise_rms_dbfs “mixture”, adc_bits variés, hp varié)

normalize_peak

stats

7) Tests à ajouter (essentiels)
tests SNR compute: connu (drone=sinus, ambient=sinus plus faible) => snr_est_db proche attendu

dropout deterministic: même seed => même sortie

softclip: amplitude bornée et monotonicité

validate_distribution strict: échoue si label0_ratio ou bins non respectés

8) Commandes attendues
bash
Copier le code
pytest -q
python build_dataset_v5.py --config build_config_field_v2.json --mode full --jobs 8 --clean
python build_dataset_v5.py --config build_config_field_v2.json --mode validate