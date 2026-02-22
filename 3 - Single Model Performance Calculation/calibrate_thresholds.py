#!/usr/bin/env python3
"""
Threshold Calibration - Balanced per-class accuracy (with drone-first bias)

Problem:
- The previous calibrator primarily optimized "balanced_precision = min(PPV, NPV)".
- You want a calibration that is *balanced* in terms of per-class accuracies:
    accuracy_drone ≈ accuracy_ambient
  while still *privileging drone detection* (prefer accuracy_drone slightly higher).

Definitions:
- accuracy_drone   := recall / sensitivity (TP / (TP+FN))
- accuracy_ambient := specificity          (TN / (TN+FP))
- delta            := accuracy_drone - accuracy_ambient

Primary constraint/objective:
- Keep delta close to a target (balance_target_delta, default +0.03)
  with a maximum deviation (max_balance_gap, default 0.10).
  This means: abs(delta - target_delta) <= max_balance_gap

Hierarchical selection (lexicographic):
Tier 1 (hard constraints, in this priority order):
  1) balance constraint (gap-to-target)
  2) min_recall (drone detection floor)
  3) min_precision_drone (PPV)
  4) min_precision_ambient (NPV)  [kept, but relaxed before balance if needed]

Tier 2 (optimization among candidates):
  - minimize gap_to_target_delta = abs(delta - target_delta)
  - prefer delta >= 0 (drone not worse than ambient) as a tie-breaker
  - maximize recall (drone detection)
  - maximize balanced_accuracy = (recall + specificity)/2
  - maximize F1
  - prefer lower threshold (more permissive) last

Progressive relaxation:
- If no candidate satisfies Tier 1 constraints, relax *non-balance* constraints first,
  and relax balance constraint only as a last resort.

Outputs:
- Same JSON schema as before:
  {"version": "...", "models": {"CNN": {"threshold": ..., "metrics_at_threshold": ..., ...}}}

Usage:
  python calibrate_thresholds_balanced.py --model CNN
  python calibrate_thresholds_balanced.py --all-models
  python calibrate_thresholds_balanced.py --model CNN --max-balance-gap 0.08 --balance-target-delta 0.05
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Keep config import behavior consistent with your repo layout
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float) -> dict:
    """Compute confusion-based metrics for a given threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = int(len(y_true))
    accuracy = float((tp + tn) / n) if n else 0.0

    precision_drone = float(tp / (tp + fp)) if (tp + fp) else 0.0  # PPV
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0           # TPR / sensitivity
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0      # TNR

    precision_ambient = float(tn / (tn + fn)) if (tn + fn) else 0.0  # NPV

    f1 = float(2 * (precision_drone * recall) / (precision_drone + recall)) if (precision_drone + recall) else 0.0
    balanced_accuracy = float(0.5 * (recall + specificity))

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,

        "precision_drone": precision_drone,        # PPV
        "precision_ambient": precision_ambient,    # NPV

        "accuracy_drone": recall,                  # per-class accuracy proxy for drone
        "accuracy_ambient": specificity,           # per-class accuracy proxy for ambient

        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "balanced_accuracy": balanced_accuracy,

        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def _format_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def optimize_threshold_balanced(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    # Drone-first balance
    balance_target_delta: float = 0.01,
    max_balance_gap: float = 0.03,

    # Floors (relaxable before balance)
    min_recall: float = 0.90,
    min_precision_drone: float = 0.70,
    min_precision_ambient: float = 0.85,

    threshold_range=(0.05, 0.95),
    threshold_step: float = 0.01,
    verbose: bool = True,
):
    """
    Choose an optimal threshold with *balance gap-to-target* as the top priority.
    Returns (threshold, optimal_metrics, all_results, relaxation_info).
    """

    thresholds = np.arange(threshold_range[0], threshold_range[1] + 1e-9, threshold_step)
    all_results = []
    for t in thresholds:
        m = calculate_metrics(y_true, y_pred_proba, float(t))
        delta = m["accuracy_drone"] - m["accuracy_ambient"]
        m["delta_drone_minus_ambient"] = float(delta)
        m["gap_to_target_delta"] = float(abs(delta - balance_target_delta))
        m["abs_accuracy_gap"] = float(abs(delta))
        all_results.append(m)

    if verbose:
        print("\n" + "=" * 80)
        print("  THRESHOLD OPTIMIZATION — Balanced per-class accuracy (drone-first bias)")
        print("=" * 80)
        print("Tier 1 (hard, highest priority):")
        print(f"  • Balance target delta (drone-ambient): {balance_target_delta:+.2f}")
        print(f"  • Max balance gap-to-target:           {_format_pct(max_balance_gap)}")
        print(f"  • Min recall (drone accuracy):         {_format_pct(min_recall)}")
        print(f"  • Min precision drone (PPV):           {_format_pct(min_precision_drone)}")
        print(f"  • Min precision ambient (NPV):         {_format_pct(min_precision_ambient)}")
        print("Tier 2 (optimize among candidates):")
        print("  • Min gap_to_target → prefer delta>=0 → max recall → max balanced_accuracy → max F1 → lower threshold")
        print("=" * 80)
        print(f"Testing {len(thresholds)} thresholds from {threshold_range[0]:.2f} to {threshold_range[1]:.2f} step {threshold_step:.2f}")

    def satisfies(m, *, gap, rec, ppv, npv):
        if m["gap_to_target_delta"] > gap:
            return False
        if m["recall"] < rec:
            return False
        if m["precision_drone"] < ppv:
            return False
        if m["precision_ambient"] < npv:
            return False
        return True

    # Progressive relaxation: relax PPV/NPV/recall first, keep balance fixed as long as possible.
    relaxation_plan = [
        {
            "name": "Base constraints",
            "gap": max_balance_gap,
            "rec": min_recall,
            "ppv": min_precision_drone,
            "npv": min_precision_ambient,
        },
        {
            "name": "Relax NPV slightly (-5%)",
            "gap": max_balance_gap,
            "rec": min_recall,
            "ppv": min_precision_drone,
            "npv": min_precision_ambient * 0.95,
        },
        {
            "name": "Relax NPV (-10%) and PPV (-5%)",
            "gap": max_balance_gap,
            "rec": min_recall,
            "ppv": min_precision_drone * 0.95,
            "npv": min_precision_ambient * 0.90,
        },
        {
            "name": "Relax recall (-3%), keep balance",
            "gap": max_balance_gap,
            "rec": min_recall * 0.97,
            "ppv": min_precision_drone * 0.95,
            "npv": min_precision_ambient * 0.90,
        },
        {
            "name": "Last resort: relax balance gap (+5pp)",
            "gap": max_balance_gap + 0.05,
            "rec": min_recall * 0.97,
            "ppv": min_precision_drone * 0.95,
            "npv": min_precision_ambient * 0.90,
        },
        {
            "name": "Fallback: minimize gap_to_target, ignore floors",
            "gap": 1.0,   # effectively no gap constraint (but we still minimize it in ranking)
            "rec": -1.0,
            "ppv": -1.0,
            "npv": -1.0,
            "fallback_no_floors": True,
        },
    ]

    candidates = []
    relaxation_applied = None

    for level, plan in enumerate(relaxation_plan, start=1):
        candidates = [m for m in all_results if satisfies(m, gap=plan["gap"], rec=plan["rec"], ppv=plan["ppv"], npv=plan["npv"])]
        if candidates:
            relaxation_applied = {
                "level": level,
                "name": plan["name"],
                "effective_constraints": {
                    "balance_target_delta": balance_target_delta,
                    "max_balance_gap": plan["gap"],
                    "min_recall": plan["rec"] if plan["rec"] >= 0 else None,
                    "min_precision_drone": plan["ppv"] if plan["ppv"] >= 0 else None,
                    "min_precision_ambient": plan["npv"] if plan["npv"] >= 0 else None,
                }
            }
            break

    # Ranking (lexicographic)
    def rank(m):
        gap = m["gap_to_target_delta"]               # minimize
        drone_worse = 1 if m["delta_drone_minus_ambient"] < 0 else 0  # prefer not worse
        recall = -m["recall"]                        # maximize
        bal_acc = -m["balanced_accuracy"]            # maximize
        f1 = -m["f1_score"]                          # maximize
        thr = m["threshold"]                         # prefer lower last
        return (gap, drone_worse, recall, bal_acc, f1, thr)

    candidates.sort(key=rank)
    optimal = candidates[0]

    if verbose:
        print("\n" + "=" * 80)
        print("  OPTIMAL THRESHOLD FOUND")
        print("=" * 80)
        print(f"Chosen threshold: {optimal['threshold']:.4f}")
        print(f"delta (drone-ambient): {optimal['delta_drone_minus_ambient']:+.4f}  | gap_to_target: {optimal['gap_to_target_delta']:.4f}")
        print("\nMetrics @ threshold:")
        print(f"  Recall (drone acc):         {optimal['recall']:.4f}")
        print(f"  Specificity (ambient acc):  {optimal['specificity']:.4f}")
        print(f"  Balanced accuracy:          {optimal['balanced_accuracy']:.4f}")
        print(f"  Precision drone (PPV):      {optimal['precision_drone']:.4f}")
        print(f"  Precision ambient (NPV):    {optimal['precision_ambient']:.4f}")
        print(f"  F1-score:                   {optimal['f1_score']:.4f}")
        print("\nConfusion:")
        print(f"  TP={optimal['tp']} TN={optimal['tn']} FP={optimal['fp']} FN={optimal['fn']}")
        if relaxation_applied:
            print("\nConstraints / relaxation used:")
            print(f"  Level {relaxation_applied['level']}: {relaxation_applied['name']}")
            for k, v in relaxation_applied["effective_constraints"].items():
                if v is None:
                    continue
                print(f"  - {k}: {v}")
        print("=" * 80)

    return optimal["threshold"], optimal, all_results, relaxation_applied


def _refine_thresholds_adaptive(y_true: np.ndarray, y_pred_proba: np.ndarray, seed_thresholds: list,
                                threshold_min: float, threshold_max: float,
                                iterations: int = 30, points: int = 200, window: float = 0.05,
                                balance_target_delta: float = 0.03,
                                max_balance_gap: float = 0.10,
                                min_recall: float = 0.90,
                                min_precision_drone: float = 0.70,
                                min_precision_ambient: float = 0.85,
                                stop_tol: float = 1e-7,
                                verbose: bool = False):
    """Refine thresholds by local adaptive grid search around seed_thresholds.

    This method evaluates dense grids around each seed, selects the best candidate
    using the same ranking used in `optimize_threshold_balanced`, and repeats
    shrinking the window for a few iterations.
    """
    def rank_key(m):
        gap = m["gap_to_target_delta"]
        drone_worse = 1 if m["delta_drone_minus_ambient"] < 0 else 0
        recall = -m["recall"]
        bal_acc = -m["balanced_accuracy"]
        f1 = -m["f1_score"]
        thr = m["threshold"]
        return (gap, drone_worse, recall, bal_acc, f1, thr)

    best_overall = None
    for seed in seed_thresholds:
        lo = max(threshold_min, seed - window)
        hi = min(threshold_max, seed + window)
        best_local = None
        prev_candidate_thr = None
        for it in range(iterations):
            grid = np.linspace(lo, hi, points)
            results = []
            for t in grid:
                m = calculate_metrics(y_true, y_pred_proba, float(t))
                delta = m["accuracy_drone"] - m["accuracy_ambient"]
                m["delta_drone_minus_ambient"] = float(delta)
                m["gap_to_target_delta"] = float(abs(delta - balance_target_delta))
                results.append(m)
            # filter by base floors/balance (kept simple: accept all, ranking will prefer good ones)
            results.sort(key=rank_key)
            candidate = results[0]
            best_local = candidate if (best_local is None or rank_key(candidate) < rank_key(best_local)) else best_local
            # shrink around candidate for next iter
            cthr = float(candidate["threshold"])
            span = max(1e-4, (hi - lo) / 4.0)
            lo = max(threshold_min, cthr - span)
            hi = min(threshold_max, cthr + span)
            # (silent) early stopping: if candidate threshold hasn't moved significantly, stop
            if prev_candidate_thr is not None:
                if abs(prev_candidate_thr - cthr) < stop_tol:
                    break
            prev_candidate_thr = cthr

        if best_overall is None or rank_key(best_local) < rank_key(best_overall):
            best_overall = best_local

    return best_overall


def load_validation_predictions(model_name: str):
    """
    Load model and validation features, compute probabilities.
    Uses MEL_VAL_NPZ_PATH like the original script for compatibility.
    """
    import tensorflow as tf

    # Allow unsafe deserialization for local/trusted environment (Lambda layers)
    try:
        import keras
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    model_paths = {
        "CNN": config.CNN_MODEL_PATH,
        "RNN": config.RNN_MODEL_PATH,
        "CRNN": config.CRNN_MODEL_PATH,
        "ATTENTION_CRNN": config.ATTENTION_CRNN_MODEL_PATH,
        "EFFICIENTNET": config.EFFICIENTNET_MODEL_PATH,
        "MOBILENET": config.MOBILENET_MODEL_PATH,
        "CONFORMER": config.CONFORMER_MODEL_PATH,
        "TCN": config.TCN_MODEL_PATH,
    }

    model_path = model_paths.get(model_name.upper())
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    val_features_path = config.MEL_VAL_NPZ_PATH
    if not val_features_path.exists():
        raise FileNotFoundError(f"Validation features not found: {val_features_path}")

    print(f"Loading validation features: {val_features_path}")
    data = np.load(val_features_path, allow_pickle=True)

    if "features" in data:
        X_val = data["features"]
        y_val = data["labels"]
    elif "mels" in data:
        X_val = data["mels"]
        y_val = data["labels"]
    else:
        raise KeyError(f"Could not find features/mels in NPZ. Available keys: {list(data.keys())}")

    print(f"Validation samples: {len(y_val)}")
    print(f"Feature shape: {X_val.shape}")

    models_needing_3d = ["RNN", "TCN", "CONFORMER"]
    if X_val.ndim == 3 and model_name.upper() not in models_needing_3d:
        X_val = X_val[..., np.newaxis]
        print(f"Added channel dimension: {X_val.shape}")

    print("Generating predictions...")
    preds = model.predict(X_val, batch_size=32, verbose=0)

    if preds.ndim > 1 and preds.shape[1] > 1:
        y_pred_proba = preds[:, 1]
    else:
        y_pred_proba = preds.reshape(-1)

    return np.asarray(y_val).astype(int), np.asarray(y_pred_proba).astype(float)


def save_calibration_results(calibration_data: dict, output_path: Path):
    """Atomic save, merge with existing calibration file if present."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_data = None
    if output_path.exists():
        try:
            existing_data = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️  Could not read existing calibration file ({e}). Creating new.")

    if existing_data is not None and isinstance(existing_data, dict) and "models" in existing_data:
        existing_data["models"].update(calibration_data["models"])
        existing_data["calibration_date"] = calibration_data["calibration_date"]
        existing_data["calibration_mode"] = calibration_data.get("calibration_mode", existing_data.get("calibration_mode"))
        existing_data["constraints"] = calibration_data.get("constraints", existing_data.get("constraints", {}))
        final_data = existing_data
    else:
        final_data = calibration_data

    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(final_data, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)

    print(f"\n✓ Calibration saved: {output_path}")
    print(f"   Models in file: {', '.join(final_data.get('models', {}).keys())}")


def main():
    p = argparse.ArgumentParser(description="Calibrate classification thresholds (balanced per-class accuracy, drone-first)")

    p.add_argument("--model", type=str,
                   choices=["CNN", "RNN", "CRNN", "ATTENTION_CRNN", "EFFICIENTNET", "MOBILENET", "CONFORMER", "TCN"],
                   help="Model to calibrate")
    p.add_argument("--all-models", action="store_true", help="Calibrate all models")

    # Floors (relaxable)
    p.add_argument("--min-recall", type=float, default=0.90, help="Minimum recall floor (default: 0.90)")
    p.add_argument("--min-precision-drone", type=float, default=0.70, help="Minimum drone precision (PPV) floor (default: 0.70)")
    p.add_argument("--min-precision-ambient", type=float, default=0.85, help="Minimum ambient precision (NPV) floor (default: 0.85)")

    # Balance is now central
    p.add_argument("--balance-target-delta", type=float, default=0.03,
                   help="Target delta = accuracy_drone - accuracy_ambient. Positive favors drone. (default: +0.03)")
    p.add_argument("--max-balance-gap", type=float, default=0.10,
                   help="Max allowed abs(delta - target_delta). (default: 0.10)")

    p.add_argument("--threshold-min", type=float, default=0.05, help="Minimum threshold to test (default: 0.05)")
    p.add_argument("--threshold-max", type=float, default=0.95, help="Maximum threshold to test (default: 0.95)")
    p.add_argument("--threshold-step", type=float, default=0.01, help="Threshold step (default: 0.01)")
    # Refinement: enabled by default; provide --no-refine to disable
    p.add_argument("--no-refine", dest="refine", action="store_false",
                   help="Disable local adaptive refinement (enabled by default)")
    p.set_defaults(refine=True)
    # Increase sensible defaults for high-precision refinement
    p.add_argument("--refine-iterations", type=int, default=30,
                   help="Refinement iterations (default: 30)")
    p.add_argument("--refine-points", type=int, default=200,
                   help="Points per refinement grid (default: 200)")
    p.add_argument("--refine-window", type=float, default=0.05,
                   help="Initial refinement half-window (default: 0.05)")
    p.add_argument("--refine-stop-tol", type=float, default=1e-8,
                   help="Stop tolerance for refinement when threshold change is below this (default: 1e-8)")

    p.add_argument("--output", type=str, default=None,
                   help="Output JSON file (default: results/calibrated_thresholds.json)")

    args = p.parse_args()

    if not args.model and not args.all_models:
        p.error("Either --model or --all-models must be specified")

    if args.all_models:
        models = ["CNN", "RNN", "CRNN", "ATTENTION_CRNN", "EFFICIENTNET", "MOBILENET", "CONFORMER", "TCN"]
    else:
        models = [args.model]

    output_path = Path(args.output) if args.output else config.CALIBRATION_FILE_PATH

    calibration_data = {
        "version": "2.0",
        "calibration_date": datetime.now().isoformat(),
        "calibration_mode": "balanced_per_class_accuracy_gap_to_target",
        "constraints": {
            "min_recall": args.min_recall,
            "min_precision_drone": args.min_precision_drone,
            "min_precision_ambient": args.min_precision_ambient,
            "balance_target_delta": args.balance_target_delta,
            "max_balance_gap": args.max_balance_gap,
            "threshold_range": [args.threshold_min, args.threshold_max],
            "threshold_step": args.threshold_step,
        },
        "models": {},
    }

    for model_name in models:
        print("\n" + "#" * 80)
        print(f"  CALIBRATING: {model_name}")
        print("#" * 80)

        try:
            y_val, y_pred_proba = load_validation_predictions(model_name)

            thr, optimal, all_results, relax_info = optimize_threshold_balanced(
                y_val,
                y_pred_proba,
                balance_target_delta=args.balance_target_delta,
                max_balance_gap=args.max_balance_gap,
                min_recall=args.min_recall,
                min_precision_drone=args.min_precision_drone,
                min_precision_ambient=args.min_precision_ambient,
                threshold_range=(args.threshold_min, args.threshold_max),
                threshold_step=args.threshold_step,
                verbose=True,
            )

            # Optional refinement stage: run local adaptive search around top candidates
            if args.refine:
                # choose top seeds from the discrete grid (few best by original ranking)
                def rank_key_for_sort(m):
                    gap = m["gap_to_target_delta"]
                    drone_worse = 1 if m["delta_drone_minus_ambient"] < 0 else 0
                    recall = -m["recall"]
                    bal_acc = -m["balanced_accuracy"]
                    f1 = -m["f1_score"]
                    thrv = m["threshold"]
                    return (gap, drone_worse, recall, bal_acc, f1, thrv)

                all_results_sorted = sorted(all_results, key=rank_key_for_sort)
                top_seeds = [r["threshold"] for r in all_results_sorted[:3]]  # refine around top-3
                if len(top_seeds) > 0:
                    refined = _refine_thresholds_adaptive(
                        y_val, y_pred_proba, top_seeds,
                        threshold_min=args.threshold_min, threshold_max=args.threshold_max,
                        iterations=args.refine_iterations, points=args.refine_points, window=args.refine_window,
                        balance_target_delta=args.balance_target_delta, max_balance_gap=args.max_balance_gap,
                        min_recall=args.min_recall, min_precision_drone=args.min_precision_drone,
                        min_precision_ambient=args.min_precision_ambient,
                        stop_tol=args.refine_stop_tol,
                        verbose=True,
                    )
                    if refined is not None:
                        print(f"\nRefined threshold candidate: {refined['threshold']:.6f} (gap {refined['gap_to_target_delta']:.4f})")
                        # replace optimal with refined if better by rank
                        if rank_key_for_sort(refined) < rank_key_for_sort(optimal):
                            thr = float(refined["threshold"]) 
                            optimal = refined
                            print("Refinement improved selection — using refined threshold.")
                        else:
                            print("Refinement did not improve selection; keeping discrete optimum.")

            calibration_data["models"][model_name] = {
                "threshold": float(thr),
                "validation_samples": int(len(y_val)),
                "metrics_at_threshold": optimal,
                "relaxation_applied": relax_info,
                "optimization_summary": {
                    "num_thresholds_tested": int(len(all_results)),
                    "best_f1_threshold": float(max(all_results, key=lambda x: x["f1_score"])["threshold"]),
                    "best_balanced_accuracy_threshold": float(max(all_results, key=lambda x: x["balanced_accuracy"])["threshold"]),
                    "chosen_threshold": float(thr),
                },
                "all_tested_thresholds": all_results,  # keep for visualization/debug
            }

            print(f"\n✓ {model_name} calibration complete")

        except Exception as e:
            import traceback
            print(f"\n✗ {model_name} calibration failed: {e}")
            traceback.print_exc()
            calibration_data["models"][model_name] = {"error": str(e), "threshold": 0.5}

    save_calibration_results(calibration_data, output_path)

    print("\n" + "=" * 80)
    print("  CALIBRATION SUMMARY")
    print("=" * 80)
    for model_name, data in calibration_data["models"].items():
        if "error" in data:
            print(f"{model_name}: ✗ FAILED - {data['error']}")
            continue
        m = data["metrics_at_threshold"]
        print(f"\n{model_name}: threshold={data['threshold']:.4f}  "
              f"delta={m.get('delta_drone_minus_ambient', 0.0):+.3f}  "
              f"gap={m.get('gap_to_target_delta', 0.0):.3f}  "
              f"recall={m['recall']:.3f}  spec={m['specificity']:.3f}  "
              f"f1={m['f1_score']:.3f}")

    print("\nCalibration file:", output_path)
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
