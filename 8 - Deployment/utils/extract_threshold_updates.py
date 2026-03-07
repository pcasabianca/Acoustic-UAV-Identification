#!/usr/bin/env python3
"""
Extract calibrated threshold data and update thresholds.json.

This utility reads the trainer calibration output (e.g. models/calibrated_thresholds.json)
and updates deployment thresholds in thresholds.json using the model mapping from
deployment_config.json.
"""

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, Tuple, Any


DEFAULT_CALIBRATED_FILE = "models/calibrated_thresholds.json"
DEFAULT_THRESHOLDS_FILE = "thresholds.json"
DEFAULT_CONFIG_FILE = "deployment_config.json"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize_name(value: str) -> str:
    return value.replace("-", "_").replace(" ", "_").upper()


def build_model_key_mapping(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Build mapping from calibration model key -> thresholds.json key.

    Example:
        ATTENTION_CRNN -> attention_crnn_model
        CNN -> cnn_model
    """
    mapping: Dict[str, str] = {}
    available_models = config.get("models", {}).get("available_models", {})

    for model_name, model_info in available_models.items():
        calibrated_key = model_info.get("calibrated_key", model_name)
        mapping[normalize_name(model_name)] = calibrated_key

        architecture = model_info.get("architecture")
        if architecture:
            mapping[normalize_name(architecture)] = calibrated_key

        filename = model_info.get("filename")
        if filename:
            stem = Path(filename).stem
            mapping[normalize_name(stem)] = calibrated_key

    return mapping


def extract_updates(
    calibrated_data: Dict[str, Any],
    model_mapping: Dict[str, str],
    decimals: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """
    Extract updates from calibrated file.

    Returns:
        updates_by_threshold_key, raw_thresholds_by_calibration_key
    """
    updates: Dict[str, Dict[str, Any]] = {}
    raw_thresholds: Dict[str, float] = {}

    models = calibrated_data.get("models", {})
    for calibration_model_key, payload in models.items():
        threshold = payload.get("threshold")
        if threshold is None:
            continue

        normalized = normalize_name(calibration_model_key)
        thresholds_key = model_mapping.get(normalized)

        if not thresholds_key:
            # fallback: if already in deployment format
            if calibration_model_key.endswith("_model"):
                thresholds_key = calibration_model_key.lower()
            else:
                continue

        metrics = payload.get("metrics_at_threshold", {})
        update_entry = {
            "threshold": round(float(threshold), decimals),
            "f1_score": round(float(metrics.get("f1_score", 0.0)), 3),
            "accuracy": round(float(metrics.get("accuracy", 0.0)), 3),
            "balanced_accuracy": round(float(metrics.get("balanced_accuracy", 0.0)), 3),
        }
        updates[thresholds_key] = update_entry
        raw_thresholds[calibration_model_key] = float(threshold)

    return updates, raw_thresholds


def apply_updates(
    thresholds_data: Dict[str, Any],
    updates: Dict[str, Dict[str, Any]],
    source_file: Path,
    calibration_mode: str,
) -> Dict[str, Any]:
    models = thresholds_data.setdefault("models", {})

    for model_key, update in updates.items():
        existing = models.get(model_key, {})
        description = existing.get("description", model_key.replace("_", " ").title())
        merged = {"description": description}
        merged.update(existing)
        merged.update(update)
        models[model_key] = merged

    thresholds_data["last_updated"] = str(date.today())
    thresholds_data["description"] = (
        f"Deployment thresholds configuration - CALIBRATED from {source_file.as_posix()}"
    )
    thresholds_data["calibration_mode"] = calibration_mode or thresholds_data.get("calibration_mode", "unknown")

    return thresholds_data


def print_preview(
    updates: Dict[str, Dict[str, Any]],
    raw_thresholds: Dict[str, float],
) -> None:
    print("\nProposed updates for thresholds.json")
    print("-" * 60)
    for model_key in sorted(updates.keys()):
        data = updates[model_key]
        print(
            f"{model_key:24s} threshold={data['threshold']:.3f} "
            f"f1={data['f1_score']:.3f} acc={data['accuracy']:.3f} "
            f"bal_acc={data['balanced_accuracy']:.3f}"
        )

    if raw_thresholds:
        print("\nRaw calibrated keys found:")
        for key, value in sorted(raw_thresholds.items()):
            print(f"  {key:20s} -> {value:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract calibrated threshold data and update thresholds.json"
    )
    parser.add_argument(
        "--calibrated-file",
        type=Path,
        default=Path(DEFAULT_CALIBRATED_FILE),
        help="Path to calibrated_thresholds.json",
    )
    parser.add_argument(
        "--thresholds-file",
        type=Path,
        default=Path(DEFAULT_THRESHOLDS_FILE),
        help="Path to thresholds.json",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path(DEFAULT_CONFIG_FILE),
        help="Path to deployment_config.json for model mapping",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimals for threshold values in thresholds.json",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write updates to thresholds.json (default is preview only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.calibrated_file.exists():
        print(f"[ERROR] Calibrated file not found: {args.calibrated_file}")
        return 1

    if not args.config_file.exists():
        print(f"[ERROR] Config file not found: {args.config_file}")
        return 1

    if not args.thresholds_file.exists():
        print(f"[ERROR] Thresholds file not found: {args.thresholds_file}")
        return 1

    calibrated_data = load_json(args.calibrated_file)
    config_data = load_json(args.config_file)
    thresholds_data = load_json(args.thresholds_file)

    mapping = build_model_key_mapping(config_data)
    updates, raw_thresholds = extract_updates(
        calibrated_data=calibrated_data,
        model_mapping=mapping,
        decimals=args.decimals,
    )

    if not updates:
        print("[WARN] No model updates extracted. Check mapping or input file.")
        return 2

    print_preview(updates, raw_thresholds)

    if not args.write:
        print("\nDry-run only. Use --write to update thresholds.json")
        return 0

    updated = apply_updates(
        thresholds_data=thresholds_data,
        updates=updates,
        source_file=args.calibrated_file,
        calibration_mode=calibrated_data.get("calibration_mode", ""),
    )
    save_json(args.thresholds_file, updated)
    print(f"\n[OK] Updated: {args.thresholds_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
