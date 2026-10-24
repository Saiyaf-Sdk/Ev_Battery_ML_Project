"""Member C: calibrate Member B's model and choose a validation-only alarm threshold.

This stage deliberately leaves the final test split untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .config import IDENTIFIERS, RANDOM_STATE, REDUNDANT_EXACT, SUSPECT_SCORES, TARGET
from .data import load_dataset, make_features, split_data
from .evaluation import choose_threshold, evaluate


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(data_path: Path, base_path: Path, outdir: Path) -> dict:
    data_path, base_path, outdir = map(Path, (data_path, base_path, outdir))
    df = load_dataset(data_path)
    splits = split_data(df, seed=RANDOM_STATE)

    # Loading joblib uses pickle; only open artifacts created by your team.
    bundle = joblib.load(base_path)
    base = bundle["base_pipeline"]
    previous = bundle["metadata"]
    expected_features = list(previous["feature_names"])

    if previous["training_data_sha256"] != checksum(data_path):
        raise ValueError("Dataset checksum differs from Member B's training dataset")
    if previous["random_seed"] != RANDOM_STATE:
        raise ValueError("The split seed differs from Member B's training seed")
    if previous["train_records"] != len(splits.train):
        raise ValueError("Training partition size differs from Member B's run")

    def get_xy(frame):
        x = make_features(frame, include_suspect_scores=False, drop_redundant=True)
        if list(x.columns) != expected_features:
            raise ValueError("Feature columns/order differ from Member B's training")
        return x, frame[TARGET].astype(int)

    x_cal, y_cal = get_xy(splits.calibration)
    x_val, y_val = get_xy(splits.threshold)

    # FrozenEstimator ensures Member B's fitted model is not retrained.
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(x_cal, y_cal)
    probability = calibrated.predict_proba(x_val)[:, 1]
    baseline_probability = base.predict_proba(x_val)[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("Non-finite calibrated probabilities")

    threshold, validation_f2 = choose_threshold(y_val, probability, beta=2.0)
    results = {
        "base_model": bundle["model_name"],
        "calibration_method": "sigmoid",
        "calibration_records": len(y_cal),
        "threshold_validation_records": len(y_val),
        "uncalibrated_at_0_5": evaluate(y_val, baseline_probability, 0.5),
        "calibrated_at_0_5": evaluate(y_val, probability, 0.5),
        "calibrated_at_selected_threshold": evaluate(y_val, probability, threshold),
        "selected_threshold": float(threshold),
        "selected_validation_f2": float(validation_f2),
        "final_test_evaluated": False,
    }
    banned = {TARGET, *IDENTIFIERS, *SUSPECT_SCORES, *REDUNDANT_EXACT}
    metadata = {
        "winner": bundle["model_name"],
        "calibration_method": "sigmoid",
        "threshold": float(threshold),
        "training_data_sha256": checksum(data_path),
        "raw_feature_columns": [c for c in df.columns if c not in banned],
        "engineered_feature_columns": expected_features,
        "include_suspect_scores": False,
        "fast_smoke_test_only": False,
        "final_test_evaluated": False,
        "member_b_metadata": previous,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": calibrated, "threshold": float(threshold), "metadata": metadata},
        outdir / "staged_model.joblib",
    )
    (outdir / "validation_metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Member B model: {bundle['model_name']}")
    print(f"Validation Brier: {results['uncalibrated_at_0_5']['brier_score']:.4f} -> "
          f"{results['calibrated_at_0_5']['brier_score']:.4f}")
    print(f"Selected threshold: {threshold:.2f}; validation F2: {validation_f2:.4f}")
    print("Untouched test partition: NOT evaluated")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ev_battery.csv"))
    parser.add_argument("--base-model", type=Path,
                        default=Path("artifacts/member_b/best_base_model.joblib"))
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/member_c"))
    args = parser.parse_args()
    run(args.data, args.base_model, args.outdir)


if __name__ == "__main__":
    main()
