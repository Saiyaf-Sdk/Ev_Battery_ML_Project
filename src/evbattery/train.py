"""Train/evaluate once with four separate partitions and reproducible artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from .config import IDENTIFIERS, RANDOM_STATE, SUSPECT_SCORES, TARGET
from .data import load_dataset, make_features, split_data
from .evaluation import choose_threshold, evaluate
from .modeling import candidate_models, cv_leaderboard, pipe_for


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", encoding="utf-8", delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_training(data_path: Path, outdir: Path, fast: bool = False,
                 include_suspect_scores: bool = False, tuning_file: Path | None = None,
                 baseline_only: bool = False, cv_max_rows: int | None = None) -> dict:
    # The full raw data are needed for schema metadata; the target never enters features.
    df = load_dataset(data_path)
    splits = split_data(df, seed=RANDOM_STATE)
    x_train = make_features(splits.train, include_suspect_scores)
    y_train = splits.train[TARGET].astype(int)
    if len(y_train) != len(x_train):
        raise RuntimeError("Misaligned training target")
    # Optional resource budget: stratified TRAIN-only subset for CV ranking.
    # The final winning estimator is still fitted using ALL training rows.
    if cv_max_rows is not None and cv_max_rows < len(splits.train):
        if cv_max_rows < 1000:
            raise ValueError('--cv-max-rows must be >= 1000')
        cv_frame, _ = train_test_split(splits.train, train_size=cv_max_rows,
                                       stratify=splits.train[TARGET], random_state=RANDOM_STATE)
        x_cv = make_features(cv_frame, include_suspect_scores)
        y_cv = cv_frame[TARGET].astype(int)
    else:
        x_cv, y_cv = x_train, y_train
    leaderboard = cv_leaderboard(x_cv, y_cv, fast=fast,
                                 model_names={"dummy_prior", "logistic_regression"} if baseline_only else None)
    winner = next(name for name in leaderboard["model"] if name != "dummy_prior")
    base = pipe_for(x_train, candidate_models(fast=fast)[winner])
    tune_info = None
    if tuning_file is not None:
        if fast or include_suspect_scores:
            raise ValueError('Tune file not supported with --fast or --include-suspect-scores')
        tune_info = json.loads(tuning_file.read_text(encoding='utf-8'))
        if tune_info.get('model') != winner:
            raise ValueError('Tuning file model differs from the CV-ranked winner')
        if not isinstance(tune_info.get('best_params'), dict):
            raise ValueError('Invalid tuning file')
        # Reject unexpected pipeline keys; scikit-learn also validates each value.
        valid = set(base.get_params())
        if any(k not in valid or not k.startswith('model__') for k in tune_info['best_params']):
            raise ValueError('Tuning file has unsupported hyperparameter names')
        base.set_params(**tune_info['best_params'])
    base.fit(x_train, y_train)

    x_cal = make_features(splits.calibration, include_suspect_scores).reindex(columns=x_train.columns)
    x_thr = make_features(splits.threshold, include_suspect_scores).reindex(columns=x_train.columns)
    x_test = make_features(splits.test, include_suspect_scores).reindex(columns=x_train.columns)
    # FrozenEstimator ensures fitting calibration NEVER retrains the base model.
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(x_cal, splits.calibration[TARGET])
    p_thr = calibrated.predict_proba(x_thr)[:, 1]
    threshold, threshold_f2 = choose_threshold(splits.threshold[TARGET], p_thr)
    thr_metrics = evaluate(splits.threshold[TARGET], p_thr, threshold)
    # ONLY NOW is the holdout test used. Never choose models based on test metrics.
    p_test = calibrated.predict_proba(x_test)[:, 1]
    test_metrics = evaluate(splits.test[TARGET], p_test, threshold)

    outdir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "synthetic_EV_battery_failure_classification_NOT_verified_forecasting",
        "training_data_sha256": _sha256(data_path),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "winner": winner, "seed": RANDOM_STATE,
        "tuned_params": None if tune_info is None else tune_info["best_params"],
        "fast_smoke_test_only": fast,
        "baseline_only": baseline_only,
        "include_suspect_scores": include_suspect_scores,
        "excluded_identifiers": list(IDENTIFIERS),
        "excluded_suspect_scores": [] if include_suspect_scores else list(SUSPECT_SCORES),
        "raw_feature_columns": [c for c in df.columns if c not in (TARGET, *IDENTIFIERS)],
        "engineered_feature_columns": x_train.columns.tolist(),
        "train_n": len(splits.train), "cv_ranking_train_n": len(x_cv),
        "calibration_n": len(splits.calibration),
        "threshold_n": len(splits.threshold), "test_n": len(splits.test),
        "failure_prevalence": float(df[TARGET].mean()),
        "threshold": threshold,
        "threshold_validation_f2": threshold_f2,
        "evaluation_warning": "Synthetic, non-temporal labels. Do not claim real early warning lead time.",
    }
    bundle = {"model": calibrated, "threshold": threshold, "metadata": metadata}
    # Joblib uses pickle internally. Only load trusted, locally generated artifacts.
    with tempfile.NamedTemporaryFile(dir=outdir, suffix=".joblib.tmp", delete=False) as f:
        temp_model = Path(f.name)
    try:
        joblib.dump(bundle, temp_model)
        os.replace(temp_model, outdir / "model.joblib")
    finally:
        temp_model.unlink(missing_ok=True)
    _atomic_json(outdir / "metadata.json", metadata)
    _atomic_json(outdir / "test_metrics.json", test_metrics)
    _atomic_json(outdir / "threshold_validation_metrics.json", thr_metrics)
    leaderboard.to_csv(outdir / "model_cv_leaderboard.csv", index=False)
    pd.DataFrame({
        "split": ["train", "calibration", "threshold", "test"],
        "records": [len(splits.train), len(splits.calibration), len(splits.threshold), len(splits.test)],
        "failures": [int(z[TARGET].sum()) for z in
                     (splits.train, splits.calibration, splits.threshold, splits.test)],
    }).to_csv(outdir / "split_counts.csv", index=False)
    return {"winner": winner, "threshold": threshold, "threshold_metrics": thr_metrics,
            "test_metrics": test_metrics, "leaderboard": leaderboard.to_dict("records")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"))
    parser.add_argument("--fast", action="store_true", help="Quick SMOKE TEST; not reportable results")
    parser.add_argument("--baseline-only", action="store_true", help="Full-data logistic baseline only; not a multi-model comparison")
    parser.add_argument("--cv-max-rows", type=int, help="Limit CV ranking to a stratified training-only subset; fit winner on all train")
    parser.add_argument("--tuning-file", type=Path, help="JSON from evbattery.tune; winner must match")
    parser.add_argument("--include-suspect-scores", action="store_true",
                        help="Only for leakage sensitivity experiments; do NOT deploy")
    args = parser.parse_args()
    if args.include_suspect_scores:
        warnings.warn("Composite risk scores may leak target-like information; research only!", stacklevel=1)
    result = run_training(args.data, args.outdir, args.fast, args.include_suspect_scores, args.tuning_file, args.baseline_only, args.cv_max_rows)
    print(json.dumps({k: v for k, v in result.items() if k != "leaderboard"}, indent=2))
    print("\nCV LEADERBOARD\n", pd.DataFrame(result["leaderboard"]).to_string(index=False))
    print(f"\nSaved results to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
