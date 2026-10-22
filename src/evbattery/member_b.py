"""Member B: Supervised Model Selection, Preprocessing Experiments, and Tuning.

This module executes Member B's responsibilities:
- Strictly operates on the 60% training partition (no test peeking).
- Evaluates candidate classifiers using 3-fold Stratified CV scored by Average Precision.
- Runs preprocessing comparisons (median, KNN, correlation pruning, L1 selection).
- Tunes the winning classifier using RandomizedSearchCV on training data only.
- Fits the chosen UNCALIBRATED pipeline on the full training set.
- Exports handoff model artifacts and selection metadata for Member C.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import IDENTIFIERS, RANDOM_STATE, REDUNDANT_EXACT, SUSPECT_SCORES, TARGET
from .data import load_dataset, make_features, split_data
from .experiments import compare as compare_preprocessing
from .modeling import candidate_models, cv_leaderboard, pipe_for, MODEL_PROPERTIES
from .tune import tune_model, PARAM_DISTRIBUTIONS


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", encoding="utf-8", delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def run_member_b(
    data_path: Path,
    outdir: Path,
    fast: bool = False,
    tune: bool = True,
    tune_trials: int = 10,
    include_knn: bool = False,
    run_prep_experiments: bool = True,
    prep_max_rows: int = 2500,
    seed: int = RANDOM_STATE,
) -> dict[str, object]:
    """Execute Member B supervised model selection and tuning workflow.
    
    CRITICAL LEAKAGE INVARIANT:
    This function accesses ONLY splits.train. It NEVER reads, transforms,
    evaluates, or predicts upon splits.calibration, splits.threshold, or splits.test.
    """
    data_path = Path(data_path).expanduser()
    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    splits = split_data(df, seed=seed)

    # Strict isolation: Member B only uses the training split
    train_df = splits.train
    x_train = make_features(train_df, include_suspect_scores=False, drop_redundant=True)
    y_train = train_df[TARGET].astype(int)

    if TARGET in x_train.columns:
        raise RuntimeError("Target leakage detected: target column present in feature matrix")
    if set(IDENTIFIERS).intersection(x_train.columns):
        raise RuntimeError("Identifier leakage detected: ID columns present in feature matrix")
    if set(SUSPECT_SCORES).intersection(x_train.columns):
        raise RuntimeError("Suspect score leakage: composite risk scores present in feature matrix")
    if set(REDUNDANT_EXACT).intersection(x_train.columns):
        raise RuntimeError("Redundant exact duplicate columns present in feature matrix")

    # 1. Candidate Supervised Model Comparison via 3-Fold Stratified CV
    leaderboard = cv_leaderboard(
        x_train, y_train,
        fast=fast,
        include_knn=include_knn,
        include_properties=True,
        seed=seed,
    )
    leaderboard_csv_path = outdir / "model_cv_leaderboard.csv"
    leaderboard.to_csv(leaderboard_csv_path, index=False)

    # 2. Select Winning Candidate by Mean Average Precision (excluding dummy_prior)
    eligible = leaderboard[leaderboard["model"] != "dummy_prior"]
    if eligible.empty:
        raise RuntimeError("No eligible supervised models found in CV leaderboard")
    winner_row = eligible.iloc[0]
    winner_name = str(winner_row["model"])
    winner_untuned_ap_mean = float(winner_row["cv_average_precision_mean"])
    winner_untuned_ap_std = float(winner_row["cv_average_precision_std"])
    winner_untuned_roc_auc_mean = float(winner_row["cv_roc_auc_mean"])
    winner_untuned_roc_auc_std = float(winner_row["cv_roc_auc_std"])

    # 3. Hyperparameter Tuning (RandomizedSearchCV on Training Data Only)
    tuning_info = None
    tuned_ap_mean = winner_untuned_ap_mean
    tuned_ap_std = winner_untuned_ap_std
    tuned_roc_auc_mean = winner_untuned_roc_auc_mean
    tuned_roc_auc_std = winner_untuned_roc_auc_std
    best_params = {}

    if tune and winner_name in PARAM_DISTRIBUTIONS:
        tune_res = tune_model(
            x_train, y_train,
            model_name=winner_name,
            n_iter=tune_trials,
            seed=seed,
        )
        best_params = tune_res["best_params"]
        best_estimator = tune_res["best_estimator"]

        # Cross-validate the tuned pipeline to get full std dev and ROC AUC metrics
        cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        tuned_scores = cross_validate(
            best_estimator, x_train, y_train,
            cv=cv_folds,
            scoring={"ap": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1, error_score="raise", return_train_score=False,
        )
        tuned_ap_mean = float(np.mean(tuned_scores["test_ap"]))
        tuned_ap_std = float(np.std(tuned_scores["test_ap"]))
        tuned_roc_auc_mean = float(np.mean(tuned_scores["test_roc_auc"]))
        tuned_roc_auc_std = float(np.std(tuned_scores["test_roc_auc"]))

        tuning_info = {
            "model": winner_name,
            "best_params": best_params,
            "best_cv_average_precision": float(tune_res["best_cv_average_precision"]),
            "n_iter_requested": tune_trials,
            "n_iter_executed": tune_res["n_iter_executed"],
            "param_distributions": tune_res["param_distributions"],
            "cv_results_summary": tune_res["cv_results_summary"],
            "delta_ap_vs_untuned": tuned_ap_mean - winner_untuned_ap_mean,
            "note": "Tuning performed strictly on 60% training partition via StratifiedKFold(3).",
        }
        _atomic_json(outdir / "tuning_results.json", tuning_info)
    else:
        # Untuned pipeline fitted on full training data
        candidates = candidate_models(fast=fast, include_knn=include_knn)
        best_estimator = pipe_for(x_train, candidates[winner_name])
        best_estimator.fit(x_train, y_train)

    # 4. Preprocessing Comparison Experiments
    prep_csv_path = None
    if run_prep_experiments:
        prep_sample_x = x_train
        prep_sample_y = y_train
        if len(x_train) > prep_max_rows:
            sample_train, _ = train_test_split(
                train_df, train_size=prep_max_rows,
                stratify=train_df[TARGET], random_state=seed,
            )
            prep_sample_x = make_features(sample_train, include_suspect_scores=False, drop_redundant=True)
            prep_sample_y = sample_train[TARGET].astype(int)
        prep_scores = compare_preprocessing(prep_sample_x, prep_sample_y, cv_splits=3, seed=seed)
        prep_csv_path = outdir / "preprocessing_comparison.csv"
        prep_scores.to_csv(prep_csv_path, index=False)

    # 5. Metadata and Trusted Local Model Artifact (UNCALIBRATED)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Member_B_Supervised_Model_Selection_And_Tuning",
        "description": "Uncalibrated base preprocessing and classifier pipeline trained on 60% train split",
        "training_data_sha256": _sha256(data_path),
        "random_seed": seed,
        "selected_model": winner_name,
        "is_tuned": bool(best_params),
        "best_hyperparameters": best_params,
        "train_records": int(len(train_df)),
        "cv_folds": 3,
        "feature_count": int(x_train.shape[1]),
        "feature_names": x_train.columns.tolist(),
        "cv_performance": {
            "untuned": {
                "average_precision_mean": winner_untuned_ap_mean,
                "average_precision_std": winner_untuned_ap_std,
                "roc_auc_mean": winner_untuned_roc_auc_mean,
                "roc_auc_std": winner_untuned_roc_auc_std,
            },
            "final_selected": {
                "average_precision_mean": tuned_ap_mean,
                "average_precision_std": tuned_ap_std,
                "roc_auc_mean": tuned_roc_auc_mean,
                "roc_auc_std": tuned_roc_auc_std,
            },
        },
        "model_properties": MODEL_PROPERTIES.get(winner_name, {}),
        "environment": {
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "leakage_audit_guarantee": {
            "training_split_only": True,
            "calibration_split_touched": False,
            "threshold_split_touched": False,
            "test_split_touched": False,
            "test_metrics_computed": False,
            "alarm_threshold_computed": False,
        },
        "member_c_handoff_instructions": (
            "Load base_pipeline from best_base_model.joblib. "
            "Wrap inside CalibratedClassifierCV(FrozenEstimator(base_pipeline), method='sigmoid') "
            "and fit on splits.calibration. Next, use splits.threshold to select alarm cutoff (optimizing F2). "
            "Finally, evaluate once on untouched splits.test."
        ),
        "limitations": [
            "Dataset is synthetic with no observation timestamps or verified failure horizon.",
            "This model classifies current operational telemetry patterns, NOT validated time-ahead failure lead time.",
            "Predictions from this base model are uncalibrated raw scores; Member C must apply Platt scaling before deployment.",
        ],
    }

    # Save model artifact atomically
    bundle = {
        "base_pipeline": best_estimator,
        "model_name": winner_name,
        "is_tuned": bool(best_params),
        "best_params": best_params,
        "metadata": metadata,
    }
    model_artifact_path = outdir / "best_base_model.joblib"
    with tempfile.NamedTemporaryFile(dir=outdir, suffix=".joblib.tmp", delete=False) as f:
        temp_model = Path(f.name)
    try:
        joblib.dump(bundle, temp_model)
        os.replace(temp_model, model_artifact_path)
    finally:
        temp_model.unlink(missing_ok=True)

    _atomic_json(outdir / "selected_model.json", metadata)

    return {
        "selected_model": winner_name,
        "is_tuned": bool(best_params),
        "best_params": best_params,
        "cv_average_precision_mean": tuned_ap_mean,
        "cv_average_precision_std": tuned_ap_std,
        "cv_roc_auc_mean": tuned_roc_auc_mean,
        "cv_roc_auc_std": tuned_roc_auc_std,
        "leaderboard": leaderboard.to_dict("records"),
        "artifacts_saved": [
            str(model_artifact_path),
            str(outdir / "selected_model.json"),
            str(leaderboard_csv_path),
            str(outdir / "tuning_results.json") if tuning_info else None,
            str(prep_csv_path) if prep_csv_path else None,
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to fleet telemetry CSV")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/member_b"),
                        help="Output directory for Member B artifacts")
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke-test run with reduced iterations")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip RandomizedSearchCV hyperparameter tuning")
    parser.add_argument("--tune-trials", type=int, default=10,
                        help="Number of tuning parameter combinations to sample (default: 10)")
    parser.add_argument("--include-knn", action="store_true",
                        help="Include computationally intensive KNeighborsClassifier in leaderboard")
    parser.add_argument("--no-prep-experiments", action="store_true",
                        help="Skip preprocessing comparison experiments")
    parser.add_argument("--prep-max-rows", type=int, default=2500,
                        help="Maximum training rows used for KNN imputation preprocessing experiment")
    args = parser.parse_args()

    result = run_member_b(
        data_path=args.data,
        outdir=args.outdir,
        fast=args.fast,
        tune=not args.no_tune,
        tune_trials=args.tune_trials,
        include_knn=args.include_knn,
        run_prep_experiments=not args.no_prep_experiments,
        prep_max_rows=args.prep_max_rows,
    )

    print("\n" + "=" * 60)
    print("MEMBER B: MODEL SELECTION AND CV LEADERBOARD")
    print("=" * 60)
    leaderboard_df = pd.DataFrame(result["leaderboard"])
    cols_to_print = [c for c in ["model", "cv_average_precision_mean", "cv_average_precision_std",
                                 "cv_roc_auc_mean", "fit_time_mean"] if c in leaderboard_df.columns]
    print(leaderboard_df[cols_to_print].to_string(index=False))
    print("\n" + "-" * 60)
    print(f"SELECTED BASE MODEL: {result['selected_model']}")
    print(f"TUNED: {result['is_tuned']}")
    if result["is_tuned"]:
        print(f"HYPERPARAMETERS: {json.dumps(result['best_params'], indent=2)}")
    print(f"CV AVERAGE PRECISION: {result['cv_average_precision_mean']:.4f} ± {result['cv_average_precision_std']:.4f}")
    print(f"CV ROC AUC:           {result['cv_roc_auc_mean']:.4f} ± {result['cv_roc_auc_std']:.4f}")
    print("-" * 60)
    print("ARTIFACTS SAVED:")
    for a in result["artifacts_saved"]:
        if a:
            print(f"  - {a}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
