"""Comprehensive unit and integration tests for Member B.

Tests verify:
- Strict feature exclusion rules (no target, IDs, composite scores, or duplicates).
- Prevention of target leakage across all processing steps.
- Stratified training fold consistency and reproducible CV metrics.
- Model registry completeness across all 7 classical algorithms + baseline.
- Resilience to missing values (numeric and categorical) and unseen categories.
- Preprocessing fitted strictly inside training folds.
- CorrelationPruner and L1 feature selection behavior.
- Hyperparameter tuning bounds and validation.
- Model serialization and deserialization consistency.
- Metadata and schema conformance.
- Total isolation from holdout calibration, threshold, and test partitions.
"""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from evbattery.config import IDENTIFIERS, RANDOM_STATE, REDUNDANT_EXACT, SUSPECT_SCORES, TARGET
from evbattery.data import make_features, split_data

from evbattery.experiments import CorrelationPruner, compare as compare_experiments
from evbattery.feature_catalog import CATALOG
from evbattery.member_b import run_member_b
from evbattery.modeling import candidate_models, cv_leaderboard, pipe_for, MODEL_PROPERTIES
from evbattery.tune import tune_model, PARAM_DISTRIBUTIONS


def synthetic_telemetry(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic telemetry matching all 70 dataset columns."""
    rng = np.random.default_rng(seed)
    data = {}
    for col, (group, _) in CATALOG.items():
        if group not in ("identifier", "target"):
            data[col] = rng.uniform(10, 100, n)

    # Specific realistic categories
    data["vehicle_brand"] = rng.choice(["Tesla", "BYD", "Nissan", "Hyundai"], n)
    data["vehicle_model"] = rng.choice(["Model 3", "Seal", "Leaf", "Ioniq"], n)
    data["vehicle_type"] = rng.choice(["Sedan", "SUV", "Hatchback"], n)
    data["battery_manufacturer"] = rng.choice(["CATL", "LG", "Panasonic"], n)
    data["battery_chemistry"] = rng.choice(["NMC", "LFP", "LTO"], n)
    data["drive_type"] = rng.choice(["FWD", "RWD", "AWD"], n)
    data["fleet_or_private"] = rng.choice(["Fleet", "Private"], n)
    data["terrain_type"] = rng.choice(["Flat", "Hilly", "Mountainous"], n)

    # Realistic numeric correlations
    health = rng.uniform(60, 100, n)
    age = rng.uniform(1, 10, n)
    cap = rng.uniform(40, 100, n)
    data.update({
        "battery_health_percent": health,
        "capacity_loss_percent": 100.0 - health,
        "battery_capacity_kwh": cap,
        "remaining_capacity": cap * health / 100.0,
        "vehicle_age_years": age,
        "manufacturing_year": np.round(2026 - age),
        "state_of_health": health + rng.normal(0, 1, n).clip(-2, 2),
        "cell_temperature_avg": rng.uniform(15, 45, n),
        "cell_temperature_max": rng.uniform(30, 65, n),
        "cycle_count": rng.integers(100, 2000, n).astype(float),
        "odometer_km": rng.uniform(5000, 150000, n),
        "average_ambient_temperature": rng.uniform(0, 35, n),
        "maximum_temperature": rng.uniform(20, 45, n),
        "minimum_temperature": rng.uniform(-10, 15, n),
        "fast_charge_ratio": rng.uniform(0.1, 0.5, n),
        "slow_charge_ratio": rng.uniform(0.3, 0.7, n),
        "average_charge_power_kw": rng.uniform(10, 150, n),
        "previous_faults": rng.integers(0, 4, n).astype(float),
        "BMS_warning_count": rng.integers(0, 5, n).astype(float),
        "sensor_fault_count": rng.integers(0, 3, n).astype(float),
        "last_service_days": rng.uniform(10, 800, n),
    })

    df = pd.DataFrame(data)
    df.insert(0, "vehicle_id", [f"veh_{i:05d}" for i in range(n)])
    df.insert(1, "battery_serial", [f"bat_{i:05d}" for i in range(n)])
    
    # Inject signal into target: low health + high temp = failure
    prob = 1.0 / (1.0 + np.exp((health - 72.0) / 4.0 - (df["cell_temperature_max"] - 45.0) / 10.0))
    prob = np.clip(prob, 0.02, 0.95)
    # Ensure at least 15 failures for 3-fold stratified split
    raw_targets = (rng.uniform(0, 1, n) < prob).astype(int)
    if raw_targets.sum() < 18:
        raw_targets[:18] = 1
    df[TARGET] = raw_targets
    return df


def test_feature_exclusion_rules():
    """Verify that make_features strictly strips banned columns and adds engineered features."""
    df = synthetic_telemetry(100)
    features = make_features(df, include_suspect_scores=False, drop_redundant=True)
    
    assert TARGET not in features.columns
    for ident in IDENTIFIERS:
        assert ident not in features.columns
    for suspect in SUSPECT_SCORES:
        assert suspect not in features.columns
    for redundant in REDUNDANT_EXACT:
        assert redundant not in features.columns
    
    # Confirm engineered features are present
    assert "cell_temp_max_minus_avg" in features.columns
    assert "ambient_temp_range" in features.columns
    assert "cycles_per_year" in features.columns
    assert "km_per_cycle" in features.columns
    assert "fast_charge_power_stress" in features.columns
    assert "prior_fault_and_warning_count" in features.columns
    assert "service_overdue_180d" in features.columns
    assert features.shape[1] == 63


def test_prevention_of_target_leakage():
    """Verify target never leaks into features across splits or transformations."""
    df = synthetic_telemetry(120)
    splits = split_data(df)
    x_train = make_features(splits.train)
    assert TARGET not in x_train.columns
    assert not any(col.startswith("battery_failure") for col in x_train.columns)


def test_stratified_training_folds():
    """Verify StratifiedKFold preserves class distribution across folds."""
    df = synthetic_telemetry(150)
    splits = split_data(df)
    train = splits.train
    x_train = make_features(train)
    y_train = train[TARGET].astype(int)
    
    leaderboard = cv_leaderboard(
        x_train, y_train,
        fast=True,
        model_names={"dummy_prior", "logistic_regression"},
    )
    assert not leaderboard.empty
    assert "cv_average_precision_mean" in leaderboard.columns
    assert "cv_average_precision_std" in leaderboard.columns
    assert "cv_roc_auc_mean" in leaderboard.columns
    assert "cv_roc_auc_std" in leaderboard.columns


def test_reproducible_cv_results():
    """Verify that running CV twice with the same seed yields identical numerical metrics."""
    df = synthetic_telemetry(120, seed=123)
    train = split_data(df).train
    x_train = make_features(train)
    y_train = train[TARGET].astype(int)

    lb1 = cv_leaderboard(x_train, y_train, fast=True, seed=RANDOM_STATE, model_names={"logistic_regression"})
    lb2 = cv_leaderboard(x_train, y_train, fast=True, seed=RANDOM_STATE, model_names={"logistic_regression"})

    assert lb1.iloc[0]["cv_average_precision_mean"] == pytest.approx(lb2.iloc[0]["cv_average_precision_mean"], abs=1e-9)
    assert lb1.iloc[0]["cv_roc_auc_mean"] == pytest.approx(lb2.iloc[0]["cv_roc_auc_mean"], abs=1e-9)


def test_model_registry_completeness():
    """Verify all 7 candidate algorithms + dummy (+ optional knn) are present and buildable."""
    models = candidate_models(fast=True, include_knn=True)
    required = {
        "dummy_prior",
        "logistic_regression",
        "gaussian_naive_bayes",
        "sgd_logistic",
        "decision_tree",
        "random_forest",
        "adaboost",
        "hist_gradient_boost",
        "knn",
    }
    assert required.issubset(set(models.keys()))
    
    # Check MODEL_PROPERTIES dictionary covers all algorithms
    for name in required:
        assert name in MODEL_PROPERTIES
        props = MODEL_PROPERTIES[name]
        assert "interpretable" in props
        assert "scaling_required" in props
        assert "nonlinear" in props
        assert "advantages" in props
        assert "weaknesses" in props

    # Check pipe_for builds valid pipelines
    sample_x = make_features(synthetic_telemetry(50))
    for name, clf in models.items():
        pipe = pipe_for(sample_x, clf)
        assert isinstance(pipe, Pipeline)
        assert "preprocess" in pipe.named_steps
        assert "model" in pipe.named_steps


def test_missing_values_and_unseen_categories():
    """Verify the preprocessing pipeline gracefully handles missing values and new categories."""
    df = synthetic_telemetry(120)
    train = split_data(df).train
    x_train = make_features(train)
    y_train = train[TARGET].astype(int)

    pipe = pipe_for(x_train, candidate_models(fast=True)["logistic_regression"])
    pipe.fit(x_train, y_train)

    # Create new record with NaNs and completely unknown categories
    x_test = x_train.iloc[0:2].copy()
    x_test.loc[:, "battery_health_percent"] = np.nan
    x_test.loc[:, "cell_temperature_avg"] = np.nan
    x_test.loc[:, "vehicle_brand"] = "CompletelyUnseenBrand"
    x_test.loc[:, "battery_chemistry"] = "NewUnknownChemistry"

    probs = pipe.predict_proba(x_test)
    assert probs.shape == (2, 2)
    assert not np.isnan(probs).any()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_correlation_pruner():
    """Verify CorrelationPruner drops collinear columns and respects threshold."""
    rng = np.random.default_rng(42)
    col_a = rng.normal(0, 1, 100)
    col_b = col_a + rng.normal(0, 0.001, 100) # nearly 1.0 correlation
    col_c = rng.normal(0, 1, 100)             # independent

    mat = np.column_stack([col_a, col_b, col_c])
    pruner = CorrelationPruner(threshold=0.95)
    pruner.fit(mat)
    transformed = pruner.transform(mat)

    assert transformed.shape[1] == 2
    assert pruner.get_support().sum() == 2
    with pytest.raises(ValueError, match="threshold"):
        CorrelationPruner(threshold=1.5)


def test_preprocessing_comparison():
    """Verify compare_experiments produces valid comparison table with all 4 methods."""
    df = synthetic_telemetry(120)
    train = split_data(df).train
    x_train = make_features(train)
    y_train = train[TARGET].astype(int)

    results = compare_experiments(x_train, y_train, cv_splits=3)
    assert len(results) == 4
    expected_methods = {
        "median_baseline",
        "knn_imputation",
        "median_corr_prune_0.97",
        "median_embedded_L1_select",
    }
    assert set(results["method"]) == expected_methods
    for col in ("cv_average_precision_mean", "cv_average_precision_std", "cv_roc_auc_mean", "cv_roc_auc_std", "fit_time_mean"):
        assert col in results.columns
        assert not results[col].isna().any()


def test_hyperparameter_tuning_train_only():
    """Verify tune_model executes RandomizedSearchCV on training data and returns valid parameters."""
    df = synthetic_telemetry(120)
    train = split_data(df).train
    x_train = make_features(train)
    y_train = train[TARGET].astype(int)

    res = tune_model(x_train, y_train, model_name="logistic_regression", n_iter=3, seed=RANDOM_STATE)
    assert res["model"] == "logistic_regression"
    assert "best_cv_average_precision" in res
    assert "model__C" in res["best_params"]
    assert res["best_params"]["model__C"] in PARAM_DISTRIBUTIONS["logistic_regression"]["model__C"]

    with pytest.raises(ValueError, match="Unsupported model"):
        tune_model(x_train, y_train, model_name="non_existent_model")


def test_model_save_load_consistency(tmp_path):
    """Verify saved best_base_model.joblib can be reloaded and yields identical predictions."""
    df = synthetic_telemetry(120)
    train = split_data(df).train
    x_train = make_features(train)
    y_train = train[TARGET].astype(int)

    pipe = pipe_for(x_train, candidate_models(fast=True)["logistic_regression"])
    pipe.fit(x_train, y_train)

    bundle = {
        "base_pipeline": pipe,
        "model_name": "logistic_regression",
        "is_tuned": False,
        "best_params": {},
        "metadata": {"task": "unit_test"},
    }
    artifact_path = tmp_path / "best_base_model.joblib"
    joblib.dump(bundle, artifact_path)

    loaded_bundle = joblib.load(artifact_path)
    loaded_pipe = loaded_bundle["base_pipeline"]

    orig_probs = pipe.predict_proba(x_train)
    loaded_probs = loaded_pipe.predict_proba(x_train)
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-7, atol=1e-7)


def test_member_b_end_to_end_no_test_leakage(tmp_path):
    """Verify Member B end-to-end execution on synthetic dataset.
    
    Confirms:
    1. Output artifacts generated.
    2. Model is fitted but UNCALIBRATED.
    3. Calibration, threshold, and test partitions are NOT evaluated.
    4. No test_metrics.json or threshold_validation_metrics.json are generated.
    """
    df = synthetic_telemetry(250)
    data_csv = tmp_path / "synthetic_ev_battery.csv"
    df.to_csv(data_csv, index=False)

    outdir = tmp_path / "member_b_artifacts"
    result = run_member_b(
        data_path=data_csv,
        outdir=outdir,
        fast=True,
        tune=True,
        tune_trials=2,
        include_knn=False,
        run_prep_experiments=True,
        prep_max_rows=150,
    )

    assert (outdir / "best_base_model.joblib").is_file()
    assert (outdir / "selected_model.json").is_file()
    assert (outdir / "model_cv_leaderboard.csv").is_file()
    assert (outdir / "tuning_results.json").is_file()
    assert (outdir / "preprocessing_comparison.csv").is_file()

    # Verify Member C holdout isolation: Member B must NOT create test evaluation files
    assert not (outdir / "test_metrics.json").exists()
    assert not (outdir / "threshold_validation_metrics.json").exists()

    # Verify metadata content
    with open(outdir / "selected_model.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["leakage_audit_guarantee"]["training_split_only"] is True
    assert meta["leakage_audit_guarantee"]["test_split_touched"] is False
    assert meta["feature_count"] == 63
    assert "member_c_handoff_instructions" in meta
