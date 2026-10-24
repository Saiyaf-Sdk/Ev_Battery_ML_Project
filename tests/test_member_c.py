"""Unit checks for Member C's held-out calibration and threshold workflow."""
import joblib
import numpy as np
import pandas as pd
import pytest

from evbattery.config import RANDOM_STATE, TARGET
from evbattery.data import make_features, split_data
from evbattery.member_c import checksum, run
from evbattery.modeling import candidate_models, pipe_for


def base_artifacts(tmp_path):
    rng = np.random.default_rng(123)
    n = 360
    df = pd.DataFrame({
        "vehicle_id": [f"vehicle-{i}" for i in range(n)],
        "battery_serial": [f"battery-{i}" for i in range(n)],
        "battery_health_percent": rng.uniform(60, 100, n),
        "cell_temperature_max": rng.uniform(20, 65, n),
        "vehicle_brand": rng.choice(["Brand A", "Brand B"], n),
        TARGET: (rng.random(n) < 0.2).astype(int),
    })
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    training = split_data(df).train
    x = make_features(training)
    base = pipe_for(x, candidate_models(fast=True)["logistic_regression"])
    base.fit(x, training[TARGET])
    bundle = {
        "base_pipeline": base,
        "model_name": "logistic_regression",
        "metadata": {
            "feature_names": list(x.columns),
            "training_data_sha256": checksum(csv),
            "random_seed": RANDOM_STATE,
            "train_records": len(training),
        },
    }
    model_path = tmp_path / "member_b.joblib"
    joblib.dump(bundle, model_path)
    return csv, model_path


def test_calibration_and_threshold_do_not_evaluate_test(tmp_path):
    csv, base_path = base_artifacts(tmp_path)
    outputs = tmp_path / "member_c"
    result = run(csv, base_path, outputs)
    assert (outputs / "staged_model.joblib").exists()
    assert (outputs / "validation_metrics.json").exists()
    assert not (outputs / "test_metrics.json").exists()
    assert result["final_test_evaluated"] is False
    assert 0.02 <= result["selected_threshold"] <= 0.98
    assert 0 <= result["selected_validation_f2"] <= 1
    bundle = joblib.load(outputs / "staged_model.joblib")
    assert bundle["metadata"]["final_test_evaluated"] is False
    assert np.all(np.isfinite(bundle["model"].predict_proba(
        make_features(split_data(pd.read_csv(csv)).calibration))))


def test_reject_modified_dataset(tmp_path):
    csv, base_path = base_artifacts(tmp_path)
    df = pd.read_csv(csv)
    df.loc[0, "battery_health_percent"] = 0
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Dataset checksum"):
        run(csv, base_path, tmp_path / "member_c")
