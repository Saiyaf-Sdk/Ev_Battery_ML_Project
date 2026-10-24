"""Member C final-stage safeguards with synthetic records only."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from evbattery.config import TARGET, RANDOM_STATE
from evbattery.data import make_features, split_data
from evbattery.member_c import checksum, run
from evbattery.member_c_finalize import finalize
from evbattery.modeling import candidate_models, pipe_for


def staged_artifact(tmp_path):
    rng = np.random.default_rng(123)
    n = 360
    frame = pd.DataFrame({
        "vehicle_id": [f"vehicle-{i}" for i in range(n)],
        "battery_serial": [f"battery-{i}" for i in range(n)],
        "battery_health_percent": rng.uniform(60, 100, n),
        "cell_temperature_max": rng.uniform(20, 65, n),
        "vehicle_brand": rng.choice(["Brand A", "Brand B"], n),
        TARGET: (rng.random(n) < 0.2).astype(int),
    })
    data = tmp_path / "data.csv"
    frame.to_csv(data, index=False)
    train = split_data(frame).train
    x_train = make_features(train)
    base = pipe_for(x_train, candidate_models(fast=True)["logistic_regression"])
    base.fit(x_train, train[TARGET])
    base_path = tmp_path / "base.joblib"
    joblib.dump({
        "base_pipeline": base,
        "model_name": "logistic_regression",
        "metadata": {
            "feature_names": list(x_train.columns),
            "training_data_sha256": checksum(data),
            "random_seed": RANDOM_STATE,
            "train_records": len(train),
        },
    }, base_path)
    staged_dir = tmp_path / "stage"
    run(data, base_path, staged_dir)
    return data, staged_dir / "staged_model.joblib"


def test_frozen_model_finalizes_and_exports_handoff(tmp_path):
    data, staged = staged_artifact(tmp_path)
    result = finalize(data, staged, tmp_path / "final")
    final_bundle = joblib.load(tmp_path / "final/model.joblib")
    assert result["n"] == 72
    assert sum(result[k] for k in ("tp", "fp", "tn", "fn")) == 72
    assert result["threshold"] == final_bundle["threshold"]
    assert final_bundle["metadata"]["final_test_evaluated"] is True
    assert (tmp_path / "final/test_metrics.json").exists()
    assert (tmp_path / "final/metadata.json").exists()


def test_finalization_refuses_second_test_evaluation(tmp_path):
    data, staged = staged_artifact(tmp_path)
    outdir = tmp_path / "final"
    finalize(data, staged, outdir)
    with pytest.raises(FileExistsError, match="do not rerun"):
        finalize(data, staged, outdir)


def test_finalization_rejects_changed_dataset(tmp_path):
    data, staged = staged_artifact(tmp_path)
    frame = pd.read_csv(data)
    frame.loc[0, "battery_health_percent"] = 0.0
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError, match="CSV checksum"):
        finalize(data, staged, tmp_path / "final")
