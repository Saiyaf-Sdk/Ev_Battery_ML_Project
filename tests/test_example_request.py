
import json

import pandas as pd

from evbattery.example_request import generate_request


def test_generator_excludes_unknown_features(tmp_path):
    data_path = tmp_path / "sample.csv"
    metadata_path = tmp_path / "metadata.json"

    row = {f"feature_{i}": i for i in range(12)}
    row["manufacturing_year"] = 2022

    pd.DataFrame([row]).to_csv(data_path, index=False)

    metadata = {
        "raw_feature_columns": [f"feature_{i}" for i in range(12)]
    }

    metadata_path.write_text(json.dumps(metadata))

    request = generate_request(data_path, metadata_path)

    assert len(request["features"]) == 12
    assert "manufacturing_year" not in request["features"]
    assert request["features"]["feature_0"] == 0


def test_generator_rejects_insufficient_features(tmp_path):
    import pytest

    data_path = tmp_path / "small.csv"
    metadata_path = tmp_path / "metadata.json"

    pd.DataFrame([{"temperature": 25}]).to_csv(
        data_path, index=False
    )

    metadata_path.write_text(
        json.dumps({"raw_feature_columns": ["temperature"]})
    )

    with pytest.raises(ValueError, match="At least 10"):
        generate_request(data_path, metadata_path)
