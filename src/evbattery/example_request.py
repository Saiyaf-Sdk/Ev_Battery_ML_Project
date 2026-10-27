
"""Create a valid example API request from the first dataset row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import IDENTIFIERS, TARGET, SUSPECT_SCORES


def generate_request(data_path: Path, metadata_path: Path) -> dict:
    """Generate an API request using only model-approved raw features."""

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    allowed = set(metadata["raw_feature_columns"])

    excluded = {TARGET, *IDENTIFIERS, *SUSPECT_SCORES}
    allowed -= excluded

    dataframe = pd.read_csv(data_path, nrows=1)

    if dataframe.empty:
        raise ValueError("Dataset is empty")

    row = dataframe.iloc[0]

    features = {
        column: (
            None
            if pd.isna(value)
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for column, value in row.items()
        if column in allowed
    }

    if sum(value is not None for value in features.values()) < 10:
        raise ValueError("At least 10 nonmissing features are required")

    return {"features": features}


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data", required=True, type=Path)

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/metadata.json"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/example_request.json"),
    )

    args = parser.parse_args()

    request = generate_request(args.data, args.metadata)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(request, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {args.out}")
    print(f"Accepted features: {len(request['features'])}")


if __name__ == "__main__":
    main()
