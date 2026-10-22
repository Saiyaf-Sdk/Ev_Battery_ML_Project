"""Generate machine-readable and readable exploratory data quality evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from .config import TARGET
from .data import load_dataset


def profile(frame: pd.DataFrame) -> dict:
    missing = frame.isna().sum().sort_values(ascending=False)
    correlated = frame.select_dtypes(include=np.number).drop(columns=TARGET).corr().abs()
    pairs = []
    for i, a in enumerate(correlated.columns):
        for b in correlated.columns[i+1:]:
            v = correlated.loc[a, b]
            if pd.notna(v) and v >= 0.97:
                pairs.append({"a": a, "b": b, "abs_pearson_r": round(float(v), 4)})
    return {
        "n_rows": len(frame), "n_columns": frame.shape[1],
        "label_counts": {str(k): int(v) for k, v in frame[TARGET].value_counts().items()},
        "failure_prevalence": float(frame[TARGET].mean()),
        "duplicates": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_percent": round(float(100 * frame.isna().sum().sum() / frame.size), 2),
        "top_missing_percent": {
            k: round(float(100 * v / len(frame)), 2) for k, v in missing.head(15).items()
        },
        "strongly_correlated_numeric_pairs": pairs,
        "invalid_ambient_max_below_min": int((frame['maximum_temperature'] < frame['minimum_temperature']).sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("artifacts/data_profile.json"))
    args = p.parse_args()
    result = profile(load_dataset(args.data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
