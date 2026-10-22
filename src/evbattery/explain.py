"""Permutation importances on threshold-validation data, not final test data."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from .data import load_dataset, make_features, split_data
from .config import TARGET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/permutation_importance.csv"))
    args = parser.parse_args()
    # IMPORTANT: Only load a model artifact you personally created and trust.
    bundle = joblib.load(args.model)
    metadata = bundle["metadata"]
    if metadata["fast_smoke_test_only"]:
        print("Warning: examining a fast smoke-test model, not reportable model")
    s = split_data(load_dataset(args.data))
    x = make_features(s.threshold, metadata["include_suspect_scores"])
    x = x.reindex(columns=metadata["engineered_feature_columns"])
    results = permutation_importance(bundle["model"], x, s.threshold[TARGET],
                                     scoring="average_precision", n_repeats=3,
                                     random_state=42, n_jobs=1)
    imp = pd.DataFrame({"feature": x.columns,
                        "importance_mean": results.importances_mean,
                        "importance_std": results.importances_std})
    imp = imp.sort_values("importance_mean", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(args.out, index=False)
    print(imp.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
