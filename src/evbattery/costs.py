"""Compare illustrative FN-vs-FP warning costs on threshold validation ONLY."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .config import TARGET
from .data import load_dataset, split_data, make_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--model', type=Path, default=Path('artifacts/model.joblib'))
    p.add_argument('--out', type=Path, default=Path('artifacts/illustrative_costs.csv'))
    args = p.parse_args()
    bundle = joblib.load(args.model)  # trusted locally trained artifact ONLY
    m = bundle['metadata']
    thr = split_data(load_dataset(args.data)).threshold
    x = make_features(thr, m['include_suspect_scores']).reindex(columns=m['engineered_feature_columns'])
    probs = bundle['model'].predict_proba(x)[:, 1]
    y = thr[TARGET].to_numpy()
    rows = []
    # Hypothetical cost ratios; NEVER present as actual monetary maintenance costs.
    for missed_failure_cost in (2, 5, 10):
        for cutoff in np.linspace(.02, .98, 97):
            pred = probs >= cutoff
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            rows.append({'missed_failure_cost_units': missed_failure_cost,
                         'false_warning_cost_units': 1,
                         'threshold': float(cutoff),
                         'total_assumed_cost': int(missed_failure_cost * fn + fp),
                         'false_alarms': fp, 'missed_failures': fn})
    frame = pd.DataFrame(rows)
    optimal = frame.sort_values(['missed_failure_cost_units', 'total_assumed_cost', 'threshold']).groupby('missed_failure_cost_units').head(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    optimal.to_csv(args.out, index=False)
    print(optimal.to_string(index=False))
    print('Illustrative units ONLY, determined without using held-out test data.')


if __name__ == '__main__':
    main()
