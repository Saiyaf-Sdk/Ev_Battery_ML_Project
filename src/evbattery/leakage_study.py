"""Composite risk-feature ablation using TRAINING CV ONLY (no test access)."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import TARGET, RANDOM_STATE
from .data import load_dataset, make_features, split_data
from .modeling import candidate_models, pipe_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--model', choices=['logistic_regression', 'decision_tree', 'random_forest'], default='logistic_regression')
    p.add_argument('--out', type=Path, default=Path('artifacts/leakage_ablation.csv'))
    args = p.parse_args()
    train = split_data(load_dataset(args.data)).train
    rows = []
    for allow in [False, True]:
        x = make_features(train, include_suspect_scores=allow)
        model = candidate_models()[args.model]
        result = cross_validate(pipe_for(x, model), x, train[TARGET],
                                cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                                scoring='average_precision', n_jobs=1, error_score='raise')
        rows.append({'model': args.model,
                     'risk_composites_included': allow,
                     'features': x.shape[1],
                     'train_cv_ap_mean': float(result['test_score'].mean()),
                     'train_cv_ap_std': float(result['test_score'].std())})
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print('Caution: any difference demonstrates sensitivity; it does not establish definite leakage.')


if __name__ == '__main__':
    main()
