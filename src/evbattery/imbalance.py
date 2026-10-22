"""Compare weighting, undersampling, and category-aware SMOTENC by train-only CV."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import TARGET, RANDOM_STATE
from .data import load_dataset, make_features, split_data
from .modeling import columns_by_type, preprocessing


def experiments(x: pd.DataFrame, y: pd.Series, fast: bool = False) -> pd.DataFrame:
    """DO NOT resample before cross-validation; samplers are inside the folds."""
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    basic = preprocessing(x)
    plain = LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)
    weighted = LogisticRegression(max_iter=1500, class_weight='balanced', random_state=RANDOM_STATE)
    models = {
        'no_resampling': ImbPipeline([('prep', basic), ('clf', plain)]),
        'class_weight_balanced': ImbPipeline([('prep', preprocessing(x)), ('clf', weighted)]),
        'random_undersampling': ImbPipeline([
            ('prep', preprocessing(x)),
            ('resample', RandomUnderSampler(random_state=RANDOM_STATE)),
            ('clf', LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)),
        ]),
    }
    numerical, categorical = columns_by_type(x)
    if categorical:
        # For true mixed tabular data, one-hot BEFORE SMOTE would produce fractional
        # fake dummy values. Instead impute and ordinal-encode, use SMOTENC to treat
        # categorical indices categorically, then one-hot afterward.
        ord_prep = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), numerical),
            ('cat', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ]), categorical),
        ], sparse_threshold=0.0)
        n_num, n_cat = len(numerical), len(categorical)
        onehot_post = ColumnTransformer([
            ('num', StandardScaler(), list(range(n_num))),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             list(range(n_num, n_num + n_cat))),
        ], sparse_threshold=0.0)
        models['smotenc'] = ImbPipeline([
            ('ordinal_preprocess', ord_prep),
            ('resample', SMOTENC(categorical_features=list(range(n_num, n_num + n_cat)),
                                 random_state=RANDOM_STATE, k_neighbors=3)),
            ('onehot_and_scale', onehot_post),
            ('clf', LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)),
        ])
    rows = []
    for name, pipeline in models.items():
        r = cross_validate(pipeline, x, y, cv=folds, scoring=['average_precision', 'roc_auc'],
                           error_score='raise', n_jobs=1)
        rows.append({"imbalance_method": name,
                     "cv_average_precision_mean": float(np.mean(r['test_average_precision'])),
                     "cv_roc_auc_mean": float(np.mean(r['test_roc_auc']))})
    return pd.DataFrame(rows).sort_values('cv_average_precision_mean', ascending=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('artifacts/imbalance_comparison.csv'))
    args = p.parse_args()
    train = split_data(load_dataset(args.data)).train
    scores = experiments(make_features(train), train[TARGET])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out, index=False)
    print(scores.to_string(index=False))


if __name__ == '__main__':
    main()
