"""Train-fold-only comparison of imputation, correlation pruning, and feature selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TARGET, RANDOM_STATE
from .data import load_dataset, make_features, split_data
from .modeling import columns_by_type


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Learn redundant-column removals INSIDE a CV training fold only."""
    def __init__(self, threshold=0.97):
        self.threshold = threshold

    def fit(self, x, y=None):
        arr = np.asarray(x, dtype=float)
        corr = np.abs(np.corrcoef(arr, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        selected = []
        for index in range(arr.shape[1]):
            if all(corr[index, previous] < self.threshold for previous in selected):
                selected.append(index)
        self.selected_ = np.array(selected, dtype=int)
        return self

    def transform(self, x):
        return np.asarray(x)[:, self.selected_]


def make_preprocessor(x: pd.DataFrame, numeric_imputer: str) -> ColumnTransformer:
    numerical, categorical = columns_by_type(x)
    imp = (KNNImputer(n_neighbors=5) if numeric_imputer == 'knn'
           else SimpleImputer(strategy='median'))
    parts = [('num', Pipeline([('imputer', imp), ('scaler', StandardScaler())]), numerical)]
    if categorical:
        parts.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(parts, sparse_threshold=0)


def compare(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, imputer, extra in [
        ('median_baseline', 'median', None),
        ('knn_imputation', 'knn', None),
        ('median_corr_prune_0.97', 'median', CorrelationPruner(threshold=0.97)),
        ('median_embedded_L1_select', 'median',
         SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.3,
                                            random_state=RANDOM_STATE))),
    ]:
        steps = [('prep', make_preprocessor(x, imputer))]
        if extra is not None:
            steps.append(('select', extra))
        steps.append(('clf', LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)))
        scores = cross_validate(Pipeline(steps), x, y, cv=cv,
                                scoring=['average_precision', 'roc_auc'],
                                n_jobs=1, error_score='raise')
        rows.append({
            'method': name,
            'cv_average_precision_mean': float(scores['test_average_precision'].mean()),
            'cv_roc_auc_mean': float(scores['test_roc_auc'].mean()),
            'n_records': int(len(x)),
        })
    return pd.DataFrame(rows).sort_values('cv_average_precision_mean', ascending=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--max-rows', type=int, default=2500,
                   help='KNN imputation can be costly; use the same capped sample for fair comparison')
    p.add_argument('--out', type=Path, default=Path('artifacts/preprocessing_comparison.csv'))
    args = p.parse_args()
    if args.max_rows < 200:
        raise ValueError('--max-rows must be at least 200')
    train = split_data(load_dataset(args.data)).train
    if len(train) > args.max_rows:
        train, _ = train_test_split(train, train_size=args.max_rows,
                                   random_state=RANDOM_STATE, stratify=train[TARGET])
    x = make_features(train)
    scores = compare(x, train[TARGET])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out, index=False)
    print(scores.to_string(index=False))


if __name__ == '__main__':
    main()
