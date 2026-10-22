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
    """Learn redundant-column removals INSIDE a CV training fold only.
    
    Greedily keeps features whose absolute Pearson correlation with already
    selected features is strictly below the given threshold. All correlation
    statistics are computed exclusively on the data passed to fit().
    """
    def __init__(self, threshold: float = 0.97):
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.threshold = float(threshold)

    def fit(self, x, y=None):
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Expected 2D array for correlation pruning")
        n_features = arr.shape[1]
        if n_features == 0:
            self.selected_ = np.array([], dtype=int)
            return self
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.abs(np.corrcoef(arr, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        if np.ndim(corr) == 0:
            corr = np.array([[corr]])
        selected = []
        for index in range(n_features):
            if all(corr[index, previous] < self.threshold for previous in selected):
                selected.append(index)
        self.selected_ = np.array(selected, dtype=int)
        self.n_features_in_ = n_features
        return self

    def transform(self, x):
        arr = np.asarray(x)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: fit on {self.n_features_in_}, received {arr.shape[1]}"
            )
        return arr[:, self.selected_]

    def get_support(self, indices: bool = False):
        if indices:
            return self.selected_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_] = True
        return mask


def make_preprocessor(x: pd.DataFrame, numeric_imputer: str) -> ColumnTransformer:
    numerical, categorical = columns_by_type(x)
    imp = (KNNImputer(n_neighbors=5) if numeric_imputer == 'knn'
           else SimpleImputer(strategy='median', add_indicator=True))
    parts = [('num', Pipeline([('imputer', imp), ('scaler', StandardScaler())]), numerical)]
    if categorical:
        parts.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(parts, sparse_threshold=0)


def compare(x: pd.DataFrame, y: pd.Series, cv_splits: int = 3, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Compare preprocessing variations strictly inside cross-validation training folds."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    rows = []
    configurations = [
        ('median_baseline', 'median', None),
        ('knn_imputation', 'knn', None),
        ('median_corr_prune_0.97', 'median', CorrelationPruner(threshold=0.97)),
        ('median_embedded_L1_select', 'median',
         SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.3,
                                            random_state=seed))),
    ]
    for name, imputer, extra in configurations:
        steps = [('prep', make_preprocessor(x, imputer))]
        if extra is not None:
            steps.append(('select', extra))
        steps.append(('clf', LogisticRegression(max_iter=1500, random_state=seed)))
        scores = cross_validate(
            Pipeline(steps), x, y, cv=cv,
            scoring={'ap': 'average_precision', 'roc_auc': 'roc_auc'},
            n_jobs=1, error_score='raise', return_train_score=False,
        )
        rows.append({
            'method': name,
            'cv_average_precision_mean': float(np.mean(scores['test_ap'])),
            'cv_average_precision_std': float(np.std(scores['test_ap'])),
            'cv_roc_auc_mean': float(np.mean(scores['test_roc_auc'])),
            'cv_roc_auc_std': float(np.std(scores['test_roc_auc'])),
            'fit_time_mean': float(np.mean(scores['fit_time'])),
            'n_records': int(len(x)),
        })
    return pd.DataFrame(rows).sort_values('cv_average_precision_mean', ascending=False).reset_index(drop=True)


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

