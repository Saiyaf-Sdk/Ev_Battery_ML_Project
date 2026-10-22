"""Optional train-only RandomizedSearchCV for a preselected classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import RANDOM_STATE, TARGET
from .data import load_dataset, make_features, split_data
from .modeling import pipe_for, candidate_models


PARAM_DISTRIBUTIONS: dict[str, dict[str, list[object]]] = {
    'logistic_regression': {
        'model__C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        'model__class_weight': [None, 'balanced'],
    },
    'sgd_logistic': {
        'model__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        'model__penalty': ['l2', 'l1', 'elasticnet'],
        'model__class_weight': [None, 'balanced'],
    },
    'gaussian_naive_bayes': {
        'model__var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
    },
    'random_forest': {
        'model__n_estimators': [50, 90, 150, 200],
        'model__max_depth': [None, 8, 12, 16],
        'model__min_samples_leaf': [1, 2, 4, 8],
    },
    'decision_tree': {
        'model__max_depth': [3, 5, 7, 10, None],
        'model__min_samples_leaf': [2, 5, 10, 20],
    },
    'knn': {
        'model__n_neighbors': [5, 9, 15, 21, 31],
        'model__weights': ['uniform', 'distance'],
    },
    'adaboost': {
        'model__n_estimators': [40, 80, 120, 160],
        'model__learning_rate': [0.03, 0.1, 0.3, 1.0],
    },
    'hist_gradient_boost': {
        'model__max_iter': [50, 70, 100, 150],
        'model__learning_rate': [0.03, 0.075, 0.1, 0.15],
        'model__max_leaf_nodes': [7, 15, 31],
        'model__l2_regularization': [0.0, 1.0, 5.0],
    },
}


def tune_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    n_iter: int = 10,
    seed: int = RANDOM_STATE,
) -> dict[str, object]:
    """Conduct training-only RandomizedSearchCV for a designated candidate classifier."""
    if model_name not in PARAM_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: {sorted(PARAM_DISTRIBUTIONS)}"
        )
    candidates = candidate_models(include_knn=True)
    if model_name not in candidates:
        raise ValueError(f"Model '{model_name}' not available in candidate models")
    
    pipeline = pipe_for(x_train, candidates[model_name])
    param_dist = PARAM_DISTRIBUTIONS[model_name]
    total_combinations = int(np.prod([len(v) for v in param_dist.values()]))
    effective_n_iter = min(n_iter, total_combinations)

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=effective_n_iter,
        scoring='average_precision',
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        n_jobs=1,
        refit=True,
        random_state=seed,
        error_score='raise',
        return_train_score=False,
    )
    search.fit(x_train, y_train)

    return {
        'model': model_name,
        'best_cv_average_precision': float(search.best_score_),
        'best_params': search.best_params_,
        'n_iter_requested': n_iter,
        'n_iter_executed': effective_n_iter,
        'param_distributions': {k: list(v) for k, v in param_dist.items()},
        'best_estimator': search.best_estimator_,
        'cv_results_summary': {
            'mean_test_score': [float(s) for s in search.cv_results_['mean_test_score']],
            'std_test_score': [float(s) for s in search.cv_results_['std_test_score']],
            'params': search.cv_results_['params'],
        },
        'note': 'Only training-fold CV used. Calibration, threshold, and test partitions not touched.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', required=True, choices=sorted(PARAM_DISTRIBUTIONS))
    p.add_argument('--n-iter', type=int, default=10)
    p.add_argument('--out', type=Path, default=Path('artifacts/tuned_params.json'))
    args = p.parse_args()
    if args.n_iter < 1 or args.n_iter > 100:
        raise ValueError('--n-iter must be between 1 and 100')
    raw = split_data(load_dataset(args.data)).train
    x = make_features(raw)
    result = tune_model(x, raw[TARGET], args.model, n_iter=args.n_iter)
    # Exclude non-serializable best_estimator from JSON output
    serializable = {k: v for k, v in result.items() if k != 'best_estimator'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(serializable, indent=2), encoding='utf-8')
    print(json.dumps(serializable, indent=2))


if __name__ == '__main__':
    main()

