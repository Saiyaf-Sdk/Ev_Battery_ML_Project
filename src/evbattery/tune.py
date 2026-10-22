"""Optional train-only RandomizedSearchCV for a preselected classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import RANDOM_STATE, TARGET
from .data import load_dataset, make_features, split_data
from .modeling import pipe_for, candidate_models


PARAM_DISTRIBUTIONS = {
    'logistic_regression': {
        'model__C': [0.01, 0.03, 0.1, 0.3, 1., 3., 10.],
        'model__class_weight': [None, 'balanced'],
    },
    'random_forest': {
        'model__n_estimators': [100, 150, 250],
        'model__max_depth': [None, 8, 12, 20],
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
        'model__n_estimators': [50, 100, 150, 200],
        'model__learning_rate': [0.03, 0.1, 0.3, 1.],
    },
    'hist_gradient_boost': {
        'model__max_iter': [70, 120, 170],
        'model__learning_rate': [0.03, 0.07, 0.12],
        'model__max_leaf_nodes': [7, 15, 31],
        'model__l2_regularization': [0., 1., 5.],
    },
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
    pipeline = pipe_for(x, candidate_models(include_knn=True)[args.model])
    search = RandomizedSearchCV(
        pipeline, PARAM_DISTRIBUTIONS[args.model], n_iter=args.n_iter,
        scoring='average_precision',
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1, refit=True, random_state=RANDOM_STATE,
        error_score='raise', return_train_score=False,
    )
    search.fit(x, raw[TARGET])
    output = {'model': args.model,
              'best_cv_average_precision': float(search.best_score_),
              'best_params': search.best_params_,
              'note': 'Only training-fold CV used. Calibration/threshold/test not touched.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
