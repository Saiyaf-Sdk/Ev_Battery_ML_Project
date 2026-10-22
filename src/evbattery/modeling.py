"""Leakage-safe training preprocessing and classical ML model factories."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import RANDOM_STATE


def columns_by_type(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = x.columns.difference(numeric, sort=False).tolist()
    if not numeric:
        raise ValueError("No numerical telemetry features available")
    return numeric, categorical


def preprocessing(x: pd.DataFrame) -> ColumnTransformer:
    numerical, categorical = columns_by_type(x)
    parts = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numerical)
    ]
    if categorical:
        parts.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0)


def candidate_models(fast: bool = False, include_knn: bool = False) -> dict[str, object]:
    """Seven efficient classifiers by default; optional expensive KNN + dummy."""
    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
        "gaussian_naive_bayes": GaussianNB(),
        "sgd_logistic": SGDClassifier(loss="log_loss", max_iter=2000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=7, min_samples_leaf=10, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=35 if fast else 90, min_samples_leaf=3,
            max_features="sqrt", n_jobs=1, random_state=RANDOM_STATE,
        ),
        "adaboost": AdaBoostClassifier(n_estimators=35 if fast else 80, random_state=RANDOM_STATE),
        "hist_gradient_boost": HistGradientBoostingClassifier(
            max_iter=35 if fast else 70, max_leaf_nodes=15,
            learning_rate=0.075, random_state=RANDOM_STATE,
        ),
    }
    if include_knn:
        models["knn"] = KNeighborsClassifier(n_neighbors=15)
    return models


def pipe_for(x: pd.DataFrame, estimator: object) -> Pipeline:
    return Pipeline([("preprocess", preprocessing(x)), ("model", estimator)])


def cv_leaderboard(x: pd.DataFrame, y: pd.Series, *, fast: bool = False, seed: int = RANDOM_STATE, model_names: set[str] | None = None):
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows = []
    models = candidate_models(fast=fast)
    if model_names is not None:
        models = {n: m for n, m in models.items() if n in model_names}
        if not models:
            raise ValueError("No matching model names")
    for name, estimator in models.items():
        pipeline = pipe_for(x, estimator)
        scores = cross_validate(
            pipeline, x, y, cv=folds,
            scoring={"ap": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1, error_score="raise", return_train_score=False,
        )
        rows.append({
            "model": name,
            "cv_average_precision_mean": float(np.mean(scores["test_ap"])),
            "cv_average_precision_std": float(np.std(scores["test_ap"])),
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        })
    # Dummy is only a reference, never a deployable winner.
    return pd.DataFrame(rows).sort_values("cv_average_precision_mean", ascending=False).reset_index(drop=True)
