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


MODEL_PROPERTIES: dict[str, dict[str, object]] = {
    "dummy_prior": {
        "family": "Baseline",
        "interpretable": "Trivial baseline",
        "scaling_required": False,
        "nonlinear": False,
        "advantages": "Fast no-skill sanity check; establishes minimum baseline prevalence AP",
        "weaknesses": "Zero predictive capacity; predicts constant prior probability",
    },
    "logistic_regression": {
        "family": "Linear",
        "interpretable": "High (signed odds-ratio coefficients)",
        "scaling_required": True,
        "nonlinear": False,
        "advantages": "Well-calibrated probabilities, fast training, convex loss, transparent weights",
        "weaknesses": "Cannot model nonlinear sensor interactions without explicit cross-terms",
    },
    "gaussian_naive_bayes": {
        "family": "Probabilistic / Generative",
        "interpretable": "High (feature-conditional Gaussian likelihoods)",
        "scaling_required": False,
        "nonlinear": True,
        "advantages": "Extremely fast closed-form fit, no optimization hyperparameters",
        "weaknesses": "Strong conditional independence assumption; sensitive to correlated telemetry",
    },
    "sgd_logistic": {
        "family": "Linear / Stochastic Online",
        "interpretable": "High (linear weights)",
        "scaling_required": True,
        "nonlinear": False,
        "advantages": "O(N) linear time streaming solver, minimal memory footprint",
        "weaknesses": "Sensitive to learning rate / alpha schedule; noisier convergence than L-BFGS",
    },
    "decision_tree": {
        "family": "Tree",
        "interpretable": "High (visualizable if shallow)",
        "scaling_required": False,
        "nonlinear": True,
        "advantages": "Handles mixed scales natively, captures orthogonal feature thresholds",
        "weaknesses": "High variance, step-function decision boundaries, prone to overfitting",
    },
    "random_forest": {
        "family": "Ensemble (Bagging)",
        "interpretable": "Moderate (Gini / permutation importance)",
        "scaling_required": False,
        "nonlinear": True,
        "advantages": "Reduces variance via bootstrap aggregation; robust to outliers and noisy sensors",
        "weaknesses": "Slower training with many trees, uncalibrated risk probabilities near 0 and 1",
    },
    "adaboost": {
        "family": "Ensemble (Boosting)",
        "interpretable": "Moderate (weighted sum of decision stumps)",
        "scaling_required": False,
        "nonlinear": True,
        "advantages": "Sequentially focuses weight on hard/borderline battery failure samples",
        "weaknesses": "Sensitive to label noise and severe outliers; can overfit without learning-rate shrinkage",
    },
    "hist_gradient_boost": {
        "family": "Ensemble (Gradient Boosting)",
        "interpretable": "Low-Moderate (tree ensemble / partial dependence)",
        "scaling_required": False,
        "nonlinear": True,
        "advantages": "State-of-the-art tabular accuracy, histogram binning (fast), native NaN handling",
        "weaknesses": "Complex hyperparameter space, requires monotonic constraints or regularization",
    },
    "knn": {
        "family": "Instance-based",
        "interpretable": "Low-Moderate (nearest neighbor exemplars)",
        "scaling_required": True,
        "nonlinear": True,
        "advantages": "Non-parametric; no explicit training phase; local neighborhood boundaries",
        "weaknesses": "O(N·D) query inference time; curse of dimensionality across 63 features",
    },
}


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


def cv_leaderboard(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    fast: bool = False,
    include_knn: bool = False,
    include_properties: bool = False,
    seed: int = RANDOM_STATE,
    model_names: set[str] | None = None,
) -> pd.DataFrame:
    """Evaluate candidate classifiers with 3-fold stratified CV on training data only."""
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows = []
    models = candidate_models(fast=fast, include_knn=include_knn)
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
        row = {
            "model": name,
            "cv_average_precision_mean": float(np.mean(scores["test_ap"])),
            "cv_average_precision_std": float(np.std(scores["test_ap"])),
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "fit_time_mean": float(np.mean(scores["fit_time"])),
            "score_time_mean": float(np.mean(scores["score_time"])),
        }
        if include_properties and name in MODEL_PROPERTIES:
            row.update(MODEL_PROPERTIES[name])
        rows.append(row)
    # Dummy is only a reference, never a deployable winner.
    return pd.DataFrame(rows).sort_values("cv_average_precision_mean", ascending=False).reset_index(drop=True)

