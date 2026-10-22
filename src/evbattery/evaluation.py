"""Rare-event evaluation and validation-only alarm threshold optimisation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, fbeta_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


def choose_threshold(y_true, prob, *, beta: float = 2.0) -> tuple[float, float]:
    """Select threshold on a dedicated validation set, NOT on test data."""
    y = np.asarray(y_true)
    p = np.asarray(prob)
    if np.isnan(p).any() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    thresholds = np.linspace(0.02, 0.98, 97)
    scores = [fbeta_score(y, p >= t, beta=beta, zero_division=0) for t in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def evaluate(y_true, prob, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y_true)
    p = np.asarray(prob)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "failures": int((y == 1).sum()),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
