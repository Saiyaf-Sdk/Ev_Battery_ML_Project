"""Generate report-ready charts; never use final test to choose settings."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.calibration import calibration_curve
from .config import TARGET
from .data import load_dataset, split_data, make_features


def eda_charts(data_path: Path, outdir: Path):
    train = split_data(load_dataset(data_path)).train
    outdir.mkdir(parents=True, exist_ok=True)
    counts = train[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(['Healthy (0)', 'Failure (1)'], counts.values)
    ax.set(xlabel='Target class', ylabel='Count', title='EV battery class distribution — training subset')
    fig.tight_layout()
    fig.savefig(outdir / 'target_distribution.png', dpi=160)
    plt.close(fig)

    missing = (train.isna().mean() * 100).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(missing.index[::-1], missing.values[::-1])
    ax.set(xlabel='Missing (%)', title='15 columns with most missing values — training subset')
    fig.tight_layout()
    fig.savefig(outdir / 'missingness.png', dpi=160)
    plt.close(fig)

    for feature in ('battery_health_percent', 'cell_temperature_max'):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for label in [0, 1]:
            vals = train.loc[train[TARGET] == label, feature].dropna()
            ax.hist(vals, bins=35, alpha=.55, density=True, label=f'Failure label {label}')
        ax.set(xlabel=feature, ylabel='Density', title=f'{feature} by synthetic failure class')
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f'{feature}_by_label.png', dpi=160)
        plt.close(fig)


def final_charts(data_path: Path, artifact: Path, outdir: Path):
    """Visualize a frozen model's held-out predictions. Do NOT retune using plots."""
    # Only load models you personally trained. joblib is based on pickle.
    bundle = joblib.load(artifact)
    meta = bundle['metadata']
    if meta.get('fast_smoke_test_only'):
        raise ValueError('Do not prepare final charts from smoke-test models')
    test = split_data(load_dataset(data_path)).test
    x = make_features(test, meta['include_suspect_scores'])
    x = x.reindex(columns=meta['engineered_feature_columns'])
    y = test[TARGET]
    probabilities = bundle['model'].predict_proba(x)[:, 1]
    preds = probabilities >= bundle['threshold']
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y, preds, display_labels=['Healthy', 'Failure'], ax=ax, colorbar=False)
    ax.set_title('Frozen-model confusion matrix (final holdout)')
    fig.tight_layout()
    fig.savefig(outdir / 'confusion_matrix.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=ax)
    ax.axhline(float(y.mean()), linestyle='--', label='Positive prevalence baseline')
    ax.legend()
    ax.set_title('Precision–recall curve (final holdout)')
    fig.tight_layout()
    fig.savefig(outdir / 'precision_recall.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    RocCurveDisplay.from_predictions(y, probabilities, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle='--', label='Chance')
    ax.legend()
    ax.set_title('ROC curve (final holdout)')
    fig.tight_layout()
    fig.savefig(outdir / 'roc.png', dpi=160)
    plt.close(fig)

    observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    ax.plot(predicted, observed, marker='o', label='Model')
    ax.set(xlabel='Average predicted score per bin', ylabel='Observed synthetic positive frequency',
           title='Calibration curve (final holdout)', xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'calibration.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--mode', required=True, choices=['eda', 'final'])
    p.add_argument('--model', type=Path)
    p.add_argument('--outdir', type=Path, default=Path('artifacts/figures'))
    args = p.parse_args()
    if args.mode == 'eda':
        eda_charts(args.data, args.outdir)
    else:
        if args.model is None:
            p.error('--model is required for --mode final')
        final_charts(args.data, args.model, args.outdir)
    print(f'Figures saved under {args.outdir}')


if __name__ == '__main__':
    main()
