"""PCA / K-means / Isolation Forest: fit without access to failure labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import TARGET, RANDOM_STATE
from .data import load_dataset, make_features, split_data
from .modeling import columns_by_type


def unsupervised_report(data_path: Path, outdir: Path) -> dict:
    raw = load_dataset(data_path)
    split = split_data(raw)
    x_train = make_features(split.train)
    numeric, _ = columns_by_type(x_train)
    prep = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scale', StandardScaler())])
    # No y argument is supplied to any unsupervised fit.
    z_train = prep.fit_transform(x_train[numeric])
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(z_train)
    mapped = pca.transform(z_train)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit(z_train)
    iso = IsolationForest(n_estimators=150, contamination=0.07, random_state=RANDOM_STATE).fit(z_train)
    # Known target is ONLY used after fitting, for an external diagnostic.
    train_cluster_summary = pd.DataFrame({"cluster": kmeans.labels_,
                                           "failure": split.train[TARGET].to_numpy()}).groupby("cluster")["failure"].agg(["count", "sum", "mean"])
    unusual_score = -iso.decision_function(z_train)
    anomaly_roc_auc = roc_auc_score(split.train[TARGET], unusual_score)
    rng = np.random.default_rng(RANDOM_STATE)
    ids = rng.choice(len(z_train), size=min(1500, len(z_train)), replace=False)
    silhouette = float(silhouette_score(z_train[ids], kmeans.predict(z_train[ids])))
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(mapped[:, 0], mapped[:, 1], c=kmeans.labels_, s=4, alpha=.35)
    ax.set(xlabel="PC1", ylabel="PC2", title="PCA of training records, coloured by unsupervised K-means cluster")
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(outdir / "pca_clusters.png", dpi=170)
    plt.close(fig)
    train_cluster_summary.to_csv(outdir / "cluster_failure_summary.csv")
    stats = {
        "pca_explained_variance_two_components": float(pca.explained_variance_ratio_.sum()),
        "kmeans_sample_silhouette": silhouette,
        "isolation_forest_anomaly_vs_label_roc_auc": float(anomaly_roc_auc),
        "note": "Unsupervised fits used only training predictors; failure labels used only afterward for diagnostics."
    }
    (outdir / "unsupervised_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=Path("artifacts/unsupervised"))
    args = p.parse_args()
    print(json.dumps(unsupervised_report(args.data, args.outdir), indent=2))


if __name__ == "__main__":
    main()
