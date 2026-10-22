"""Generate presentation-quality figures for Member B's report and slides."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import apply_style, save, HEALTHY, FAILURE, INK, INK_2, MUTED


def generate_member_b_figures(artifacts_dir: Path = Path("artifacts/member_b"),
                              outdir: Path = Path("docs/figures/member_b")) -> list[Path]:
    apply_style()
    outdir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Leaderboard figure
    lb_path = artifacts_dir / "model_cv_leaderboard.csv"
    if lb_path.is_file():
        df = pd.read_csv(lb_path)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        models = df["model"].tolist()
        y_pos = np.arange(len(models))
        ap_means = df["cv_average_precision_mean"].to_numpy()
        ap_stds = df["cv_average_precision_std"].to_numpy() if "cv_average_precision_std" in df.columns else np.zeros_like(ap_means)

        colors = [FAILURE if m == df.iloc[0]["model"] and m != "dummy_prior" else (MUTED if m == "dummy_prior" else HEALTHY) for m in models]
        bars = ax.barh(y_pos, ap_means, xerr=ap_stds, align="center", color=colors, alpha=0.88, capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.invert_yaxis()
        ax.set_xlabel("Cross-Validation Average Precision (Mean ± 1 SD)")
        ax.set_title("3-Fold Stratified CV Model Comparison (Training Split Only)")

        # Baseline reference line
        dummy_row = df[df["model"] == "dummy_prior"]
        if not dummy_row.empty:
            base_val = dummy_row.iloc[0]["cv_average_precision_mean"]
            ax.axvline(base_val, color="#d9534f", linestyle="--", linewidth=1.2, label=f"No-Skill Baseline ({base_val:.3f})")
            ax.legend(loc="lower right")

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + (ap_stds[i] if len(ap_stds) > i else 0) + 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}", va="center", ha="left", color=INK_2, fontsize=8.5)

        fig_path = outdir / "01_model_cv_comparison.png"
        save(fig, fig_path)
        generated.append(fig_path)

    # 2. Preprocessing figure
    prep_path = artifacts_dir / "preprocessing_comparison.csv"
    if prep_path.is_file():
        pdf = pd.read_csv(prep_path)
        fig, ax = plt.subplots(figsize=(7.5, 4))
        methods = pdf["method"].tolist()
        ap_means = pdf["cv_average_precision_mean"].to_numpy()
        ap_stds = pdf["cv_average_precision_std"].to_numpy()

        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, ap_means, yerr=ap_stds, capsize=4, color="#184f95", alpha=0.82, width=0.55)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Mean Average Precision (AP)")
        ax.set_title("Preprocessing & Feature Selection Comparison (Within CV Folds)")

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)

        fig_path = outdir / "02_preprocessing_experiments.png"
        save(fig, fig_path)
        generated.append(fig_path)

    # 3. Tuning results figure
    tune_path = artifacts_dir / "tuning_results.json"
    if tune_path.is_file():
        with open(tune_path, "r", encoding="utf-8") as f:
            tune_data = json.load(f)
        if "cv_results_summary" in tune_data and "mean_test_score" in tune_data["cv_results_summary"]:
            scores = tune_data["cv_results_summary"]["mean_test_score"]
            stds = tune_data["cv_results_summary"]["std_test_score"]
            trials = np.arange(1, len(scores) + 1)
            best_idx = int(np.argmax(scores))

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.errorbar(trials, scores, yerr=stds, fmt="o-", color=HEALTHY, ecolor=MUTED,
                        capsize=3, linewidth=1.5, markersize=6, label="Trial AP")
            ax.scatter([trials[best_idx]], [scores[best_idx]], color=FAILURE, s=120, zorder=5,
                       label=f"Best Config (Trial {best_idx+1})")
            ax.set_xlabel("RandomizedSearchCV Trial Number")
            ax.set_ylabel("3-Fold CV Average Precision")
            ax.set_title(f"Hyperparameter Optimization Trials ({tune_data['model']})")
            ax.set_xticks(trials)
            ax.legend(loc="lower right")

            fig_path = outdir / "03_hyperparameter_tuning.png"
            save(fig, fig_path)
            generated.append(fig_path)

    # 4. Accuracy vs Runtime Engineering Tradeoff Map
    if lb_path.is_file():
        df = pd.read_csv(lb_path)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for _, row in df.iterrows():
            m = row["model"]
            if m == "dummy_prior":
                continue
            x_time = row["fit_time_mean"] * 1000.0 # ms
            y_ap = row["cv_average_precision_mean"]
            c = FAILURE if row["model"] == df.iloc[0]["model"] else HEALTHY
            ax.scatter(x_time, y_ap, color=c, s=110, alpha=0.9)
            ax.annotate(m, (x_time, y_ap), textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=INK)

        ax.set_xlabel("Mean Training Time per Fold (milliseconds)")
        ax.set_ylabel("Cross-Validation Average Precision (AP)")
        ax.set_title("ML Engineering Tradeoff: Predictive Power vs Computational Latency")

        fig_path = outdir / "04_ap_vs_runtime_tradeoff.png"
        save(fig, fig_path)
        generated.append(fig_path)

    return generated


if __name__ == "__main__":
    figs = generate_member_b_figures()
    print(f"Generated {len(figs)} figures successfully under docs/figures/member_b/")
