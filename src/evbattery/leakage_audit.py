"""Member A — leakage and feature-set audit (training data only).

Run:
    python -m evbattery.leakage_audit --data "data/ev_battery.csv" --outdir artifacts/leakage_audit

What it produces
----------------
split_check.csv              60/10/10/20 split sizes, failure counts, prevalence, ID overlap
univariate_screen.csv        how well each single column separates the classes (ROC AUC)
redundancy_check.csv         exact formula relationships between columns
high_correlation_pairs.csv   |Pearson r| >= 0.95 pairs (training split)
composite_reconstruction.csv can each suspect score be rebuilt from raw telemetry?
feature_set_ablation.csv     3-fold CV average precision for four candidate feature sets
feature_set.json             the agreed feature list handed to Members B, C and D
figures/*.png

Nothing here touches the 20% test split. All CV happens inside the 60% train split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate

from .config import IDENTIFIERS, RANDOM_STATE, REDUNDANT_EXACT, SUSPECT_SCORES, TARGET
from .data import load_dataset, make_features, split_data
from .modeling import candidate_models, pipe_for
from . import plot_style as ps

SUSPICIOUS_AUC = 0.90   # a single raw column this strong deserves a provenance check

FEATURE_SETS = {
    "A_baseline_no_composites": dict(include_suspect_scores=False, drop_redundant=False),
    "B_recommended_no_composites_no_duplicates": dict(include_suspect_scores=False, drop_redundant=True),
    "C_with_composites": dict(include_suspect_scores=True, drop_redundant=False),
    "D_with_composites_no_duplicates": dict(include_suspect_scores=True, drop_redundant=True),
}
ABLATION_MODELS = ("logistic_regression", "random_forest", "hist_gradient_boost")


def split_check(df: pd.DataFrame) -> pd.DataFrame:
    s = split_data(df)
    parts = {"train (fit + CV)": s.train, "calibration": s.calibration,
             "threshold": s.threshold, "test (final, untouched)": s.test}
    rows = [{"split": k, "rows": len(v), "percent_of_data": round(100 * len(v) / len(df), 1),
             "failures": int(v[TARGET].sum()), "failure_prevalence": round(float(v[TARGET].mean()), 4)}
            for k, v in parts.items()]
    ids = [set(v["vehicle_id"]) for v in parts.values()]
    overlap = sum(len(ids[i] & ids[j]) for i in range(4) for j in range(i + 1, 4))
    out = pd.DataFrame(rows)
    out["vehicle_id_overlap_between_splits"] = overlap
    return out


def univariate_screen(train: pd.DataFrame) -> pd.DataFrame:
    """Single-column ROC AUC on the training split.

    AUC 0.5 = no signal. Strength = max(AUC, 1 - AUC), so direction does not matter.
    A raw column with strength >= 0.90 would be suspicious (could be a disguised label).
    """
    rows = []
    y = train[TARGET]
    for c in train.columns:
        if c in (TARGET, *IDENTIFIERS):
            continue
        s = train[c]
        mask = s.notna()
        if pd.api.types.is_numeric_dtype(s):
            score = s[mask]
        else:  # categorical: score each level by its training failure rate
            score = s[mask].map(y[mask].groupby(s[mask]).mean())
        auc = roc_auc_score(y[mask], score)
        strength = max(auc, 1 - auc)
        group = ("composite_score" if c in SUSPECT_SCORES else
                 "redundant_copy" if c in REDUNDANT_EXACT else "raw")
        rows.append({"feature": c, "group": group, "type": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical",
                     "auc": round(auc, 4), "strength": round(strength, 4),
                     "direction": "higher -> more failure" if auc >= 0.5 else "lower -> more failure",
                     "flag": "INVESTIGATE (near-perfect separator)" if strength >= SUSPICIOUS_AUC else ""})
    out = pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def redundancy_check(df: pd.DataFrame) -> pd.DataFrame:
    """Verify exact formulas between columns (label-free -> safe on full CSV)."""
    d = df
    rules = {
        "capacity_loss_percent = 100 - battery_health_percent":
            d["capacity_loss_percent"] - (100 - d["battery_health_percent"]),
        "remaining_capacity = battery_capacity_kwh * battery_health_percent / 100":
            d["remaining_capacity"] - d["battery_capacity_kwh"] * d["battery_health_percent"] / 100,
        "manufacturing_year = 2026 - vehicle_age_years":
            d["manufacturing_year"] - (2026 - d["vehicle_age_years"]),
        "state_of_health ~ battery_health_percent":
            d["state_of_health"] - d["battery_health_percent"],
        "pack_voltage / cell_voltage_avg = cells in series":
            d["pack_voltage"] / d["cell_voltage_avg"] - (d["pack_voltage"] / d["cell_voltage_avg"]).round(),
    }
    rows = []
    for name, diff in rules.items():
        diff = diff.dropna()
        rows.append({"relationship": name, "rows_checked": int(len(diff)),
                     "max_abs_difference": round(float(diff.abs().max()), 4),
                     "mean_abs_difference": round(float(diff.abs().mean()), 4),
                     "verdict": "EXACT (drop one)" if diff.abs().max() < 1.2 and "state_of_health" not in name
                     and "pack_voltage" not in name else "close but not identical (keep both)"})
    return pd.DataFrame(rows)


def high_correlation_pairs(train: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    num = train.select_dtypes(include="number").drop(columns=[TARGET])
    corr = num.corr().abs()
    rows = []
    cols = corr.columns
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "abs_pearson_r": round(float(r), 4)})
    return pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False)


def composite_reconstruction(train: pd.DataFrame) -> pd.DataFrame:
    """Can each suspect score be predicted from the raw telemetry we keep?

    High R^2 -> the score is mostly a recombination of raw columns (little new info).
    Low R^2 + high AUC -> the score holds information not in the raw columns: the
    strongest leakage warning sign, because we do not know where that info came from.
    """
    x = make_features(train, include_suspect_scores=False, drop_redundant=True)
    x = x.select_dtypes(include="number")
    rows = []
    for score in SUSPECT_SCORES:
        y = train[score]
        mask = y.notna()
        model = HistGradientBoostingRegressor(max_iter=150, random_state=RANDOM_STATE)
        r2 = cross_val_score(model, x[mask], y[mask], cv=KFold(3, shuffle=True, random_state=RANDOM_STATE),
                             scoring="r2", n_jobs=1)
        auc = roc_auc_score(train.loc[mask, TARGET], y[mask])
        rows.append({"suspect_score": score, "cv_r2_from_raw_features": round(float(r2.mean()), 4),
                     "single_feature_auc_strength": round(max(auc, 1 - auc), 4)})
    return pd.DataFrame(rows).sort_values("cv_r2_from_raw_features", ascending=False)


def feature_set_ablation(train: pd.DataFrame, models=ABLATION_MODELS) -> pd.DataFrame:
    """Same 3 stratified folds, same models; only the feature set changes."""
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    y = train[TARGET]
    rows = []
    for set_name, kwargs in FEATURE_SETS.items():
        x = make_features(train, **kwargs)
        for m in models:
            res = cross_validate(pipe_for(x, candidate_models()[m]), x, y, cv=folds,
                                 scoring={"ap": "average_precision", "roc_auc": "roc_auc"},
                                 n_jobs=1, error_score="raise")
            rows.append({"feature_set": set_name, "model": m, "n_features": x.shape[1],
                         "cv_ap_mean": round(float(res["test_ap"].mean()), 4),
                         "cv_ap_std": round(float(res["test_ap"].std()), 4),
                         "cv_roc_auc_mean": round(float(res["test_roc_auc"].mean()), 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def fig_univariate(screen: pd.DataFrame, path, top: int = 25):
    plt = ps.plt
    d = screen.head(top).iloc[::-1]
    colors = [ps.FAILURE if g == "composite_score" else ps.NEUTRAL if g == "redundant_copy" else ps.HEALTHY
              for g in d["group"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(d["feature"], d["strength"], color=colors, height=0.62)
    ax.axvline(0.5, color=ps.INK_2, linestyle="--", linewidth=1)
    ax.axvline(SUSPICIOUS_AUC, color=ps.INK_2, linestyle=":", linewidth=1)
    ax.text(0.503, len(d) - 0.1, "0.5 = no signal", fontsize=8, color=ps.INK_2)
    ax.text(SUSPICIOUS_AUC + 0.003, len(d) - 0.1, "0.90 = investigate", fontsize=8, color=ps.INK_2)
    for y, v in enumerate(d["strength"]):
        ax.annotate(f"{v:.3f}", (v, y), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=8, color=ps.INK_2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ps.HEALTHY, label="raw telemetry (kept)"),
                       Patch(color=ps.NEUTRAL, label="exact duplicate (dropped)"),
                       Patch(color=ps.FAILURE, label="composite score (excluded)")],
              loc="lower right")
    ax.set(xlim=(0.45, 1.0), ylim=(-0.7, len(d) + 0.6), xlabel="Single-feature ROC AUC strength  max(AUC, 1 - AUC)",
           title=f"Top {top} single-feature separators (training split)")
    ax.grid(axis="y", visible=False)
    ps.save(fig, path)


def fig_ablation(ablation: pd.DataFrame, path):
    plt = ps.plt
    sets = list(FEATURE_SETS)
    labels = {"A_baseline_no_composites": "A: starter\nbaseline",
              "B_recommended_no_composites_no_duplicates": "B: recommended\n(no composites,\nno duplicates)",
              "C_with_composites": "C: + composite\nscores",
              "D_with_composites_no_duplicates": "D: + composites,\nno duplicates"}
    models = list(dict.fromkeys(ablation["model"]))
    model_colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(sets))
    for i, m in enumerate(models):
        sub = ablation[ablation.model == m].set_index("feature_set").reindex(sets)
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, sub["cv_ap_mean"], width * 0.92,
                      yerr=sub["cv_ap_std"], capsize=3, color=model_colors[i], label=m,
                      error_kw={"ecolor": ps.INK_2, "elinewidth": 1})
        for b, v, sd in zip(bars, sub["cv_ap_mean"], sub["cv_ap_std"]):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v + sd), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=7.5, color=ps.INK_2)
    ax.axhline(0.0692, color=ps.INK_2, linestyle="--", linewidth=1, label="no-skill AP = prevalence (0.069)")
    ax.set_xticks(x, [labels[s] for s in sets], fontsize=9)
    ax.set(ylabel="3-fold CV average precision (train split)", ylim=(0, 1.0),
           title="Feature-set ablation: do composites or duplicates change performance?")
    ax.legend(loc="upper left", ncols=4, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ps.save(fig, path)


# --------------------------------------------------------------------------- runner
def recommended_feature_set(df: pd.DataFrame) -> dict:
    x = make_features(df.head(50), **FEATURE_SETS["B_recommended_no_composites_no_duplicates"])
    numeric = x.select_dtypes(include="number").columns.tolist()
    return {
        "description": "Agreed model input columns after Member A audit. Use make_features("
                       "raw, include_suspect_scores=False, drop_redundant=True).",
        "n_features": int(x.shape[1]),
        "numeric_features": numeric,
        "categorical_features": [c for c in x.columns if c not in numeric],
        "engineered_features": [c for c in x.columns if c not in df.columns],
        "dropped_identifiers": list(IDENTIFIERS),
        "excluded_composite_scores": list(SUSPECT_SCORES),
        "dropped_exact_duplicates": list(REDUNDANT_EXACT),
        "target": TARGET,
    }


def run(data_path: Path, outdir: Path, skip_ablation: bool = False) -> dict:
    ps.apply_style()
    df = load_dataset(data_path)
    train = split_data(df).train
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    split = split_check(df)
    split.to_csv(outdir / "split_check.csv", index=False)
    screen = univariate_screen(train)
    screen.to_csv(outdir / "univariate_screen.csv", index=False)
    redundancy_check(df).to_csv(outdir / "redundancy_check.csv", index=False)
    high_correlation_pairs(train).to_csv(outdir / "high_correlation_pairs.csv", index=False)
    recon = composite_reconstruction(train)
    recon.to_csv(outdir / "composite_reconstruction.csv", index=False)
    fig_univariate(screen, outdir / "figures" / "08_univariate_auc.png")
    feature_set = recommended_feature_set(df)
    (outdir / "feature_set.json").write_text(json.dumps(feature_set, indent=2), encoding="utf-8")

    summary = {"split": split.to_dict(orient="records"),
               "raw_features_flagged": screen.loc[(screen.flag != "") & (screen.group == "raw"), "feature"].tolist(),
               "top5_single_features": screen.head(5)[["feature", "group", "strength"]].to_dict(orient="records"),
               "recommended_n_features": feature_set["n_features"]}
    if not skip_ablation:
        ablation = feature_set_ablation(train)
        ablation.to_csv(outdir / "feature_set_ablation.csv", index=False)
        fig_ablation(ablation, outdir / "figures" / "09_feature_set_ablation.png")
        summary["ablation"] = ablation.to_dict(orient="records")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=Path("artifacts/leakage_audit"))
    p.add_argument("--skip-ablation", action="store_true", help="skip the slower CV ablation (~2-5 min)")
    args = p.parse_args()
    print(json.dumps(run(args.data, args.outdir, args.skip_ablation), indent=2))
    print("Reminder: a performance jump with composites shows SENSITIVITY, not proof of leakage.")


if __name__ == "__main__":
    main()
