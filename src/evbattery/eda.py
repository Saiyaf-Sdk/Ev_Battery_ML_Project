"""Member A — data profiling, plausibility checks and EDA figures.

Run:
    python -m evbattery.eda --data "data/ev_battery.csv" --outdir artifacts/eda

Two rules keep this EDA honest:
1. Label-free facts (shape, missing cells, duplicates, value ranges) are
   computed on the full CSV: they do not use the target, so they cannot leak.
2. Anything that compares healthy vs failed rows is computed on the 60%
   TRAINING split only. We never look at the 20% test set while making
   decisions, otherwise the final test score would be optimistic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IDENTIFIERS, SUSPECT_SCORES, TARGET
from .data import load_dataset, split_data
from .feature_catalog import CATALOG, decision_for
from . import plot_style as ps

KEY_NUMERIC = [
    "battery_health_percent", "cell_temperature_max", "internal_resistance",
    "cell_temperature_avg", "vehicle_age_years", "charge_efficiency",
]
HEATMAP_FEATURES = [
    "battery_health_percent", "state_of_health", "capacity_loss_percent",
    "remaining_capacity", "internal_resistance", "cell_temperature_avg",
    "cell_temperature_max", "charge_efficiency", "discharge_efficiency",
    "vehicle_age_years", "manufacturing_year", "cycle_count", "odometer_km",
    "cell_voltage_std", "voltage_imbalance", "average_ambient_temperature",
    "maximum_temperature", "minimum_temperature", "previous_faults",
    "BMS_warning_count", TARGET,
]
CATEGORICAL_PLOTS = ["battery_chemistry", "vehicle_type", "fleet_or_private", "terrain_type"]
RATIO_COLUMNS = [
    "fast_charge_ratio", "slow_charge_ratio", "overnight_charging_ratio",
    "home_charging_ratio", "highway_driving_ratio", "city_driving_ratio",
]
PERCENT_COLUMNS = [
    "battery_health_percent", "state_of_charge", "depth_of_discharge", "state_of_health",
    "charge_efficiency", "discharge_efficiency", "capacity_loss_percent", "humidity",
    "cooling_system_health", "regenerative_braking_usage",
]


# --------------------------------------------------------------------------- tables
def overview(df: pd.DataFrame) -> dict:
    """Dataset-level facts. Uses the label only to count classes."""
    counts = df[TARGET].value_counts().sort_index()
    numeric = df.select_dtypes(include="number").columns
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_numeric_columns": int(len(numeric)),
        "n_text_columns": int(df.shape[1] - len(numeric)),
        "healthy": int(counts.get(0, 0)),
        "failures": int(counts.get(1, 0)),
        "failure_prevalence": round(float(df[TARGET].mean()), 4),
        "all_healthy_baseline_accuracy": round(float(1 - df[TARGET].mean()), 4),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percent_of_table": round(float(100 * df.isna().sum().sum() / df.size), 2),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
        "rows_with_any_missing_percent": round(float(100 * df.isna().any(axis=1).mean()), 2),
        "columns_with_missing": int((df.isna().sum() > 0).sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_ids": int(df.drop(columns=list(IDENTIFIERS)).duplicated().sum()),
        "unique_vehicle_ids": bool(df["vehicle_id"].is_unique),
        "unique_battery_serials": bool(df["battery_serial"].is_unique),
    }


def column_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, missing %, cardinality, range, group, decision."""
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        group, meaning = CATALOG.get(col, ("unknown", ""))
        decision, reason = decision_for(col)
        rows.append({
            "column": col,
            "group": group,
            "meaning": meaning,
            "type": "numeric" if is_num else "categorical/text",
            "missing_percent": round(100 * s.isna().mean(), 2),
            "n_unique": int(s.nunique(dropna=True)),
            "min": round(float(s.min()), 3) if is_num else None,
            "median": round(float(s.median()), 3) if is_num else None,
            "max": round(float(s.max()), 3) if is_num else None,
            "decision": decision,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def plausibility_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Physical / logical sanity rules. Label-free, so safe on the full CSV."""
    d = df
    checks: list[tuple[str, pd.Series, str]] = []

    def add(name, bad_mask, note):
        checks.append((name, pd.Series(bad_mask, index=d.index).fillna(False).astype(bool), note))

    add("cell_temperature_avg > cell_temperature_max",
        d["cell_temperature_avg"] > d["cell_temperature_max"], "average can never exceed maximum")
    add("ambient: maximum_temperature < minimum_temperature",
        d["maximum_temperature"] < d["minimum_temperature"], "max must be >= min")
    add("ambient average outside [min, max]",
        (d["average_ambient_temperature"] > d["maximum_temperature"])
        | (d["average_ambient_temperature"] < d["minimum_temperature"]), "average must lie between min and max")
    for col in RATIO_COLUMNS:
        add(f"{col} outside 0-1", (d[col] < 0) | (d[col] > 1), "ratio must be a share")
    for col in PERCENT_COLUMNS:
        add(f"{col} outside 0-100", (d[col] < 0) | (d[col] > 100), "percentage")
    add("fast_charge_ratio + slow_charge_ratio > 1",
        d["fast_charge_ratio"] + d["slow_charge_ratio"] > 1.0001, "shares of the same sessions")
    add("highway_driving_ratio + city_driving_ratio > 1",
        d["highway_driving_ratio"] + d["city_driving_ratio"] > 1.0001,
        "shares of the same driving should not exceed 100%")
    add("remaining_capacity > battery_capacity_kwh",
        d["remaining_capacity"] > d["battery_capacity_kwh"], "cannot hold more than nominal")
    add("vehicle_age_years <= 0", d["vehicle_age_years"] <= 0, "age must be positive")
    add("cycle_count <= 0", d["cycle_count"] <= 0, "used battery must have cycles")
    for col in ("odometer_km", "average_speed", "daily_distance", "internal_resistance",
                "battery_capacity_kwh", "pack_voltage", "altitude"):
        add(f"{col} < 0", d[col] < 0, "cannot be negative")
    add("temperature_variance at ceiling 32.94",
        np.isclose(d["temperature_variance"], d["temperature_variance"].max()),
        "values piled up at the maximum = clipped/saturated by the data generator")
    add("temperature_variance exactly 0",
        d["temperature_variance"] == 0, "values piled up at 0 = clipped at the lower end")
    add("average_charging_time at ceiling 600 min",
        d["average_charging_time"] == d["average_charging_time"].max(), "clipped at 600")
    # brand-model consistency: a model should belong to exactly one brand
    pairs = d.dropna(subset=["vehicle_brand", "vehicle_model"])
    multi = pairs.groupby("vehicle_model")["vehicle_brand"].nunique()
    shared = multi[multi > 1].index.tolist()
    add(f"vehicle_model listed under >1 brand {shared}",
        d["vehicle_model"].isin(shared) & d["vehicle_brand"].notna(), "naming/labelling inconsistency")
    types_per_model = d.dropna(subset=["vehicle_model", "vehicle_type"]).groupby("vehicle_model")["vehicle_type"].nunique()
    add(f"vehicle_type varies within a model (max {int(types_per_model.max())} types per model)",
        d["vehicle_model"].isin(types_per_model[types_per_model > 1].index) & d["vehicle_type"].notna(),
        "e.g. a sedan model also labelled 'Truck' -> body type is randomly generated")

    out = pd.DataFrame([
        {"check": n, "rows_flagged": int(m.sum()),
         "percent_of_rows": round(100 * float(m.mean()), 2), "why_it_matters": note}
        for n, m, note in checks
    ])
    out["status"] = np.where(out["rows_flagged"] == 0, "PASS", "FLAG")
    return out[["status", "check", "rows_flagged", "percent_of_rows", "why_it_matters"]]


def class_comparison(train: pd.DataFrame) -> pd.DataFrame:
    """Median/mean per class + standardised mean difference (training split only)."""
    numeric = train.select_dtypes(include="number").columns.drop(TARGET)
    h, f = train[train[TARGET] == 0], train[train[TARGET] == 1]
    rows = []
    for c in numeric:
        pooled = np.sqrt((h[c].var() + f[c].var()) / 2)
        smd = (f[c].mean() - h[c].mean()) / pooled if pooled > 0 else np.nan
        rows.append({
            "feature": c, "healthy_median": round(h[c].median(), 3),
            "failure_median": round(f[c].median(), 3),
            "healthy_mean": round(h[c].mean(), 3), "failure_mean": round(f[c].mean(), 3),
            "standardised_mean_difference": round(float(smd), 3),
            "is_suspect_score": c in SUSPECT_SCORES,
        })
    out = pd.DataFrame(rows)
    return out.reindex(out["standardised_mean_difference"].abs().sort_values(ascending=False).index)


def categorical_failure_rates(train: pd.DataFrame) -> pd.DataFrame:
    cats = [c for c in train.select_dtypes(exclude="number").columns if c not in IDENTIFIERS]
    rows = []
    for c in cats:
        g = train.groupby(train[c].fillna("(missing)"))[TARGET].agg(["size", "sum", "mean"])
        for level, r in g.iterrows():
            rows.append({"column": c, "level": level, "rows": int(r["size"]),
                         "failures": int(r["sum"]), "failure_rate": round(float(r["mean"]), 4)})
    return pd.DataFrame(rows)


def missingness_vs_target(train: pd.DataFrame) -> pd.DataFrame:
    """Is a value being missing itself related to failure? (training split)"""
    rows = []
    for c in train.columns:
        miss = train[c].isna()
        if miss.sum() == 0:
            continue
        rows.append({"column": c, "missing_rows": int(miss.sum()),
                     "failure_rate_when_missing": round(float(train.loc[miss, TARGET].mean()), 4),
                     "failure_rate_when_present": round(float(train.loc[~miss, TARGET].mean()), 4)})
    out = pd.DataFrame(rows)
    out["difference"] = (out["failure_rate_when_missing"] - out["failure_rate_when_present"]).round(4)
    return out.reindex(out["difference"].abs().sort_values(ascending=False).index)


def dictionary_markdown(dictionary: pd.DataFrame) -> str:
    """Human-readable provenance table (copied to docs/FEATURE_DICTIONARY.md)."""
    lines = [
        "# Feature dictionary and provenance (all 70 columns)",
        "",
        "Generated by `python -m evbattery.eda`. Missing % and ranges are from the full CSV.",
        "Decisions: **keep** = model input; **drop_id** / **drop_target** = never an input;",
        "**exclude_suspect** = composite score of unknown origin (ablation only);",
        "**drop_redundant** = exact formula of another kept column.",
        "",
        "| # | Column | Group | Meaning | Type | Missing % | Unique | Min | Median | Max | Decision |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:g}"
    for i, r in enumerate(dictionary.itertuples(index=False), 1):
        lines.append(f"| {i} | `{r.column}` | {r.group} | {r.meaning} | {r.type} | {r.missing_percent:.2f} | "
                     f"{r.n_unique:,} | {fmt(r.min)} | {fmt(r.median)} | {fmt(r.max)} | **{r.decision}** |")
    lines += ["", "## Decision reasons", ""]
    for decision, group in dictionary.groupby("decision", sort=False):
        lines.append(f"- **{decision}** ({len(group)}): " + "; ".join(sorted(set(group["reason"]))))
        lines.append(f"  Columns: {', '.join('`' + c + '`' for c in group['column'])}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures
def fig_target(df, path):
    plt = ps.plt
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar([ps.CLASS_NAMES[0], ps.CLASS_NAMES[1]], counts.values,
                  color=[ps.HEALTHY, ps.FAILURE], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v:,}  ({100 * v / counts.sum():.2f}%)", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", color=ps.INK)
    ax.set(ylabel="Number of batteries", title="Class balance: only 6.92% of batteries failed",
           ylim=(0, counts.max() * 1.15))
    ax.grid(axis="x", visible=False)
    ps.save(fig, path)


def fig_missing(df, path):
    plt = ps.plt
    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(miss.index[::-1], miss.values[::-1], color=ps.HEALTHY, height=0.6)
    for y, v in enumerate(miss.values[::-1]):
        ax.annotate(f"{v:.2f}%", (v, y), xytext=(4, 0), textcoords="offset points",
                    va="center", color=ps.INK_2, fontsize=9)
    ax.set(xlabel="Missing values (% of 20,000 rows)",
           title="15 columns with the most missing values (full CSV)", xlim=(0, miss.max() * 1.18))
    ax.grid(axis="y", visible=False)
    ps.save(fig, path)


def fig_distributions(train, path):
    plt = ps.plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, feat in zip(axes.ravel(), KEY_NUMERIC):
        vals = train[feat].dropna()
        bins = np.linspace(vals.min(), vals.max(), 36)
        for label in (0, 1):
            v = train.loc[train[TARGET] == label, feat].dropna()
            ax.hist(v, bins=bins, density=True, alpha=0.55, color=ps.CLASS_COLORS[label],
                    label=ps.CLASS_NAMES[label])
        ax.set(title=feat, ylabel="Density")
        ax.title.set_fontsize(10.5)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Key telemetry distributions by class (training split, n = 12,000)",
                 fontweight="bold", color=ps.INK)
    ps.save(fig, path)


def fig_boxplots(train, path):
    plt = ps.plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, feat in zip(axes.ravel(), KEY_NUMERIC):
        data = [train.loc[train[TARGET] == k, feat].dropna() for k in (0, 1)]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": ps.INK, "linewidth": 1.5})
        ax.set_xticks([1, 2], ["Healthy", "Failure"])
        for patch, k in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(ps.CLASS_COLORS[k])
            patch.set_alpha(0.75)
            patch.set_edgecolor(ps.CLASS_COLORS[k])
        ax.set_title(feat, fontsize=10.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Boxplots by class (training split; whiskers 1.5 IQR, outliers hidden)",
                 fontweight="bold", color=ps.INK)
    ps.save(fig, path)


def fig_correlation(train, path):
    plt = ps.plt
    cols = [c for c in HEATMAP_FEATURES if c in train.columns]
    corr = train[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(11, 9.4))
    im = ax.imshow(corr.values, cmap=ps.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if abs(v) >= 0.3 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.7 else ps.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Pearson r")
    ax.set_title("Correlation of selected numeric features (training split; |r| >= 0.30 printed)")
    ps.save(fig, path)


def fig_categorical(train, path):
    plt = ps.plt
    base = train[TARGET].mean()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax, col in zip(axes.ravel(), CATEGORICAL_PLOTS):
        g = train.groupby(train[col].fillna("(missing)"))[TARGET].agg(["mean", "size"]).sort_values("mean")
        ax.barh(g.index.astype(str), 100 * g["mean"], color=ps.HEALTHY, height=0.55)
        ax.axvline(100 * base, color=ps.INK_2, linestyle="--", linewidth=1)
        for y, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.annotate(f"{100 * m:.1f}%  (n={n:,})", (100 * m, y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=ps.INK_2,
                        bbox={"boxstyle": "square,pad=0.15", "fc": ps.SURFACE, "ec": "none"})
        ax.set(title=col, xlabel="Failure rate (%)", xlim=(0, 100 * g["mean"].max() * 1.75))
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Failure rate by category (training split; dashed line = overall {100 * base:.2f}%)",
                 fontweight="bold", color=ps.INK)
    ps.save(fig, path)


def fig_temperature_variance(df, path):
    plt = ps.plt
    v = df["temperature_variance"].dropna()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.hist(v, bins=60, color=ps.HEALTHY)
    ax.set(xlabel="temperature_variance", ylabel="Rows",
           title="temperature_variance is clipped at 0 and at 32.94 (full CSV)")
    top = int(np.isclose(v, v.max()).sum())
    zero = int((v == 0).sum())
    ax.annotate(f"{top:,} rows = 32.94", (v.max(), top), xytext=(-150, -10),
                textcoords="offset points", color=ps.INK_2, arrowprops={"arrowstyle": "->", "color": ps.MUTED})
    ax.annotate(f"{zero:,} rows = 0", (0, zero), xytext=(30, 20),
                textcoords="offset points", color=ps.INK_2, arrowprops={"arrowstyle": "->", "color": ps.MUTED})
    ps.save(fig, path)


# --------------------------------------------------------------------------- runner
def run(data_path: Path, outdir: Path) -> dict:
    ps.apply_style()
    df = load_dataset(data_path)
    train = split_data(df).train
    outdir.mkdir(parents=True, exist_ok=True)
    figs = outdir / "figures"
    figs.mkdir(exist_ok=True)

    facts = overview(df)
    (outdir / "overview.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    dictionary = column_dictionary(df)
    dictionary.to_csv(outdir / "column_dictionary.csv", index=False)
    (outdir / "FEATURE_DICTIONARY.md").write_text(dictionary_markdown(dictionary), encoding="utf-8")
    checks = plausibility_checks(df)
    checks.to_csv(outdir / "plausibility_checks.csv", index=False)
    df.describe().T.round(3).to_csv(outdir / "numeric_summary_full.csv")
    class_comparison(train).to_csv(outdir / "class_comparison_train.csv", index=False)
    categorical_failure_rates(train).to_csv(outdir / "categorical_failure_rates_train.csv", index=False)
    missingness_vs_target(train).to_csv(outdir / "missingness_vs_target_train.csv", index=False)
    (df.isna().mean() * 100).round(2).sort_values(ascending=False).rename("missing_percent") \
        .to_csv(outdir / "missing_by_column.csv")

    fig_target(df, figs / "01_target_distribution.png")
    fig_missing(df, figs / "02_missingness_top15.png")
    fig_distributions(train, figs / "03_distributions_by_class.png")
    fig_boxplots(train, figs / "04_boxplots_by_class.png")
    fig_correlation(train, figs / "05_correlation_heatmap.png")
    fig_categorical(train, figs / "06_failure_rate_by_category.png")
    fig_temperature_variance(df, figs / "07_temperature_variance_clipping.png")
    return {"overview": facts, "flags": checks[checks.status == "FLAG"]["check"].tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=Path("artifacts/eda"))
    args = p.parse_args()
    result = run(args.data, args.outdir)
    print(json.dumps(result, indent=2))
    print(f"Tables and figures saved under {args.outdir}")


if __name__ == "__main__":
    main()
