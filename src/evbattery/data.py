"""Strict CSV loading, reproducible holdout splits and stateless features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import IDENTIFIERS, RANDOM_STATE, REDUNDANT_EXACT, SUSPECT_SCORES, TARGET


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    calibration: pd.DataFrame
    threshold: pd.DataFrame
    test: pd.DataFrame


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find CSV: {path}")
    frame = pd.read_csv(path, low_memory=False)
    expected = {TARGET, *IDENTIFIERS}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported")
    if frame.empty:
        raise ValueError("CSV has no rows")
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 and 1, without nulls")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Both healthy and failure cases are needed")
    if frame[list(IDENTIFIERS)].isna().any().any():
        raise ValueError("Identifiers have missing values")
    if any(frame[col].duplicated().any() for col in IDENTIFIERS):
        raise ValueError("Duplicate vehicle/battery identifiers: inspect record grain")
    if frame.duplicated().any():
        raise ValueError("Exact duplicate records: investigate before training")
    return frame


def split_data(frame: pd.DataFrame, seed: int = RANDOM_STATE) -> Splits:
    """60% train, 10% calibration, 10% threshold, 20% untouched test."""
    if len(frame) < 100:
        raise ValueError("Insufficient rows to establish robust splits")
    development, test = train_test_split(
        frame, test_size=0.20, stratify=frame[TARGET], random_state=seed
    )
    train, cal_and_thr = train_test_split(
        development, test_size=0.25, stratify=development[TARGET], random_state=seed
    )
    calibration, threshold = train_test_split(
        cal_and_thr, test_size=0.50, stratify=cal_and_thr[TARGET], random_state=seed
    )
    for subset in (train, calibration, threshold, test):
        if subset[TARGET].nunique() != 2:
            raise ValueError("A split is missing a class; need more failure samples")
    return Splits(train, calibration, threshold, test)


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        answer = a / b.where(b > 0, np.nan)
    return answer.replace([np.inf, -np.inf], np.nan)


def make_features(raw: pd.DataFrame, include_suspect_scores: bool = False,
                  drop_redundant: bool = True) -> pd.DataFrame:
    """Only per-row transformations: no fitting, no target access, no global statistics.

    drop_redundant=True (default since the Member A audit) removes columns that are
    exact formulas of other columns (config.REDUNDANT_EXACT). The 3-fold CV ablation
    showed no measurable loss from removing them; see docs/MEMBER_A_DATA_AUDIT.md.
    """
    banned = {TARGET, *IDENTIFIERS}
    if not include_suspect_scores:
        banned.update(SUSPECT_SCORES)
    if drop_redundant:
        banned.update(REDUNDANT_EXACT)
    x = raw.drop(columns=list(banned), errors="ignore").copy()
    additions = {
        "cell_temp_max_minus_avg": ("cell_temperature_max", "cell_temperature_avg"),
        "ambient_temp_range": ("maximum_temperature", "minimum_temperature"),
    }
    for new_col, (a, b) in additions.items():
        if a in x.columns and b in x.columns:
            x[new_col] = pd.to_numeric(x[a], errors="coerce") - pd.to_numeric(x[b], errors="coerce")
    if {"cycle_count", "vehicle_age_years"}.issubset(x):
        x["cycles_per_year"] = _safe_divide(x["cycle_count"], x["vehicle_age_years"])
    if {"odometer_km", "cycle_count"}.issubset(x):
        x["km_per_cycle"] = _safe_divide(x["odometer_km"], x["cycle_count"])
    if {"fast_charge_ratio", "average_charge_power_kw"}.issubset(x):
        x["fast_charge_power_stress"] = (
            pd.to_numeric(x["fast_charge_ratio"], errors="coerce")
            * pd.to_numeric(x["average_charge_power_kw"], errors="coerce")
        )
    fault_cols = [c for c in ("previous_faults", "BMS_warning_count", "sensor_fault_count") if c in x]
    if len(fault_cols) == 3:
        # Do not falsely claim zero faults when *all* component values are absent.
        x["prior_fault_and_warning_count"] = x[fault_cols].sum(axis=1, min_count=1)
    if "last_service_days" in x:
        days = pd.to_numeric(x["last_service_days"], errors="coerce")
        x["service_overdue_180d"] = (days > 180).astype(float).where(days.notna())
    for col in x.select_dtypes(include=np.number).columns:
        x[col] = x[col].replace([np.inf, -np.inf], np.nan)
    # No feature may itself be a nested object or python list.
    if any(x[c].map(lambda z: isinstance(z, (dict, list, set))).any() for c in x.select_dtypes('object')):
        raise ValueError("Nested values are not supported as input features")
    return x
