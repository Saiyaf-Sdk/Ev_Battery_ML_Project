"""Tests for Member A's profiling, EDA and leakage-audit code.

They use a small synthetic frame with the same 70 columns as the real CSV,
so they run without the dataset (and in CI) in a few seconds.
"""
import numpy as np
import pandas as pd

from evbattery.config import IDENTIFIERS, REDUNDANT_EXACT, SUSPECT_SCORES, TARGET
from evbattery.data import make_features
from evbattery.eda import categorical_failure_rates, missingness_vs_target, overview, plausibility_checks
from evbattery.feature_catalog import CATALOG, decision_for
from evbattery.leakage_audit import split_check, univariate_screen

CATEGORICAL = {
    "vehicle_brand": ["Tesla", "BYD", "Kia"], "vehicle_model": ["Model 3", "Seal", "EV6"],
    "vehicle_type": ["SUV", "Sedan"], "battery_manufacturer": ["CATL", "LG"],
    "battery_chemistry": ["NMC", "LFP"], "drive_type": ["AWD", "FWD"],
    "fleet_or_private": ["Fleet", "Private"], "terrain_type": ["Flat", "Hilly"],
}


def full_frame(n=400, seed=0):
    """Synthetic data with every real column and internally consistent physics."""
    rng = np.random.default_rng(seed)
    d = {c: rng.uniform(0, 1, n) for c, (g, _) in CATALOG.items() if g not in ("identifier", "target")}
    for c, levels in CATEGORICAL.items():
        d[c] = rng.choice(levels, n)
    d["vehicle_model"] = np.array(["Model 3", "Seal", "EV6"])[
        pd.Series(d["vehicle_brand"]).map({"Tesla": 0, "BYD": 1, "Kia": 2}).to_numpy()]
    d["vehicle_type"] = np.array(["SUV", "Sedan", "SUV"])[
        pd.Series(d["vehicle_brand"]).map({"Tesla": 0, "BYD": 1, "Kia": 2}).to_numpy()]
    health = rng.uniform(60, 100, n)
    cap = rng.uniform(40, 120, n)
    age = rng.uniform(1, 10, n)
    d.update({
        "battery_health_percent": health, "capacity_loss_percent": 100 - health,
        "battery_capacity_kwh": cap, "remaining_capacity": cap * health / 100,
        "vehicle_age_years": age, "manufacturing_year": np.round(2026 - age),
        "state_of_health": health + rng.normal(0, 1, n).clip(-3, 3),
        "cell_temperature_avg": rng.uniform(10, 40, n),
        "average_ambient_temperature": rng.uniform(0, 30, n),
        "cycle_count": rng.integers(100, 2000, n).astype(float),
        "odometer_km": rng.uniform(1000, 200000, n),
        "pack_voltage": 96 * rng.uniform(3.4, 3.8, n),
        "temperature_variance": rng.uniform(0, 30, n),
        "average_charging_time": rng.uniform(10, 500, n),
        "last_service_days": rng.uniform(1, 900, n),
    })
    d["cell_voltage_avg"] = d["pack_voltage"] / 96
    d["cell_temperature_max"] = d["cell_temperature_avg"] + rng.uniform(0, 10, n)
    d["maximum_temperature"] = d["average_ambient_temperature"] + rng.uniform(0, 10, n)
    d["minimum_temperature"] = d["average_ambient_temperature"] - rng.uniform(0, 10, n)
    d["fast_charge_ratio"] = rng.uniform(0, 0.5, n)
    d["slow_charge_ratio"] = rng.uniform(0, 0.5, n)
    d["highway_driving_ratio"] = rng.uniform(0, 1, n)
    d["city_driving_ratio"] = 1 - d["highway_driving_ratio"]
    for c in ("humidity", "charge_efficiency", "discharge_efficiency", "state_of_charge",
              "depth_of_discharge", "cooling_system_health", "regenerative_braking_usage"):
        d[c] = rng.uniform(5, 95, n)
    frame = pd.DataFrame(d)
    frame.insert(0, "vehicle_id", [f"v{i}" for i in range(n)])
    frame.insert(1, "battery_serial", [f"b{i}" for i in range(n)])
    # failure depends on health, so a real signal exists
    frame[TARGET] = (health + rng.normal(0, 8, n) < 70).astype(int)
    return frame


def test_catalog_covers_every_column_and_decisions_are_consistent():
    frame = full_frame(50)
    assert set(frame.columns) == set(CATALOG), "every CSV column needs a provenance entry"
    assert len(CATALOG) == 70
    assert decision_for(TARGET)[0] == "drop_target"
    assert all(decision_for(c)[0] == "drop_id" for c in IDENTIFIERS)
    assert all(decision_for(c)[0] == "exclude_suspect" for c in SUSPECT_SCORES)
    assert all(decision_for(c)[0] == "drop_redundant" for c in REDUNDANT_EXACT)


def test_make_features_drops_redundant_by_default_and_can_keep_them():
    frame = full_frame(60)
    default = make_features(frame)
    kept = make_features(frame, drop_redundant=False)
    assert not set(REDUNDANT_EXACT) & set(default.columns)
    assert set(REDUNDANT_EXACT) <= set(kept.columns)
    assert not {TARGET, *IDENTIFIERS, *SUSPECT_SCORES} & set(default.columns)


def test_overview_counts():
    frame = full_frame(200)
    frame.loc[0:4, "humidity"] = np.nan
    facts = overview(frame)
    assert facts["n_rows"] == 200 and facts["n_columns"] == 70
    assert facts["healthy"] + facts["failures"] == 200
    assert facts["missing_cells"] == 5
    assert facts["exact_duplicate_rows"] == 0


def test_plausibility_checks_pass_on_clean_data_and_catch_injected_errors():
    clean = plausibility_checks(full_frame(200))
    must_pass = ["cell_temperature_avg > cell_temperature_max",
                 "fast_charge_ratio + slow_charge_ratio > 1",
                 "highway_driving_ratio + city_driving_ratio > 1"]
    assert (clean.set_index("check").loc[must_pass, "status"] == "PASS").all()

    bad = full_frame(200)
    bad.loc[0, "cell_temperature_avg"] = bad.loc[0, "cell_temperature_max"] + 5
    bad.loc[1, "humidity"] = 140
    bad.loc[2, ["vehicle_brand", "vehicle_model"]] = ["BYD", "Model 3"]  # model under two brands
    res = plausibility_checks(bad).set_index("check")
    assert res.loc["cell_temperature_avg > cell_temperature_max", "rows_flagged"] == 1
    assert res.loc["humidity outside 0-100", "rows_flagged"] == 1
    assert any(c.startswith("vehicle_model listed under >1 brand") and res.loc[c, "status"] == "FLAG"
               for c in res.index)


def test_missing_values_do_not_count_as_violations():
    frame = full_frame(100)
    frame.loc[:10, ["cell_temperature_avg", "cell_temperature_max"]] = np.nan
    res = plausibility_checks(frame).set_index("check")
    assert res.loc["cell_temperature_avg > cell_temperature_max", "rows_flagged"] == 0


def test_target_tables_use_both_classes():
    frame = full_frame(300)
    frame.loc[0:20, "battery_chemistry"] = None
    rates = categorical_failure_rates(frame)
    assert "(missing)" in set(rates.loc[rates.column == "battery_chemistry", "level"])
    assert rates["failure_rate"].between(0, 1).all()
    miss = missingness_vs_target(frame)
    assert "battery_chemistry" in set(miss["column"])


def test_univariate_screen_never_scores_target_or_ids_and_finds_signal():
    screen = univariate_screen(full_frame(400))
    assert not {TARGET, *IDENTIFIERS} & set(screen["feature"])
    assert screen["strength"].between(0.5, 1).all()
    # failure was generated from health, so a health column must rank first
    assert screen.iloc[0]["feature"] in {"battery_health_percent", "capacity_loss_percent",
                                         "state_of_health", "remaining_capacity"}


def test_split_check_has_no_overlap_and_keeps_prevalence():
    frame = full_frame(500)
    sc = split_check(frame)
    assert sc["rows"].sum() == 500
    assert (sc["vehicle_id_overlap_between_splits"] == 0).all()
    overall = frame[TARGET].mean()
    assert (sc["failure_prevalence"] - overall).abs().max() < 0.03
