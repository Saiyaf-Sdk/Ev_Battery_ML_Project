# Member A — Data Profiling, EDA and Leakage / Feature-Set Audit

**Owner:** Member A (Saiyaf) · **Covers plan days:** 1–3 (+ the composite-score sensitivity study from Day 9)
**Dataset:** `ev battery_failure prediction Dataset (1).csv` — SHA-256 `ba0be61cab0260bbc120668fba20fad1185c79978dd0b4111a722214bdb02e29`
**Reproduce everything in this file:**

```bash
python -m evbattery.quality       --data data/ev_battery.csv
python -m evbattery.eda           --data data/ev_battery.csv --outdir artifacts/eda
python -m evbattery.leakage_audit --data data/ev_battery.csv --outdir artifacts/leakage_audit
```

All numbers below were produced by those commands (random seed 42). Sections 1–3 can be pasted into Report §3 "Dataset and EDA"; sections 4–6 into Report §4 "Methods".

> **Rule we followed:** label-free facts (size, missing values, ranges) use the full CSV. Anything that compares healthy vs failed batteries uses **only the 60 % training split (12,000 rows)**. The 20 % test split was never opened by Member A.

---

## 1. Dataset profile

| Fact | Value |
|---|---:|
| Rows × columns | 20,000 × 70 (60 numeric, 10 text) |
| Healthy (0) / Failure (1) | 18,616 / **1,384 (6.92 %)** |
| Accuracy of a model that always says "healthy" | **93.08 %** (with 0 % failure recall) |
| Missing cells | 53,344 (**3.81 %** of the table) |
| Columns with missing values | 67 of 70 (only `vehicle_id`, `battery_serial`, `battery_failure` are complete) |
| Missing per column | between 3.05 % and 4.94 % |
| Rows with at least one missing value | 18,663 (**93.31 %**) |
| Exact duplicate rows | 0 (also 0 when IDs are ignored) |
| `vehicle_id` / `battery_serial` unique | yes / yes → one row per battery, no repeated history |

**What this means**
- This is **binary classification** (`battery_failure` is 0 or 1), and a **rare-event** one. Accuracy is useless as a headline metric: 93 % is free. Use average precision, recall, precision, F2 (Member C).
- Missing values are spread almost evenly (3–5 % in every column), which looks like values were removed at random by the data generator. Because 93 % of rows have a gap somewhere, **deleting incomplete rows is not an option** — we must impute (inside training folds only).
- No row repeats and every battery appears once, so a random stratified split is valid. (Real fleet data with repeated readings per vehicle would need a grouped/time-based split.)

Figures: `docs/figures/member_a/01_target_distribution.png`, `02_missingness_top15.png`.

## 2. Data-quality and physical plausibility checks

36 rules were checked (`artifacts/eda/plausibility_checks.csv`). **30 passed** — e.g. no cell average above cell maximum, ambient average always between min and max, all ratios in 0–1, all percentages in 0–100, fast + slow charge share ≤ 1, no negative distances/voltages, remaining capacity never above nominal. Negative ambient and cell temperatures exist but are physically plausible (winter).

Six issues were **flagged**:

| Flag | Rows | Interpretation / decision |
|---|---:|---|
| `highway_driving_ratio + city_driving_ratio > 1` | 9,254 (46 %) | The two shares were generated independently (sum ranges 0–1.99). Physically impossible, but we **keep both** and document it: we cannot know which one is wrong. |
| `temperature_variance` exactly 32.94 (its maximum) | 8,746 (44 %) | Values are **clipped** at an upper limit. |
| `temperature_variance` exactly 0 | 2,573 (13 %) | Clipped at the lower limit. Together: ~57 % of this column is saturated, so it carries less detail than it seems. Kept. |
| `average_charging_time` = 600 min (max) | 884 (4.4 %) | Clipped at 600. Kept. |
| `vehicle_model` "BE 6" listed under both Mahindra and BYD | 713 | Labelling error in the generator (BE 6 is a Mahindra model). Kept; tree/one-hot models are unaffected. |
| `vehicle_type` varies inside every model (all 6 body types per model) | 18,571 | e.g. a "Mustang Mach-E" labelled "Truck". Body type was assigned randomly → `vehicle_type` is close to noise. Kept (harmless), but do **not** interpret it physically. |

These flags are strong evidence that the dataset is **synthetic** (simulated), which is exactly how the project is framed.

Figure: `07_temperature_variance_clipping.png`.

## 3. What separates failed from healthy batteries (training split)

Strongest raw signals (standardised mean difference, SMD; |SMD| ≈ 1 is a large difference):

| Feature | Healthy median | Failure median | SMD |
|---|---:|---:|---:|
| battery_health_percent | 83.7 % | 71.2 % | −1.32 |
| state_of_health | 83.6 % | 71.1 % | −1.30 |
| internal_resistance | 0.418 | 0.630 | +1.21 |
| cell_temperature_max | 39.6 °C | 54.7 °C | +1.19 |
| cell_temperature_avg | 26.3 °C | 38.8 °C | +1.08 |
| charge_efficiency | 96.3 % | 95.1 % | −1.05 |
| discharge_efficiency | 95.5 % | 94.0 % | −1.03 |
| vehicle_age_years | 6.2 | 9.3 | +0.98 |

Failed batteries are older, hotter, have lower health, higher internal resistance and lower efficiency — physically sensible. Fault history (`previous_faults`, `BMS_warning_count`) helps moderately. Charging, driving and environment columns show almost **no** individual difference (single-feature AUC 0.50–0.54).

Categorical columns matter little: failure rates by chemistry range 4.9 % (LTO) to 7.7 % (missing); by vehicle type 6.5–7.9 %; terrain 6.6–7.4 %.

**Is "missing" itself informative?** Barely: the largest gap is `previous_faults` (4.4 % failure rate when missing vs 7.0 % when present, only 367 rows). The starter pipeline already adds missing-indicator columns, so the models can use this if it helps.

**Correlation structure** (training split, `high_correlation_pairs.csv`): `battery_health_percent`–`capacity_loss_percent` |r| = 1.00; `manufacturing_year`–`vehicle_age_years` 0.995; `battery_health_percent`–`state_of_health` 0.99; `cell_voltage_std`–`voltage_imbalance` 0.99; ambient avg/max/min 0.95–0.98; `thermal_runaway_risk`–`thermal_health_score` 0.99. Correlation ≠ causation and ≠ leakage — it means redundancy.

Figures: `03_distributions_by_class.png`, `04_boxplots_by_class.png`, `05_correlation_heatmap.png`, `06_failure_rate_by_category.png`.

## 4. Splitting (verified)

| Split | Rows | % | Failures | Prevalence | Used for |
|---|---:|---:|---:|---:|---|
| Train | 12,000 | 60 | 830 | 0.0692 | fitting + 3-fold CV (Member B) |
| Calibration | 2,000 | 10 | 139 | 0.0695 | sigmoid calibration (Member C) |
| Threshold | 2,000 | 10 | 138 | 0.0690 | choosing the alarm cut-off (Member C) |
| Test | 4,000 | 20 | 277 | 0.0693 | **one final evaluation only** |

Stratification keeps the failure rate ≈ 6.9 % in every split, and no `vehicle_id` appears in two splits (overlap = 0).

```
20,000 rows ──stratified, seed 42──► 80 % development ─┬─ 60 % train   (fit, CV, tuning, EDA decisions)
                                                       ├─ 10 % calibration
                                                       └─ 10 % threshold
                                   └► 20 % TEST (locked until the very end)
```

## 5. Leakage audit

**Data leakage** = the model gets information during training that it would not have at prediction time (or that is derived from the answer), so offline scores look better than reality.

| Risk | Where | Control |
|---|---|---|
| Identifiers | `vehicle_id`, `battery_serial` | Dropped. Unique per row → a model could only memorise them. |
| Target in the inputs | `battery_failure` | Dropped from X by `make_features()`. |
| Composite scores with unknown origin | 8 columns: `thermal_runaway_risk`, `thermal_health_score`, `battery_stress_index`, `aging_score`, `charging_quality_score`, `driving_stress_score`, `maintenance_score`, `predicted_remaining_life_cycles` | **Excluded** from the baseline; studied only in the ablation below. |
| Preprocessing fitted on all data | imputers, scaler, encoder | Inside a scikit-learn `Pipeline`, refit in every CV fold. |
| Test-set peeking | EDA, feature choice | All target-based EDA and audits on the train split only. |
| Resampling before split | SMOTE etc. (Member C) | Must happen inside training folds only. |

**Single-feature screen** (`univariate_screen.csv`, figure `08_univariate_auc.png`): the strongest single column is a composite score, `thermal_runaway_risk` (AUC 0.850), followed by `thermal_health_score` 0.845 and `predicted_remaining_life_cycles` 0.823. The best raw column is `battery_health_percent` (0.822). **No column reaches 0.90**, so there is no obvious "disguised label" among the raw telemetry.

**Can the composites be rebuilt from raw telemetry?** (`composite_reconstruction.csv`, gradient-boosted regression, 3-fold CV R² on train): `aging_score` 0.96, `thermal_runaway_risk` 0.91, `predicted_remaining_life_cycles` 0.89, `thermal_health_score` 0.88, `maintenance_score` 0.88, `battery_stress_index` 0.87, `driving_stress_score` 0.86, `charging_quality_score` 0.75. So they are mostly recombinations of raw columns — but not completely.

**Ablation** (same 3 stratified folds on train, average precision mean ± SD; `feature_set_ablation.csv`, figure `09_feature_set_ablation.png`):

| Feature set | # features | Logistic reg. | Random forest | Hist. gradient boosting |
|---|---:|---:|---:|---:|
| A — starter baseline (no composites) | 66 | 0.721 ± 0.042 | 0.630 ± 0.062 | 0.731 ± 0.038 |
| **B — recommended (no composites, no duplicates)** | **63** | **0.719 ± 0.042** | **0.639 ± 0.058** | **0.729 ± 0.034** |
| C — with composites | 74 | 0.768 ± 0.031 | 0.663 ± 0.048 | 0.778 ± 0.004 |
| D — with composites, no duplicates | 71 | 0.766 ± 0.032 | 0.682 ± 0.044 | 0.777 ± 0.004 |

No-skill AP = prevalence = 0.069.

**Interpretation**
1. Adding the composite scores raises AP by about **+0.03 to +0.05** for every model. Since they are ~75–96 % reconstructable from raw columns, the extra lift must come from information *not* in the raw columns we have. One plausible explanation: the scores were computed by the generator **before** random missing values were injected, or directly from the hidden failure mechanism. We cannot tell which — and that uncertainty is the reason to exclude them.
2. This is **sensitivity, not proof of leakage**. We report it honestly and keep the safer baseline.
3. Removing the three exact duplicates changes AP by −0.002 to +0.009 without composites (A→B) and at most +0.019 with them (random forest, C→D) — every change is smaller than one standard deviation, i.e. no measurable difference. We drop them for a simpler, more interpretable model (cleaner coefficients and permutation importances).
4. The starter baseline number is reproduced exactly (logistic regression, set A: 0.7207 — same as `docs/BASELINE_RESULTS.md`), which confirms the audit code uses the same pipeline.

## 6. Final feature-set decision (handed to Members B, C, D)

| Decision | Columns | Count |
|---|---|---:|
| Keep — raw telemetry & specification | everything else | 56 |
| Add — engineered per-row features (`make_features`) | `cell_temp_max_minus_avg`, `ambient_temp_range`, `cycles_per_year`, `km_per_cycle`, `fast_charge_power_stress`, `prior_fault_and_warning_count`, `service_overdue_180d` | 7 |
| Drop — identifiers | `vehicle_id`, `battery_serial` | 2 |
| Drop — target | `battery_failure` | 1 |
| Exclude — suspect composites (ablation only) | 8 listed above | 8 |
| Drop — exact duplicates | `capacity_loss_percent` (=100 − health), `remaining_capacity` (= capacity × health/100), `manufacturing_year` (= 2026 − age) | 3 |

**Model input = 63 features** (55 numeric, 8 categorical). Machine-readable list: `artifacts/leakage_audit/feature_set.json`. Every column's provenance and decision: `docs/FEATURE_DICTIONARY.md`.

Code change: `make_features()` now drops the three duplicates by default (`drop_redundant=True`; set `False` to reproduce the old 66-feature baseline). The API still accepts the duplicate fields in requests and ignores them.

## 7. Limitations (for Report §11)

- **Synthetic data**: clipped columns, impossible ratio sums and random body types show the rows were simulated; results do not transfer to real EVs.
- **No timestamps / no failure horizon**: we cannot claim "predicts failure X days early". This is a cross-sectional classifier.
- **Unknown provenance of composites**: excluded, but we cannot fully rule out that some *raw* columns were also generated using the label. Only the dataset author could confirm.
- **Univariate screen is optimistic for categorical columns** (their AUC uses category failure rates computed on the same training rows). It is a screening tool, not a model.
- EDA comparisons are descriptive; correlations and class differences are associations, not causes.
