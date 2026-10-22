# Team Handoff — from Member A to Members B, C and D

Member A's part (data profiling, EDA, leakage and feature-set audit) is complete. Read `docs/MEMBER_A_DATA_AUDIT.md` first (10 minutes). This page lists what you can rely on and what each of you does next.

## What everyone can rely on

| Item | Agreed value |
|---|---|
| Dataset | 20,000 × 70, 1,384 failures (6.92 %), 3.81 % missing cells, 0 duplicates |
| Dataset location | `data/ev_battery.csv` (not in Git — copy it in yourself; see `data/README.md`) |
| Split | `split_data()` → 60 % train / 10 % calibration / 10 % threshold / 20 % test, stratified, seed 42 — **do not change** |
| Features | `make_features(raw)` with defaults → **63 inputs** (55 numeric, 8 categorical); list in `artifacts/leakage_audit/feature_set.json` |
| Excluded | IDs, target, 8 composite scores, 3 exact duplicates (`capacity_loss_percent`, `remaining_capacity`, `manufacturing_year`) |
| Preprocessing | `modeling.pipe_for()` — median imputation + missing indicators + scaling; most-frequent + one-hot (`handle_unknown="ignore"`). Always inside the Pipeline. |
| Primary metric | average precision (AP). No-skill AP = 0.069. Report recall, precision, F2 too. |
| Reference CV AP (train, 3-fold, 63 features) | logistic regression 0.719 · random forest 0.639 · hist. gradient boosting 0.729 |

**Golden rules:** never open the test split until the final evaluation; fit every imputer/scaler/sampler/selector inside CV training folds; never quote smoke-test (`--fast`) numbers.

## Member B — supervised models, CV and tuning (plan days 4–6, 9)

1. `python -m evbattery.experiments --data data/ev_battery.csv --max-rows 2500` (median vs KNN imputation, L1 selection, correlation pruning).
2. `python -m evbattery.train --data data/ev_battery.csv --outdir artifacts` — the 7-model CV leaderboard (`model_cv_leaderboard.csv`). Build the comparison table (algorithm, interpretable?, needs scaling?, nonlinear?, AP mean/SD, ROC AUC).
3. Tune the CV winner: `python -m evbattery.tune --data data/ev_battery.csv --model <winner> --n-iter 10`, then retrain with `--tuning-file`.
4. Tip from the audit: random forest is weakest (AP ≈ 0.64) and boosting/logistic are close (≈ 0.72–0.73). Strongly correlated groups (health / SoH, ambient avg / max / min, cell_voltage_std / voltage_imbalance) remain — mention this when interpreting coefficients.

## Member C — imbalance, calibration, thresholds, evaluation (plan days 7, 8, 10)

1. `python -m evbattery.imbalance --data data/ev_battery.csv` (none vs class weights vs undersampling vs SMOTENC — inside folds).
2. Use Member B's final `artifacts/model.joblib`: calibration plot, threshold chosen on the **threshold** split (F2), then the **one** test evaluation (`test_metrics.json`).
3. `python -m evbattery.costs --data data/ev_battery.csv --model artifacts/model.joblib` and `python -m evbattery.charts --mode final ...`.
4. Note from the audit: 8 categorical columns exist, so use SMOTENC, not plain SMOTE.

## Member D — unsupervised, explanation, API, integration/testing (plan days 11–13)

1. `python -m evbattery.unsupervised --data data/ev_battery.csv` (PCA, K-means, Isolation Forest — no labels in fit).
2. `python -m evbattery.explain --data data/ev_battery.csv --model artifacts/model.joblib` (permutation importance). Expect importance shared between correlated pairs — the audit already removed exact duplicates to reduce this.
3. API demo: `python -m evbattery.example_request ...`, then `uvicorn evbattery.api:app --host 127.0.0.1 --port 8000`. Requests may still include the 3 dropped duplicate fields; the API ignores them.
4. Own `python -m pytest` for the whole repo; keep `tests/test_member_a.py` green.

## Open questions Member A could not resolve (mention in the report)

- Where do the 8 composite scores come from, and were they computed before missing values were injected? (Explains their extra +0.03–0.05 AP.)
- `highway_driving_ratio + city_driving_ratio` exceeds 1 in 46 % of rows; `temperature_variance` is clipped in 57 % of rows.
- The proposal PDF in the shared folder is empty (0 bytes); please re-share it so references can be copied into the report.
