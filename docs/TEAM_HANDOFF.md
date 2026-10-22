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

## Member B — supervised models, CV and tuning (plan days 4–6, 9) — COMPLETED

1. Preprocessing experiments completed (`artifacts/member_b/preprocessing_comparison.csv`): confirmed median imputation with missing indicators matches KNN imputation without quadratic latency penalty.
2. Supervised model comparison leaderboard generated across 7 classical algorithms + dummy + KNN (`artifacts/member_b/model_cv_leaderboard.csv`).
3. Dedicated leakage-safe entrypoint created: `python -m evbattery.member_b --data data/ev_battery.csv --outdir artifacts/member_b`. It strictly operates on the 60% training partition; calibration, threshold, and test partitions are never accessed.
4. Hyperparameter tuning completed on training partition via `RandomizedSearchCV` (`artifacts/member_b/tuning_results.json`).
5. Handoff model artifact exported: `artifacts/member_b/best_base_model.joblib` and selection metadata `artifacts/member_b/selected_model.json`.
6. Full documentation delivered: `docs/MEMBER_B_MODELING_REPORT.md`, `docs/MEMBER_B_LEARNING_GUIDE.md`, `docs/MEMBER_B_HANDOFF.md`, and automated test suite `tests/test_member_b.py`.


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
