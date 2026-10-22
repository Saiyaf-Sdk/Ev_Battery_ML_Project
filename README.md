# EV Battery Failure — Machine Learning Mini Project

**Target:** `battery_failure` (`0` = healthy, `1` = failure) from 20,000 **simulated** fleet-telemetry rows. This is a research/teaching project, **not** a real EV safety monitoring product. No observation timestamp or failure horizon is provided, so the dataset does **not** validate early warning lead time.

## Project contents

```
README.md
requirements.txt, pyproject.toml, .gitignore, .env.example
src/evbattery/
  config.py         # guarded target/identifier/composite score exclusions
  data.py           # data validation, 60/10/10/20 stratified split, features
  quality.py        # missing values, correlations, data-quality profile
  modeling.py       # 7 efficient classifiers + optional KNN + dummy, CV inside train only
  evaluation.py     # precision, recall, F1/F2, AP, ROC, MCC, Brier score
  train.py          # model selection + separate calibration + alarm tuning + one test
  experiments.py    # imputation, L1 selection, correlation-pruning comparison
  tune.py           # optional train-CV RandomizedSearchCV hyperparameter search
  imbalance.py      # baseline, weights, undersampling, categorical-aware SMOTENC
  unsupervised.py   # PCA, K-means, IsolationForest, unsupervised fitting
  explain.py        # validation-set permutation importance
  leakage_study.py  # training-CV risk-score ablation
  charts.py         # EDA and frozen-test visualizations
  costs.py          # hypothetical warning costs, threshold validation only
  example_request.py # example API payload generator
  api.py            # authenticated FastAPI educational demonstration

artifacts/           # generated locally, ignored by Git

tests/               # automated assertions
docs/                # 14-day learning schedule, interview bank, report template
```

## 1. Setup (Python 3.11+ recommended)

Unzip the folder. Open a terminal in its root. Place the existing CSV anywhere convenient: **do not edit the original**.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest
```

## 2. Dataset quality checks

Use your actual path to the uploaded CSV in place of `PATH/TO/ev_battery.csv` below. Quotes are important when your filename contains spaces.

```bash
python -m evbattery.quality --data "PATH/TO/ev_battery.csv"
```

Outputs: `artifacts/data_profile.json`.

## 3. Working model — from data to evaluation

```bash
# Quick functional smoke test; DO NOT quote its results in your report.
python -m evbattery.train --data "PATH/TO/ev_battery.csv" --outdir artifacts

# Resource-conscious comparison: rank all default algorithms on 4k TRAIN-only rows,
# still fit winning model on 12k train rows and evaluate on full 4k holdout.
# Document the smaller CV sample explicitly when reporting:
python -m evbattery.train --data "PATH/TO/ev_battery.csv" --outdir artifacts/resource_limited --cv-max-rows 4000/smoke --fast

# Fast first checkpoint on ALL 20k rows: logistic regression + dummy only.
# Do not present this as the required 5+-algorithm comparison.
python -m evbattery.train --data "PATH/TO/ev_battery.csv" --outdir artifacts/baseline --baseline-only

# Full, CV-ranked classical ML comparison, once you are ready:
python -m evbattery.train --data "PATH/TO/ev_battery.csv" --outdir artifacts
```

**Full training** holds out 20% test data. The optional `--cv-max-rows` reduces only the computational sample used for model ranking; that should be disclosed in the report. It uses stratified 60% fitting, 10% independent Platt/sigmoid calibration, 10% F2-based decision-threshold choice and 20% untouched final test. The train-only 3-fold CV ranks 7 classical algorithms (KNN optional due to computational expense) using **average precision** (scikit-learn AP, which is not a trapezoidal PR-AUC estimate), plus a dummy reference. Class-imbalance comparisons and imputation variations are separate development experiments. The default final model does not automatically incorporate a sampler winner: if that experiment proves helpful, implement its winning method in the final CV training pipeline, then rerun the documented procedure **without selecting on final test outcomes**.

Generated outputs: `model_cv_leaderboard.csv`, `threshold_validation_metrics.json`, `test_metrics.json`, `split_counts.csv`, `metadata.json`, `model.joblib`.

## 4. Other experiments — use development folds, not holdout test

```bash
python -m evbattery.experiments --data "PATH/TO/ev_battery.csv" --max-rows 2500
# Run after CV identifies the most promising model (e.g., adaboost).
python -m evbattery.tune --data "PATH/TO/ev_battery.csv" --model adaboost --n-iter 10
# When the tuning-file model matches the baseline CV winner, train again using:
# python -m evbattery.train --data "PATH/TO/ev_battery.csv" --outdir artifacts --tuning-file artifacts/tuned_params.json
python -m evbattery.imbalance --data "PATH/TO/ev_battery.csv"
python -m evbattery.unsupervised --data "PATH/TO/ev_battery.csv"
python -m evbattery.explain --data "PATH/TO/ev_battery.csv" --model artifacts/model.joblib
python -m evbattery.leakage_study --data "PATH/TO/ev_battery.csv"
python -m evbattery.costs --data "PATH/TO/ev_battery.csv" --model artifacts/model.joblib
python -m evbattery.charts --data "PATH/TO/ev_battery.csv" --mode eda
python -m evbattery.charts --data "PATH/TO/ev_battery.csv" --mode final --model artifacts/model.joblib --outdir artifacts/final_figures
```

The capped `experiments.py` sample supports faster KNN imputation experiments; treat those as preliminary, not a full-dataset conclusion. The unsupervised models never see labels during fit. Label-based enrichment is analyzed afterward for research.

### Model inspection

`artifacts/test_metrics.json` is the final holdout **only once**. Prefer average precision, recall, F2, precision and false positives, not accuracy alone. The split counts and class baseline should always be included. A positive result on a simulated dataset cannot prove performance on physical vehicles.

## 5. Demonstration API (LOCAL DEVELOPMENT ONLY)

After full training, generate a sample request:

```bash
python -m evbattery.example_request --data "PATH/TO/ev_battery.csv"
# Linux/macOS example — replace this placeholder with a real secret kept outside Git:
export EV_API_KEY='your-long-random-secret'
export EV_MODEL_PATH='artifacts/model.joblib'
uvicorn evbattery.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "X-API-Key: your-long-random-secret" \
  -H "Content-Type: application/json" \
  --data-binary @artifacts/example_request.json
```

For Windows PowerShell: `$env:EV_API_KEY="..."`; `$env:EV_MODEL_PATH="artifacts/model.joblib"`. The API intentionally refuses smoke-test artifacts, optional composite-score experiments, missing API keys and unknown features. A prediction is an **experimental score**, not a certified physical failure probability.

## Security / reliability guidance

- No real EV, BMS or vehicle access should be connected. Bind `127.0.0.1` only. **Do not deploy publicly** without TLS termination, a trusted identity provider, request body limits, rate limiting, input range rules, observability, a dependency vulnerability review and risk assessment.
- `joblib` uses Python pickle. **NEVER load model files uploaded by clients or taken from untrusted sources.** Load only your locally trained artifact.
- Code is checked by automated tests, but crash-free or secure production operation cannot be guaranteed without environment-specific testing, threat modeling and monitoring.
- Keep secrets in environment variables and raw fleet telemetry outside your public Git repository. `.gitignore` ignores CSV and model artifacts.
- Save source revision, dataset hash, random seed and Python package versions for reproducibility. Keep a **frozen final test** and don't repeatedly compare configurations on it.
- For real forecasting, request timestamps, a formal prediction horizon, battery-level repeated histories, and a forward-in-time/grouped evaluation.

## What this starter does NOT claim

1. No guaranteed prediction lead time: there are no temporal labels or event horizons.
2. Synthetic risk scores are removed by default due to unknown provenance, but leakage risk in other sensor-derived features remains to be audited with the data generator or domain owner.
3. The code does not guarantee a model above any metric threshold. Report actual measured results, including weak findings.
4. Model comparison does not prove that a particular imbalance strategy is best until experiments demonstrate it.
5. A local API demo is not a safety-critical deployment.

See `docs/DAY_BY_DAY_PLAN.md` for exactly what to learn and accomplish each day and `docs/INTERVIEW_PREP.md` for interview questions and explanations.
