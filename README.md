# EV Battery Failure — Machine Learning Mini Project

**Target:** `battery_failure` (`0` = healthy, `1` = failure) from 20,000 **simulated** fleet-telemetry rows. This is a research/teaching project, **not** a real EV safety monitoring product. No observation timestamp or failure horizon is provided, so the dataset does **not** validate early warning lead time.

## Project contents

```
README.md, menu.ps1
requirements.txt, pyproject.toml, .gitignore, .env.example
src/evbattery/
  config.py         # guarded target/identifier/composite score exclusions
  data.py           # data validation, 60/10/10/20 stratified split, features
  quality.py        # missing values, correlations, data-quality profile
  eda.py            # Member A: profiling, 36 plausibility checks, EDA tables + figures
  feature_catalog.py# Member A: provenance + keep/drop decision for all 70 columns
  leakage_audit.py  # Member A: split check, AUC screen, composite study, feature-set ablation
  plot_style.py     # shared colour-blind-safe chart style (Healthy = blue, Failure = orange)
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
  ui.py             # interactive console dashboard with human-readable cards

data/                # put ev_battery.csv here (ignored by Git, see data/README.md)
artifacts/           # generated locally, ignored by Git
learning_log.md      # one observation per working day

tests/               # automated assertions
docs/                # 14-day learning schedule, interview bank, report template
```

## Team status

| Member | Area | Status |
|---|---|---|
| A | Data profiling, EDA, leakage and feature-set audit | **Done** — see `docs/MEMBER_A_DATA_AUDIT.md`, `docs/TEAM_HANDOFF.md` |
| B | Supervised models, CV, tuning | **Done** — see `docs/MEMBER_B_HANDOFF.md`, `docs/MEMBER_B_MODELING_REPORT.md` |
| C | Imbalance, calibration, thresholds, evaluation | **Done** — see `docs/MEMBER_C_FINAL_RESULTS.md` |
| D | Unsupervised, explanation, API, integration/testing | **Done** — see `docs/member_d_results.md` |

**Feature set:** `make_features()` drops three exact duplicate columns by default (63 model inputs). See `docs/FEATURE_DICTIONARY.md`.

---

## ⚡ Quick Start: Interactive Dashboard (Recommended)

To run the project with an easy-to-use menu that shows clean, color-coded summaries instead of raw JSON walls:

### Windows PowerShell:
```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Run the interactive menu
.\menu.ps1
```
*(Or run `python -m evbattery.ui`)*

```text
========================================================================
   [*] EV BATTERY HEALTH & FAILURE ML -- INTERACTIVE DASHBOARD [*]
========================================================================
  What would you like to do?

  [1] Code Health Check         Run unit tests (clean & quiet)
  [2] Dataset Quality Check     Vehicle counts, missing data, health %
  [3] Explore Telemetry (EDA)   Key sensor patterns & failure graphs
  [4] Leakage & Safety Audit    Verify fair, cheat-free evaluation
  [5] Train & Compare Models    Leaderboard of 7 ML algorithms
  [6] Explain Model Decisions   Top sensors driving risk alerts
  [7] Plain-English ML Guide    Demystify machine learning jargon
  [0] Exit
========================================================================
Select an option [0-7]: 
```

---

## 🛠️ Setup Instructions (Python 3.11+ recommended)

Unzip the folder. Open a terminal in its root. Place `ev_battery.csv` in `data/`:

### Windows PowerShell:
```powershell
python -m venv .venv

# If script execution is restricted on Windows:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
pytest
```

### Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pytest
```

---

## 📋 Step-by-Step Running Instructions (Command Line)

If you prefer running individual commands directly:

### 1. Run Automated Unit Tests
```bash
pytest
```
Verifies data split integrity, preprocessing pipelines, model factories, and leakage protections across 33 tests.

### 2. Dataset Quality Checks
```bash
python -m evbattery.quality --data data/ev_battery.csv
```
Profiles fleet records (20,000 cars), class imbalance (6.92% failure rate), missing cells across 67 columns, and collinear sensor pairs. Saves `artifacts/data_profile.json`.

### 3. Exploratory Data Analysis (EDA)
```bash
python -m evbattery.eda --data data/ev_battery.csv --outdir artifacts/eda
```
Generates 7 statistical figures under `artifacts/eda/figures/` (distributions, boxplots, correlation heatmap) and checks 36 dataset plausibility flags.

### 4. Leakage & Feature Safety Audit
```bash
python -m evbattery.leakage_audit --data data/ev_battery.csv --outdir artifacts/leakage_audit
```
Audits univariate ROC AUC separation, verifies zero split overlap across 60/10/10/20 partitions, and evaluates feature-set ablation (justifying the exclusion of composite risk scores and dropping 3 redundant copies to arrive at 63 features).

### 5. Supervised Model Training & Evaluation

#### A. Full Training Pipeline (Model selection, Platt calibration, threshold tuning, and holdout test):
```bash
python -m evbattery.train --data data/ev_battery.csv --outdir artifacts
```
* Uses 3-fold Stratified CV on 60% train partition to rank 7 algorithms + dummy baseline using **Average Precision (AP)**.
* Fits Platt/sigmoid calibration on the 10% calibration partition.
* Tunes decision threshold for $F_2$-score on the 10% threshold partition.
* Evaluates **once** on the untouched 20% holdout test partition.
* Outputs: `artifacts/model_cv_leaderboard.csv`, `artifacts/threshold_validation_metrics.json`, `artifacts/test_metrics.json`, `artifacts/model.joblib`.

#### B. Resource-Conscious / Fast Benchmark:
```bash
python -m evbattery.train --data data/ev_battery.csv --outdir artifacts/resource_limited --cv-max-rows 4000 --fast
```

### 6. Explainability & Feature Importance
```bash
python -m evbattery.explain --data data/ev_battery.csv --model artifacts/model.joblib
```
Computes permutation feature importance on the validation partition to identify the top sensors driving failure warnings. Outputs `artifacts/permutation_importance.csv`.

### 7. Other Research Experiments (Development Folds Only)

```bash
# Imputation and correlation pruning experiments on training sample:
python -m evbattery.experiments --data data/ev_battery.csv --max-rows 2500

# Hyperparameter tuning via RandomizedSearchCV (train CV only):
python -m evbattery.tune --data data/ev_battery.csv --model hist_gradient_boost --n-iter 10

# Class-imbalance strategy comparison (SMOTE-NC, class weighting, undersampling):
python -m evbattery.imbalance --data data/ev_battery.csv

# Unsupervised anomaly detection & clustering (PCA, K-Means, Isolation Forest):
python -m evbattery.unsupervised --data data/ev_battery.csv

# Illustrative false-alarm vs missed-failure economic costs:
python -m evbattery.costs --data data/ev_battery.csv --model artifacts/model.joblib
```

---

## 🌐 8. Demonstration API (Local Development Only)

After full training, test the authenticated FastAPI inference server:

### 1. Generate an example request payload:
```bash
python -m evbattery.example_request --data data/ev_battery.csv
```

### 2. Start the API server:

**Windows PowerShell:**
```powershell
$env:EV_API_KEY="test-secret-key"
$env:EV_MODEL_PATH="artifacts/model.joblib"
uvicorn evbattery.api:app --host 127.0.0.1 --port 8000
```

**Linux / macOS:**
```bash
export EV_API_KEY='test-secret-key'
export EV_MODEL_PATH='artifacts/model.joblib'
uvicorn evbattery.api:app --host 127.0.0.1 --port 8000
```

### 3. Send a test prediction in another terminal:
```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "X-API-Key: test-secret-key" ^
  -H "Content-Type: application/json" ^
  --data-binary @artifacts/example_request.json
```

---

## 🔒 Security & Reliability Guidance

- Bind `127.0.0.1` only. **Do not deploy publicly** without TLS termination, trusted auth, rate limiting, payload validation, and threat modeling.
- `joblib` uses Python pickle. **NEVER load model files from untrusted sources.** Load only locally trained artifacts.
- Keep secrets in environment variables and raw fleet telemetry outside Git.
- Save dataset hash, random seed (`42`), and package versions for reproducibility. Keep a **frozen final test** and never repeatedly tune configurations on it.

## What this starter does NOT claim

1. No guaranteed prediction lead time: there are no temporal labels or event horizons.
2. Synthetic risk scores are removed by default due to unknown provenance.
3. The code does not guarantee a model above any metric threshold; report actual measured results.
4. Model comparison does not prove that a particular imbalance strategy is best until experiments demonstrate it.
5. A local API demo is not a safety-critical deployment.

See `docs/DAY_BY_DAY_PLAN.md` for learning objectives and `docs/INTERVIEW_PREP.md` for viva questions and answers.
