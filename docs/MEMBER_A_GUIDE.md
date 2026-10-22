# Member A — Your Personal Guide

You are **Member A: data profiling, visualisations, leakage and feature-set audit**. Your work comes first because everyone else builds on the dataset facts, the split and the feature list you hand over.

## 1. What has been done (and where)

| Deliverable | File(s) |
|---|---|
| Data profiling + 36 plausibility checks | `src/evbattery/eda.py` → `artifacts/eda/*.csv`, `overview.json` |
| 7 EDA figures | `artifacts/eda/figures/` (copies in `docs/figures/member_a/`) |
| Feature provenance for all 70 columns | `src/evbattery/feature_catalog.py`, `docs/FEATURE_DICTIONARY.md` |
| Split verification, leakage screen, composite study, ablation | `src/evbattery/leakage_audit.py` → `artifacts/leakage_audit/` |
| Final feature set (63 inputs) | `artifacts/leakage_audit/feature_set.json`; `make_features(drop_redundant=True)` in `data.py` |
| Written results with real numbers | `docs/MEMBER_A_DATA_AUDIT.md` (paste into report §3–4) |
| Unit tests for your code | `tests/test_member_a.py` (8 tests) |
| Handover for B, C, D | `docs/TEAM_HANDOFF.md` |

**Your job now is to understand it well enough to explain every line in the viva, run it on your own laptop, and push it to GitHub.** Examiners will ask *why*, not just *what*.

## 2. Run it on your Windows laptop (first time)

Install **Python 3.11** (tick "Add python.exe to PATH"), **Git**, and **VS Code**. Then in PowerShell:

```powershell
cd "D:\Projects\Machine Learning\ev_battery_ml_project"
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest                     # expect all tests to pass

# your three commands (data/ev_battery.csv is already in place)
python -m evbattery.quality       --data data/ev_battery.csv
python -m evbattery.eda           --data data/ev_battery.csv --outdir artifacts/eda
python -m evbattery.leakage_audit --data data/ev_battery.csv --outdir artifacts/leakage_audit
```

The last command takes 2–5 minutes (it trains 12 cross-validated models). Add `--skip-ablation` for a quick run. Compare your output numbers with `docs/MEMBER_A_DATA_AUDIT.md` — they should match.

## 3. What you must be able to explain (study list)

Work through these in order; each links to the code that does it.

1. **Row, column, feature, label.** One row = one battery. 70 columns. Features `X` = inputs; label `y` = `battery_failure`. Why binary classification, not regression: the answer is yes/no.
2. **Class imbalance.** 1,384 / 20,000 = 6.92 %. A model that always says "healthy" scores 93.08 % accuracy and catches zero failures. → Use precision, recall, F2, average precision.
3. **Missing data.** 3.81 % of cells; but 93 % of rows have a gap, so we can't drop rows. Imputation (median / most frequent) is *fitted on training folds only*.
4. **Plausibility checks** (`eda.plausibility_checks`). Know at least three passed checks and the six flags, and why the flags suggest synthetic data.
5. **Reading your figures.** Histograms show distribution shape; boxplots show median/IQR; heatmap shows linear correlation; category chart shows failure rate per group with the 6.92 % reference line.
6. **Stratified split** 60/10/10/20 with seed 42. Why stratify (keep 6.9 % failures everywhere). Why a seed (reproducible). Why the test set is locked (an honest final score).
7. **Why EDA uses only the training split** for class comparisons: if you choose features by looking at test data, the test score is no longer unbiased.
8. **Leakage** — definition and our examples: IDs, the target, composite scores, preprocessing on all data, SMOTE before splitting, picking the threshold on test.
9. **Univariate AUC screen.** AUC 0.5 = random, 1.0 = perfect. A raw column ≥ 0.90 would be suspicious. None found; best raw = health 0.822.
10. **Composite study.** Composites are 75–96 % rebuildable from raw columns, yet add +0.03–0.05 AP → they contain something extra of unknown origin → excluded. "Sensitivity, not proof."
11. **Redundancy vs leakage.** `capacity_loss_percent = 100 − health` is a duplicate, not leakage. Dropped because it adds nothing (ablation: change < 1 SD).
12. **Engineered features** in `make_features()`: temperature gap, ambient range, cycles/year, km/cycle, fast-charge stress, total fault count, service overdue flag. All are per-row formulas — no fitting, so no leakage.

## 4. Likely viva questions for you (answer aloud, then check)

- *Why is 93 % accuracy misleading here?* → majority-class baseline.
- *What is data leakage? Two examples from our project.* → composite scores; fitting the imputer on all 20k rows.
- *Why did you exclude `thermal_runaway_risk` if it is the best predictor?* → unknown generation; could encode the label; ablation shows +0.05 AP we cannot explain.
- *Is a highly correlated variable leakage?* → No; e.g. health vs SoH r = 0.99 is shared physics/redundancy.
- *Why drop `capacity_loss_percent`?* → exact `100 − health`; adds no information; splits importance between two copies.
- *Why not drop rows with missing values?* → would lose 93 % of rows.
- *Why is the split random rather than by time?* → no timestamps exist; a limitation.
- *What does the `highway + city > 1` flag tell you?* → independently simulated columns; synthetic data.
- *Could raw columns still leak?* → possible; only the data author could confirm; none exceeded AUC 0.90.
- *Can this model forecast failure a week early?* → No: no timestamps or horizon; cross-sectional only.

## 5. Your slides in the final presentation (suggested 3)

1. **Problem & dataset** — 20,000 × 70 synthetic records, 6.92 % failures, 3.81 % missing; figure 01.
2. **EDA findings** — health, temperature, resistance, age separate the classes; categories barely matter; data-quality flags; figures 03 + 05.
3. **Leakage policy & split** — what was removed and why, split diagram, ablation figure 09.

## 6. Git commits for your part (suggested)

```text
data: add data/README and keep raw CSV out of Git
eda: add profiling, plausibility checks and EDA figures
audit: add feature catalogue and leakage/feature-set audit
features: drop exact duplicate columns by default (ablation evidence)
docs: Member A data audit, feature dictionary and team handoff
test: unit tests for Member A modules
```

See `docs/GITHUB_SETUP.md` for creating the repository and pushing.
