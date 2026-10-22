# Learning log

One observation per working day: what was done, what was learned, and why a decision matters.

## Member A — Day 1 (setup + data loading)
- Loaded the CSV with `load_dataset()`: 20,000 rows × 70 columns; IDs unique; label only 0/1.
- Learned: one row = one battery; `X` = 63 input features after cleaning, `y` = `battery_failure`. It is binary classification because the answer is yes/no.

## Member A — Day 2 (EDA)
- 1,384 failures = 6.92 %. Always predicting "healthy" gives 93.08 % accuracy and catches no failures, so accuracy is the wrong metric.
- 93 % of rows have at least one missing value → imputation, not row deletion.
- Failed batteries: lower health (median 71 % vs 84 %), hotter cells (max 55 °C vs 40 °C), higher internal resistance, older vehicles. Categories (chemistry, terrain, body type) barely change the failure rate.
- Data-quality flags (highway + city share > 1, clipped `temperature_variance`, random body types) show the data is simulated.

## Member A — Day 3 (leakage, features, splitting)
- Verified the 60/10/10/20 stratified split: prevalence 0.069 in every part, no vehicle in two parts.
- No raw column separates the classes suspiciously well (best AUC 0.822 < 0.90).
- Composite scores add +0.03–0.05 AP even though they are mostly rebuildable from raw columns → unknown extra information → excluded. Lesson: a jump in score is a warning sign, not a win.
- Three columns are exact formulas of others; dropping them did not change AP (difference < 1 SD) → simpler feature set of 63.

<!-- Members B, C, D: add your own entries below -->
