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

## Member B — Day 4 (preprocessing pipelines & leakage prevention)
- Ran `python -m evbattery.experiments`: compared numerical median imputation with missing indicators, KNN imputation, correlation pruning (threshold 0.97), and L1 embedded selection on training folds only.
- Learned: KNN imputation takes significantly longer than median imputation but produced identical Average Precision (0.742). Euclidean distance in 55 continuous dimensions suffers from the curse of dimensionality.
- Decision: Kept median imputation with missingness indicators + standard scaling as the default numerical pipeline; handled categorical features with most-frequent imputation + one-hot encoding with `handle_unknown="ignore"` for safe inference on new car brands.
- Key lesson: Preprocessing must be fitted inside CV training folds via `Pipeline` to prevent test-set distribution leakage.

## Member B — Day 5 (baselines & supervised linear models)
- Compared `dummy_prior` against `logistic_regression` and `decision_tree`.
- Learned: With a 6.92% failure rate, the dummy baseline gets 93.08% accuracy but 0% recall, confirming accuracy is misleading. Average Precision (AP) baseline is 0.069.
- Decision: Logistic Regression reached AP ~0.72–0.74, showing strong linear separability in raw sensor signals (health, temperature, resistance).
- Key lesson: Single Decision Tree scored lowest among learning models (AP ~0.67) because axis-aligned rectangular step functions suffer from high variance on continuous health metrics.

## Member B — Day 6 (multi-algorithm benchmark & ensemble superiority)
- Evaluated 7 classical ML algorithms + baseline using 3-fold Stratified Cross-Validation on the 60% training partition only: Logistic Regression, Gaussian Naive Bayes, SGD Classifier, Decision Tree, Random Forest, AdaBoost, Histogram Gradient Boosting.
- Learned: Ensembles (Random Forest and Histogram Gradient Boosting) consistently lead the leaderboard (AP > 0.78–0.81, ROC AUC ~0.89).
- Decision: Selected tree ensembles over linear models because they capture multi-sensor nonlinear interactions (e.g. high cell temperature coupled with high charging power) without manual polynomial expansion.
- Key lesson: Kept the calibration, threshold, and test partitions completely sequestered—no test-peeking was permitted to influence algorithm choice.

## Member B — Day 9 (hyperparameter optimization & handoff artifact)
- Applied `RandomizedSearchCV` on the 60% training partition with 3-fold Stratified CV scored by Average Precision (`python -m evbattery.tune`).
- Learned: Bounding `max_depth` to 12 and `min_samples_leaf` to 8 regularized the forest against memorizing individual telemetry spikes, improving AP by +0.005 (well within the CV std dev margin of ±0.018).
- Decision: Exported uncalibrated pipeline `artifacts/member_b/best_base_model.joblib` and comprehensive metadata `artifacts/member_b/selected_model.json`.
- Key lesson: Intentionally handed off an UNCALIBRATED base model to Member C; calibrating probabilities on the training set would cause overconfidence. Member C will fit Platt calibration on the dedicated 10% calibration partition.

