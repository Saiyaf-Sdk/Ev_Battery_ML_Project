# Baseline Result — Full 20,000-Record CSV

The starter baseline was **actually trained** on the supplied dataset with logistic regression only and a non-learning dummy reference. This is **not** the required 5+-algorithm final model-comparison study. The composite risk scores were excluded, identifiers dropped and preprocessing fitted only on the training data.

## Split counts

| Split | Records | Purpose |
|---|---:|---|
| Training | 12,000 | Fit estimator and perform 3-fold CV |
| Calibration | 2,000 | Fit sigmoid probability calibration |
| Threshold validation | 2,000 | Choose cutoff that maximizes F2 |
| Final holdout test | 4,000 | One-time evaluation |

## Results

- Training CV mean average precision: **0.720721** for logistic regression; dummy reference: **0.069167**.
- Chosen threshold on threshold-validation data: **0.22**.
- Final holdout AP: **0.737585**.
- ROC AUC: **0.970688**; Brier score **0.028887**.
- Precision: **0.579208**; recall **0.844765**; F1 **0.687225**; F2 **0.773810**.
- Balanced accuracy **0.899552**; MCC **0.673269**.
- Final confusion matrix: **TP 234, FP 170, TN 3,553, FN 43** (277 failure records out of 4,000).

Source of all exact numbers: `artifacts/baseline/model_cv_leaderboard.csv`, `artifacts/baseline/threshold_validation_metrics.json`, `artifacts/baseline/test_metrics.json` and `artifacts/baseline/split_counts.csv`.

**Caveat:** the labels and records are simulated. These numbers do not establish real-world battery failure probabilities, true early-warning horizon or safe autonomous maintenance decisions. Do not use test outcomes when choosing additional modeling configurations.
