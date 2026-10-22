# Mini Project Final Report — Template

**Title:** Early Warning of Electric Vehicle Battery Failure from Fleet Telemetry Using Classical Machine Learning

**IMPORTANT:** Replace every [PLACEHOLDER] with **actual experiment output**. Never publish made-up metric values or treat simulated labels as physical events.

## 1. Executive summary

One paragraph: research question, simulated data, main leakage-aware methods, best observed validation and final holdout results, and what cannot be inferred.

## 2. Problem and constraints

Describe why battery faults matter, class imbalance, missing values and timing ambiguity. State that the dataset lacks observation/failure timestamps and real-world failure lead-time cannot be validated.

## 3. Dataset and EDA

- Source: supplied Kaggle-sourced simulated EV telemetry CSV referenced in proposal.
- Shape: 20,000 × 70; target `battery_failure`; failures 1,384 (6.92%).
- Overall missing cells: 53,344 (3.81%); 0 exact duplicates; vehicle IDs and battery serials unique.
- Insert labeled plots for target distribution, telemetry histograms, missingness, and selected correlations.
- Note strongly related variables and suspicious generated score features; justify feature exclusions.

## 4. Methods

Present stratified train/calibration/threshold/test split (60/10/10/20), CV, median/categorical imputers, one-hot encoding, standardization, derived features and leakage policy. Explain **when** features are available and which risks remain unknown.

## 5. Supervised comparison

Fill from `artifacts/model_cv_leaderboard.csv`:

| Model | CV AP mean | CV AP SD | CV ROC AUC |
|---|---:|---:|---:|
| [MODEL 1] | [ ] | [ ] | [ ] |
| [MODEL 2] | [ ] | [ ] | [ ] |
| [MODEL 3] | [ ] | [ ] | [ ] |
| [MODEL 4] | [ ] | [ ] | [ ] |
| [MODEL 5] | [ ] | [ ] | [ ] |

Mark a dummy positive-prevalence baseline as a reference, not a production model. `average_precision` is scikit-learn AP, not trapezoidal PR AUC.

## 6. Preprocessing, feature selection and imbalance

Describe train-fold-only comparison of median vs KNN imputation, correlation pruning/L1 selection, no resampling vs class weights vs random undersampling vs SMOTENC. Include the sample size if comparison was made on a reduced subset; don't generalize the winner beyond evidence.

## 7. Calibration and alert policy

State sigmoid/Platt calibration on a separate 10%, chosen threshold on other 10%, selected F2, Brier score and any illustrative cost ratios. Illustrative assumptions are not actual operating costs. Insert reliability plot if prepared.

## 8. Final holdout evaluation

Copy **once-only** metrics from `artifacts/test_metrics.json`:

| Metric | Value |
|---|---:|
| Average precision | [ ] |
| ROC AUC | [ ] |
| Precision | [ ] |
| Recall | [ ] |
| F1 | [ ] |
| F2 | [ ] |
| Balanced accuracy | [ ] |
| MCC | [ ] |
| Brier | [ ] |
| True positives | [ ] |
| False positives | [ ] |
| True negatives | [ ] |
| False negatives | [ ] |

Analyze false negatives and false alarms without leaking test-specific decisions into model selection.

## 9. Unsupervised studies

Show PCA variance explained, K-means cluster scatter, sampled silhouette score, and IsolationForest anomaly association with failure labels. Confirm labels were excluded during fitting. Discuss lack of clear separation if that is what results show.

## 10. Interpretability and API demonstration

Describe permutation importance (what it measures, correlation limitations). Show input schema, example request, token authentication and sample response. State local educational demo only.

## 11. Limitations, safety and future work

Synthetic record generation, no temporal horizon, missing measurement availability metadata, high dimensionality, possible score leakage, single dataset, potential sampling bias, hypothetical maintenance-cost assumptions. Propose timestamped fleet data, battery-level history, out-of-time holdouts, real-world validation and monitoring.

## 12. Reproducibility

Include Git commit hash, dataset SHA256 (`metadata.json`), Python/library versions, random seed, exact commands, train/validation/test strategy and output file references.

## 13. Conclusion

Distinguish what the experiment **demonstrated** from what it **did not demonstrate**.

## 14. References

Use the references in the original uploaded proposal, and document any additional references you actually consulted.
