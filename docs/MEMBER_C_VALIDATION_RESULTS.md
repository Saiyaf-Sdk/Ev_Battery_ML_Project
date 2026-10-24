# Member C — Probability Calibration and Threshold Validation

## Purpose

This stage extends Member B's trained **Histogram Gradient Boosting** model without retraining the base classifier. It fits sigmoid probability calibration on an independent 10% calibration partition (2,000 records) and selects a decision threshold by maximizing F2 on a separate 10% threshold-validation partition (2,000 records). The 20% final test set was not evaluated at this stage.

## Dataset and model

- Dataset: synthetic 20,000-row EV battery telemetry dataset; target `battery_failure`.
- Positive cases in threshold-validation partition: 138 / 2,000 (6.9%).
- Selected supervised classifier: `hist_gradient_boost` (Member B's frozen, trained pipeline).
- Calibration method: sigmoid (Platt-style calibration).
- F2-based alarm threshold chosen on threshold validation: **0.10**.

## Threshold-validation results (not final test results)

| Metric | Uncalibrated, threshold 0.50 | Calibrated, threshold 0.50 | Calibrated, threshold 0.10 |
|---|---:|---:|---:|
| Precision | 0.7245 | 0.7257 | 0.4674 |
| Recall | 0.5145 | 0.5942 | **0.9348** |
| F1 | 0.6017 | 0.6534 | 0.6232 |
| F2 | 0.5462 | 0.6165 | **0.7790** |
| Average precision (AP) | 0.7161 | 0.7161 | 0.7161 |
| ROC AUC | 0.9750 | 0.9750 | 0.9750 |
| Brier score | 0.03202 | **0.03122** | **0.03122** |
| True positives | 71 | 82 | 129 |
| False positives | 27 | 31 | 147 |
| True negatives | 1,835 | 1,831 | 1,715 |
| False negatives | 67 | 56 | 9 |

The Brier score improved slightly following calibration. AP and ROC AUC were unchanged in this experiment; they assess ranking, which was preserved by the fitted sigmoid calibration. Changing the alert threshold changes confusion-matrix metrics but does not change these probability-ranking scores.

## Interpretation and trade-off

At the original uncalibrated 0.50 cutoff, the model detected 71 of 138 failures. With calibrated probabilities and a 0.10 cutoff, it detected **129 of 138 failures** (93.48% recall), leaving **9 missed failures**. The cost of increased recall was **147 false alarms**, rather than 27 originally. Precision fell to 46.74%. This may be sensible for a screening use case where missing failures is costly, but any practical choice would need validated operational costs and real-world deployment evidence.

## Safeguards and limits

- Training, calibration and threshold selection used separate partitions.
- Threshold selection happened entirely on the validation partition; results here are **not a measure of final held-out generalization**.
- The source dataset is synthetic and cross-sectional; the study does not establish real-world EV failure rates, a lead-time warning capability or safe vehicle deployment.
- Exact experiment records are local generated files, excluded from Git: `artifacts/member_c/validation_metrics.json` and `artifacts/member_c/staged_model.joblib`.
- Final evaluation is done once, with model and threshold frozen, in Batch 3.

## Reproduce the calibration and threshold-selection stage

```bash
python -m evbattery.member_c --data data/ev_battery.csv --base-model artifacts/member_b/best_base_model.joblib --outdir artifacts/member_c
python -m pytest
```
