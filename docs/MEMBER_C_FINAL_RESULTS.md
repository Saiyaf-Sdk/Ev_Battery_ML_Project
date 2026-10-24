# Member C — Final Held-Out Evaluation and Handoff

**Do not fill in this report by inventing results.** After the one-time final evaluation, copy measured values from `artifacts/test_metrics.json`.

## Frozen evaluation protocol

- Base supervised model: `hist_gradient_boost` (from Member B).
- Sigmoid calibrated using 10% calibration records.
- F2 threshold selected using a separate 10% validation set: **0.10**.
- Final test set: 20% of the dataset (4,000 records).
- No additional fitting, hyperparameter tuning or threshold selection on test data.

## Final test metrics (fill with actual output)

| Metric | Measured value |
|---|---|
| Average precision | 0.751260 |
| ROC AUC | 0.975719 |
| Brier score | 0.028534 |
| Precision | 0.468354 |
| Recall | 0.935018 |
| F1 | 0.624096 |
| F2 | 0.779651 |
| True positives | 259 |
| False positives | 294 |
| True negatives | 3429 |
| False negatives | 18 |

## Discussion

### Discussion of Final Results

The final model was evaluated using 4,000 unseen test records. The selected classification threshold was 0.10, which had been chosen using the separate threshold-validation dataset.

Out of 277 actual battery failure records, the model correctly detected 259 and missed 18. The recall was 93.50%, while precision was 46.84%. The F2 score was 0.7797.

The model achieved a high failure detection rate, but it also produced 294 false-positive warnings. This shows the trade-off between detecting more failures and generating unnecessary alerts.

The final results were close to those obtained during validation. However, the dataset is simulated, so further testing with real EV battery data would be required before practical deployment.

## Figures

Generated locally (excluded from Git unless explicitly copied into `docs/figures/member_c/`):
- `artifacts/member_c/figures/confusion_matrix.png`
- `artifacts/member_c/figures/precision_recall.png`
- `artifacts/member_c/figures/roc.png`
- `artifacts/member_c/figures/calibration.png`

## Member D handoff

- Final calibrated and frozen model: `artifacts/model.joblib`
- Model metadata: `artifacts/metadata.json`
- Final evaluation: `artifacts/test_metrics.json`
- Expected default API model location: `artifacts/model.joblib`
- Never load serialized model files from an untrusted source.

## Scientific limits

This is a model evaluated on synthetic cross-sectional EV telemetry records, not verified future failure prediction. It cannot establish a real-world early-warning horizon or safety assurance.
