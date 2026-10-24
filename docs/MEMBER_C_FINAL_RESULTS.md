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
| Average precision | TODO |
| ROC AUC | TODO |
| Brier score | TODO |
| Precision | TODO |
| Recall | TODO |
| F1 | TODO |
| F2 | TODO |
| True positives | TODO |
| False positives | TODO |
| True negatives | TODO |
| False negatives | TODO |

## Discussion

Describe the missed-failure vs false-alarm trade-off using **final test** values. Explain whether performance changed relative to threshold-validation results. Do not change the threshold based on the observed test metrics.

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
