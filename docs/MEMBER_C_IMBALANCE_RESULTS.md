# Member C — Class Imbalance Experiment Results

## 1. Objective

The purpose of this experiment was to compare different class imbalance handling methods for EV battery failure classification.

The dataset contains 20,000 battery records, with 1,384 failure records (6.92%). Since failure cases are less common than healthy cases, a classification model may become biased toward the majority class.

## 2. Methods

Four strategies were compared:

- No resampling
- Balanced class weighting
- Random undersampling
- SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features)

The experiments used Logistic Regression with three-fold stratified cross-validation on the training partition only.

All resampling operations were applied within the training folds to avoid data leakage.

## 3. Experimental Results

| Imbalance Method | Mean Average Precision | Mean ROC-AUC |
|---|---:|---:|
| SMOTENC | 0.736630 | 0.976565 |
| No Resampling | 0.718699 | 0.974349 |
| Class Weight Balanced | 0.709194 | 0.973955 |
| Random Undersampling | 0.673553 | 0.966923 |

## 4. Discussion

SMOTENC achieved the highest Average Precision (0.736630) and ROC-AUC (0.976565) among the tested strategies.

Compared with the baseline without resampling, SMOTENC improved mean Average Precision by approximately 0.0179.

Class weighting produced slightly lower Average Precision than the baseline. Random undersampling gave the lowest score among the tested strategies.

These results suggest that SMOTENC is worth considering for this dataset. However, the experiment used Logistic Regression, so these results do not prove that SMOTENC will improve Member B's selected final model.

## 5. Conclusion

Among the four tested methods, SMOTENC achieved the best cross-validation results.

The results were obtained from training-only cross-validation. The calibration, threshold-validation, and final test partitions were not used in this experiment.

Further evaluation will consider probability calibration, warning threshold selection, and final performance on the held-out test set.

## 6. Reproducibility

The experiment can be repeated using:

`python -m evbattery.imbalance --data data/ev_battery.csv --out artifacts/member_c/imbalance_comparison.csv`

The generated CSV is stored locally in the ignored `artifacts/member_c/` directory. The results above are retained in this report for version control.