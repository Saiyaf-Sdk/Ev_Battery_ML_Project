# Member B to Member C — Technical Interface & Model Handoff Specification

**From:** Member B (Supervised ML Modeling & Pipeline Architecture)  
**To:** Member C (Class Imbalance, Probability Calibration, Threshold Tuning & Final Evaluation)  
**Status:** Ready for Integration  
**Primary Artifact:** `artifacts/member_b/best_base_model.joblib`  
**Metadata:** `artifacts/member_b/selected_model.json`  
**Cross-Validation Comparison:** `artifacts/member_b/model_cv_leaderboard.csv`

---

## 1. What Member B Has Delivered

Member B has completed the training-partition supervised learning phase:
1. **Evaluated Preprocessing Strategies:** Confirmed that median imputation with missingness indicators + standard scaling + one-hot encoding (`handle_unknown='ignore'`) provides optimal predictive capability and millisecond latency.
2. **Benchmarked Candidate Algorithms:** Conducted 3-fold Stratified Cross-Validation on the 60% training partition across seven classical algorithms (Logistic Regression, Gaussian Naive Bayes, SGD Classifier, Decision Tree, Random Forest, AdaBoost, Histogram Gradient Boosting) plus Dummy and KNN baselines.
3. **Hyperparameter Tuning:** Tuned the winning tree ensemble using `RandomizedSearchCV` on training folds only.
4. **Fitted Uncalibrated Base Model:** Trained the complete winning pipeline (preprocessing + classifier) across the full 60% training partition (12,000 rows).
5. **Zero Test Peeking:** Calibration (10%), threshold-validation (10%), and final test (20%) partitions remain completely untouched.

---

## 2. Model Bundle Interface Contract

The file `artifacts/member_b/best_base_model.joblib` is a serialized dictionary bundle created with `joblib.dump`:

```python
{
    "base_pipeline": Pipeline(...),    # Complete, fitted, UNCALIBRATED scikit-learn Pipeline
    "model_name": "random_forest",     # Name of the winning algorithm
    "is_tuned": True,                  # Whether hyperparameter tuning was applied
    "best_params": {...},              # Best hyperparameters discovered via 3-fold CV
    "metadata": {...}                  # Environment, feature names, seeds, and training stats
}
```

### Feature Specification
- **Input Dimensions:** Exactly 63 features (55 numeric, 8 categorical).
- **Approved Feature Columns:** Listed in exact order in `selected_model.json["feature_names"]`.
- **Pre-transformations:** Must be generated via `evbattery.data.make_features(df, include_suspect_scores=False, drop_redundant=True)`.
- **Target / IDs / Composites:** Excluded by `make_features()`.

---

## 3. Step-by-Step Instructions for Member C

Member C can integrate this model immediately. Follow these five steps:

### Step 1: Load the Base Pipeline and Partitions
```python
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from evbattery.config import TARGET
from evbattery.data import load_dataset, split_data, make_features
from evbattery.evaluation import choose_threshold, evaluate

# 1. Load Member B's uncalibrated model bundle
bundle = joblib.load("artifacts/member_b/best_base_model.joblib")
base_pipeline = bundle["base_pipeline"]

# 2. Load the dataset and extract the four agreed partitions
df = load_dataset("data/ev_battery.csv")
splits = split_data(df, seed=42)
```

### Step 2: Fit Probability Calibration on the Calibration Partition ONLY
Do NOT retrain the base model. Use `FrozenEstimator` to guarantee that only the Platt sigmoid calibrator is fitted:

```python
# Extract calibration features and labels
x_cal = make_features(splits.calibration)
y_cal = splits.calibration[TARGET].astype(int)

# Fit Platt (sigmoid) probability calibration on the 10% calibration set (2,000 rows)
calibrated_model = CalibratedClassifierCV(
    FrozenEstimator(base_pipeline), method="sigmoid"
)
calibrated_model.fit(x_cal, y_cal)
```

### Step 3: Select Alarm Cutoff Threshold on the Threshold Partition ONLY
Do NOT use the test set to choose your decision threshold:

```python
# Extract threshold-validation features and labels
x_thr = make_features(splits.threshold)
y_thr = splits.threshold[TARGET].astype(int)

# Predict calibrated failure probabilities
p_thr = calibrated_model.predict_proba(x_thr)[:, 1]

# Optimize decision threshold for F2 score (prioritizing recall of rare failures)
optimal_threshold, best_f2 = choose_threshold(y_thr, p_thr, beta=2.0)
print(f"Optimal Alarm Threshold: {optimal_threshold:.4f} (Validation F2: {best_f2:.4f})")
```

### Step 4: Final One-Time Test Evaluation
Now—and only now—evaluate the locked holdout test set:

```python
# Extract test features and labels
x_test = make_features(splits.test)
y_test = splits.test[TARGET].astype(int)

# Predict calibrated probabilities on untouched test set
p_test = calibrated_model.predict_proba(x_test)[:, 1]

# Calculate final test metrics using the threshold chosen on the threshold set
test_metrics = evaluate(y_test, p_test, threshold=optimal_threshold)
print("Final Test Metrics:")
print(f"  Average Precision: {test_metrics['average_precision']:.4f}")
print(f"  ROC AUC:           {test_metrics['roc_auc']:.4f}")
print(f"  Recall:            {test_metrics['recall']:.4f}")
print(f"  Precision:         {test_metrics['precision']:.4f}")
print(f"  F2 Score:          {test_metrics['f2']:.4f}")
print(f"  Brier Score:       {test_metrics['brier_score']:.4f}")
```

### Step 5: Save Final Deployed Artifact
Save the calibrated bundle for Member D's API service:
```python
final_bundle = {
    "model": calibrated_model,
    "threshold": optimal_threshold,
    "metadata": {
        **bundle["metadata"],
        "calibration_method": "sigmoid",
        "threshold": optimal_threshold,
        "test_metrics": test_metrics,
    }
}
joblib.dump(final_bundle, "artifacts/model.joblib")
```

---

## 4. Verification and Sanity Checks for Member C

Before proceeding, run these three checks:
1. **Uncalibrated Output Check:** Calling `base_pipeline.predict_proba(x_cal)` returns raw model probabilities.
2. **Feature Shape Invariant:** `base_pipeline.predict_proba(x)` will accept any DataFrame with the approved 63 columns, even if it contains missing numerical values or previously unseen categorical levels.
3. **Target Invariant:** Verify that `splits.calibration[TARGET]` and `splits.threshold[TARGET]` both have failure prevalence $\approx 6.92\%$.
