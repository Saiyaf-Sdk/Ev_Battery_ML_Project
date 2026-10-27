
# Member D — Experimental Results and API Integration

## Project
Early Warning of Electric Vehicle Battery Failure from Fleet Telemetry Using Classical Machine Learning

## 1. Responsibilities

Member D completed:
- Unsupervised learning experiments
- PCA dimensionality reduction
- K-Means clustering
- Isolation Forest anomaly detection
- Permutation feature importance analysis
- Integration with the trained classification model
- FastAPI prediction and authentication testing
- Sample request generator improvements
- Automated regression testing

## 2. Dataset

- Total records: 20,000
- Total columns: 70
- Target variable: battery_failure
- Missing values: 53,344 cells

The dataset SHA256 matches the hash stored in the trained model metadata.

## 3. Unsupervised Learning Results

### PCA

Explained variance of the first two components:
0.170121 (17.01%)

The first two principal components capture approximately
17.01% of the analyzed variation.

### K-Means Clustering

Sample silhouette score:
0.053544

This low score indicates weak cluster separation.

### Isolation Forest

Anomaly score versus failure label ROC-AUC:
0.709879

The anomaly scores show some ability to distinguish
failure-labelled records from non-failure records.

Failure labels were not used for unsupervised model fitting.
They were used afterward for diagnostic evaluation.

## 4. Model Explainability

Permutation importance identified these leading features:

| Feature | Mean Importance |
|---------|-----------------|
| cell_temperature_max | 0.241198 |
| battery_health_percent | 0.138004 |
| previous_faults | 0.136654 |
| BMS_warning_count | 0.119449 |
| cell_temperature_avg | 0.066746 |

The model relies strongly on temperature, battery-health,
previous fault and BMS warning information.

These importance scores do not establish causation.

## 5. Model Integration

Trained classification model:
Histogram Gradient Boosting

Calibration:
Sigmoid

Classification threshold:
0.10

The API loads the saved trained model and returns a
failure risk score, classification alert, model name
and research disclaimer.

## 6. API Verification

Health endpoint:
GET /health — Passed

Prediction endpoint:
POST /predict — Passed

Authentication:
Requests without a valid X-API-Key were rejected.

Example prediction:
- Failure risk score: 0.005056
- Alarm threshold: 0.10
- Alert: False
- Model: hist_gradient_boost

This is a demonstration prediction on synthetic data.

## 7. Integration Bug Fix

The original example request generator included three
fields not accepted by the trained model:

- manufacturing_year
- remaining_capacity
- capacity_loss_percent

The generator was updated to filter features using the
saved model metadata.

Two regression tests were added to verify its behavior.

## 8. Testing

All 33 automated tests passed after integration changes.

Some dependency deprecation warnings remain, but there
were no test failures.

## 9. Generated Outputs

- artifacts/unsupervised/pca_clusters.png
- artifacts/unsupervised/cluster_failure_summary.csv
- artifacts/unsupervised/unsupervised_summary.json
- artifacts/permutation_importance.csv

## 10. Limitations

This project uses synthetic cross-sectional data.

The results do not demonstrate actual prediction lead time
or real-world EV battery safety performance.

The API is a research demonstration and should not be used
for operational safety decisions without external validation.
