# Machine Learning Interview + Viva Preparation

Study these questions by **answering out loud before reading the notes**. Each answer is intentionally short; elaborate using results you measured on *your* dataset, not invented percentages.

## Fundamentals

1. **AI vs ML vs deep learning?** AI is a broad field of intelligent systems; ML learns statistical patterns from data; deep learning is a subset based on multi-layer neural networks. This project uses classical ML on tabular data.
2. **Classification vs regression?** Classification predicts a discrete class (failure yes/no). Regression estimates a continuous value (e.g. remaining cycles).
3. **Feature vs target?** Features are input telemetry (`X`); the target is `battery_failure` (`y`). Avoid feeding target or identifiers into the model.
4. **Supervised vs unsupervised learning?** Supervised uses labels during fit. Unsupervised PCA/K-means/Isolation Forest finds structure without labels. Later label-based evaluation is allowed as long as fitting did not use labels.
5. **Training, validation and test?** Training learns model weights. Validation chooses/calibrates/tunes settings. Test is an untouched final unbiased check, if used once after development.
6. **What is cross-validation?** Repeatedly train on folds and validate on other folds for robust, low-leakage model comparisons. All fitted preprocessing belongs inside each training fold.
7. **What is overfitting?** The model memorizes idiosyncrasies/noise and generalizes poorly. Control via regularization, tree depth, held-out data and representative validation.
8. **Bias vs variance?** High bias underfits; high variance overfits sampling noise. Complex models often trade one against the other.
9. **Why scale?** Distance and linear regularization depend on input scale; tree splits are less sensitive. Standardize with training-fitted means/std.
10. **How is missing data handled?** Training-fold median for numbers and most-frequent for categories; compare alternatives on training CV, never compute replacement values from test data.

## Algorithms and evaluation

11. **Logistic regression?** Linear log-odds with sigmoid transformation; useful interpretable baseline and often strong on tabular data.
12. **Gaussian Naive Bayes?** Models each feature's class-conditional distribution as Gaussian and assumes conditional independence; simple and fast but assumptions can fail with correlated telemetry.
13. **KNN?** Predicts from nearest training samples; vulnerable to scaling, high dimensions and class imbalance.
14. **Decision Tree vs Random Forest?** One tree makes recursive splits; forest averages randomly varied trees to reduce variance, at the cost of interpretability and computation.
15. **Bagging vs boosting?** Bagging trains varied models (often in parallel) and aggregates; boosting sequentially focuses on correcting errors.
16. **Precision, recall and F2?** Precision = true alerts/all alerts; recall = caught failures/all failures; F2 weights recall more heavily than precision, useful when misses are particularly costly.
17. **Why not accuracy?** 93.08% of our simulated records are healthy; predicting “healthy” for every record gives ~93.08% accuracy but 0% failure recall.
18. **PR AUC vs ROC AUC?** PR centers the minority positive class and false alert tradeoff; ROC describes sensitivity and false-positive rate. For this dataset, report sklearn average precision precisely (not identical to trapezoidal PR AUC).
19. **What is MCC?** Correlation-based metric accounting for all four confusion matrix cells, useful with imbalance.
20. **What is the Brier score?** Mean squared error between predicted probabilities and actual binary labels; lower generally means better combined calibration and discrimination.
21. **What is class imbalance?** Failures are 6.92% of rows. Compare class weights, training-only undersampling and SMOTENC; a naive minority-class oversample before split leaks.
22. **What is SMOTENC?** A SMOTE variant designed for datasets with categorical features; ordinary interpolation of one-hot dummy columns can produce invalid synthetic categories.
23. **Why change the threshold?** A probability 0.5 cutoff may miss too many failures. Pick threshold from a separate validation set after calibration based on F2 or specified costs.
24. **Calibration vs discrimination?** Discrimination ranks positives above negatives; calibration assesses whether predicted frequencies match observed frequencies. A model can rank well but output unreliable probabilities.
25. **What is a false negative here?** A record labeled failed that gets a healthy/no-alert decision — a missed potential warning under the synthetic labels.
26. **What does SHAP/permutation importance tell you?** Which features influence performance/predictions; neither proves a causal mechanism. Correlation can split importances across redundant columns.
27. **PCA vs K-means?** PCA projects onto high-variance linear directions; K-means groups points by distance to learned centroids. Neither uses failure labels when fitting.
28. **Isolation Forest?** Anomaly detector that quickly isolates observations in random decision trees. Anomalies are not necessarily actual battery failures.

## Project-specific hard questions

29. **Can your model forecast failure one week early?** No. The CSV contains no timestamped observation-to-failure horizon, so that claim cannot be validated. We demonstrate cross-sectional synthetic classification only.
30. **Why remove thermal_runaway_risk and predicted_remaining_life_cycles?** They are precomputed scores with undocumented generation timing/provenance. Using them could inflate performance if they encode the target. Compare separately as a sensitivity experiment, not the baseline.
31. **Is a highly correlated variable necessarily leakage?** No. Correlation can simply reflect shared physics or redundant measurement; leakage depends on when and how a feature was generated.
32. **How do you avoid leakage?** Drop IDs/target/suspect scores, split first, fit all preprocessing/selection/resampling within training CV, calibrate and threshold on separate validation parts and open holdout once.
33. **What is your train/validation/test design?** 60% train (with 3-fold internal stratified CV), 10% calibration, 10% threshold selection, 20% final test. Set fixed random seed 42.
34. **Why is a random split a limitation?** Future EV performance may shift by year, vehicle fleet, battery chemistry and site. Without chronology we cannot measure true temporal generalization.
35. **What about repeated vehicles?** Current identifiers are unique per row; there is no per-vehicle longitudinal record. Real telemetry would require group-aware and time-aware splitting.
36. **Would you trust a 98% ROC AUC in real service?** Not on its own. Check leakages, prevalence/precision, sensor timing, future fleet shift, calibration, domain validation, and false-negative outcomes. Simulated data only.
37. **What would a deployable system need?** Labeled real-world longitudinal telemetry, defined prediction window, out-of-time/out-of-vehicle validation, calibrated alerts, drift monitoring, observability, rollback, access control, privacy and safety governance.
38. **Why is `joblib.load` risky?** The underlying pickle format can execute code. Only load trusted internally created files; never load arbitrary uploads from users.
39. **What if the API receives new category/missing telemetry?** `OneHotEncoder(handle_unknown='ignore')` tolerates new categories and fitted imputers handle missing values; API enforces schema and returns controlled errors for invalid fields. Out-of-distribution monitoring remains needed.
40. **How do you know the code will not crash?** You can't guarantee it. Establish test coverage, reproducible dependencies, input contracts, controlled failures, logging/monitoring and deployment-specific testing. Local unit/smoke tests are evidence, not proof of perfection.

## Practice plan (last 4 days)

- Day 11: Answer Q1–20 aloud; write your own examples.
- Day 12: Answer Q21–30; demo data split and training flow.
- Day 13: Answer Q31–40; trace missingness and unknown-category handling.
- Day 14: Ask teammates to cross-question each other in a 15-minute mock interview, then repeat each unclear answer in 60 seconds or less.
