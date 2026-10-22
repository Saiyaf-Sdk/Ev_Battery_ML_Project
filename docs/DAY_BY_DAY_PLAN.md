# 14-Day EV Battery Machine Learning Project — Zero to Viva

**How to study:** aim for 3–5 focused hours/day: 45–60 min theory, 90–150 min coding, 45 min experiments, 30 min explaining your findings aloud, 20 min Git/notes. If your Python experience is minimal, give extra time to Days 1–4. Every day: make a Git commit with a descriptive message, write one observation in `learning_log.md`, and explain *why* each design decision matters.

**Project question:** Can classical supervised ML distinguish failed batteries from healthy ones in a **simulated** 20,000-row EV telemetry dataset? What happens when we exclude suspect composite scores, address missing data/imbalance, calibrate risk estimates, and study structure without labels?

**Scientific constraint:** There is no observation timestamp, known failure time or defined lead-time horizon. This is a **cross-sectional failure classifier**, not empirically verified time-ahead forecasting. Make that distinction explicit in your report.

## Day 1 — Set up + Python/pandas basics

**Learn:** AI vs ML vs deep learning; feature (`X`) vs label (`y`); classification vs regression; supervised vs unsupervised; Python functions, variables, lists/dictionaries, `venv`, imports, pandas DataFrame/Series, `head()`, `shape`, `dtypes`.

**Build:** Install Python 3.11+, VS Code or PyCharm, Git. Open project. Create a virtual environment, install requirements. Run `python -m pytest` and `python -m evbattery.quality --data "PATH/TO/CSV"`. Examine the CSV without changing it. Open `src/evbattery/data.py` and trace `load_dataset()`.

**Hands-on:** In a new notebook or scratch script, print column names, first 5 rows, `df.shape`, `df.battery_failure.value_counts()`, and a numeric `describe()` table. Do not use IDs as predictors.

**Done when:** You can say what one row, one column, a feature, a label, and a model mean. Your terminal commands run on your machine.

**Viva question:** Why is battery failure a *binary classification* problem rather than regression?

## Day 2 — Exploratory data analysis (EDA)

**Learn:** mean/median, standard deviation, distributions, class imbalance, missing percentages, histograms, boxplots, categorical frequency tables, Pearson correlation and why association is not causation.

**Build:** Run the data-quality script. Plot failure-class counts, histogram of battery health vs class, maximum cell temperature vs class, and a correlation heatmap for selected numeric fields. List the ten features with the most missing data. Check plausible physical ranges (temperature min ≤ max; percentage fields in 0–100 if they truly represent percentages).

**Hands-on:** Study *actual* dataset findings: 20,000 rows; 70 columns; 1,384 failures (6.92%); 53,344 missing cells (3.81% of table); zero exact duplicate rows. Confirm them yourself. Explain what “93% accuracy” could mean for a trivial healthy-only model.

**Done when:** You have at least four correctly titled charts and a short EDA summary with limitations.

**Viva question:** Why could very high accuracy be misleading for this dataset?

## Day 3 — Leakage, features and splitting

**Learn:** target leakage, train/validation/test contamination, feature provenance, stratified split, reproducibility via random seeds; distinction between random splits and time-based splits.

**Build:** Review `src/evbattery/config.py` for excluded identifiers and suspect engineered/composite risk scores. Check `make_features()` in `data.py`: temperature gap, ambient-temperature range, cycles/year, distance/cycle, fast-charge stress, previous warning counts and service flag.

**Hands-on:** Run `split_data()` and verify 60% fitting, 10% calibration, 10% threshold, 20% test, all with similar failure prevalence. Never explore the test set when selecting models. Explain why `predicted_remaining_life_cycles` could unfairly help if generated after or from labels.

**Done when:** You can draw the split diagram and explain why all fitted transformations belong within training folds.

**Viva question:** What is data leakage? Give two examples from our project.

## Day 4 — Preprocessing as ML pipelines

**Learn:** median vs KNN imputation; one-hot encoding; unknown categories; standardization `(x-mean)/std`; why trees don't need scaling but logistic regression/KNN do; `ColumnTransformer` and `Pipeline`.

**Build:** Read `modeling.py` and `experiments.py`. Run `python -m evbattery.experiments --data "PATH/TO/CSV" --max-rows 2500`. The same stratified training sample is used across all methods; KNN imputation is limited for reasonable runtime. Note that selecting a feature based on all 20k rows before CV would be leakage.

**Hands-on:** Write down an example of imputing one missing cell. Show how three categorical values become one-hot columns. Compare selected CV average-precision scores.

**Done when:** You can explain every pipeline stage in plain language and why it must be refit within each fold.

**Viva question:** What happens when the model receives an unseen vehicle brand?

## Day 5 — Baselines and supervised learning

**Learn:** how `fit(X,y)`, `predict(X)` and `predict_proba(X)` differ; logistic regression, sigmoid probability, regularization; decision tree splits; dummy baseline; underfitting vs overfitting.

**Build:** Read `modeling.py`. Optionally run the complete `train.py --fast` on a small development extract as a functional smoke test; do not publish those model scores. Compare dummy, logistic and decision-tree CV performance (training only).

**Hands-on:** Explain the meaning of a logistic regression coefficient and why it isn't automatically causal. Sketch a simple decision tree rule based on temperature and health.

**Done when:** You can describe the flow from raw row to model risk score.

**Viva question:** Why include a simple logistic regression if more complex models are available?

## Day 6 — Compare seven supervised classifiers

**Learn:** Gaussian Naive Bayes (conditional independence assumption), KNN (distance), Decision Tree (splits), Random Forest (bagging), AdaBoost (sequential weak learners), Histogram Gradient Boosting (residual correction) and SGD logistic regression (stochastic optimization), plus Logistic Regression. KNN is optional because repeated distance computations can be expensive.

**Build:** Run full supervised training when ready: `python -m evbattery.train --data "PATH/TO/CSV" --outdir artifacts`. Examine `model_cv_leaderboard.csv`. The dummy baseline is an eighth non-learning reference.

**Hands-on:** Make a comparison table: algorithm, interpretable?, scaling sensitive?, nonlinear?, average precision mean/std, ROC AUC mean. Do not name a winner before seeing your results.

**Done when:** You have ≥5 actual trained algorithms, CV results, and a reasoned model selection.

**Viva question:** Why does Random Forest reduce single-tree variance? Does it eliminate overfitting?

## Day 7 — Evaluation for rare events

**Learn:** TP/TN/FP/FN; precision = TP/(TP+FP); recall = TP/(TP+FN); F1 harmonic mean; F2 weights recall more; ROC AUC; precision–recall curve; average precision (AP); Matthews correlation coefficient; balanced accuracy; Brier score.

**Build:** Read `evaluation.py`. Use `model_cv_leaderboard.csv`, `threshold_validation_metrics.json`, and `test_metrics.json`. Explain the confusion matrix in practical battery-warning terms: FN = missed failing battery; FP = unnecessary warning.

**Hands-on:** Manually compute precision/recall/F2 for a made-up confusion matrix. Compare AP to baseline failure prevalence (~0.0692). **Important:** sklearn `average_precision_score` is a specific AP estimator, not trapezoidal PR AUC; label your graphs/results accurately.

**Done when:** You can choose useful rare-class metrics instead of blindly reporting accuracy.

**Viva question:** If recall improves but false alarms double, what tradeoff has occurred?

## Day 8 — Class imbalance experiments

**Learn:** weighted loss; random undersampling; random oversampling; SMOTE and its assumptions; why ordinary SMOTE on one-hot categories can make nonsensical synthetic categories; SMOTENC for mixed tabular data; why resampling is done inside CV training folds only.

**Build:** `python -m evbattery.imbalance --data "PATH/TO/CSV"`. It compares no sampling, logistic class weights, random undersampling and SMOTENC. Review `imbalance.py` to identify where sampling occurs relative to validation folds.

**Hands-on:** Discuss cost ratios (missed failure vs false alarm) as hypothetical assumptions, not real service costs. Inspect tradeoffs across methods. These separate experiments do not silently replace your final trained model; record the outcome for a justified follow-up iteration.

**Done when:** You can explain why running SMOTE before train–test splitting is invalid.

**Viva question:** If only 6.92% fail, why not simply oversample to 50/50 and use accuracy?

## Day 9 — Feature selection and hyperparameter tuning

**Learn:** hyperparameters vs learned parameters; `RandomizedSearchCV`, cross-validation grid; regularization; overfitting to validation; L1 embedded selection vs filtering vs recursive feature elimination; collinearity vs leakage.

**Build:** Review `experiments.py`: it fits median/KNN imputers, correlation pruning and L1 feature selection **inside training folds**. Run model-specific tuning with `evbattery.tune` (see README). Do **not** compare configurations on the final holdout.

**Hands-on:** Compare features with and without suspect composite scores in a development-only sensitivity study. Do not train a final “winner” by repeatedly choosing the setting that looks best on test data.

**Done when:** You can explain which parameters were tuned and what evidence supports any change.

**Viva question:** What is the difference between cross-validation and hyperparameter tuning?

## Day 10 — Calibration and threshold selection

**Learn:** discrimination vs probability calibration; Platt/sigmoid vs isotonic; Brier score; reliability diagram; choosing a threshold to optimize F2 on an *independent* threshold-validation set.

**Build:** Inspect `train.py`: the model is trained on 60%, sigmoid-calibrated on 10%, threshold is chosen on a different 10%, and finally tested on untouched 20%. Compare a 0.5 threshold with your selected threshold using **threshold-validation** results only.

**Hands-on:** Explain why an alert threshold of 0.25 is not the same as saying 25% of battery packs will fail; calibration is only as meaningful as the synthetic label definition and representative data.

**Done when:** You can distinguish probability ranking from probability accuracy and warning decisions.

**Viva question:** Why shouldn't we select the alarm threshold using the test set?

## Day 11 — Unsupervised ML

**Learn:** what happens when there is no training label; PCA (variance directions), K-means (centroids), silhouette score, anomalies with Isolation Forest; why clusters aren't automatically healthy/failing groups.

**Build:** `python -m evbattery.unsupervised --data "PATH/TO/CSV"`. Open `artifacts/unsupervised/pca_clusters.png`, `cluster_failure_summary.csv` and `unsupervised_summary.json`. Labels are only used *after* unsupervised fitting to check enrichment.

**Hands-on:** Describe two dimensions of PCA, whether 3 clusters separate clearly, and whether higher anomaly scores are associated with known failures.

**Done when:** You have an unsupervised visual and a cautious interpretation, without claiming unsupervised training used labels.

**Viva question:** Why is K-means unsupervised if you later compare clusters against `battery_failure`?

## Day 12 — Interpretability and serving a model

**Learn:** feature importance vs causation, permutation importance, model serialization, feature schema contracts, API request/response, safe deployment limitations and secrets.

**Build:** `python -m evbattery.explain --data "PATH/TO/CSV"`; view `permutation_importance.csv`. Use `python -m evbattery.example_request --data "PATH/TO/CSV"`; configure `EV_API_KEY` and `EV_MODEL_PATH`, and run `uvicorn evbattery.api:app --host 127.0.0.1 --port 8000`. Test `/health` and authenticated `/predict` with a sample JSON request.

**Hands-on:** Predict for the first row, then experiment with a different realistic sensor value and explain that this is not a causal counterfactual experiment. Never accept uploaded arbitrary `.joblib` artifacts.

**Done when:** You can load your own trained model and obtain a structured result from a local endpoint.

**Viva question:** What's the difference between building a good classifier and maintaining a reliable production ML system?

## Day 13 — Engineering tests, robustness, reporting

**Learn:** assertions, unit tests, integration tests, reproducibility, dependency control, safe inference, train–serve skew, failure modes, input validation, operational monitoring and drift.

**Build:** Run `python -m pytest`. Test missing values, unknown categories, zero divisors, invalid labels, duplicate keys, API missing token/unknown fields. Save final figures and comparisons. Fill `docs/REPORT_TEMPLATE.md` using **real measured results** and include their limitations.

**Hands-on:** Simulate a malformed request; verify you get a controlled error rather than a stack trace sent to the user. Commit code, not data, passwords or serialized model files.

**Done when:** All local tests pass, outputs are reproducible, and your report clearly distinguishes observed results from assumptions.

**Viva question:** How can data drift make a valid offline model unreliable after deployment?

## Day 14 — Semester presentation and interview readiness

**Learn:** explaining results simply, making scientific claims at the right strength, tradeoffs in rare-event classification, threat modeling and ML system architecture.

**Build:** Present 10–12 slides: problem → dataset → EDA → label imbalance → leakage policy → pipeline/split → models → metrics/imbalance → calibration/threshold → unsupervised → final demo → limitations/future work. Demo two API requests, including one intentionally invalid request to show validation.

**Hands-on:** Practice a 7-minute technical walkthrough, then 15 minutes of viva questions without reading notes. Use `docs/INTERVIEW_PREP.md`. State what the dataset cannot prove: real EV failure rates, forecasting horizon, and deployment readiness.

**Done when:** Every team member can explain the data, model, metrics, leakage controls, calibration, limitations and code without copying a generated script.

---

## Suggested four-person division (adapt to your actual team)

- **Member A:** data profiling, visualizations, leakage and feature-set audit.
- **Member B:** supervised models, CV and tuning.
- **Member C:** imbalance handling, thresholding, calibration and evaluation.
- **Member D:** unsupervised learning, explanation, API, integration/testing.

Everyone reviews each other's changes, learns the complete pipeline and can describe the scientific limitations. Rotate presentation responsibilities; avoid silos.

## If you only have seven days

1. **Day 1:** Python setup, load/validate, EDA and leakage policy.
2. **Day 2:** splitting, preprocessing and feature engineering.
3. **Day 3:** baseline + 7 algorithms, 3-fold CV and metrics.
4. **Day 4:** class imbalance experiments, feature comparisons/tuning.
5. **Day 5:** calibration, thresholds, final test; interpret failures.
6. **Day 6:** PCA/IsolationForest, interpretability and local API.
7. **Day 7:** tests, slides, report and viva rehearsal.

Seven days is a compressed implementation sprint, not enough to master all machine-learning theory for interviews. Fourteen days is the better option.

## Definition of done / grading checklist

- [ ] Raw CSV preserved and loaded by one documented command.
- [ ] Problem framed correctly as synthetic binary classification.
- [ ] Missing values, duplicate checks, distributions and feature relationships explained.
- [ ] IDs and suspicious composite scores handled explicitly, with rationale.
- [ ] No imputation/scaling/selection/resampling fit on held-out samples.
- [ ] At least five supervised algorithms compared via stratified CV.
- [ ] Imbalance study compares none, weighting, undersampling and SMOTENC.
- [ ] Missing-data and correlated-feature/selection experiments documented.
- [ ] Hyperparameter tuning and selection strategy reproducible.
- [ ] Precision, recall, F1, F2, AP, ROC AUC, MCC, balanced accuracy evaluated.
- [ ] Calibration, Brier score and threshold strategy documented.
- [ ] PCA, clustering and anomaly detection fitted without labels.
- [ ] Feature explanations, false-negative analysis, limitations included.
- [ ] Model saved with metadata, test results and source revision.
- [ ] Tests, simple demo, final report and oral walkthrough delivered.
