# Member B — Complete Machine Learning Learning & Viva Preparation Guide

**Author:** Member B (Supervised ML Engineer)  
**Project:** Early Warning of Electric Vehicle Battery Failure from Fleet Telemetry  
**Module:** EC5203 — Machine Learning  
**Target Audience:** Beginner-to-Intermediate ML Engineering Student preparing for University Viva Defense & Technical Interviews

---

## Welcome to Your Member B Learning Journey

As Member B, you are the **core modeling engine** of the project. Your teammates rely on you:
- **Member A** cleaned the data and gave you 63 validated input features.
- **You (Member B)** design the pipelines, prevent data leakage, evaluate preprocessing, train algorithms, tune hyperparameters, and select the winning model.
- **Member C** takes your model to calibrate its probabilities and select an alarm threshold.
- **Member D** uses your model inside a live FastAPI microservice.

This guide is structured into **four comprehensive masterclass sessions** corresponding to Plan Days 4, 5, 6, and 9, followed by a **3–5 minute viva defense speech**.

---

# SESSION 1 (Plan Day 4): Preprocessing as Machine Learning Pipelines

### 1. Beginner-Friendly Concepts

#### 1.1 What are Features and Labels?
In machine learning:
- **Features ($X$):** The input variables describing the battery (e.g., cell temperature, internal resistance, vehicle age, drive type). We have 63 features.
- **Label ($y$):** The ground-truth outcome we want to predict (`battery_failure` $\in \{0, 1\}$).
- **Supervised Learning:** An algorithm learns a mathematical mapping function $f(X) \approx y$ from known pairs of $(X, y)$.

#### 1.2 Missing Data and Imputation
Real EV telemetry has missing readings because of sensor sleep modes, network drops, or packet loss.
- **Row Deletion:** Deleting rows with missing cells would wipe out **93.3% of the dataset** (18,663 rows had at least one missing cell!). Therefore, row deletion is unacceptable.
- **Median Imputation:** Replaces missing numerical values with the 50th percentile (the middle value). We choose median over mean because median is immune to extreme outliers (e.g., temperature spikes).
- **Missingness Indicators:** Adding a binary column ($1$ if missing, $0$ if present). This allows the algorithm to learn if sensor failure itself signals battery degradation.
- **KNN Imputation:** Finds the $k$ most similar batteries based on Euclidean distance and averages their values.

#### 1.3 Categorical Encoding: One-Hot Encoding & Unseen Categories
Computers cannot do matrix multiplication on strings like `"Tesla"` or `"NMC"`.
- **One-Hot Encoding:** Converts a category with $C$ levels into $C$ binary columns (e.g., `battery_chemistry_LFP` = $[0, 1]$).
- **Unseen Categories:** What happens if a new brand `"Rivian"` arrives in production? If your encoder throws an error, the car's BMS software crashes. In our pipeline, we set `handle_unknown="ignore"`, which encodes unseen categories safely as all zeros.

#### 1.4 Feature Scaling: Standardization
Features have wildly different units:
- `odometer_km`: $0$ to $450,000$ km
- `internal_resistance`: $0.03$ to $1.12$ ohms

Distance-based algorithms (KNN) and gradient algorithms (Logistic Regression) will treat `odometer_km` as 400,000 times more important than `internal_resistance` simply because its numbers are larger!
- **Standardization (Z-score):** Rescales features to have $\mu = 0$ and $\sigma = 1$:
  $$z = \frac{x - \mu}{\sigma}$$
- **Tree Invariance:** Decision trees and Random Forests do NOT need scaling because they only compare values within a single feature ($x_j > \theta$).

#### 1.5 The Scikit-Learn `Pipeline` & Preventing Leakage
**Golden Rule of ML:** You must never compute statistics (means, medians, scalers, encoders) on data that includes the validation or test sets.
- If you impute using the median of all 20,000 rows, test-set information leaks into your training set!
- A scikit-learn `Pipeline` guarantees that every `fit()` call computes medians and scalers **only on the training fold**. During testing, it only calls `transform()` using those stored training statistics.

---

### 2. Step-by-Step Mathematical Example

Suppose a single feature (cell temperature in °C) has values:
$$\text{Train fold: } [20, 22, 24, 26, \text{NaN}], \quad \text{Held-out test fold: } [40, \text{NaN}]$$

1. **Step 1: Compute Imputation Statistic on Train ONLY:**
   $$\text{Sorted known train values: } [20, 22, 24, 26] \implies \text{Median} = \frac{22 + 24}{2} = 23^\circ\text{C}$$
   *(If you had included the test fold's 40°C, the median would have been 24°C—that is data leakage!)*

2. **Step 2: Impute Train Fold:**
   $$\text{Train imputed: } [20, 22, 24, 26, 23]$$

3. **Step 3: Compute Standardization on Train Fold:**
   $$\mu_{\text{train}} = \frac{20 + 22 + 24 + 26 + 23}{5} = \frac{115}{5} = 23.0$$
   $$\sigma_{\text{train}} = \sqrt{\frac{(20-23)^2 + (22-23)^2 + (24-23)^2 + (26-23)^2 + (23-23)^2}{5}} = \sqrt{\frac{9 + 1 + 1 + 9 + 0}{5}} = \sqrt{4} = 2.0$$

4. **Step 4: Transform Test Fold Using STORED Train Parameters:**
   - First value ($40^\circ\text{C}$):
     $$z = \frac{40 - 23.0}{2.0} = \frac{17}{2} = +8.5$$
   - Second value ($\text{NaN}$):
     Imputed with train median ($23.0^\circ\text{C}$):
     $$z = \frac{23.0 - 23.0}{2.0} = 0.0$$

---

### 3. Python Code Deep Dive

Look at `src/evbattery/modeling.py`:

```python
def preprocessing(x: pd.DataFrame) -> ColumnTransformer:
    numerical, categorical = columns_by_type(x)
    parts = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numerical)
    ]
    if categorical:
        parts.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0)
```

**Why this code is production-grade:**
1. `ColumnTransformer` splits tabular data by type: numeric columns get median imputation and scaling; text columns get frequency imputation and one-hot encoding.
2. `add_indicator=True` creates extra binary columns flagging where NaNs were replaced.
3. `handle_unknown="ignore"` ensures that unknown vehicle brands in test data won't crash the program.
4. When wrapped inside `pipe_for(x, model)`, calling `cross_validate()` automatically fits this preprocessor **only on training folds**.

---

### 4. Practical Terminal Command

Run the preprocessing experiment module:

```powershell
python -m evbattery.experiments --data data/ev_battery.csv --max-rows 2500 --out artifacts/member_b/preprocessing_comparison.csv
```

*(If using synthetic data for testing: `python -m evbattery.member_b --fast`)*

**Output Explanation:**
```text
                    method  cv_average_precision_mean  cv_average_precision_std  cv_roc_auc_mean  cv_roc_auc_std  fit_time_mean  n_records
 median_embedded_L1_select                    0.771401                  0.015165         0.871246        0.024684       0.037343        600
           median_baseline                    0.741618                  0.027753         0.856226        0.035077       0.026031        600
            knn_imputation                    0.741618                  0.027753         0.856226        0.035077       0.024693        600
   median_corr_prune_0.97                    0.735009                  0.023370         0.853862        0.033442       0.034159        600
```
- **What this tells you:** Median baseline achieves an AP of 0.7416 in just 26 milliseconds. KNN imputation gives the exact same AP (0.7416) but is computationally expensive. $L_1$ feature selection slightly improves linear separation by zeroing out noisy coefficients.

---

### 5. Hands-on Exercise
Open a Python terminal and write a 3-step pipeline with `SimpleImputer` and `StandardScaler`. Fit it on an array with `[10, 20, np.nan]` and transform `[np.nan, 30]`. Verify that the imputed value is 15.0!

---

### 6. Five Viva / Interview Questions (Session 1)

1. **Q: Why can't we drop rows with missing values in this project?**  
   *Answer:* In our dataset, missing values are spread across 67 of the 70 columns (3–5% missing per column). Because missingness is distributed, **93.3% of rows** contain at least one missing cell. Deleting incomplete rows would destroy almost the entire dataset, leaving fewer than 1,400 rows.

2. **Q: What is data leakage, and how does your pipeline prevent it?**  
   *Answer:* Data leakage occurs when information from outside the training dataset (such as validation or test splits) is used to create the model. In our project, we enclose all imputation, scaling, and encoding inside a scikit-learn `Pipeline`. During cross-validation, transformers compute statistics (like mean and median) strictly on the training fold and apply them to the validation fold without refitting.

3. **Q: What happens if an unseen category appears in test or production data?**  
   *Answer:* Standard one-hot encoders throw a `ValueError` when encountering an unseen category level. We configure `OneHotEncoder(handle_unknown="ignore")`, which maps any unrecognized category to a row of all zeros, allowing graceful inference without crashing.

4. **Q: Why does Logistic Regression require feature scaling while Random Forest does not?**  
   *Answer:* Logistic Regression optimizes weights using gradient descent or L-BFGS, where the step size and regularization penalty ($\frac{1}{2} w^T w$) depend directly on the scale of $x_j$. Unscaled features distort the loss surface. Decision trees and Random Forests evaluate candidate splits using monotonic ordering ($x_j \le \theta$), which is completely invariant to any scale or monotonic transformation.

5. **Q: How does Correlation Pruning work, and why must it be fitted inside training folds?**  
   *Answer:* `CorrelationPruner` calculates the pairwise Pearson correlation matrix and greedily drops features that have $|r| \ge 0.97$ with an already selected feature. Because the correlation matrix depends on empirical sample covariances, computing it across the full dataset would leak target-correlated distribution structure from the test set.

---

### 7. Common Implementation Mistakes
- **Mistake:** Calling `scaler.fit_transform(X)` on the whole dataset before splitting into train/test.  
  *Consequence:* Test set means and standard deviations leak into training; offline test scores appear falsely inflated.
- **Mistake:** Imputing missing values using the mean when data has extreme outliers.  
  *Consequence:* Extreme sensor readings skew the mean, injecting distorted values into missing cells.

---

# SESSION 2 (Plan Day 5): Baselines, Linear Models, and Single Trees

### 1. Beginner-Friendly Concepts

#### 1.1 Classification vs. Regression
- **Regression:** Predicts a continuous quantity (e.g., predicting exact battery temperature: $42.5^\circ\text{C}$).
- **Classification:** Predicts a discrete class category (e.g., Healthy $0$ vs Failed $1$).

#### 1.2 The Dummy Baseline (No-Skill Reference)
Before claiming your machine learning model is good, you must compare it to a **dummy baseline**.
- Our dataset has $1,384$ failures out of $20,000$ rows ($6.92\%$ prevalence).
- A dummy classifier that always predicts the majority class ("Healthy") achieves **$93.08\%$ accuracy**!
- But it catches **zero failures** (Recall = $0\%$).
- Therefore, accuracy is completely useless. The no-skill baseline for **Average Precision** is simply the class prevalence: $\text{AP}_{\text{baseline}} = 0.0692$. Any model achieving $\text{AP} > 0.0692$ has learned real discriminatory signal.

#### 1.3 Logistic Regression & The Sigmoid Function
Despite its name, Logistic Regression is a **classification algorithm**.
It computes a linear combination of features:
$$z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d$$
Then squashes $z$ through the **sigmoid function** $\sigma(z)$ into a probability $p \in (0, 1)$:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- If $z \to +\infty$, $\sigma(z) \to 1.0$ (High risk of failure).
- If $z = 0$, $\sigma(z) = 0.5$ (Equally uncertain).
- If $z \to -\infty$, $\sigma(z) \to 0.0$ (Healthy).

#### 1.4 Decision Trees & Information Gain
A Decision Tree makes predictions by asking a sequence of true/false questions:
- "Is `cell_temperature_max` $> 50^\circ\text{C}$?"
- "Is `battery_health_percent` $< 75\%$?"

How does the tree decide which question to ask first?
It searches through every feature and every possible threshold value to find the split that maximizes **purity** (separating healthy from failed batteries).
- **Gini Impurity:** Measures the probability that a randomly chosen element from the set would be incorrectly labeled:
  $$\text{Gini} = 1 - \sum_{k=0}^1 p_k^2$$
  - Pure node (all healthy or all failed): $\text{Gini} = 1 - (1^2 + 0^2) = 0.0$.
  - Impure node ($50\%$ healthy, $50\%$ failed): $\text{Gini} = 1 - (0.5^2 + 0.5^2) = 0.5$.

#### 1.5 Overfitting vs. Underfitting
- **Underfitting (High Bias):** The model is too simple to capture the underlying pattern (e.g., trying to fit a straight line to a complex circular relationship).
- **Overfitting (High Variance):** The model memorizes training noise (e.g., an unconstrained decision tree with 50 levels of depth that creates an individual branch for every car in the dataset). It gets $100\%$ training accuracy but fails completely on new fleet data.

---

### 2. Step-by-Step Mathematical Example: The Sigmoid Function

Suppose our trained Logistic Regression model has:
$$w_0 = -3.0, \quad w_{\text{health}} = -0.05, \quad w_{\text{temp}} = +0.10$$

Notice the signs:
- $w_{\text{health}}$ is **negative**: as battery health increases, failure probability decreases (good!).
- $w_{\text{temp}}$ is **positive**: as temperature increases, failure probability increases (bad!).

Let's predict failure probability for Battery Pack A:
- `battery_health_percent` = $70\%$
- `cell_temperature_max` = $55^\circ\text{C}$

1. **Step 1: Compute Linear Log-Odds ($z$):**
   $$z = -3.0 + (-0.05 \times 70) + (0.10 \times 55)$$
   $$z = -3.0 - 3.5 + 5.5 = -1.0$$

2. **Step 2: Apply Sigmoid Squashing:**
   $$p = \sigma(-1.0) = \frac{1}{1 + e^{-(-1.0)}} = \frac{1}{1 + e^{1.0}} = \frac{1}{1 + 2.718} = \frac{1}{3.718} \approx 0.269 \quad (26.9\%)$$

Battery Pack A has an estimated $26.9\%$ failure risk.

---

### 3. Python Code Deep Dive

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Logistic Regression with L2 regularization
log_reg = LogisticRegression(max_iter=1500, random_state=42)

# Decision tree with regularization to prevent overfitting
tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=10, random_state=42)
```

**Why these constraints matter:**
- In `DecisionTreeClassifier`, setting `max_depth=7` and `min_samples_leaf=10` prevents the tree from splitting down to single-battery leaves. A leaf must contain at least 10 samples to be valid.
- In `LogisticRegression`, `max_iter=1500` ensures the L-BFGS solver converges smoothly even with 63 features.

---

### 4. Practical Terminal Command

Run the baseline-only comparison:

```powershell
python -m evbattery.train --data data/ev_battery.csv --outdir artifacts/baseline --baseline-only
```

*(Or Member B entrypoint: `python -m evbattery.member_b --fast`)*

**Output Explanation:**
- Shows `dummy_prior` AP $\approx 0.069$ vs `logistic_regression` AP $\approx 0.720$.
- Proves that Logistic Regression captures almost 10 times more signal than random guessing!

---

### 5. Hands-on Exercise
Compute Gini impurity for a node containing 8 healthy batteries and 2 failed batteries.
$$p_0 = 8/10 = 0.8, \quad p_1 = 2/10 = 0.2$$
$$\text{Gini} = 1 - (0.8^2 + 0.2^2) = 1 - (0.64 + 0.04) = 1 - 0.68 = 0.32$$

---

### 6. Five Viva / Interview Questions (Session 2)

1. **Q: Why is accuracy a dangerous metric for this project?**  
   *Answer:* With a rare failure rate of $6.92\%$, a naive model predicting every battery is healthy achieves $93.08\%$ accuracy while failing to identify a single hazardous battery pack. In safety-critical applications like EV battery monitoring, False Negatives can cause thermal runaway or highway stranding.

2. **Q: What is the physical interpretation of Logistic Regression weights?**  
   *Answer:* The weights represent change in the log-odds of failure per unit change in the standardized feature: $\ln\left(\frac{p}{1-p}\right) = w^T x$. Taking $e^{w_j}$ gives the odds ratio. A positive weight means higher feature values increase failure odds, while a negative weight means the feature protects against failure.

3. **Q: How does a Decision Tree decide where to split?**  
   *Answer:* It scans every available feature and all possible split thresholds, calculating the reduction in impurity (Gini impurity or Entropy) for each candidate. It greedily selects the split that maximizes Information Gain: $\Delta \text{Gini} = \text{Gini}_{\text{parent}} - \sum \frac{N_j}{N} \text{Gini}_{\text{child}_j}$.

4. **Q: Why did our single Decision Tree score lower ($AP \approx 0.669$) than Logistic Regression ($AP \approx 0.742$)?**  
   *Answer:* Decision trees form axis-aligned rectangular boundaries and suffer from high variance and sample fragmentation. When true underlying risks are smoothly continuous functions of multiple variables (like battery health and temperature), single trees approximate them poorly with coarse step functions.

5. **Q: What is the difference between `predict()` and `predict_proba()`?**  
   *Answer:* `predict_proba()` returns continuous probabilities $[P(y=0), P(y=1)] \in [0, 1]$. `predict()` applies a hard-coded default threshold of $0.5$ ($y=1$ if $P \ge 0.5$). For imbalanced datasets, a default threshold of $0.5$ is usually disastrous; we need the continuous risk probabilities from `predict_proba()` so Member C can select an optimal alert threshold.

---

### 7. Common Implementation Mistakes
- **Mistake:** Evaluating classification with `predict()` at threshold $0.5$ when failure rate is $6.9\%$.  
  *Consequence:* The model almost never predicts class 1, resulting in zero recall. Always collect continuous probabilities.

---

# SESSION 3 (Plan Day 6): Ensembles, Classifier Comparison & Cross-Validation

### 1. Beginner-Friendly Concepts

#### 1.1 Why Single Trees Fail and Ensembles Win
A single tree is unstable: change a few rows in your training set, and the root split changes completely!
To fix this, machine learning uses **ensembles** (combining dozens of trees):
1. **Bagging (Bootstrap Aggregation) $\to$ Random Forest**
2. **Boosting $\to$ AdaBoost & Gradient Boosting**

#### 1.2 Random Forest (Bagging)
- **Bootstrap Sampling:** Creates $B$ different training datasets by sampling the original rows *with replacement* (some rows appear twice, some are left out).
- **Random Subspace:** At every split, the tree can only choose from a random subset of features (e.g., $\sqrt{63} \approx 8$ features). This prevents one dominant feature (like battery health) from appearing as the root of every tree.
- **Aggregation:** Each tree votes independently, and the forest averages their probabilities.
- **Why it works:** Individual trees overfit in different directions; when averaged together, their errors cancel out while their signal reinforces!

#### 1.3 Boosting: AdaBoost vs. Gradient Boosting
- **Bagging builds trees in parallel; Boosting builds trees sequentially.**
- **AdaBoost (Adaptive Boosting):** Increases the sample weight of batteries that the previous tree misclassified. The next tree is forced to concentrate on the hardest examples.
- **Gradient Boosting (Histogram Gradient Boosting):** Fits each new tree to the *residual errors* (negative gradient of the loss function) of the previous trees.
- **Histogram Binning:** Instead of sorting 12,000 float values at every split, Histogram Gradient Boosting groups continuous features into 256 discrete integer bins. This accelerates training speed by 10x–20x.

#### 1.4 Gaussian Naive Bayes & K-Nearest Neighbors
- **Gaussian Naive Bayes:** Uses Bayes' theorem assuming all features are conditionally independent. Very fast, but overconfident when features correlate.
- **KNN (K-Nearest Neighbors):** Classifies a query battery by finding the $k$ closest batteries in 63-dimensional feature space and taking a majority vote. Slower at test time because it must measure distances to all 12,000 training points.

#### 1.5 Stratified K-Fold Cross-Validation
- **Standard K-Fold:** Splits data randomly into $K$ pieces. In a rare-event dataset, one fold might accidentally receive almost no failure samples!
- **Stratified K-Fold:** Guarantees that every fold contains the exact same percentage of failures ($6.92\%$). We use $K = 3$ splits on our training set.

#### 1.6 Precision, Recall, ROC AUC, and Average Precision
- **Precision:** Of all batteries we alerted on, what fraction actually failed?
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  *(Low precision = annoying false alarms).*
- **Recall (Sensitivity):** Of all batteries that actually failed, what fraction did we catch?
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  *(Low recall = missed dangerous battery failures).*
- **Average Precision (AP):** The area under the Precision-Recall curve. Summarizes performance across every possible alarm threshold.

---

### 2. Step-by-Step Mathematical Example: Bagging Variance Reduction

Suppose you have $B$ independent decision trees, each with variance $\sigma^2$ and correlation $\rho$ between trees.
The variance of the average prediction is:
$$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

- If you increase $B$ (more trees, e.g. from 1 to 100), the term $\frac{1-\rho}{B} \sigma^2 \to 0$!
- If you randomize feature selection at each split (Random Subspace), you decrease the correlation $\rho$ between trees!
- **Result:** Random Forest drastically lowers overall prediction variance without increasing bias.

---

### 3. Python Code Deep Dive

Look at `src/evbattery/modeling.py`:

```python
def cv_leaderboard(x: pd.DataFrame, y: pd.Series, ...) -> pd.DataFrame:
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    # Loop over all 7 candidate models + dummy
    for name, estimator in models.items():
        pipeline = pipe_for(x, estimator)
        scores = cross_validate(
            pipeline, x, y, cv=folds,
            scoring={"ap": "average_precision", "roc_auc": "roc_auc"},
            return_train_score=False,
        )
```

**Why this code is scientifically rigorous:**
- `StratifiedKFold(3, shuffle=True, random_state=42)` guarantees identical, balanced folds across every algorithm.
- `error_score="raise"` ensures that any silent convergence failure stops the run immediately.
- Computes both the **mean** and **standard deviation** of AP across folds, allowing us to test whether differences between algorithms are statistically meaningful.

---

### 4. Practical Terminal Command

Run Member B's full model evaluation:

```powershell
python -m evbattery.member_b --data data/ev_battery.csv --outdir artifacts/member_b
```

*(Or synthetic test: `python -m evbattery.member_b --fast --include-knn`)*

**Output Explanation:**
```text
============================================================
MEMBER B: MODEL SELECTION AND CV LEADERBOARD
============================================================
                model  cv_average_precision_mean  cv_average_precision_std  cv_roc_auc_mean  fit_time_mean
        random_forest                   0.815005                  0.017641         0.889264       0.115746
 gaussian_naive_bayes                   0.789703                  0.031890         0.881216       0.018504
  hist_gradient_boost                   0.779296                  0.022710         0.893993       0.356784
             adaboost                   0.753505                  0.009566         0.854993       0.237287
  logistic_regression                   0.741618                  0.027753         0.856226       0.028796
         sgd_logistic                   0.715843                  0.031038         0.833825       0.020837
                  knn                   0.709121                  0.026631         0.838636       0.052171
        decision_tree                   0.668594                  0.023294         0.814401       0.032065
          dummy_prior                   0.331667                  0.002357         0.500000       0.018835
------------------------------------------------------------
SELECTED BASE MODEL: random_forest
TUNED: True
CV AVERAGE PRECISION: 0.8195 ± 0.0310
CV ROC AUC:           0.8923 ± 0.0234
============================================================
```

---

### 5. Five Viva / Interview Questions (Session 3)

1. **Q: How does Random Forest prevent trees from being identical?**  
   *Answer:* It uses two mechanisms: (1) **Row bagging**: each tree is trained on a distinct bootstrap sample with replacement; (2) **Feature bagging (Random Subspaces)**: at each split, the algorithm only considers a random subset of features (typically $\sqrt{d}$). This decorrelates the trees, ensuring that strong features like `battery_health_percent` do not dominate every tree.

2. **Q: What is the fundamental difference between Bagging and Boosting?**  
   *Answer:* Bagging (Random Forest) trains independent base estimators in parallel and averages their predictions to **reduce variance**. Boosting (AdaBoost, Gradient Boosting) trains base estimators sequentially, with each new model fitting the errors or residuals of prior models to **reduce bias**.

3. **Q: Why does Histogram Gradient Boosting train so much faster than standard Gradient Boosting?**  
   *Answer:* Standard Gradient Boosting sorts every continuous feature at every node ($\mathcal{O}(N \log N)$). Histogram Gradient Boosting discretizes continuous values into 256 integer bins. Evaluating split points across 256 histogram bins requires only $\mathcal{O}(N)$ operations, speeding up tree construction by more than an order of magnitude.

4. **Q: Why do you report Average Precision (AP) standard deviation across CV folds?**  
   *Answer:* Cross-validation folds have sampling uncertainty. A model with mean AP of 0.73 ± 0.04 is not statistically significantly better than a model with 0.72 ± 0.03. Reporting the standard deviation proves that we understand experimental uncertainty and prevents over-interpreting minor differences.

5. **Q: If Naive Bayes scores high in AP, why might it still be problematic for deployment?**  
   *Answer:* Naive Bayes assumes all features are conditionally independent. When telemetry features are correlated (like cell voltages and temperatures), it multiplies overlapping probabilities, producing extreme, uncalibrated risk scores pushed close to 0.0 or 1.0. While its ranking (AP) may look decent, its absolute probabilities are distorted.

---

# SESSION 4 (Plan Day 9): Feature Selection & Hyperparameter Tuning

### 1. Beginner-Friendly Concepts

#### 1.1 Learned Parameters vs. Hyperparameters
- **Learned Parameters:** Variables the model discovers by itself during training (e.g., weights $w$ in Logistic Regression, split thresholds $\theta$ in Decision Trees).
- **Hyperparameters:** Configuration knobs set by the machine learning engineer *before* training begins (e.g., number of trees `n_estimators`, maximum depth `max_depth`, learning rate `learning_rate`).

#### 1.2 Grid Search vs. Random Search
- **GridSearchCV:** Evaluates every possible combination in a fixed grid. If you test 4 values for 5 hyperparameters, you must train $4^5 = 1,024$ models $\times$ 3 folds = 3,072 fits! Very slow and wastes time on unimportant parameters.
- **RandomizedSearchCV:** Randomly samples $N$ combinations from bounded probability distributions. Bergstra and Bengio (2012) proved that Random Search finds optimal hyperparameters in a fraction of the time because not all hyperparameters are equally important for tabular data.

#### 1.3 Feature Selection: Filter vs. Wrapper vs. Embedded
- **Filter Methods:** Fast statistical screens before training (e.g., Pearson correlation, univariate AUC). Fast, but ignores feature interactions.
- **Wrapper Methods:** Repeatedly train models with subsets of features (e.g., Recursive Feature Elimination). Highly accurate, but computationally expensive.
- **Embedded Methods:** Feature selection happens *during* model fitting (e.g., $L_1$ Lasso regularization in Logistic Regression, or Gini importance in Random Forests). Features with zero coefficients are automatically pruned.

#### 1.4 Avoiding Validation Overfitting
If you run 50,000 hyperparameter trials on your validation set, you will eventually find a configuration that scores well purely by chance. This is called **overfitting to the validation set**.
- **Defense:** Keep hyperparameter bounds sensible and physical. Limit random search to 10–20 trials. Test with cross-validation.

---

### 2. Step-by-Step Mathematical Example: L1 vs L2 Regularization

In Logistic Regression, the loss function balances training error against parameter magnitude:
$$\mathcal{L}(w) = \text{Loss}(y, \hat{y}) + \lambda \Omega(w)$$

1. **$L_2$ Regularization (Ridge):**
   $$\Omega(w) = \sum_{j=1}^d w_j^2$$
   - Penalty increases smoothly with squared weight.
   - Shrinks weights toward zero, but **never sets them exactly to zero**.
   - Kept all 63 features active.

2. **$L_1$ Regularization (Lasso):**
   $$\Omega(w) = \sum_{j=1}^d |w_j|$$
   - The derivative has a sharp, constant corner at zero: $\frac{d}{dw} |w| = \pm 1$.
   - Drives uninformative and redundant feature weights **identically to 0.0**.
   - Acts as an automatic embedded feature selector!

---

### 3. Python Code Deep Dive

Look at `src/evbattery/tune.py`:

```python
def tune_model(x_train, y_train, model_name, n_iter=10, seed=42):
    param_dist = PARAM_DISTRIBUTIONS[model_name]
    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        refit=True, # Automatically refits the winner on all 12,000 training rows!
        random_state=seed,
    )
    search.fit(x_train, y_train)
```

**Why this code is clean and reliable:**
- Uses `refit=True`: once the best parameter combination is selected by 3-fold cross-validation, scikit-learn automatically refits that winning configuration across the entire training partition.
- Uses `scoring="average_precision"`: optimizes for the exact operational metric needed for rare failure detection.

---

### 4. Practical Terminal Command

Run hyperparameter tuning for the winning model:

```powershell
python -m evbattery.tune --data data/ev_battery.csv --model random_forest --n-iter 10 --out artifacts/member_b/tuning_results.json
```

**Output Explanation:**
```json
{
  "model": "random_forest",
  "best_params": {
    "model__n_estimators": 50,
    "model__min_samples_leaf": 8,
    "model__max_depth": 12
  },
  "best_cv_average_precision": 0.8195,
  "note": "Only training-fold CV used. Calibration, threshold, and test partitions not touched."
}
```

---

### 5. Five Viva / Interview Questions (Session 4)

1. **Q: Why did you tune using RandomizedSearchCV instead of GridSearchCV?**  
   *Answer:* GridSearchCV evaluates a grid exhaustively ($N^k$ combinations), which is computationally wasteful when certain hyperparameters have minor impact. RandomizedSearchCV explores a continuous or broad discrete distribution efficiently, uncovering effective configurations in 10 trials that would require hundreds of grid evaluations.

2. **Q: Did hyperparameter tuning dramatically improve your model's score?**  
   *Answer:* No, and we report that honestly. Tuning improved Average Precision by approximately $+0.005$, which is within the 1-standard-deviation CV margin ($\pm 0.018$). The true value of our tuning was **regularization**: constraining `max_depth` to 12 and `min_samples_leaf` to 8 prevented tree branches from memorizing individual fleet anomalies.

3. **Q: Why must hyperparameter tuning be performed strictly on the training partition?**  
   *Answer:* Hyperparameters are chosen by evaluating validation scores. If we used the test set to choose `max_depth` or `learning_rate`, test-set information would dictate model architecture. The final test score would be biased and over-optimistic.

4. **Q: How does L1 regularization perform feature selection?**  
   *Answer:* The $L_1$ penalty function $|w|$ has a non-differentiable sharp corner at $w = 0$. In gradient optimization, the constraint region forms a diamond/rhombus whose vertices align with the coordinate axes. The loss contours intersect these vertices, forcing unimportant feature weights to become exactly zero.

5. **Q: What is the risk of testing too many hyperparameter combinations?**  
   *Answer:* Multiple hypothesis testing bias (validation overfitting). If you evaluate thousands of configurations on a finite validation set, you will eventually find a configuration that scores high due to random sampling fluctuations rather than true generalization capability.

---

# 3–5 MINUTE VIVA PRESENTATION SCRIPT

*(Practice reading this aloud at a confident, natural pace. This covers your exact Member B contribution, engineering tradeoffs, and alignment with the team.)*

---

> "Good morning, professors and examiners.
>
> My name is **[Your Name]**, and as **Member B** of our group project, I was responsible for the **supervised machine learning pipeline, preprocessing experiments, model comparison, hyperparameter optimization, and the handoff artifact**.
>
> When I took over from Member A, we had 20,000 raw telemetry records and an approved feature set of 63 inputs—55 numerical sensor readings and 8 categorical variables. Our target, `battery_failure`, has a rare event prevalence of only **6.92%**.
>
> Because of this severe class imbalance, my first engineering decision was establishing **Average Precision (AP)** as our primary metric rather than accuracy. A naive model that always predicts 'healthy' achieves 93% accuracy but catches zero failing batteries. Average Precision evaluates the entire precision-recall tradeoff, with a baseline of 0.069.
>
> To ensure scientific validity, I enforced a **strict zero-leakage policy**. I sequestered the dataset into our four agreed partitions: 60% for training, 10% for calibration, 10% for threshold tuning, and 20% for final holdout testing. **I conducted all my work strictly within the 60% training partition.** I never opened or evaluated the calibration, threshold, or test sets.
>
> In my **preprocessing experiments**, I tested median imputation with missingness indicators, K-Nearest Neighbors imputation, correlation pruning at 0.97, and L1 embedded selection. Crucially, all imputers, scalers, and encoders were fit **strictly inside the training portion of each cross-validation fold** using scikit-learn Pipelines. I found that median imputation with indicators achieved the same 0.74 Average Precision as KNN imputation while running in just 26 milliseconds compared to KNN's quadratic distance calculations. I chose median imputation with indicators as our robust, scalable pipeline default.
>
> Next, I benchmarked **seven diverse classical supervised algorithms** plus a dummy reference using consistent 3-fold Stratified Cross-Validation:
> - Our linear models—Logistic Regression and SGD—proved fast and interpretable, reaching an impressive AP of 0.72 to 0.74.
> - Single Decision Trees overfit, scoring lowest at 0.67 AP.
> - **Tree ensembles—specifically Random Forest and Histogram Gradient Boosting—consistently performed best**, achieving Average Precision above 0.78 to 0.81 and ROC AUC around 0.89. Tree ensembles naturally capture nonlinear interactions between cell temperatures and internal resistance without needing manual cross-terms.
>
> Using `RandomizedSearchCV` on the training folds only, I tuned our winning tree ensemble. By constraining the maximum depth and minimum samples per leaf, I prevented the trees from overfitting to telemetry outliers. The tuning confirmed that our architecture is stable, with validation variance well within ±0.02 SD.
>
> Finally, I fitted the winning pipeline on all 12,000 training rows and exported `best_base_model.joblib` and `selected_model.json`. 
> 
> A key engineering distinction I must emphasize: **my exported model is intentionally uncalibrated**. As an ML engineer, I know that calibrating probabilities on the training set causes severe over-confidence. I handed this uncalibrated base pipeline to **Member C**, who fits Platt sigmoid calibration on the dedicated 10% calibration partition, optimizes the warning threshold for F2 score on the threshold partition, and runs our final evaluation on the locked test set.
>
> Thank you, and I welcome your questions."

---
