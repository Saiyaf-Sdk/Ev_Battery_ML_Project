# data/

## Jupyter notebooks

- `01_EDA.ipynb`: data quality, missing values, class balance, distributions, correlations, and categorical analysis.
- `02_Model_Training.ipynb`: preprocessing, classifier comparison, calibration, threshold selection, evaluation, and model saving.

Run each notebook from top to bottom using a Python 3.11+ kernel with the project requirements installed (`python -m pip install -e .` from the project root). Install Jupyter support with `python -m pip install notebook ipykernel` if needed.

Both notebooks prefer `ev_battery.csv`. If it is absent, they explicitly use `synthetic_benchmark.csv` for demonstration only. Benchmark outputs are not main-dataset results. Generated tables and models go under `artifacts/notebooks/benchmark/` or `artifacts/notebooks/dataset/`.

Put the dataset here as **`data/ev_battery.csv`** (a copy of `ev battery_failure prediction Dataset (1).csv`).

- Do **not** edit the CSV — every script reads it as-is.
- It is **not** committed to Git (`*.csv` is in `.gitignore`). Share it with teammates directly.
- Expected SHA-256: `ba0be61cab0260bbc120668fba20fad1185c79978dd0b4111a722214bdb02e29`
  (Windows check: `Get-FileHash data\ev_battery.csv -Algorithm SHA256`)
- Expected shape: 20,000 rows × 70 columns, 1,384 failures.
