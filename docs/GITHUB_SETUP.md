# GitHub Setup — one shared repository for the team

**Yes, create a GitHub repository.** The plan expects daily commits, teammates build on Member A's work, and examiners like to see the history.

## 1. Create the repository (Member A, once)

1. github.com → **New repository**.
2. Name: `ev-battery-failure-ml` (any name is fine).
3. **Private** (recommended until submission) — then *Settings → Collaborators* → add Members B, C, D.
4. Do **not** tick "Add a README", ".gitignore" or "license" — the project already has them.
5. Copy the HTTPS URL, e.g. `https://github.com/<your-username>/ev-battery-failure-ml.git`.

## 2. First push (from `D:\Projects\Machine Learning\ev_battery_ml_project`)

```powershell
git config --global user.name  "Saiyaf"
git config --global user.email "your-github-email@example.com"

git init
git branch -M main
git add .
git status          # CHECK: no .csv, no .joblib, no .venv, no artifacts/ in the list
git commit -m "Starter project + Member A: data profiling, EDA, leakage and feature-set audit"
git remote add origin https://github.com/<your-username>/ev-battery-failure-ml.git
git push -u origin main
```

`.gitignore` already blocks `*.csv`, `*.joblib`, `.venv/`, `.env` and `artifacts/` — the dataset and trained models must never go to GitHub. Member A's figures are committed under `docs/figures/member_a/` so the report can use them.

## 3. How teammates join

```powershell
git clone https://github.com/<your-username>/ev-battery-failure-ml.git
cd ev-battery-failure-ml
# copy the CSV to data\ev_battery.csv, then follow README section 1
```

## 4. Simple team workflow

- One branch per member: `git checkout -b member-b-models`.
- Commit small and often with clear messages (`models: add CV leaderboard table`).
- Push the branch and open a **Pull Request**; one other member reviews before merging to `main`.
- Before starting work each day: `git checkout main && git pull`.
- Never commit the CSV, `model.joblib`, API keys or `.env`.
