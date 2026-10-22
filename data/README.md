# data/

Put the dataset here as **`data/ev_battery.csv`** (a copy of `ev battery_failure prediction Dataset (1).csv`).

- Do **not** edit the CSV — every script reads it as-is.
- It is **not** committed to Git (`*.csv` is in `.gitignore`). Share it with teammates directly.
- Expected SHA-256: `ba0be61cab0260bbc120668fba20fad1185c79978dd0b4111a722214bdb02e29`
  (Windows check: `Get-FileHash data\ev_battery.csv -Algorithm SHA256`)
- Expected shape: 20,000 rows × 70 columns, 1,384 failures.
