# MSDS 451 — Programming Assignment 1

Predict next‑day **direction** of returns for a chosen asset (ticker: **NVDA** in this template) using lagged price features and gradient boosting (XGBoost) with **time‑series cross‑validation** and (optionally) **AIC‑based** feature subset selection.

## Repo structure (suggested)

```
.
├─ 451_pa1_jump_start_v001.ipynb      # working notebook 
├─ 451_pa1_jump_start_v001.py         # script version 
├─ 451_pa1_jump_start_v001.html       # HTML export of the notebook
├─ getdata_yfinance.py                # optional helper to download CSV
├─ data/
│  └─ msds_getdata_yfinance_nvda.csv  # your asset CSV (example: NVDA)
├─ reports/
│  ├─ 451_pa1_report_yourname.pdf     # final PDF report
│  └─ figures/                        # plots saved by the notebook
├─ requirements.txt
├─ README.md                          
└─ .gitignore
```

## How to run

1. **Install Python packages**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. **Fetch data for a new ticker**
   - Edit `getdata_yfinance.py` to set `symbol = "NVDA"` (or your ticker).
   - Run:
     ```bash
     python3 getdata_yfinance.py
     ```
   - Move the output CSV into `data/`.

3. **Open and run the notebook**
   - Launch Jupyter:
     ```bash
     jupyter lab
     ```
   - Open `451_pa1_jump_start_v001.ipynb`.
   - Ensure the CSV load path points to `data/msds_getdata_yfinance_<ticker>.csv`.
   - Run all cells. Plots & metrics are produced inline.

## Reproducibility details

- **No leakage**: features use lags; scaling (if any) is inside a Pipeline during CV.
- **CV**: `TimeSeriesSplit(gap=10, n_splits=5)`.
- **Model**: `XGBClassifier` tuned via `RandomizedSearchCV`.
- **Random seeds**: `random_state=2025`.

## Data

- Source: Yahoo! Finance via `yfinance` (CSV included in `data/`).
- Asset: **AAPL** .  
- Frequency: Daily.

## AI assistance disclosure

I used ChatGPT to help plan the project structure, identify leakage risks, design cross‑validation, and draft README/report templates. All modeling decisions, code execution, and interpretation of results are my own.
