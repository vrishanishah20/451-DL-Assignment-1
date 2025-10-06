# MSDS 451 — Programming Assignment 1

Predict next‑day **direction** of returns for a chosen asset (ticker: **NVDA** in this template) using lagged price features and gradient boosting (XGBoost) with **time‑series cross‑validation** and (optionally) **AIC‑based** feature subset selection.

## Repo structure (suggested)

```
.
├─ 451_pa1_jump_start_v001.ipynb      # your working notebook (edited for your ticker)
├─ 451_pa1_jump_start_v001.py         # script version (optional)
├─ 451_pa1_jump_start_v001.html       # HTML export of the notebook
├─ getdata_yfinance.py                # optional helper to download CSV
├─ data/
│  └─ msds_getdata_yfinance_nvda.csv  # your asset CSV (example: NVDA)
├─ reports/
│  ├─ 451_pa1_report_yourname.pdf     # your final PDF report
│  └─ figures/                        # plots saved by the notebook
├─ requirements.txt
├─ README.md                          # this file
└─ .gitignore
```

## How to run

1. **Install Python packages**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. **(Optional) Fetch data for a new ticker**
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

4. **Export notebook to HTML**
   - Jupyter: `File → Export Notebook As → HTML`
   - Save as `451_pa1_jump_start_v001.html` at repo root.

5. **Save plots to `reports/figures/` (optional but recommended)**
   - In plotting cells, after displaying a figure, add:
     ```python
     plt.savefig("reports/figures/roc_curve.png", dpi=150, bbox_inches="tight")
     ```

6. **Write your PDF report**
   - Use the template in `reports/451_pa1_report_template.md` (provided).
   - Export it to PDF (VS Code “Markdown PDF”, or print-to-PDF), and save as:
     `reports/451_pa1_report_yourname.pdf`.

## Reproducibility details

- **No leakage**: features use lags; scaling (if any) is inside a Pipeline during CV.
- **CV**: `TimeSeriesSplit(gap=10, n_splits=5)`.
- **Model**: `XGBClassifier` tuned via `RandomizedSearchCV`.
- **Random seeds**: `random_state=2025`.

## Results (fill these in)

- Train/CV mean accuracy: `XX.XXX ± YY.YYY`
- Best hyperparameters: `{...}`
- Final confusion matrix (labels: [0,1]):  
  `[[TN, FP], [FN, TP]]`
- ROC AUC: `X.XXX` (if computed with probabilities)
- Brief interpretation: _2–3 sentences on what features mattered and how stable the signal looked across folds._

## Data

- Source: Yahoo! Finance via `yfinance` (CSV included in `data/`).
- Asset: **NVDA** (replace with your chosen ticker; do not use WTI).  
- Frequency: Daily.

## AI assistance disclosure

I used ChatGPT to help plan the project structure, identify leakage risks, design cross‑validation, and draft README/report templates. All modeling decisions, code execution, and interpretation of results are my own.

---

**Submission reminder**: In the assignment comments, paste the **cloneable** repo URL ending in `.git` (e.g., `https://github.com/yourname/msds-451-pa1.git`).

