# Consumer Credit Scorecard

A retail Probability-of-Default (PD) scorecard built end-to-end on real,
messy Kaggle data — WOE binning, logistic regression, scaled to an actual
points-based scorecard, benchmarked against XGBoost, and validated with
KS/Gini/PSI. Built the way a retail credit risk team actually builds one,
prioritizing interpretability and auditability over black-box accuracy.

**Dataset:** [Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) (Kaggle) — two files with no ready-made
target; the label had to be derived from monthly payment-status history.

## Results

| Metric | Train | Validation | Test |
|---|---|---|---|
| KS | 18.10 | 6.10 | 12.14 |
| Gini | 0.2151 | 0.0183 | 0.1087 |

- 438,557 raw applicants → 45,985 labeled (had credit history) → 9,997 after
  deduplication → 5,998 / 1,999 / 2,000 train/val/test split
- Final bad rate ~4.4%, 5-variable logistic scorecard, all coefficients
  correctly signed and validated against VIF and business intuition
- XGBoost challenger benchmarked head-to-head: outperforms on train but
  **not** out-of-sample — evidence the performance ceiling is the data,
  not the model choice, which is the actual justification for shipping
  the interpretable model
- Full write-up, including every limitation and assumption:
  [`reports/model_memo_template.md`](reports/model_memo_template.md)

## What this project demonstrates

- **Target engineering from raw behavioral data** — no label existed in
  the source data; built one from monthly payment-status codes
- **A real bug found and fixed, not glossed over:** WOE application
  (`woebin_ply`) silently produced NaN for 30% of rows on a category that
  was correctly binned at fit time — traced to a fit/apply inconsistency
  in how missing values are recognized, and fixed rather than patched
  around by dropping data
- **A genuine multicollinearity diagnosis:** a coefficient flipped sign
  when two correlated variables were both included; diagnosed via VIF,
  resolved by dropping the weaker-IV variable, not just forced to a
  "correct" sign
- **An IV-inflation finding, tested rather than assumed:** three variables
  showed suspiciously high Information Value; rather than assuming
  leakage, they were re-binned under the same tuned settings as the rest
  of the model and shown to have no coherent trend — small-sample noise,
  not signal, a materially different (and more defensible) conclusion
  than the initial suspicion
- Full validation suite (KS, Gini, lift, calibration) with honest
  reporting of where the model is weak, not just where it's strong

## Project structure

```
credit-scorecard/
├── data/                    # raw CSVs (gitignored) + processed data
├── src/
│   ├── config.py            # paths, constants, scorecard scaling params
│   ├── eda.py                # exploratory analysis
│   ├── data_prep.py          # target derivation, cleaning, split
│   ├── woe_binning.py        # WOE/IV binning
│   ├── model.py               # logistic regression, VIF, XGBoost challenger
│   ├── scorecard.py           # scaling to points (Offset/Factor/PDO)
│   ├── validation.py          # KS, Gini, lift, calibration
│   ├── monitoring.py          # PSI stability check
│   └── run_*.py                # end-to-end pipeline runners for each stage
├── app/streamlit_app.py       # interactive scoring demo
├── reports/model_memo_template.md   # full write-up
└── requirements.txt            # pinned dependency versions
```

## Running it

```bash
python -m venv venv
venv\Scripts\activate              # Windows; use source venv/bin/activate on Mac/Linux
python -m pip install -r requirements.txt
```

Download `application_record.csv` and `credit_record.csv` from the
[Kaggle dataset page](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
into `data/raw/`, then run the pipeline stages in order:

```bash
python -m src.run_binning         # target derivation, cleaning, WOE binning
python -m src.run_model           # logistic regression, VIF, sign validation
python -m src.run_scorecard       # scale to points
python -m src.run_validation      # KS, Gini, lift, calibration
python -m src.run_challenger      # XGBoost benchmark
python -m src.run_monitoring      # PSI check
streamlit run app/streamlit_app.py   # interactive scoring UI
```

## Known limitations

- No true out-of-time split — this dataset has no usable time axis, so
  train/val/test are stratified random splits rather than genuine
  chronological cohorts. This is the single biggest methodological gap.
- Small, imbalanced sample (~265 bad events in training) drives real
  instability in validation metrics across splits.
- PSI monitoring compares in-sample vs. out-of-sample distributions as a
  proxy for drift, since no real longitudinal data exists.
- Expected Loss / cutoff analysis not yet built.

Full details, reasoning, and every assumption made: [`reports/model_memo_template.md`](reports/model_memo_template.md)
