# Consumer Credit Scorecard

A retail Probability-of-Default (PD) scorecard built on the Home Credit Default Risk
dataset, following the traditional WOE / logistic-regression scorecard methodology
used in retail credit risk teams.

## Project structure

```
credit-scorecard/
├── data/
│   ├── raw/              # original downloaded CSVs go here (gitignored)
│   └── processed/        # cleaned / WOE-transformed data (gitignored)
├── src/
│   ├── config.py         # paths, constants, scorecard scaling parameters
│   ├── data_prep.py      # cleaning, target definition, train/test/val split
│   ├── eda.py             # exploratory analysis helpers
│   ├── woe_binning.py     # fine/coarse classing, IV, WOE transform
│   ├── model.py           # logistic regression + VIF check + challenger XGBoost
│   ├── scorecard.py       # scaling model to points (Offset/Factor/PDO)
│   ├── validation.py      # KS, Gini, lift chart, calibration
│   └── monitoring.py      # PSI / CSI stability checks
├── app/
│   └── streamlit_app.py  # scoring UI (Week 3)
├── reports/
│   └── model_memo_template.md   # fill in as you go — becomes your final write-up
├── outputs/
│   ├── models/            # saved model artifacts (.pkl)
│   └── figures/           # saved plots for the memo / README
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Dataset: [Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) (Kaggle, rikdifos).
Download both `application_record.csv` and `credit_record.csv` and place them in `data/raw/`.
Do not commit raw data — it's gitignored.

This dataset has **no ready-made target column** — `application_record.csv` has
applicant features, `credit_record.csv` has monthly credit behavior (`STATUS`
per month). The target is derived in `data_prep.derive_target()`: an applicant
is "bad" if a `STATUS` of 2/3/4/5 (60+ days overdue) appears anywhere in their
history, "good" otherwise. This derivation — and the fact that it's a
simplified stand-in for a full vintage analysis — is worth a paragraph in the
model memo; it's a natural interview question.

## Workflow (matches the Week 1-3 plan)

1. `src/eda.py` — understand the data before touching anything else
2. `src/data_prep.py` — clean known anomalies, define target, split data
3. `src/woe_binning.py` — fine → coarse classing, compute IV, shortlist variables
4. `src/model.py` — VIF check, fit logistic regression, sanity-check coefficient signs
5. `src/scorecard.py` — scale the model into actual points
6. `src/validation.py` — KS, Gini, lift, calibration
7. `src/monitoring.py` — simulate PSI drift
8. `app/streamlit_app.py` — wrap it in a usable interface
9. `reports/model_memo_template.md` — write it up

## Status

- [ ] EDA complete
- [ ] Target defined, anomalies handled, split done
- [ ] WOE binning + IV shortlist done
- [ ] Logistic model fit, VIF clean, coefficient signs validated
- [ ] Scorecard scaled to points
- [ ] Validation suite run (KS / Gini / calibration)
- [ ] Challenger XGBoost + SHAP comparison done
- [ ] PSI monitoring simulated
- [ ] Streamlit app working
- [ ] Model memo written
