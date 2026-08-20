"""
Central config: paths, random seed, and scorecard scaling parameters.
Keep every hardcoded constant here instead of scattered across scripts,
so the whole project stays reproducible and easy to defend in an interview.
"""

from pathlib import Path

# ---- Paths ----
# Dataset: "Credit Card Approval Prediction" (Kaggle, rikdifos)
# https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

RAW_APPLICATION_RECORD = RAW_DATA_DIR / "application_record.csv"  # applicant features
RAW_CREDIT_RECORD = RAW_DATA_DIR / "credit_record.csv"            # monthly credit behavior, used to derive target

ID_COL = "ID"

# ---- Reproducibility ----
RANDOM_SEED = 42

# ---- Target ----
# There is no ready-made target column in this dataset -- it's derived from
# credit_record.csv's STATUS field per applicant:
#   0: 1-29 days past due   1: 30-59 days past due   2: 60-89 days past due
#   3: 90-119 days past due 4: 120-149 days past due 5: 150+ days / write-off
#   C: paid off that month  X: no loan that month
# Convention used here: an applicant is "bad" (TARGET=1) if STATUS 2/3/4/5
# appears ANYWHERE in their credit_record history (60+ days overdue at least
# once). Everyone else is "good" (TARGET=0). This is a simplified version of
# a proper vintage analysis -- document that simplification in the memo.
TARGET_COL = "TARGET"
BAD_STATUS_CODES = {"2", "3", "4", "5"}

# ---- Train/test/validation split ----
# No clean time axis available in this dataset -> stratified split instead of a
# true out-of-time (OOT) split. This is a documented limitation, not an oversight.
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2

# ---- Known-issue watchlist (confirm during EDA, don't assume) ----
# application_record.csv includes DAYS_EMPLOYED and DAYS_BIRTH, both counted
# backwards from "today". A sibling Kaggle credit dataset (Home Credit) has a
# documented placeholder anomaly (365243) in DAYS_EMPLOYED for
# unemployed/pensioner applicants -- check whether this dataset has the same
# or a different anomaly during eda.find_anomalies() rather than assuming.

# ---- WOE binning ----
MIN_BIN_POPULATION = 0.10      # each bin should hold >= 10% of the population
                                # (bumped up from the usual 5% -- with only
                                # ~265 bad events in train, finer bins would
                                # leave too few bads per bin to trust)
BIN_NUM_LIMIT = 5              # max bins per variable, same reasoning
IV_KEEP_THRESHOLD = 0.02       # drop variables with IV below this (not useful)
IV_SUSPICIOUS_THRESHOLD = 0.5  # flag variables above this - possible leakage

# ---- Multicollinearity ----
VIF_DROP_THRESHOLD = 5.0

# ---- Scorecard scaling (Offset / Factor / PDO) ----
# Score = Offset + Factor * ln(odds)
# Factor = PDO / ln(2)
# Offset = Base_Score - Factor * ln(Base_Odds)
BASE_SCORE = 600          # anchor score
BASE_ODDS = 50            # odds of good:bad at the base score (50:1)
PDO = 20                  # points to double the odds

# ---- Expected loss assumption ----
# No real LGD/EAD in this dataset - use a documented, published-benchmark
# assumption rather than fitting one. State this clearly in the model memo.
ASSUMED_LGD = 0.45  # ~45% loss given default, common Basel benchmark for unsecured retail

# ---- PSI thresholds ----
PSI_STABLE_MAX = 0.10
PSI_MODERATE_MAX = 0.25
