"""
Week 1, Day 3-4: build the target from credit_record.csv, join to
application_record.csv, clean anomalies, and split.

This dataset has no ready-made target -- deriving it correctly is itself
part of the "intricacies to satisfy" list, and it's a natural interview
question ("how did you define good/bad?"), so keep this logic explicit
and documented rather than buried.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    TARGET_COL, ID_COL, BAD_STATUS_CODES,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_SEED,
)


def derive_target(credit_record: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ID -> TARGET (1 = bad, 0 = good).
    Bad = STATUS in BAD_STATUS_CODES (60+ days overdue) appears at least
    once anywhere in that applicant's credit_record history.

    NOTE: this is a simplified version of a proper vintage analysis (which
    would look at bad rate by months-on-book from account opening). Good
    enough to ship a v1 scorecard on -- name the simplification explicitly
    in the model memo, and list "proper vintage analysis" as a next step.
    """
    credit_record = credit_record.copy()
    credit_record["STATUS"] = credit_record["STATUS"].astype(str)
    is_bad = credit_record["STATUS"].isin(BAD_STATUS_CODES)

    target = (
        is_bad.groupby(credit_record[ID_COL]).max()
        .astype(int)
        .rename(TARGET_COL)
        .reset_index()
    )
    return target


def merge_application_and_target(application: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join on ID. Applicants in application_record.csv with no matching
    rows in credit_record.csv have no credit history yet and can't be
    labeled good/bad -- they're dropped here, not imputed. Report how many
    are dropped; if it's a large fraction, that's worth a line in the memo.
    """
    before = len(application)
    merged = application.merge(target, on=ID_COL, how="inner")
    dropped = before - len(merged)
    print(f"Dropped {dropped} of {before} applicants with no credit_record history "
          f"({dropped / before:.1%})")
    return merged


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw DAYS_BIRTH / DAYS_EMPLOYED counters (negative, days
    from today) into human-readable years -- easier to bin and to sanity
    check during EDA."""
    df = df.copy()
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    if "DAYS_EMPLOYED" in df.columns:
        df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365.25).round(1)
    return df


def clean_known_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: fill in once eda.find_anomalies() confirms what's actually in
    this dataset -- e.g. DAYS_EMPLOYED may have a large-positive placeholder
    for unemployed/pensioner applicants (check YEARS_EMPLOYED for
    suspiciously large or positive values after add_derived_fields()).
    Don't assume the exact placeholder value from a different dataset.
    """
    df = df.copy()
    if "YEARS_EMPLOYED" in df.columns:
        suspicious = df["YEARS_EMPLOYED"] < 0  # negative years employed = anomaly, since
                                                # DAYS_EMPLOYED was positive (future-dated)
        if suspicious.sum() > 0:
            df["FLAG_EMPLOYMENT_ANOMALY"] = suspicious.astype(int)
            df.loc[suspicious, "YEARS_EMPLOYED"] = None
            print(f"Flagged {suspicious.sum()} rows with anomalous YEARS_EMPLOYED")
    return df


def deduplicate_applicants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Known quirk of this dataset: multiple ID values can map to the same
    underlying person (identical feature rows), which double-counts them at
    train time. Check for exact-duplicate rows (excluding ID) during EDA;
    drop duplicates here if confirmed, keeping the first occurrence.
    """
    feature_cols = [c for c in df.columns if c not in (ID_COL,)]
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols, keep="first")
    print(f"Dropped {before - len(df)} duplicate applicant rows")
    return df


def fillna_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN in object-dtype (categorical/string) columns with the
    literal string "missing" rather than leaving it as an actual NaN.

    Why this matters: scorecardpy's woebin() correctly bins real NaN values
    into a "missing" bin at fit time, but woebin_ply() (applying those bins
    to new data) can fail to re-recognize NaN as belonging to that bin --
    seen concretely in this project, where applying WOE to OCCUPATION_TYPE
    produced NaN for exactly the 1,836 rows that were genuinely missing,
    even though those exact rows fit into a valid "missing" bin during
    binning. Converting NaN to a literal string upfront makes both steps
    rely on plain string matching instead of NaN-detection, which is more
    robust across the pandas/scorecardpy version combination in use here.
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].fillna("missing")
    return df


def train_val_test_split(df: pd.DataFrame):
    """Stratified split -- no clean time axis here either, same documented
    limitation as before."""
    train, temp = train_test_split(
        df, train_size=TRAIN_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_SEED
    )
    relative_val_size = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val, test = train_test_split(
        temp, train_size=relative_val_size, stratify=temp[TARGET_COL], random_state=RANDOM_SEED
    )
    return train, val, test


def run_prep_pipeline(application: pd.DataFrame, credit_record: pd.DataFrame):
    """Full chain: derive target -> merge -> derive fields -> clean -> dedupe -> split."""
    target = derive_target(credit_record)
    print(f"Target derived for {len(target)} applicants, "
          f"bad rate: {target[TARGET_COL].mean():.4f}")

    df = merge_application_and_target(application, target)
    df = add_derived_fields(df)
    df = clean_known_anomalies(df)
    df = deduplicate_applicants(df)
    df = fillna_categorical(df)

    train, val, test = train_val_test_split(df)
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    print(f"Target rate - train: {train[TARGET_COL].mean():.4f}, "
          f"val: {val[TARGET_COL].mean():.4f}, test: {test[TARGET_COL].mean():.4f}")
    return train, val, test


if __name__ == "__main__":
    from src.eda import load_raw
    application, credit_record = load_raw()
    train, val, test = run_prep_pipeline(application, credit_record)