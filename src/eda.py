"""
Week 1, Day 1-2: exploratory analysis.

Goal here isn't pretty charts - it's building the list of things you'll
need to handle in data_prep.py: missingness, anomalies, target derivation
sanity checks, duplicate applicants.
"""

import pandas as pd
from src.config import RAW_APPLICATION_RECORD, RAW_CREDIT_RECORD, ID_COL


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw files. Returns (application_record, credit_record)."""
    application = pd.read_csv(RAW_APPLICATION_RECORD)
    credit_record = pd.read_csv(RAW_CREDIT_RECORD)
    return application, credit_record


def status_distribution(credit_record: pd.DataFrame) -> pd.Series:
    """STATUS value counts -- confirm the codes match documentation
    (0-5 past-due buckets, C = paid off, X = no loan that month) before
    trusting the derived target."""
    return credit_record["STATUS"].astype(str).value_counts()


def applicants_with_no_credit_history(application: pd.DataFrame, credit_record: pd.DataFrame) -> int:
    """IDs in application_record with zero rows in credit_record -- these
    get dropped in data_prep.merge_application_and_target(). Check this
    isn't an unreasonably large share of the data."""
    app_ids = set(application[ID_COL])
    credit_ids = set(credit_record[ID_COL])
    return len(app_ids - credit_ids)


def duplicate_applicant_check(application: pd.DataFrame) -> int:
    """This dataset is known to have multiple IDs mapping to the same
    underlying person (identical feature rows). Count exact duplicates
    (excluding ID) to see if data_prep.deduplicate_applicants() matters here."""
    feature_cols = [c for c in application.columns if c != ID_COL]
    return application.duplicated(subset=feature_cols).sum()


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """% missing per column, sorted descending."""
    miss = df.isnull().mean().sort_values(ascending=False)
    return miss[miss > 0].to_frame("pct_missing")


def find_anomalies(df: pd.DataFrame) -> dict:
    """
    TODO: inspect DAYS_BIRTH / DAYS_EMPLOYED ranges here directly
    (df[['DAYS_BIRTH','DAYS_EMPLOYED']].describe()) before assuming any
    specific placeholder value -- this dataset may not share the exact
    365243 anomaly seen in other Kaggle credit datasets. Look for:
      - positive DAYS_EMPLOYED values (would imply employment in the future)
      - implausible AGE_YEARS after add_derived_fields() runs
    Return a dict of {column: description_of_issue} as you find them.
    """
    anomalies = {}
    if "DAYS_EMPLOYED" in df.columns:
        positive_count = (df["DAYS_EMPLOYED"] > 0).sum()
        if positive_count > 0:
            anomalies["DAYS_EMPLOYED"] = (
                f"{positive_count} rows with positive DAYS_EMPLOYED "
                "(implies future employment date -- likely unemployed/pensioner "
                "placeholder, confirm the exact value before cleaning)"
            )
    return anomalies


def dtype_split(df: pd.DataFrame) -> tuple[list, list]:
    """Return (numeric_cols, categorical_cols), excluding target and ID."""
    exclude = {"TARGET", ID_COL}
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    categorical = [c for c in df.select_dtypes(include="object").columns if c not in exclude]
    return numeric, categorical


if __name__ == "__main__":
    application, credit_record = load_raw()
    print(f"application_record shape: {application.shape}")
    print(f"credit_record shape: {credit_record.shape}")

    print("\nSTATUS distribution:")
    print(status_distribution(credit_record))

    print(f"\nApplicants with no credit history (will be dropped): "
          f"{applicants_with_no_credit_history(application, credit_record)}")

    print(f"\nDuplicate applicant rows (excluding ID): "
          f"{duplicate_applicant_check(application)}")

    print("\nTop missing columns in application_record:")
    print(missingness_report(application).head(10))

    print("\nAnomalies found:")
    print(find_anomalies(application))
