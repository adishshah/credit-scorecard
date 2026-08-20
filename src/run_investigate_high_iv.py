"""
Investigate the 3 variables flagged as suspiciously high IV
(AMT_INCOME_TOTAL, YEARS_EMPLOYED, DAYS_EMPLOYED) before deciding whether
they represent genuine predictive signal or a small-sample artifact.

What to look for in the output:
  - GOOD SIGN: many bins, each with a reasonable population share and bad
    count, WOE moving smoothly/monotonically -- suggests real signal.
  - RED FLAG: any bin with a tiny population (a handful of rows) driving
    an extreme WOE value -- that's overfitting to noise, not genuine
    predictive power, and would explain an inflated IV.

Usage: python -m src.run_investigate_high_iv
"""

from src.eda import load_raw
from src.data_prep import run_prep_pipeline
from src.woe_binning import fine_and_coarse_classing


def main():
    application, credit_record = load_raw()
    train, val, test = run_prep_pipeline(application, credit_record)

    suspicious_vars = ["AMT_INCOME_TOTAL", "YEARS_EMPLOYED", "DAYS_EMPLOYED"]
    print(f"Binning {suspicious_vars} on train to inspect bin quality...\n")
    bins = fine_and_coarse_classing(train, suspicious_vars)

    for var, bin_df in bins.items():
        print(f"--- {var} ---")
        print(bin_df[["variable", "bin", "count", "count_distr", "bad", "badprob", "woe"]].to_string(index=False))
        min_count = bin_df["count"].min()
        min_bad = bin_df["bad"].min()
        print(f"[Smallest bin: {min_count} rows, {min_bad} bad events]\n")


if __name__ == "__main__":
    main()