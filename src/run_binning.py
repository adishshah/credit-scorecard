"""
Run the full Week 1 pipeline end to end: load raw data -> derive target ->
clean -> split -> compute IV -> shortlist -> fine/coarse class -> save bins.

Usage:  python -m src.run_binning
"""

import joblib
from src.eda import load_raw
from src.data_prep import run_prep_pipeline
from src.woe_binning import compute_iv_table, shortlist_variables, fine_and_coarse_classing
from src.config import MODELS_DIR, ID_COL, TARGET_COL

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("STEP 1: Loading raw data and building train/val/test")
    print("=" * 60)
    application, credit_record = load_raw()
    train, val, test = run_prep_pipeline(application, credit_record)

    print("\n" + "=" * 60)
    print("STEP 2: Computing Information Value per variable")
    print("=" * 60)
    iv_df = compute_iv_table(train)
    print(iv_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 3: Shortlisting variables")
    print("=" * 60)
    shortlist = shortlist_variables(iv_df)

    # DAYS_BIRTH is redundant with AGE_YEARS (same information, unconverted)
    # and its auto-binned WOE was non-monotonic -- drop it rather than fix
    # binning for a variable we don't need.
    if "DAYS_BIRTH" in shortlist:
        shortlist.remove("DAYS_BIRTH")
        print("Dropped DAYS_BIRTH: redundant with AGE_YEARS, non-monotonic WOE, not worth fixing")

    print(f"Shortlisted {len(shortlist)} variables: {shortlist}")

    if not shortlist:
        print("\nNo variables cleared the IV threshold. Stop here and "
              "investigate -- this would be unusual and worth digging into "
              "before going further, not something to push past.")
        return

    print("\n" + "=" * 60)
    print("STEP 4: Fine + coarse classing (WOE bins)")
    print("=" * 60)
    # Manual overrides for variables whose auto-binned WOE was non-monotonic.
    # AGE_YEARS: auto bins gave a zigzag trend. Merging into <35 / 35-59 / 59+
    # gives bad rates 5.03% -> 4.39% -> 3.18%, a clean monotonic
    # younger-is-riskier trend matching standard life-cycle credit risk.
    # CNT_FAM_MEMBERS: auto bins dipped in the middle. Merging into <3 / 3+
    # gives bad rates 4.22% -> 4.96%, monotonic, more-dependents-is-riskier.
    breaks_list = {
        "AGE_YEARS": [35, 59],
        "CNT_FAM_MEMBERS": [3],
    }
    bins = fine_and_coarse_classing(train, shortlist, breaks_list=breaks_list)
    for var, bin_df in bins.items():
        print(f"\n--- {var} ---")
        print(bin_df[["variable", "bin", "count", "count_distr", "bad", "badprob", "woe"]].to_string(index=False))

    joblib.dump(bins, MODELS_DIR / "woe_bins.pkl")
    joblib.dump({"train": train, "val": val, "test": test}, MODELS_DIR / "splits.pkl")
    print(f"\nSaved bins to {MODELS_DIR / 'woe_bins.pkl'}")
    print(f"Saved splits to {MODELS_DIR / 'splits.pkl'}")
    print("\nNext: eyeball each bin above for monotonic WOE trend and "
          "business-sense direction before moving to model.py.")


if __name__ == "__main__":
    main()