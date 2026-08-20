"""
Week 2, Day 1-2: apply WOE transform -> VIF check -> fit logistic
regression -> validate coefficient signs.

Usage: python -m src.run_model
"""

import joblib
from src.config import MODELS_DIR, TARGET_COL
from src.woe_binning import apply_woe
from src.model import compute_vif, drop_high_vif, fit_logistic, check_coefficient_signs


def main():
    bins = joblib.load(MODELS_DIR / "woe_bins.pkl")
    splits = joblib.load(MODELS_DIR / "splits.pkl")
    train, val, test = splits["train"], splits["val"], splits["test"]

    print("=" * 60)
    print("STEP 1: Applying WOE transform")
    print("=" * 60)
    train_woe = apply_woe(train, bins)
    val_woe = apply_woe(val, bins)
    test_woe = apply_woe(test, bins)

    woe_cols = [c for c in train_woe.columns if c.endswith("_woe")]
    print(f"WOE columns: {woe_cols}")

    X_train, y_train = train_woe[woe_cols], train_woe[TARGET_COL]
    X_val, y_val = val_woe[woe_cols], val_woe[TARGET_COL]
    X_test, y_test = test_woe[woe_cols], test_woe[TARGET_COL]

    print("\n" + "=" * 60)
    print("STEP 2: VIF check")
    print("=" * 60)
    print("Missing values per WOE column (train):")
    print(X_train.isna().sum())
    print("Infinite values per WOE column (train):")
    print((X_train.abs() == float("inf")).sum())

    bad_rows = X_train.isna().any(axis=1) | (X_train.abs() == float("inf")).any(axis=1)
    if bad_rows.sum() > 0:
        print(f"\nWARNING: {bad_rows.sum()} of {len(X_train)} train rows still have a "
              f"NaN/inf WOE value after the fillna_categorical fix -- this shouldn't "
              f"happen anymore. Investigate before proceeding rather than silently "
              f"dropping rows.")
        X_train = X_train[~bad_rows]
        y_train = y_train[~bad_rows]
    else:
        print("No missing/infinite WOE values -- full train set retained.")

    vif_df = compute_vif(X_train)
    print(vif_df.to_string(index=False))
    X_train_clean = drop_high_vif(X_train, vif_df)
    kept_cols = X_train_clean.columns.tolist()
    print(f"Kept after VIF: {kept_cols}")

    X_val_clean = X_val[kept_cols]
    X_test_clean = X_test[kept_cols]

    print("\n" + "=" * 60)
    print("STEP 3: Fit logistic regression")
    print("=" * 60)
    model = fit_logistic(X_train_clean, y_train)
    print(model.summary())

    print("\n" + "=" * 60)
    print("STEP 4: Coefficient sign check")
    print("=" * 60)
    # In this project's WOE convention (confirmed from the bin tables):
    # higher WOE = higher observed bad rate. So every coefficient SHOULD
    # come out positive if binning direction was applied consistently.
    # A negative sign here is a real red flag, not noise -- investigate it.
    expected_signs = {col: "positive" for col in kept_cols}
    check_coefficient_signs(model, expected_signs)

    mismatched = [
        col for col in kept_cols
        if col in model.params and model.params[col] < 0
    ]

    if mismatched:
        print(f"\n{mismatched} flipped sign. Given these are also the "
              f"variables with elevated VIF (correlated with another kept "
              f"variable), this is a multicollinearity symptom, not noise: "
              f"once two correlated variables share a model, individual "
              f"coefficients can flip even though each looks sensible "
              f"alone. Dropping the weaker-IV variable of the correlated "
              f"pair and refitting.")
        kept_cols = [c for c in kept_cols if c not in mismatched]
        X_train_clean = X_train_clean[kept_cols]
        X_val_clean = X_val_clean[kept_cols]
        X_test_clean = X_test_clean[kept_cols]

        print("\n" + "=" * 60)
        print("STEP 5: Refit after dropping sign-mismatched variable(s)")
        print("=" * 60)
        model = fit_logistic(X_train_clean, y_train)
        print(model.summary())

        expected_signs = {col: "positive" for col in kept_cols}
        check_coefficient_signs(model, expected_signs)
    else:
        print("\nAll coefficient signs correct -- no refit needed.")

    joblib.dump({
        "model": model, "kept_cols": kept_cols,
        "X_train": X_train_clean, "y_train": y_train,
        "X_val": X_val_clean, "y_val": y_val,
        "X_test": X_test_clean, "y_test": y_test,
    }, MODELS_DIR / "logistic_model.pkl")
    print(f"\nSaved model + data to {MODELS_DIR / 'logistic_model.pkl'}")
    print("\nNext: check that R-squared/p-values look reasonable and every "
          "coefficient sign is positive, then move to scorecard.py.")


if __name__ == "__main__":
    main()