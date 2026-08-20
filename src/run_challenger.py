"""
Week 2, Day 5-7 (part 2): XGBoost challenger + SHAP comparison against the
logistic scorecard.

Trains on all 6 shortlisted WOE variables (not just the 5 that survived
VIF for the logistic model) -- tree models don't suffer from
multicollinearity the way linear models do, so there's no reason to drop
CNT_FAM_MEMBERS here.

Usage: python -m src.run_challenger
"""

import numpy as np
import joblib
from src.config import MODELS_DIR, TARGET_COL
from src.woe_binning import apply_woe
from src.model import fit_xgboost_challenger
from src.validation import ks_statistic, gini_coefficient


def main():
    bins = joblib.load(MODELS_DIR / "woe_bins.pkl")
    splits = joblib.load(MODELS_DIR / "splits.pkl")
    train, val, test = splits["train"], splits["val"], splits["test"]

    print("=" * 60)
    print("STEP 1: WOE transform (all 6 shortlisted variables)")
    print("=" * 60)
    train_woe = apply_woe(train, bins)
    val_woe = apply_woe(val, bins)
    test_woe = apply_woe(test, bins)
    woe_cols = [c for c in train_woe.columns if c.endswith("_woe")]
    print(f"Using: {woe_cols}")

    X_train, y_train = train_woe[woe_cols], train_woe[TARGET_COL]
    X_val, y_val = val_woe[woe_cols], val_woe[TARGET_COL]
    X_test, y_test = test_woe[woe_cols], test_woe[TARGET_COL]

    print("\n" + "=" * 60)
    print("STEP 2: Fit XGBoost challenger")
    print("=" * 60)
    xgb_model = fit_xgboost_challenger(X_train, y_train)
    print("Fitted.")

    print("\n" + "=" * 60)
    print("STEP 3: KS / Gini -- XGBoost vs logistic scorecard")
    print("=" * 60)
    logistic_results = {"train": (18.10, 0.2151), "val": (6.10, 0.0183), "test": (12.14, 0.1087)}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = xgb_model.predict_proba(X)[:, 1]
        ks = ks_statistic(y, proba)
        gini = gini_coefficient(y, proba)
        log_ks, log_gini = logistic_results[name]
        print(f"{name.upper():5s}: XGBoost KS={ks:5.2f} Gini={gini:.4f}   |   "
              f"Logistic KS={log_ks:5.2f} Gini={log_gini:.4f}")

    print("\n" + "=" * 60)
    print("STEP 4: Feature importance")
    print("=" * 60)
    # SHAP hit an unresolved xgboost 3.x / shap 0.49.1 compatibility bug
    # (base_score serialization mismatch that persists through internal
    # re-parsing, not fixable from the calling side without patching one
    # of the two libraries directly) -- not worth the time given it's a
    # nice-to-have for the memo, not a blocker. Using XGBoost's own
    # gain-based importance instead, which needs no workaround.
    importance = xgb_model.feature_importances_
    shap_summary = sorted(zip(woe_cols, importance), key=lambda x: -x[1])
    print("(Using XGBoost's built-in gain-based importance -- SHAP hit an "
          "unresolved xgboost/shap version compatibility bug, noted as a "
          "known limitation rather than chased further.)")
    for var, imp in shap_summary:
        print(f"{var:30s} importance = {imp:.4f}")

    joblib.dump({
        "model": xgb_model, "woe_cols": woe_cols,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }, MODELS_DIR / "xgboost_model.pkl")
    print(f"\nSaved to {MODELS_DIR / 'xgboost_model.pkl'}")
    print("\nIf XGBoost's val/test KS is similarly weak (not dramatically "
          "higher than logistic), that confirms the ceiling here is the "
          "data/variables, not the model choice -- strengthens the Week 1 "
          "finding rather than contradicting it.")


if __name__ == "__main__":
    main()