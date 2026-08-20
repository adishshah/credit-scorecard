"""
Week 2, Day 5-7: KS, Gini, lift, and calibration on the fitted logistic
scorecard -- the honest read on how much this model actually separates
good from bad, beyond eyeballing mean score differences.

Usage: python -m src.run_validation
"""

import joblib
import statsmodels.api as sm
from src.config import MODELS_DIR
from src.validation import ks_statistic, gini_coefficient, lift_table, calibration_table


def predict(model, X):
    X_const = sm.add_constant(X, has_constant="add")
    return model.predict(X_const)


def main():
    model_data = joblib.load(MODELS_DIR / "logistic_model.pkl")
    model = model_data["model"]
    kept_cols = model_data["kept_cols"]

    print("=" * 60)
    print("STEP 1: KS / Gini across train, val, test")
    print("=" * 60)
    preds = {}
    for name in ["train", "val", "test"]:
        X = model_data[f"X_{name}"][kept_cols]
        y = model_data[f"y_{name}"]
        y_pred = predict(model, X)
        preds[name] = (y, y_pred)
        ks = ks_statistic(y, y_pred)
        gini = gini_coefficient(y, y_pred)
        print(f"{name.upper():5s}: KS={ks:5.2f}   Gini={gini:.4f}")

    print("\n" + "=" * 60)
    print("STEP 2: Lift table (validation set)")
    print("=" * 60)
    y_val, y_pred_val = preds["val"]
    print(lift_table(y_val, y_pred_val).to_string())

    print("\n" + "=" * 60)
    print("STEP 3: Calibration table (validation set)")
    print("=" * 60)
    print(calibration_table(y_val, y_pred_val).to_string())

    print("\nA KS above ~40 is generally considered a strong scorecard; "
          "this one almost certainly won't clear that with only 5 weak "
          "demographic variables. That's the real number to put in the "
          "memo -- and the honest trigger for deciding whether to properly "
          "investigate AMT_INCOME_TOTAL / YEARS_EMPLOYED / DAYS_EMPLOYED "
          "instead of leaving them excluded on a suspicion that was never "
          "actually confirmed as leakage.")


if __name__ == "__main__":
    main()
    