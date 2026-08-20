"""
Week 2, Day 1-2: multicollinearity check, logistic regression fit,
coefficient sign validation. Plus the XGBoost challenger for later in Week 2.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.config import TARGET_COL, VIF_DROP_THRESHOLD, RANDOM_SEED


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF per variable on WOE-transformed features. Drop/merge anything
    above config.VIF_DROP_THRESHOLD -- this is a common interview probe."""
    vif_df = pd.DataFrame()
    vif_df["variable"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df.sort_values("VIF", ascending=False)


def drop_high_vif(X: pd.DataFrame, vif_df: pd.DataFrame) -> pd.DataFrame:
    keep = vif_df[vif_df["VIF"] < VIF_DROP_THRESHOLD]["variable"].tolist()
    dropped = vif_df[vif_df["VIF"] >= VIF_DROP_THRESHOLD]["variable"].tolist()
    if dropped:
        print(f"Dropping high-VIF variables: {dropped}")
    return X[keep]


def fit_logistic(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit logistic regression on WOE-transformed variables using
    statsmodels (not sklearn) so you get p-values and coefficient
    interpretation for free -- useful for the model memo."""
    X_const = sm.add_constant(X_train)
    model = sm.Logit(y_train, X_const).fit()
    return model


def check_coefficient_signs(model, expected_signs: dict = None) -> None:
    """
    Print each coefficient and flag anything unexpected.
    expected_signs: optional dict like {'AMT_INCOME_TOTAL_woe': 'negative'}
    to check against your own business-logic expectations.
    A WOE-transformed variable's coefficient should always come out positive
    if the binning direction was set up consistently -- a negative sign is
    the red flag to investigate, not to ignore.
    """
    print(model.params)
    if expected_signs:
        for var, expected in expected_signs.items():
            if var in model.params:
                actual = "positive" if model.params[var] > 0 else "negative"
                flag = "OK" if actual == expected else "MISMATCH -- investigate"
                print(f"{var}: expected {expected}, got {actual} [{flag}]")


def fit_xgboost_challenger(X_train, y_train):
    """Benchmark model. Compare AUC against the logistic scorecard, then use
    SHAP to compare feature importance -- write up why the interpretable
    model is still the one that ships, even if XGBoost scores higher."""
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_SEED, eval_metric="auc"
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    print("Run after woe_binning.py produces WOE-transformed train/val/test sets.")
