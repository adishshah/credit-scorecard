"""
Week 2, Day 5-7: validation suite - KS, Gini, lift/gains, calibration.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
    """Max separation between cumulative good and cumulative bad
    distributions. >40 is generally considered a strong scorecard."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return max(tpr - fpr) * 100


def gini_coefficient(y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1


def lift_table(y_true: pd.Series, y_pred_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile-based lift/gains table -- shows how much better than random
    the model is at catching bad loans in the riskiest deciles."""
    df = pd.DataFrame({"y": y_true, "proba": y_pred_proba})
    df["decile"] = pd.qcut(df["proba"], n_bins, labels=False, duplicates="drop")
    summary = df.groupby("decile").agg(
        n=("y", "size"),
        bad_rate=("y", "mean"),
        avg_proba=("proba", "mean"),
    ).sort_index(ascending=False)
    overall_bad_rate = df["y"].mean()
    summary["lift"] = summary["bad_rate"] / overall_bad_rate
    return summary


def calibration_table(y_true: pd.Series, y_pred_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted PD vs actual observed default rate per score band --
    rank-ordering well is not the same as being calibrated. Check both."""
    df = pd.DataFrame({"y": y_true, "proba": y_pred_proba})
    df["bin"] = pd.qcut(df["proba"], n_bins, duplicates="drop")
    summary = df.groupby("bin").agg(
        predicted_pd=("proba", "mean"),
        actual_default_rate=("y", "mean"),
        n=("y", "size"),
    )
    summary["gap"] = summary["actual_default_rate"] - summary["predicted_pd"]
    return summary


def full_validation_report(y_true: pd.Series, y_pred_proba: np.ndarray) -> dict:
    return {
        "KS": ks_statistic(y_true, y_pred_proba),
        "Gini": gini_coefficient(y_true, y_pred_proba),
        "AUC": roc_auc_score(y_true, y_pred_proba),
    }


if __name__ == "__main__":
    print("Run after model.py produces predicted probabilities on the validation set.")
