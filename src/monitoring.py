"""
Week 3, Day 1-2: simulate PSI drift between a development ("old") cohort
and a "recent" cohort, to imitate post-deployment monitoring.
"""

import numpy as np
import pandas as pd
from src.config import PSI_STABLE_MAX, PSI_MODERATE_MAX


def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI = sum[ (actual_pct - expected_pct) * ln(actual_pct / expected_pct) ]
    `expected` = score distribution at model development (train set)
    `actual`   = score distribution in a "recent" cohort
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # avoid div-by-zero / log(0)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def interpret_psi(psi_value: float) -> str:
    if psi_value < PSI_STABLE_MAX:
        return "No significant population shift"
    elif psi_value < PSI_MODERATE_MAX:
        return "Moderate shift -- investigate"
    else:
        return "Significant shift -- model likely needs redevelopment"


def simulate_cohort_split(df: pd.DataFrame, date_proxy_col: str = None):
    """
    This dataset has no real application date. For the simulation, split by
    row index (first 70% = 'old'/development, last 30% = 'recent') and be
    explicit in the memo that this is a proxy for a true time-based split,
    not the real thing.
    """
    split_idx = int(len(df) * 0.7)
    old_cohort = df.iloc[:split_idx]
    recent_cohort = df.iloc[split_idx:]
    return old_cohort, recent_cohort


if __name__ == "__main__":
    print("Run after scorecard.py produces final scores for the full dataset.")
