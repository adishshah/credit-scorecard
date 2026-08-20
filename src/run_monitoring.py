"""
Week 3, Day 1-2: PSI stability check.

No real time axis in this dataset, so the proxy used here is train
(development sample, what the model was fit on) vs val+test combined
(unseen data) -- a defensible stand-in for pre/post-deployment drift.
State this proxy explicitly in the memo rather than implying it's a real
longitudinal check.

Usage: python -m src.run_monitoring
"""

import joblib
import pandas as pd
from src.config import MODELS_DIR
from src.monitoring import calculate_psi, interpret_psi


def main():
    scorecard_data = joblib.load(MODELS_DIR / "scorecard.pkl")
    train_scores = scorecard_data["train_scores"]
    val_scores = scorecard_data["val_scores"]
    test_scores = scorecard_data["test_scores"]

    development = train_scores
    recent = pd.concat([val_scores, test_scores])

    print("=" * 60)
    print("PSI: train (development) vs val+test (unseen)")
    print("=" * 60)
    print(f"Development n={len(development)}, mean={development.mean():.1f}")
    print(f"Recent n={len(recent)}, mean={recent.mean():.1f}")

    psi = calculate_psi(development.values, recent.values)
    print(f"\nPSI = {psi:.4f}")
    print(f"Interpretation: {interpret_psi(psi)}")
    print("\nProxy note for the memo: this compares in-sample vs "
          "out-of-sample score distributions as a stand-in for real "
          "pre/post-deployment drift, since this dataset has no time axis "
          "to do a true vintage-based PSI check.")


if __name__ == "__main__":
    main()
    