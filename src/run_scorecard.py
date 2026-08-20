"""
Week 2, Day 3-4: build the actual points-based scorecard from the fitted
logistic model, and score train/val/test.

Usage: python -m src.run_scorecard
"""

import joblib
from src.config import MODELS_DIR
from src.scorecard import get_offset_factor, base_points, build_scorecard_points_table, score_dataframe


def main():
    bins = joblib.load(MODELS_DIR / "woe_bins.pkl")
    splits = joblib.load(MODELS_DIR / "splits.pkl")
    model_data = joblib.load(MODELS_DIR / "logistic_model.pkl")
    model = model_data["model"]
    kept_cols = model_data["kept_cols"]
    train, val, test = splits["train"], splits["val"], splits["test"]

    print("=" * 60)
    print("STEP 1: Scaling parameters")
    print("=" * 60)
    offset, factor = get_offset_factor()
    print(f"Offset: {offset:.2f}, Factor: {factor:.2f}")
    print(f"Base points (from intercept): {base_points(model)}")

    print("\n" + "=" * 60)
    print("STEP 2: Points table (one row per variable/bin)")
    print("=" * 60)
    points_table = build_scorecard_points_table(model, bins, kept_cols)
    print(points_table.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 3: Scoring train / val / test")
    print("=" * 60)
    train_scores = score_dataframe(model, bins, kept_cols, train)
    val_scores = score_dataframe(model, bins, kept_cols, val)
    test_scores = score_dataframe(model, bins, kept_cols, test)

    for name, scores, df in [("Train", train_scores, train), ("Val", val_scores, val), ("Test", test_scores, test)]:
        print(f"\n{name} scores: min={scores.min()}, max={scores.max()}, "
              f"mean={scores.mean():.1f}, median={scores.median():.1f}")
        # Sanity check: mean score for bad applicants should be LOWER than
        # for good applicants, since higher score = safer.
        bad_mean = scores[df["TARGET"] == 1].mean()
        good_mean = scores[df["TARGET"] == 0].mean()
        direction = "OK (bad < good)" if bad_mean < good_mean else "PROBLEM (bad >= good)"
        print(f"{name} mean score -- bad applicants: {bad_mean:.1f}, "
              f"good applicants: {good_mean:.1f} [{direction}]")

    joblib.dump({
        "points_table": points_table,
        "base_points": base_points(model),
        "train_scores": train_scores, "val_scores": val_scores, "test_scores": test_scores,
    }, MODELS_DIR / "scorecard.pkl")
    print(f"\nSaved scorecard to {MODELS_DIR / 'scorecard.pkl'}")
    print("\nNext: validation.py for KS/Gini/calibration on these scores, "
          "then the XGBoost challenger, then monitoring.py for PSI.")


if __name__ == "__main__":
    main()
