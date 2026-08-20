
"""
Week 2, Day 3-4: scale the fitted logistic model into an actual points-based
scorecard. This is the step that turns "a classifier" into "a scorecard".

Score formula: Score = Offset + Factor * ln(odds_good_to_bad)
Our logistic model predicts logit = ln(P(bad)/P(good)) = ln(odds_bad), since
TARGET=1 means bad. We want higher score = safer, so we use the negative:
ln(odds_good) = -(intercept + sum(beta_i * WOE_i))

This gives:
  base_points  = Offset - Factor * intercept
  points(i,bin) = -Factor * beta_i * WOE_i,bin

Sanity check on sign: in this project's WOE convention (confirmed from the
bin tables throughout), a higher WOE means a riskier bin, and every kept
coefficient came out positive. So for a risky bin (positive WOE), points
come out negative -- lower points for a riskier bin, which is exactly the
behavior a real scorecard should have (higher score = safer applicant).
"""

import math
import pandas as pd
from src.config import BASE_SCORE, BASE_ODDS, PDO
from src.woe_binning import apply_woe


def get_offset_factor():
    """Factor = PDO / ln(2); Offset = Base_Score - Factor * ln(Base_Odds)."""
    factor = PDO / math.log(2)
    offset = BASE_SCORE - factor * math.log(BASE_ODDS)
    return offset, factor


def base_points(model) -> float:
    """Points contributed by the model's intercept alone, before any
    variable-specific points are added."""
    offset, factor = get_offset_factor()
    intercept = model.params.get("const", 0.0)
    return round(offset - factor * intercept, 1)


def build_scorecard_points_table(model, bins: dict, kept_cols: list) -> pd.DataFrame:
    """
    Human-readable points table: one row per (variable, bin), showing how
    many points that bin contributes. This is the table you'd put in the
    model memo / show a non-technical underwriter -- it's what makes this
    a "scorecard" rather than just a fitted model.
    """
    offset, factor = get_offset_factor()
    rows = []
    for col in kept_cols:
        var = col.replace("_woe", "")
        beta = model.params[col]
        for _, row in bins[var].iterrows():
            rows.append({
                "variable": var,
                "bin": row["bin"],
                "points": round(-factor * beta * row["woe"], 1),
            })
    return pd.DataFrame(rows)


def score_dataframe(model, bins: dict, kept_cols: list, raw_df: pd.DataFrame) -> pd.Series:
    """
    Score any raw applicant dataframe end to end: WOE-transform it using
    the fitted bins, then compute Score = base_points + sum(points per var).
    Computed directly from WOE values (not by re-matching against the
    points table) -- simpler and avoids duplicating scorecardpy's own
    bin-matching logic.
    """
    offset, factor = get_offset_factor()
    woe_df = apply_woe(raw_df, bins)
    intercept = model.params.get("const", 0.0)
    score = pd.Series(offset - factor * intercept, index=raw_df.index)
    for col in kept_cols:
        beta = model.params[col]
        score = score - factor * beta * woe_df[col]
    return score.round(0).astype(int)


if __name__ == "__main__":
    offset, factor = get_offset_factor()
    print(f"Offset: {offset:.2f}, Factor: {factor:.2f}")
    print("Run via src.run_scorecard for the full scoring pipeline.")