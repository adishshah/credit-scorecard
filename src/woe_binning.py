import scorecardpy as sc
import pandas as pd
from typing import Optional
from src.config import (

    TARGET_COL, ID_COL, IV_KEEP_THRESHOLD, IV_SUSPICIOUS_THRESHOLD,
    MIN_BIN_POPULATION, BIN_NUM_LIMIT,
)


def compute_iv_table(df: pd.DataFrame) -> pd.DataFrame:
    """IV per variable, ID excluded. Use this to build your candidate
    shortlist -- see config.IV_KEEP_THRESHOLD / IV_SUSPICIOUS_THRESHOLD."""
    cols = [c for c in df.columns if c not in (ID_COL,)]
    iv_df = sc.iv(df[cols], y=TARGET_COL)
    return iv_df.sort_values("info_value", ascending=False)


def shortlist_variables(iv_df: pd.DataFrame) -> list:
    """Keep variables with IV in the useful-but-not-suspicious range.
    Anything above IV_SUSPICIOUS_THRESHOLD needs manual review for leakage."""
    keep = iv_df[
        (iv_df["info_value"] >= IV_KEEP_THRESHOLD)
        & (iv_df["info_value"] < IV_SUSPICIOUS_THRESHOLD)
    ]
    suspicious = iv_df[iv_df["info_value"] >= IV_SUSPICIOUS_THRESHOLD]
    if len(suspicious) > 0:
        print("SUSPICIOUSLY HIGH IV -- check these for leakage before using:")
        print(suspicious)
    return keep["variable"].tolist()


def fine_and_coarse_classing(df: pd.DataFrame, variables: list, breaks_list: Optional[dict] = None) -> dict:
    """
    Fine classing happens automatically inside woebin(); coarse classing is
    controlled here via count_distr_limit (min % of population per bin) and
    bin_num_limit (max bins per variable) -- both tuned down from
    scorecardpy's defaults given the sparse bad-event count.

    breaks_list: optional {variable: [breakpoints]} to override automatic
    binning for variables where it produced a non-monotonic WOE trend.
    Use this once you've eyeballed the output and picked business-sensible
    cutpoints -- don't guess blindly, check the resulting bad rate ordering
    first (see run_binning.py for the reasoning behind the current overrides).

    After this runs, PLOT every bin (sc.woebin_plot) and check:
      1. Is the WOE trend monotonic across bins?
      2. Does each bin hold enough bad events to trust (not just enough rows)?
      3. Does the direction make business sense?
    """
    cols = variables + [TARGET_COL]
    bins = sc.woebin(
        df[cols], y=TARGET_COL,
        count_distr_limit=MIN_BIN_POPULATION,
        bin_num_limit=BIN_NUM_LIMIT,
        breaks_list=breaks_list,
    )
    return bins



def apply_woe(df: pd.DataFrame, bins: dict) -> pd.DataFrame:
    """Transform a dataframe into WOE values using fitted bins -- use bins
    fit on train only for train/val/test, to avoid leakage."""
    return sc.woebin_ply(df, bins)


if __name__ == "__main__":
    print("Run via src.run_binning for the full end-to-end stage 1 pipeline.")