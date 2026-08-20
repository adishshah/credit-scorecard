"""
Consumer Credit Scorecard -- scoring demo.
Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import streamlit as st
import pandas as pd
import joblib

from src.config import MODELS_DIR
from src.scorecard import score_dataframe, get_offset_factor
from src.woe_binning import apply_woe

st.set_page_config(page_title="Credit Scorecard Demo", layout="centered")
st.title("Consumer Credit Scorecard")
st.caption("Retail PD scorecard demo -- WOE + logistic regression scaled to points.")


@st.cache_resource
def load_artifacts():
    bins = joblib.load(MODELS_DIR / "woe_bins.pkl")
    model_data = joblib.load(MODELS_DIR / "logistic_model.pkl")
    scorecard_data = joblib.load(MODELS_DIR / "scorecard.pkl")
    return bins, model_data, scorecard_data


try:
    bins, model_data, scorecard_data = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Model artifacts aren't built yet: "
        f"`{e.filename}` is missing. Run the pipeline through "
        "src/run_binning.py and src/model.py first, then re-run this app."
    )
    st.stop()

model = model_data["model"]
kept_cols = model_data["kept_cols"]  # WOE-suffixed cols, e.g. "AGE_YEARS_woe"
train_scores = scorecard_data["train_scores"]

# score_dataframe() expects the *_woe column names in kept_cols, but takes
# raw (unbinned) applicant data and WOE-transforms it internally via
# apply_woe() -- so the form below still collects raw values like
# AGE_YEARS, not WOE scores. We just need the raw variable names to know
# which inputs to render.
raw_vars = [c.replace("_woe", "") for c in kept_cols]

# Score bands from the actual train score distribution (quartiles) --
# adapts to whatever the model actually produces rather than hardcoded
# guesses at cutoffs.
band_cuts = train_scores.quantile([0.25, 0.5, 0.75]).tolist()


def score_to_band(score: int) -> str:
    if score <= band_cuts[0]:
        return "High Risk"
    elif score <= band_cuts[1]:
        return "Medium-High Risk"
    elif score <= band_cuts[2]:
        return "Medium-Low Risk"
    else:
        return "Low Risk"


# ---------------------------------------------------------------------
# Score -> PD, and per-variable point contributions ("why" this score).
# Both are derived directly from the same Offset/Factor/PDO scaling
# src/scorecard.py uses, so they stay consistent with the score itself
# rather than being a second, separately-calibrated estimate.
# ---------------------------------------------------------------------
_OFFSET, _FACTOR = get_offset_factor()


def score_to_pd(score: int) -> float:
    """
    Invert Score = Offset - Factor * logit(P(bad)) back to a PD.
    logit = ln(P(bad)/P(good)); score.py subtracts Factor*logit from
    Offset, so logit = (Offset - Score) / Factor, then sigmoid it.
    """
    logit = (_OFFSET - score) / _FACTOR
    return 1.0 / (1.0 + math.exp(-logit))


def get_contributions(model, bins: dict, kept_cols: list, raw_df: pd.DataFrame) -> dict:
    """
    Points contributed by each variable for this one applicant, so the
    demo can show which factors drove the score (most negative = biggest
    risk driver, mirroring the sign convention in score_dataframe).
    """
    woe_df = apply_woe(raw_df, bins)
    contributions = {}
    for col in kept_cols:
        beta = model.params[col]
        var = col.replace("_woe", "")
        contributions[var] = round(-_FACTOR * beta * woe_df[col].iloc[0], 1)
    return contributions


# ---------------------------------------------------------------------
# Input widgets, one per possible shortlisted variable. Only the ones
# that actually survived VIF (kept_cols) get rendered, so this form
# doesn't break if a different subset than expected makes it into the
# final model.
# ---------------------------------------------------------------------
def render_age_years(col):
    age = col.number_input("Age (years)", min_value=18, max_value=100, value=35)
    return {"AGE_YEARS": float(age)}, age


def render_days_birth(col, age_hint=None):
    # Dataset convention: DAYS_BIRTH is negative days since birth.
    # If AGE_YEARS is also being collected, derive DAYS_BIRTH from it
    # instead of asking twice.
    if age_hint is not None:
        return {"DAYS_BIRTH": -round(age_hint * 365.25)}
    age = col.number_input("Age (years)", min_value=18, max_value=100, value=35, key="age_for_days_birth")
    return {"DAYS_BIRTH": -round(age * 365.25)}


def render_occupation(col):
    occ = col.selectbox("Occupation type", [
        "Laborers", "Core staff", "Accountants", "Managers", "Drivers",
        "High skill tech staff", "Cleaning staff", "Private service staff",
        "Cooking staff", "Low-skill Laborers", "Medicine staff", "Secretaries",
        "HR staff", "Waiters/barmen staff", "Security staff", "Realty agents",
        "Sales staff", "IT staff", "Prefer not to say / missing",
    ])
    # WOE binning treats missing as its own bin -- represent it as a real
    # NaN, not the literal string "missing", so it hits that bin correctly.
    val = None if occ == "Prefer not to say / missing" else occ
    return {"OCCUPATION_TYPE": val}


def render_family_status(col):
    fam = col.selectbox("Family status", [
        "Married", "Single / not married", "Civil marriage", "Separated", "Widow",
    ])
    return {"NAME_FAMILY_STATUS": fam}


def render_cnt_fam_members(col):
    n = col.number_input("Number of family members", min_value=1, max_value=20, value=1)
    return {"CNT_FAM_MEMBERS": float(n)}


def render_cnt_children(col):
    n = col.number_input("Number of children", min_value=0, max_value=20, value=0)
    return {"CNT_CHILDREN": n}


def render_own_realty(col):
    val = col.radio("Owns real estate?", ["Y", "N"])
    return {"FLAG_OWN_REALTY": val}


st.subheader("Applicant Details")
col1, col2 = st.columns(2)

applicant = {}
age_value = None  # cached so DAYS_BIRTH can reuse it instead of asking twice

# Order matters only for layout; each renderer is independent.
left_fields = ["AGE_YEARS", "CNT_CHILDREN", "FLAG_OWN_REALTY", "CNT_FAM_MEMBERS"]
right_fields = ["NAME_FAMILY_STATUS", "OCCUPATION_TYPE", "DAYS_BIRTH"]

for field in left_fields:
    if field not in raw_vars:
        continue
    if field == "AGE_YEARS":
        vals, age_value = render_age_years(col1)
        applicant.update(vals)
    elif field == "CNT_CHILDREN":
        applicant.update(render_cnt_children(col1))
    elif field == "FLAG_OWN_REALTY":
        applicant.update(render_own_realty(col1))
    elif field == "CNT_FAM_MEMBERS":
        applicant.update(render_cnt_fam_members(col1))

for field in right_fields:
    if field not in raw_vars:
        continue
    if field == "NAME_FAMILY_STATUS":
        applicant.update(render_family_status(col2))
    elif field == "OCCUPATION_TYPE":
        applicant.update(render_occupation(col2))
    elif field == "DAYS_BIRTH":
        applicant.update(render_days_birth(col2, age_hint=age_value))

missing_from_form = [v for v in raw_vars if v not in applicant]
if missing_from_form:
    st.warning(
        "The trained model uses variable(s) this form doesn't have an "
        f"input for yet: {missing_from_form}. Add a renderer for these "
        "in app/streamlit_app.py before scoring will work correctly."
    )

if st.button("Score Applicant", type="primary", disabled=bool(missing_from_form)):
    applicant_df = pd.DataFrame([applicant])
    score = int(score_dataframe(model, bins, kept_cols, applicant_df).iloc[0])
    band = score_to_band(score)
    pd_estimate = score_to_pd(score)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", score)
    c2.metric("Estimated PD", f"{pd_estimate:.1%}")
    c3.metric("Risk Band", band)

    st.caption(
        f"Training data score range: {int(train_scores.min())}-{int(train_scores.max())} "
        f"(mean {train_scores.mean():.0f}). Higher score = lower predicted risk."
    )

    contributions = get_contributions(model, bins, kept_cols, applicant_df)
    # Sort ascending: most negative (biggest risk driver) first.
    ranked = sorted(contributions.items(), key=lambda kv: kv[1])
    st.subheader("Top Contributing Factors")
    for var, pts in ranked[:3]:
        direction = "reduced" if pts < 0 else "boosted"
        st.write(f"- **{var}**: {direction} the score by {abs(pts):.1f} points")

st.divider()
with st.expander("About this model"):
    st.write(
        f"Logistic regression scorecard on WOE-binned features "
        f"({', '.join(raw_vars)}). "
        "Built on a small, imbalanced sample "
        "(~9,997 labeled applicants, ~4.4% bad rate) -- see the model memo "
        "for full validation numbers (KS/Gini/PSI) and known limitations "
        "(dedup label loss, ever-bad target definition, unresolved "
        "suspicious-IV variables)."
    )
    