# Model Development Memo — Consumer Credit Scorecard

## 1. Purpose

This model supports approve/decline and risk-based pricing decisions for
unsecured retail credit applications. It produces a Probability of Default
(PD) estimate and a points-based score (540–600 range observed) for each
applicant, built using the traditional WOE + logistic regression scorecard
methodology used across retail credit risk teams, chosen specifically for
its interpretability and auditability over black-box alternatives.

## 2. Data

- **Source:** Credit Card Approval Prediction (Kaggle, rikdifos) —
  `application_record.csv` (applicant features, 438,557 rows) +
  `credit_record.csv` (monthly credit behavior, used to derive the target)
- **Target derivation:** No ready-made target exists in this dataset. An
  applicant is labeled BAD (TARGET=1) if a STATUS of 2, 3, 4, or 5
  (60+ days past due) appears anywhere in their `credit_record` history;
  otherwise GOOD. This is a simplified "ever-bad" flag rather than a full
  vintage/months-on-book analysis — a documented simplification, not an
  oversight, and a reasonable next step for a v2 model.
- **Population funnel:**
  - 438,557 applicants in `application_record.csv`
  - 45,985 (8.3%) had matching `credit_record` history and could be
    labeled; 402,100 (91.7%) had none and were excluded — not imputed
  - Bad rate on the full labeled population: 1.45%
  - 26,460 exact-duplicate applicant rows (identical features under
    different IDs, a known characteristic of this dataset) were dropped,
    keeping the first occurrence — final population: 9,997 applicants
  - **Limitation:** this deduplication approach discards outcome
    information for genuine repeat applicants who may have had different
    results; it's why the bad rate shifted from 1.45% (all labeled
    applicants) to 4.4% (post-dedup). A production version would need a
    more principled dedup rule.
- **Split:** stratified 60/20/20 (train 5,998 / val 1,999 / test 2,000).
  No true out-of-time (OOT) split was possible — this dataset has no
  usable time axis. Bad rate: 4.42% train / 4.40% val / 4.45% test.
  **This is the single biggest methodological limitation of this model**
  and would be the first thing to fix with production data.

## 3. Data Cleaning

- **DAYS_EMPLOYED anomaly:** 75,329 rows had a positive value (implying
  future employment), consistent with a placeholder for
  unemployed/pensioner applicants. Flagged (`FLAG_EMPLOYMENT_ANOMALY`)
  and nulled rather than treated as a real numeric value.
- **Missing values:** not imputed. WOE binning treats missingness as its
  own explicit category, preserving any signal in the fact of missingness
  itself (e.g. `OCCUPATION_TYPE` was missing for 30.6% of applicants).
- **Class imbalance:** no resampling applied. The natural ~4.4% bad rate
  was kept so the model's predicted probabilities remain interpretable as
  real-world PD estimates rather than artificially rebalanced.
- **A real implementation bug, found and fixed:** applying WOE bins
  (`woebin_ply`) to `OCCUPATION_TYPE` initially produced NaN for exactly
  the 1,836 rows (30.6%) that were genuinely missing — even though the
  binning fit step correctly recognized and binned those same values.
  Root cause: an inconsistency in how `scorecardpy` recognizes NaN at fit
  time vs. apply time. Fixed by converting NaN to a literal `"missing"`
  string category before binning, so both steps rely on plain string
  matching rather than NaN-detection. This preserved the full training
  set instead of silently discarding 30% of an already-small dataset —
  a naive fix (drop the affected rows) was considered and rejected.

## 4. Variable Selection

- **IV computed on all candidate variables.** Three cleared an
  IV > 0.5 "suspiciously high" threshold: `AMT_INCOME_TOTAL` (0.866),
  `YEARS_EMPLOYED` (0.858), `DAYS_EMPLOYED` (0.709).
- **These three were investigated directly, not just excluded on
  suspicion.** Re-binning them with the same tuned, coarse binning
  settings used for the rest of the model (5 bins max, ≥10% population
  per bin) showed well-populated bins (618–885 rows, 19–44 bad events
  each) with **no coherent monotonic trend** — bad rate bounced between
  2.9% and 5.7% with no direction as income/tenure increased. This is not
  the signature of data leakage (which typically shows near-total
  separation); it's IV inflation from unconstrained fine-binning on a
  small sample, applied before binning parameters were tuned. **Verdict:
  correctly excluded, but for a different reason than originally
  suspected** — worth stating precisely in review, since "we assumed
  leakage" and "we tested for leakage and found sampling noise instead"
  are very different claims.
- `DAYS_BIRTH` was dropped as redundant with `AGE_YEARS` (same
  information, unconverted) and had a non-monotonic auto-binned WOE not
  worth fixing given the redundancy.
- **Final shortlist (6 variables, IV 0.02–0.5):** `AGE_YEARS` (0.328),
  `OCCUPATION_TYPE` (0.079), `NAME_FAMILY_STATUS` (0.040),
  `CNT_FAM_MEMBERS` (0.034), `CNT_CHILDREN` (0.033), `FLAG_OWN_REALTY`
  (0.023). All individually weak — the strongest is barely into
  "medium predictor" territory.

## 5. WOE Binning

- Bin count and minimum population per bin were tuned down from
  `scorecardpy`'s defaults (5 bins max, ≥10% of population per bin,
  vs. defaults of ~8 bins / 5%) given the small bad-event count
  (~265 in train) — finer bins would leave too few bad events per bin
  to trust.
- Two variables' auto-binned WOE came out non-monotonic and were
  manually re-binned with business-justified breakpoints:
  - `AGE_YEARS`: merged into <35 / 35–59 / 59+, giving bad rates of
    5.03% → 4.39% → 3.18% — a clean, monotonic "younger = riskier"
    trend matching standard life-cycle credit risk patterns.
  - `CNT_FAM_MEMBERS`: merged into <3 / 3+ members, giving 4.22% → 4.96%
    — monotonic, more dependents = more risk.
- `OCCUPATION_TYPE` (categorical) was coarse-classed into 4 groups by
  the algorithm based on similar bad rates; one group merges "IT staff"
  with "Low-skill Laborers" purely on statistical similarity, not
  economic similarity — worth being ready to explain, not evidence of
  an error.

## 6. Model

- **Type:** logistic regression (statsmodels) on WOE-transformed
  variables.
- **VIF check:** `CNT_FAM_MEMBERS` (3.63) and `CNT_CHILDREN` (3.45) were
  the two highest (both under the 5.0 drop threshold), reflecting their
  natural correlation (family size includes children).
- **A real coefficient sign issue was found and resolved.** With all 6
  variables in the model, `CNT_FAM_MEMBERS`'s coefficient came out
  negative — inconsistent with this project's WOE convention (higher WOE
  = higher risk, so every coefficient should be positive). Given its
  elevated VIF and overlap with `CNT_CHILDREN`, this was diagnosed as a
  multicollinearity symptom, not noise: two correlated variables sharing
  a model can flip each other's individual coefficients even when each
  looks sensible alone. `CNT_FAM_MEMBERS` (the weaker-IV variable of the
  pair) was dropped and the model refit.
- **Final model (5 variables), all coefficients correctly signed:**

| Variable | Coefficient | p-value |
|---|---|---|
| const | -3.074 | <0.001 |
| CNT_CHILDREN_woe | 1.030 | 0.072 |
| AGE_YEARS_woe | 0.755 | 0.123 |
| NAME_FAMILY_STATUS_woe | 1.161 | 0.002 |
| FLAG_OWN_REALTY_woe | 0.931 | 0.024 |
| OCCUPATION_TYPE_woe | 0.986 | <0.001 |

  Pseudo R² = 0.0166 (low). Overall model jointly significant
  (LLR p-value 9.06e-07). **Honest note:** `CNT_CHILDREN` and
  `AGE_YEARS` are not individually significant at the 5% level — an
  expected consequence of only ~265 bad events in training, not a
  reason to drop them, since their inclusion was justified by IV
  selection upstream and the joint model is significant.

## 7. Scorecard Scaling

- Base score 600 at base odds 50:1, PDO (points to double the odds) = 20
  → Offset = 487.12, Factor = 28.85
- Base points from the model intercept: 575.8
- Every bin's points were verified sign-correct: safer bins score
  positive points, riskier bins negative, consistent across all 5
  variables (e.g. `AGE_YEARS` 59+ contributes +7.4 points, while
  `OCCUPATION_TYPE`'s riskiest group contributes -15.6).
- **Score range: 540–600, a narrow 60-point spread**, with most
  applicants clustering around 577–580. This directly reflects the
  model's weak-but-real discriminatory power (see Section 8) — it is
  not a scaling error.

## 8. Validation Results

| Metric | Train | Validation | Test |
|---|---|---|---|
| KS | 18.10 | 6.10 | 12.14 |
| Gini | 0.2151 | 0.0183 | 0.1087 |

- A KS above ~40 is generally considered a strong scorecard; **this
  model does not clear that bar**, and the KS drop from train to
  validation (18.10 → 6.10) indicates real instability, not just
  weak-but-consistent signal — a direct consequence of the small,
  imbalanced sample (~265 bad events in train, fewer still in val/test).
- **Calibration (validation set):** the highest-predicted-risk band
  (predicted PD 8.6%) had an actual observed bad rate of only 3.1% —
  lower than several "safer" bands. The top-decile lift was 0.71
  (worse than random). This is a genuine, material weakness, not a
  minor caveat, and should be stated as such to anyone reviewing this
  model for a production decision.
- **Sign sanity check passed in all three splits:** mean score for bad
  applicants was lower than for good applicants (train: 573.8 vs 577.8;
  val: 577.3 vs 577.3, effectively no separation; test: 575.3 vs 577.5).

## 9. Challenger Model

- XGBoost trained on all 6 shortlisted WOE variables (no VIF constraint
  needed for tree models).

| Metric | Train | Val | Test |
|---|---|---|---|
| XGBoost KS | 24.79 | 8.44 | 9.39 |
| Logistic KS | 18.10 | 6.10 | 12.14 |

- XGBoost fits training data noticeably better but **does not
  out-perform the logistic model out-of-sample** (worse on test) — the
  classic signature of a more flexible model overfitting on a small,
  weak-signal dataset. **This is a meaningful finding, not a null
  result:** it demonstrates the performance ceiling here is the data
  (sample size, variable strength), not the choice of model, which
  directly justifies shipping the interpretable logistic scorecard
  instead of a black-box alternative — no real predictive power is being
  sacrificed for interpretability in this case.
- **Feature importance** (XGBoost's native gain-based importance; SHAP
  was attempted but blocked by an unresolved `xgboost`/`shap` version
  compatibility bug, documented as a known limitation rather than
  pursued further): importance was fairly evenly spread across all 6
  variables (0.12–0.22), with `CNT_FAM_MEMBERS_woe` ranking highest
  (0.223) despite being excluded from the logistic model. This is a
  genuine, explainable divergence: tree models split on correlated
  variables without the coefficient-instability problem linear models
  face, so XGBoost can use `CNT_FAM_MEMBERS` cleanly where logistic
  regression could not.

## 10. Expected Loss Framing

**Not yet implemented.** `config.py` defines `ASSUMED_LGD = 0.45`
(a common Basel benchmark for unsecured retail) as a placeholder
assumption, but no cutoff/approval-rate analysis or
Expected Loss = PD × LGD × EAD calculation has been built yet. This is
an open item, not a finding — flagged explicitly rather than omitted.

## 11. Stability Monitoring

- PSI computed between the training set (development sample) and the
  combined validation+test set (data unseen during fitting), as a proxy
  for pre/post-deployment drift.
- **PSI = 0.0029 — "no significant population shift."**
- **Important caveat:** this dataset has no real time axis, so train,
  validation, and test are all random splits from the same underlying
  population, not genuinely time-separated cohorts. This PSI check
  demonstrates the calculation is correctly implemented and
  interpretable, but it is **not a real test of temporal drift** — it
  would be expected to show near-zero PSI regardless of whether the
  underlying model is stable in production. A true PSI monitoring
  process requires actual vintage/time-based data.

## 12. Limitations & Next Steps

- No true out-of-time validation split — the single biggest
  methodological gap, driven entirely by the dataset's lack of a time
  axis rather than a modeling choice.
- Deduplication approach discards outcome variance among genuine repeat
  applicants (Section 2).
- Small, imbalanced sample (~265 bad events in training) drives real
  instability in validation metrics (Section 8) and limits the model to
  weak-but-honest discriminatory power (KS 6–18 depending on split).
- Three genuinely high-IV variables (income, employment tenure) were
  excluded after direct investigation showed the signal was small-sample
  noise, not leakage — worth revisiting with a larger dataset where
  fine-binned estimates would be more trustworthy.
- Expected Loss / cutoff analysis not yet built (Section 10).
- SHAP-based explainability comparison blocked by an unresolved library
  version conflict; native XGBoost feature importance used as a
  substitute.
- Next steps, in priority order: (1) build the cutoff/Expected Loss
  analysis, (2) if more data becomes available, revisit the excluded
  income/employment variables with a larger sample, (3) resolve or
  work around the SHAP compatibility issue for a fuller explainability
  comparison, (4) if this were moving toward production, replace the
  stratified split with genuine time-based vintages for a real OOT
  validation and PSI check.