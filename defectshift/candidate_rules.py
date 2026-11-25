"""Regenerate the candidate domain splits from the data by rule, which shows that the candidate list is the output of a
mechanical rule and not a ranking.

Rule (matches the 78 candidates of the candidate list config.CANDIDATES_CSV, including names and the sizes of A and B;
tests/test_candidate_rules.py):
  process condition: every categorical column (cat_*) except the production-lot column (config.LOT_COL). A = most frequent
          level, B = all other levels (candidates._split_masks); a column is kept if both sides have at least
          config.MIN_DOMAIN records (in this data only cat_003 is removed, with 92 records in B).
  temporal: for each axis (x_time, y_time), 15 equally spaced quantiles from 0.10 to 0.90 (TIME_QUANTILES, linear
          interpolation in pandas) are used as cutoffs. A = up to the cutoff, B = later. Names are `<axis><=<cutoff>`;
          the 8th quantile (0.50) is the pre-specified temporal domain of the axis (its median).
  random-split controls: one per seed of config.PLACEBO_B_TARGET (candidates.build_one).
The pipeline reads the candidate list (candidates.load_candidates); this module checks that the list can be regenerated
by the rule and builds the candidate table (name, kind and sizes of A and B).
"""
import numpy as np
import pandas as pd

from defectshift import config
from defectshift.candidates import build_one, parse_candidate

TIME_AXES = ("x_time", "y_time")
TIME_QUANTILES = np.linspace(0.1, 0.9, 15)


def categorical_candidates(df):
    """Names of the process-condition candidate columns (in column-name order)."""
    out = []
    for c in sorted(c for c in df.columns if c.startswith("cat_") and c != config.LOT_COL):
        n_a = int(df[c].value_counts().iloc[0])
        if min(n_a, len(df) - n_a) >= config.MIN_DOMAIN:
            out.append(c)
    return out


def time_candidates(df):
    """Names of the temporal cutoff candidates (by axis, then by cutoff)."""
    out = []
    for ax in TIME_AXES:
        for t in sorted(set(df[ax].quantile(TIME_QUANTILES))):
            n_a = int((df[ax] <= t).sum())
            if min(n_a, len(df) - n_a) >= config.MIN_DOMAIN:
                out.append(f"{ax}<={t}")
    return out


def candidate_names(df, include_placebo=False):
    """Candidate names: process condition, then temporal, then (optionally) random-split controls."""
    names = categorical_candidates(df) + time_candidates(df)
    if include_placebo:
        names += [f"placebo@{s}" for s in config.PLACEBO_B_TARGET]
    return names


def candidate_table(df, include_placebo=False):
    """Candidate table: candidate, kind (cat, x_time, y_time, placebo), n_A, n_B."""
    rows = []
    for n in candidate_names(df, include_placebo):
        kind, _ = parse_candidate(n)
        _, mA, mB = build_one(df, n)
        rows.append(dict(candidate=n, kind=kind, n_A=int(np.sum(mA)), n_B=int(np.sum(mB))))
    return pd.DataFrame(rows)
