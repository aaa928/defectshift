"""The 78 candidate domain splits (48 process-condition and 30 temporal cutoffs) and the ten random-split controls
(production lots assigned at random, with matched sizes).

Process-condition splits take the most frequent level of a column as A and all other levels as B; temporal splits put
the records up to a cutoff in A and later records in B. Random-split controls are named `placebo@<seed>`. Each candidate
has its own feature set (`feature_sets`): the production-lot column and the column that defines a process-condition
split are removed.
"""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from defectshift import config

_TIME_RE = re.compile(r"^(x|y)_time<=(.+)$")
_PLACEBO_RE = re.compile(r"^placebo@(\d+)$")


def parse_candidate(name):
    """Candidate name -> (kind, payload). Process-condition candidates are column names, temporal cutoffs look like
    `x_time<=2099-06-04 10:07:30.5`, and random-split controls like `placebo@9001`."""
    m = _PLACEBO_RE.match(name)
    if m:
        return "placebo", int(m.group(1))
    m = _TIME_RE.match(name)
    if m:
        return m.group(1) + "_time", pd.Timestamp(m.group(2))
    return "cat", name


def _split_masks(df, kind, payload):
    """(mask of A, mask of B). Process condition: most frequent level (A) against all others (B); temporal: up to the
    cutoff (A) against later records (B)."""
    if kind == "cat":
        majority = df[payload].value_counts().idxmax()
        return df[payload] == majority, df[payload] != majority
    col = "x_time" if kind == "x_time" else "y_time"
    return df[col] <= payload, df[col] > payload


def build_one(df, name):
    """(slot name, mask of A, mask of B) of one candidate."""
    kind, payload = parse_candidate(name)
    if kind == "placebo":   # production lots assigned at random to A or B (stratified by y > 0), |B| close to the target in config
        folds = list(StratifiedGroupKFold(n_splits=config.PLACEBO_N_FOLDS, shuffle=True, random_state=payload).split(
            df.index, (df[config.TARGET_COL] > 0).astype(int), df[config.LOT_COL]))
        m = placebo_n_folds(payload, len(df))
        mB = df.index.isin(np.concatenate([te for _, te in folds[:m]]))
        return f"placebo@{payload}", ~mB, mB
    mA, mB = _split_masks(df, kind, payload)
    slot = f"cat:{payload}" if kind == "cat" else f"time_{kind[0]}@{payload.date()}"
    return slot, mA.to_numpy(), mB.to_numpy()


def subsample(idx, n, seed):
    """Reduce an index to n entries; unchanged if it already has n or fewer."""
    if len(idx) <= n:
        return idx
    rng = np.random.default_rng(seed)
    return pd.Index(rng.choice(np.asarray(idx), size=n, replace=False))


def slot_dir(slot):
    return slot.replace(":", "_").replace("@", "_")


def placebo_n_folds(seed, n_rows):
    """Number of folds that go to B for a control seed = round(PLACEBO_N_FOLDS x target |B| / N)."""
    return int(round(config.PLACEBO_N_FOLDS * config.PLACEBO_B_TARGET[seed] / n_rows))


def load_candidates(include_placebo=False):
    """Candidate names in the order of the candidate list (splits/candidates.csv), random-split controls last."""
    r = pd.read_csv(config.CANDIDATES_CSV)
    assert len(r) == 78, f"{len(r)} candidates, expected 78"
    names = list(r["candidate"])
    if include_placebo:
        names += [f"placebo@{s}" for s in config.PLACEBO_B_TARGET]
    return names


def load_data():
    df = pd.read_parquet(config.DATA_PATH, engine="pyarrow").reset_index(drop=True)
    check_categorical(df)
    return df, df[config.TARGET_COL]


def check_categorical(df):
    """Check the assumptions on the categorical columns (cat_*): no missing values and non-negative integers. Every
    training path relies on them (CatBoost categorical handling, int64 conversion for embeddings and foundation-model
    inputs, the fixed-schema cardinality = maximum + 1). A violation stops before training, so that a changed data file
    cannot break the pipeline silently."""
    import numpy as np
    bad = []
    for c in [c for c in df.columns if c.startswith("cat_")]:
        v = df[c]
        if v.isna().any():
            bad.append(f"{c}: missing {int(v.isna().sum())}")
            continue
        a = np.asarray(v, dtype=float)
        if (a < 0).any() or not np.all(np.equal(np.floor(a), a)):
            bad.append(f"{c}: values that are not non-negative integers")
    if bad:
        raise ValueError("categorical columns violate the assumptions: " + "; ".join(bad[:5]))


def feature_sets(df, name):
    """(cat_cols, num_cols) of one candidate. The production-lot column (config.LOT_COL) is removed for every candidate
    (it is used only as the grouping unit of the splits), and a process-condition split also removes the column that
    defines it. The timestamp columns (x_time, y_time) and the target have other prefixes and are never included."""
    kind, payload = parse_candidate(name)
    cat_cols = [c for c in df.columns if c.startswith("cat_") and c != config.LOT_COL]
    num_cols = [c for c in df.columns if c.startswith("num_")]
    if kind == "cat":
        assert payload in cat_cols, payload
        cat_cols = [c for c in cat_cols if c != payload]
    return cat_cols, num_cols
