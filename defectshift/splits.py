"""Splits.

A (source):  A_heldout (fixed 20%) | A_dev (80%) -> tuning split (A_dev 80/20) / A_train and A_val of repetition i (A_dev 75/25)
B (target):  B_tr / B_val / B_screen / B_te = about 48 / 12 / 15 / 25%, fixed per candidate
Internal early-stopping split: the n* records split 80/20 (early-stopping models only)

Every split is grouped by production lot (all records of a lot on one side, config.LOT_COL) and stratified by y > 0;
if that is impossible the split raises an error. Each step takes one fold of StratifiedGroupKFold(k), so every share
has the form 1/k of the remaining records, and record shares are approximate because lots differ in size (the actual
sizes are recorded in the result rows). Lots that straddle a domain boundary (a categorical level or a temporal
cutoff) can remain on both sides of A and B; the domain definition is not changed.
A_heldout never enters tuning, training or early stopping (`check_A`).
The stratification of the four subsets of B uses the labels when the benchmark is built; the labels are not exposed to
training, tuning or evaluation through it, and it only stabilizes the share of nonzero defect rates across subsets.
Each stage receives only the subsets of B it may use (`B_view`); screening never receives B_te.
"""
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from defectshift import config
from defectshift.candidates import slot_dir, subsample


def _strat_split(idx, y, groups, test_size, seed, what="split"):
    """Split idx into (rest, test_size share), grouped by lot and stratified by y > 0: the first fold of
    StratifiedGroupKFold(k = 1/test_size) is the share. If stratification is impossible (fewer lots than k, or a y > 0
    or y = 0 stratum of the input missing on one side), the function stops instead of falling back to another split,
    so that the protocol of lot-grouped, y > 0 stratified splits is never changed silently."""
    k = round(1 / test_size)
    assert abs(k * test_size - 1) < 1e-9, f"{what}: test_size {test_size} is not of the form 1/k"
    idx = np.asarray(idx)
    strat = (y.loc[idx] > 0).astype(int).to_numpy()
    g = groups.loc[idx].to_numpy()
    try:
        rest, part = next(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(idx, strat, g))
    except ValueError as e:
        raise RuntimeError(f"{what}: lot-grouped split stratified by y > 0 is impossible (n={len(idx)}, {len(np.unique(g))} lots, k={k}): {e}") from e
    for side, ix in (("rest", rest), ("share", part)):
        if len(np.unique(strat[ix])) < len(np.unique(strat)):
            raise RuntimeError(f"{what}: lot-grouped split stratified by y > 0 is impossible: the {side} side lacks the y > 0 or y = 0 "
                               f"stratum (n={len(idx)}, share of y > 0 {strat.mean():.3f}, {len(np.unique(g))} lots, k={k})")
    return pd.Index(idx[rest]), pd.Index(idx[part])


def _disjoint(*parts):
    allp = np.concatenate([np.asarray(p) for p in parts])
    return len(allp) == len(np.unique(allp))


# ---------------------------------------------------------------- source side A
def split_A(A_idx, y, groups):
    """A -> A_heldout (fixed) / A_dev."""
    A_dev, A_heldout = _strat_split(A_idx, y, groups, config.A_HELDOUT_FRAC, config.A_HELDOUT_SEED, "A_heldout split")
    assert _disjoint(A_dev, A_heldout) and len(A_dev) + len(A_heldout) == len(A_idx)
    return dict(A_heldout=A_heldout, A_dev=A_dev)


def hpo_split(A_dev, y, groups):
    """A_dev -> (tuning training rows, tuning validation rows). One per candidate, independent of the repetition."""
    return _strat_split(A_dev, y, groups, config.HPO_VAL_FRAC, config.HPO_SEED, "tuning split")


def repeat_split(A_dev, y, groups, i):
    """A_dev -> (A_train, A_val) of repetition i."""
    return _strat_split(A_dev, y, groups, config.A_VAL_FRAC_OF_DEV, config.RANDOM_STATE + i, f"repetition {i} A_train/A_val split")


def get_A(slot, A_idx, y, groups):
    """Split of A (one per candidate, cached). An existing cache is checked against a fresh computation."""
    return _cached(os.path.join(config.CACHE_DIR, SPLIT_CACHE, "A", slot_dir(slot) + ".sha256"), split_A(A_idx, y, groups))


def check_A(a, used_sets):
    """Check that A_heldout overlaps none of the sets drawn from A_dev (tuning, training, early stopping)."""
    held = set(a["A_heldout"])
    for name, s in used_sets.items():
        assert not held & set(s), f"A_heldout entered {name}"
        assert set(s) <= set(a["A_dev"]), f"{name} uses rows outside A_dev"


# ---------------------------------------------------------------- target side B
def split_B4(B_idx, y, groups, seed=config.B_SPLIT_SEED, fracs=config.B_FRACS):
    """Split B into four subsets. Order: B_te (1/4 of B) -> B_screen (1/5 of the remaining 75%) -> B_val (1/5 of the
    remaining 60%) -> the rest is B_tr."""
    rest, te = _strat_split(B_idx, y, groups, fracs["B_te"], seed, "B_te split")
    rest, screen = _strat_split(rest, y, groups, fracs["B_screen"] / (1 - fracs["B_te"]), seed, "B_screen split")
    tr, val = _strat_split(rest, y, groups, fracs["B_val"] / (fracs["B_tr"] + fracs["B_val"]), seed, "B_val/B_tr split")
    out = dict(B_tr=tr, B_val=val, B_screen=screen, B_te=te)
    assert _disjoint(*out.values()) and sum(map(len, out.values())) == len(B_idx), "the four subsets of B overlap or miss rows"
    return out


STAGE_ALLOWED = {
    "screening": ("B_tr", "B_screen"),          # screening: no B_val (target-side tuning only) and no B_te (final test only)
    "benchmark": ("B_tr", "B_val", "B_te"),
    "selection_audit": ("B_te",),               # after the domains are fixed: inference with saved models only
    "shift_statistics": ("B_tr", "B_screen"),
}


def B_view(slot, B_idx, y, groups, stage):
    """Only the subsets of B that a stage may use. The four subsets exist only inside this function, and only the
    permitted keys leave it. An existing cache is checked against a fresh computation (fixed splits)."""
    b4 = _cached(os.path.join(config.CACHE_DIR, SPLIT_CACHE, "B4", slot_dir(slot) + ".sha256"), split_B4(B_idx, y, groups))
    return {k: b4[k] for k in STAGE_ALLOWED[stage]}


def b_te_eligible(n_B):
    """Screening eligibility, computed from |B| without reading B_te: floor(0.25 |B|) >= 500."""
    return int(np.floor(config.B_FRACS["B_te"] * n_B)) >= config.B_TE_MIN


# ---------------------------------------------------------------- n* and internal early stopping
def matched(A_train, B_tr, i):
    """(n*, A_match, B_match) of repetition i. n* = min(|A_train|, |B_tr|), sampling seed RANDOM_STATE + i."""
    n_star = min(len(A_train), len(B_tr))
    A_match = subsample(A_train, n_star, config.RANDOM_STATE + i)
    B_match = subsample(B_tr, n_star, config.RANDOM_STATE + i)
    assert len(A_match) == len(B_match) == n_star
    return n_star, A_match, B_match


def es_split(pool, y, groups, i):
    """Internal split of early-stopping models: pool (n* records) -> (training 80%, early stopping 20%). Same rule for A
    and B, seed ES_SEED_BASE + i."""
    fit, es = _strat_split(pool, y, groups, config.ES_VAL_FRAC, config.ES_SEED_BASE + i, f"repetition {i} internal early-stopping split")
    assert _disjoint(fit, es) and len(fit) + len(es) == len(pool)
    return fit, es


# ---------------------------------------------------------------- cache
SPLIT_CACHE = "splits_lot"   # cache folder of the lot-grouped splits


def _digest(split):
    """Sha256 of a split (a dictionary of row-label indices): the names and the row labels of every subset, in order."""
    h = hashlib.sha256()
    for k in sorted(split):
        h.update(k.encode())
        h.update(np.asarray(split[k], dtype=np.int64).tobytes())
    return h.hexdigest()


def _cached(path, fresh):
    """The split is always computed afresh. The cache file holds only its digest: a later run checks that the split did not
    change, and no object is read back from disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    digest = _digest(fresh)
    if os.path.exists(path):
        with open(path) as f:
            assert f.read().strip() == digest, f"split differs from the one recorded in the cache: {path}"
    else:
        tmp = f"{path}.{os.getpid()}.tmp"            # written under a temporary name and renamed, so that a concurrent run never reads a partial file
        with open(tmp, "w") as f:
            f.write(digest + "\n")
        os.replace(tmp, path)
    return fresh
