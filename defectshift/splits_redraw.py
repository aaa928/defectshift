"""Redrawn target test split. B_screen of the canonical B split (splits.split_B4) is kept, and B_te', B_val' and B_tr' are drawn again
from the remaining target rows (B_tr + B_val + B_te of the canonical split) in the same way (by lot, stratified by y > 0, one fold of a
StratifiedGroupKFold).

Shares: _strat_split accepts only 1/k, so B_te' is 1/3 of the remaining rows (canonical: 25% of B = 29.4% of the remaining rows; 1/3 is
the closest 1/k value, 28.3% of B), then 1/5 of what is left is B_val' (the canonical ratio B_val / (B_tr + B_val)), and the rest is
B_tr' (45.3% of B, canonical 48%).
This split is not canonical; it serves the sensitivity analysis of the test-lot composition. The canonical split (cache B4/) is not
changed, and the redrawn splits are kept in B4_resample_s<k>/. Kept in a separate module so that the shared training code (splits.py)
is not changed.
"""
import os

import pandas as pd

from defectshift import config
from defectshift.candidates import slot_dir
from defectshift.splits import SPLIT_CACHE, STAGE_ALLOWED, B_view, _cached, _disjoint, _strat_split

RESPLIT_SEEDS = (20261105, 20261106, 20261107, 20261108, 20261109)   # split seeds of redraws 1-5
TE_FRAC = 1 / 3
VAL_FRAC = 1 / 5


def resplit_B(B_idx, y, groups, b_screen, seed):
    """Keep B_screen and draw B_te', B_val' and B_tr' again. B_screen is removed while the order of B_idx is kept (the split depends on the input order)."""
    B_idx = pd.Index(B_idx)
    rest = B_idx[~B_idx.isin(pd.Index(b_screen))]
    if len(rest) + len(b_screen) != len(B_idx):
        raise RuntimeError("B_screen is not a subset of B")
    rest, te = _strat_split(rest, y, groups, TE_FRAC, seed, "redrawn B_te split")
    tr, val = _strat_split(rest, y, groups, VAL_FRAC, seed, "redrawn B_val/B_tr split")
    out = dict(B_tr=tr, B_val=val, B_screen=pd.Index(b_screen), B_te=te)
    assert _disjoint(*out.values()) and sum(map(len, out.values())) == len(B_idx), "the redrawn B split overlaps or drops rows"
    names = list(out)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(groups.loc[out[a]]) & set(groups.loc[out[b]]):
                raise RuntimeError(f"redrawn B split: {a} and {b} share a lot")
    return out


def B_view_resplit(slot, B_idx, y, groups, stage, k):
    """B_view of redraw k (1-5). The canonical B_screen is read from the screening-stage view (B_tr, B_screen), which does not see B_te.
    An existing cache is checked against a fresh computation."""
    if k not in range(1, len(RESPLIT_SEEDS) + 1):
        raise ValueError(f"the redraw number is 1-5: {k}")
    canon = B_view(slot, B_idx, y, groups, "screening")
    path = os.path.join(config.CACHE_DIR, SPLIT_CACHE, f"B4_resample_s{k}", slot_dir(slot) + ".sha256")
    b = _cached(path, resplit_B(B_idx, y, groups, canon["B_screen"], RESPLIT_SEEDS[k - 1]))
    return {key: b[key] for key in STAGE_ALLOWED[stage]}
