"""Splits of the time-ordered configuration (the reference is fitted on earlier lots only). Kept in a separate module so that the
shared training code (splits.py) is not changed.

Target side: the target lots are ordered by axis time (median time of the target rows of the lot) and assigned by the start position of
the lot (cumulative share of rows) to B_val 12%, B_screen 15%, B_tr 48% and B_te 25% from the front (no time buffer, no stratification by y).
Training candidate rule (the same for both axes): tau_min is the earliest row time of B_te. For every lot L of the training candidate
rows (A_dev and B_tr), tau_max(L) is the latest row time of that lot among the candidate rows. A lot is used only if tau_max(L) < tau_min
and it is not a test lot (whole lots). The inspection times of y_time spread within a lot and across lots, so the lot median order alone
does not guarantee that only past information is used; this rule does.
Buffer: of the usable lots, the one with the latest tau_max (ties: last by lot name) is removed from both the source and the target side.
The two time axes are handled separately and are not compared.
"""
import numpy as np
import pandas as pd

from defectshift import config
from defectshift.splits import _disjoint


def lot_time(idx, groups, t):
    """Median axis time of every lot within the rows idx (sorted by time, ties by lot name)."""
    idx = pd.Index(idx)
    s = t.loc[idx].groupby(groups.loc[idx].to_numpy()).median()
    return s.sort_index(kind="stable").sort_values(kind="stable")


def order_B(B_idx, groups, t, fracs=config.B_FRACS):
    """Four-way split of the target rows in time order. Returns {B_val, B_screen, B_tr, B_te: row Index}, the target lot times (Series)."""
    idx = pd.Index(B_idx)
    lt = lot_time(idx, groups, t)
    g = groups.loc[idx].to_numpy()
    size = pd.Series(g).value_counts().reindex(lt.index).to_numpy()
    start = (np.cumsum(size) - size) / len(idx)
    edge_val = fracs["B_val"]
    edge_screen = edge_val + fracs["B_screen"]
    edge_tr = edge_screen + fracs["B_tr"]
    part = pd.Series(np.where(start < edge_val, "B_val", np.where(start < edge_screen, "B_screen", np.where(start < edge_tr, "B_tr", "B_te"))), index=lt.index)
    of_row = part.reindex(g).to_numpy()
    out = {k: idx[of_row == k] for k in ("B_val", "B_screen", "B_tr", "B_te")}
    assert _disjoint(*out.values()) and sum(map(len, out.values())) == len(idx), "the time-ordered B split overlaps or drops rows"
    return out, lt, part


def forward_view(A_dev, B_idx, groups, t):
    """Split that uses earlier lots only. Returns a dict:
    B_tr, B_val, B_screen, B_te (row Index; B_tr has the rule and the buffer applied), A_allowed (rows of A_dev usable for source
    training), tau_min, buffer_lot, info (numbers for reporting)."""
    A_dev = pd.Index(A_dev)
    b, lt_B, part = order_B(B_idx, groups, t)
    te_rows = b["B_te"]
    te_lots = set(part.index[part == "B_te"])
    tau_min = t.loc[te_rows].min()
    pool = A_dev.append(b["B_tr"])
    assert not pool.duplicated().any()
    g_pool = groups.loc[pool]
    tmax = t.loc[pool].groupby(g_pool.to_numpy()).max()
    late = tmax >= tau_min
    is_te = tmax.index.isin(list(te_lots))
    elig = tmax[~late & ~is_te].sort_index(kind="stable").sort_values(kind="stable")
    buf_lot = elig.index[-1]
    keep = set(elig.index) - {buf_lot}
    gA, gB = groups.loc[A_dev], groups.loc[b["B_tr"]]
    A_allowed = A_dev[gA.isin(keep).to_numpy()]
    B_tr = b["B_tr"][gB.isin(keep).to_numpy()]
    out = dict(b, B_tr=B_tr)
    train = A_allowed.append(B_tr)
    assert not set(groups.loc[train]) & te_lots, "a test lot is among the training rows"
    assert buf_lot not in set(groups.loc[train]), "the buffer lot remains on the training side"
    assert (t.loc[train] < tau_min).all(), "a training row is not earlier than the first test row"
    late_rows = pool[(t.loc[pool] >= tau_min).to_numpy()]
    late_lots = set(tmax.index[late])
    drop_late = pool[g_pool.isin(late_lots - te_lots).to_numpy()]
    drop_te = pool[g_pool.isin(te_lots).to_numpy()]
    drop_buf = pool[(g_pool == buf_lot).to_numpy()]
    rest = elig.drop(index=buf_lot)
    info = dict(
        n_A_dev=len(A_dev), n_A_allowed=len(A_allowed), n_B_tr_before=len(b["B_tr"]), n_B_tr=len(B_tr), n_B_te=len(te_rows),
        te_lots=len(te_lots), lots_pool=int(tmax.size), lots_used=len(keep),
        violating_rows=len(late_rows), violating_lots=len(late_lots - te_lots),
        rows_removed_with_violating_lots=len(drop_late), rows_removed_A=int(drop_late.isin(A_dev).sum()), rows_removed_B=int(drop_late.isin(b["B_tr"]).sum()),
        test_lots_in_pool=len(te_lots & set(tmax.index)), rows_removed_test_lots=len(drop_te),
        buffer_lot=str(buf_lot), buffer_rows=len(drop_buf), buffer_side="".join(s for s, ix in (("A", A_dev), ("B", b["B_tr"])) if drop_buf.isin(ix).any()),
        tau_min=str(tau_min), last_train_time=str(rest.max()), gap_hours=float((tau_min - rest.max()) / pd.Timedelta(hours=1)),
        te_start_median=str(lt_B[part == "B_te"].min()))
    return dict(out, A_allowed=A_allowed, tau_min=tau_min, buffer_lot=buf_lot, info=info)
