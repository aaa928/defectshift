"""Appendix diagnostics: three diagnostics of the final benchmark recomputed on lot-disjoint target test rows. It reads only saved predictions and the training-row records (no training).
The definitions follow 06c_aggregate.py (tail bins, unseen categories) and 06d_diagnostics.py (error concentration); only the evaluation rows change.
Lot-disjoint = B_te rows whose lot also occurs in the source training pool A_dev are excluded (the same definition as lot_disjoint_gaps.py).

  1. Error concentration: per fit (ID_match and OOD_match, repetition), the share of the sum of squared errors on the lot-disjoint rows that falls on the top 1 % (ceil) rows by absolute error within these rows.
     The top rows are selected again within the evaluation rows (the rows selected on all rows are not carried over).
  2. Unseen categories: per fit, the share of lot-disjoint rows that have at least one category value absent from the training rows (fit + early stopping). Categorical columns = the input categorical columns of that domain.
  3. Tail-bin MAE: the bin boundaries are fixed once per domain from the y quantiles (0, 50, 80, 90, 95, 100 %) of all B_te rows, as before (not recomputed);
     only the lot-disjoint rows are put into the bins and the MAE and the target-reference gap per bin (MAE_S / MAE_R - 1) are given.
Aggregation: per-fit value -> repetition mean per model -> domain = median over the models (or the summary stated in the table).
Output: results/analysis/<run-id>/{diag_fits.csv, diag_tail.csv, diag_domain.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/diagnostics_lot_disjoint.py --run-id diagnostics_lot_disjoint
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as cm
from defectshift import config
from defectshift.candidates import build_one, feature_sets, load_data
from defectshift.splits import get_A
from defectshift.utils import log

TAIL_EDGES = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)      # the same as aggregate (06c_aggregate.py)
TAIL_LABELS = ("0-50", "50-80", "80-90", "90-95", "95-100")
TOP_FRAC = 0.01                                     # the same as diagnostics (06d_diagnostics.py)
SRC, REF = "ID_match", "OOD_match"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    a = ap.parse_args()
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = cm.load_challenge(a.benchmark_run_id)
    fits, tails = [], []
    for name, g in pf.groupby("candidate", sort=False):
        slot, mA, _ = build_one(df, name)
        A = get_A(slot, df.index[mA], y, lot)
        dev_lots = set(lot.loc[A["A_dev"]].unique())
        cats, _ = feature_sets(df, name)
        idx0 = np.load(g.pred_path.iloc[0])["ood_idx"]
        y_all = y.loc[idx0].to_numpy()
        edges = np.quantile(y_all, TAIL_EDGES)               # existing boundaries (all B_te rows)
        keep = ~lot.loc[idx0].isin(dev_lots).to_numpy()
        yk = y_all[keep]
        ev_cats = df.loc[idx0[keep], cats]
        for r in g.itertuples():
            z = np.load(r.pred_path)
            if not np.array_equal(z["ood_idx"], idx0):
                raise SystemExit(f"{name} {r.model} r{r.repeat} {r.regime}: the B_te row order differs")
            prefix = r.pred_path[: -len(".pred.npz")]
            f = np.load(prefix + ".npz")
            train = np.concatenate([f["fit_idx"], f["es_idx"]]).astype(np.int64)
            seen = {c: set(df.loc[train, c].unique()) for c in cats}
            new = np.column_stack([~ev_cats[c].isin(seen[c]).to_numpy() for c in cats]) if cats else np.zeros((len(ev_cats), 0), bool)
            e = z["ood_pred"][keep] - yk
            se = e ** 2
            k = int(np.ceil(TOP_FRAC * len(e)))
            top = np.argsort(-np.abs(e), kind="stable")[:k]
            fits.append(dict(candidate=name, model=r.model, repeat=r.repeat, regime=r.regime, n_rows=int(keep.sum()), n_top=k,
                             top1_sse_share=float(se[top].sum() / se.sum()), unseen_cat_rows=float(new.any(axis=1).mean())))
            ae = np.abs(e)
            for b, lab in enumerate(TAIL_LABELS):
                lo, hi = edges[b], edges[b + 1]
                m = (yk >= lo) & ((yk <= hi) if b == len(TAIL_LABELS) - 1 else (yk < hi))
                tails.append(dict(candidate=name, model=r.model, repeat=r.repeat, regime=r.regime, bin=lab, n=int(m.sum()),
                                  y_lo=float(lo), y_hi=float(hi), mae=float(ae[m].mean()) if m.any() else np.nan))
        log(f"[diag_disjoint] {cm.short(name)}: evaluation rows {keep.sum()}/{len(keep)}")
    fits, tails = pd.DataFrame(fits), pd.DataFrame(tails)
    if (fits.groupby(["candidate", "model", "regime"]).repeat.nunique() != config.N_REPEATS_BENCHMARK).any():
        raise SystemExit("a (domain, model, regime) has a wrong number of repetitions")
    nb = tails.dropna(subset=["mae"]).groupby(["candidate", "model", "regime", "bin"]).repeat.nunique()
    if not nb.eq(config.N_REPEATS_BENCHMARK).all():
        raise SystemExit(f"a tail bin is empty or has a wrong number of repetitions: {nb[nb != config.N_REPEATS_BENCHMARK].head().to_dict()}")
    w = tails.pivot_table(index=["candidate", "model", "repeat", "bin"], columns="regime", values="mae").reset_index()
    w["tr_mae_pct"] = 100 * (w[SRC] / w[REF] - 1)
    tail_cells = w.groupby(["candidate", "model", "bin"])[[SRC, REF, "tr_mae_pct"]].mean().reset_index()
    fc = fits.groupby(["candidate", "model", "regime"])[["top1_sse_share", "unseen_cat_rows"]].mean().reset_index()
    rows = []
    for name, c in fc.groupby("candidate"):
        s, t = c[c.regime == SRC], c[c.regime == REF]
        tc = tail_cells[tail_cells.candidate == name]
        top_bin = tc[tc.bin == "95-100"]
        rows.append(dict(candidate=name, n_rows=int(fits[fits.candidate == name].n_rows.iloc[0]),
                         top1_share_src_med=s.top1_sse_share.median(), top1_share_ref_med=t.top1_sse_share.median(),
                         top1_share_min=c.top1_sse_share.min(), top1_share_max=c.top1_sse_share.max(),
                         unseen_src_max=s.unseen_cat_rows.max(), unseen_ref_max=t.unseen_cat_rows.max(),
                         tail95_tr_mae_med=float(top_bin.tr_mae_pct.median()),
                         tail95_n=int(tails[(tails.candidate == name) & (tails.bin == "95-100")].n.iloc[0])))
    dom = pd.DataFrame(rows)
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    fits.to_csv(os.path.join(out, "diag_fits.csv"), index=False)
    tail_cells.merge(tails.drop_duplicates(["candidate", "bin"])[["candidate", "bin", "n", "y_lo", "y_hi"]], on=["candidate", "bin"]) \
        .to_csv(os.path.join(out, "diag_tail.csv"), index=False)
    dom.to_csv(os.path.join(out, "diag_domain.csv"), index=False)
    print(dom.round(3).to_string(index=False))
    log("[diag_disjoint] done")


if __name__ == "__main__":
    main()
