"""Lot composition of the size-matched training samples (A_match, B_match). Only the splits are rebuilt (no training or prediction, CPU only).

For every repetition i the rows are rebuilt by the same rule as in the benchmark: repeat_split(A_dev, i) -> A_train, matched(A_train, B_tr, i) -> n*, A_match, B_match;
for the models with early stopping (9 models) es_split(pool, i) divides the pool into 80 % training / 20 % early stopping (by lot). The rows used in the actual optimization are that 80 %.
Reported (side = A or B, row set = match or es_fit):
  n_rows, n_lots (number of distinct lots), median, quartiles and maximum of the rows per lot, the row share of the largest lot, the effective number of lots 1/sum(p^2) (p = row share of a lot)
  shared: the number of lots that occur in both the A-side and the B-side row sets and the row share of these lots (on the A side and on the B side)
Targets: the 7 challenge domains + the 10 randomized lot-split controls. The control names are read from lot_disjoint/ctrl_values.csv.
In addition: the number of lots from which the tail rows left out in the test-row configuration "top 1% excluded" come (tail_lots.csv), and the rank correlation of the lot-count ratio and the gap over the controls (log).
Output: results/analysis/<run-id>/{lot_info_reps.csv (per repetition), lot_info_domain.csv (repetition mean, minimum, maximum), tail_lots.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/lot_composition.py --run-id lot_composition
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as cm
from defectshift import config
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data
from defectshift.splits import B_view, es_split, get_A, matched, repeat_split
from defectshift.utils import log
from verdict_robustness import TOP_FRAC


def lot_stats(rows, lot):
    c = lot.loc[rows].value_counts()
    p = c / c.sum()
    return dict(n_rows=int(c.sum()), n_lots=int(len(c)), rows_per_lot_med=float(c.median()), rows_per_lot_p25=float(c.quantile(.25)),
                rows_per_lot_p75=float(c.quantile(.75)), rows_per_lot_max=int(c.max()), max_lot_share=float(p.max()), eff_lots=float(1 / np.sum(p ** 2)))


def shared(ra, rb, lot):
    la, lb = lot.loc[ra], lot.loc[rb]
    sh = set(la.unique()) & set(lb.unique())
    return dict(n_shared_lots=len(sh), shared_row_share_A=float(la.isin(sh).mean()), shared_row_share_B=float(lb.isin(sh).mean()))


def tail_lots(name, A, bv, y, lot):
    """Lot composition of the tail rows left out in the test-row configuration 'top 1% excluded'. Tail = the ceil(1% x |B_te|) rows with the largest |y - median(y of B_tr)|,
    selected from all of B_te (the same rule as verdict_robustness.py). As in the lot-disjoint evaluation, rows of lots that also occur in A_dev are removed before counting."""
    te = np.asarray(bv["B_te"])
    yv = y.loc[te].to_numpy()
    score = np.abs(yv - float(y.loc[bv["B_tr"]].median()))
    k = max(1, int(np.ceil(TOP_FRAC * len(yv))))
    top = np.zeros(len(yv), bool)
    top[np.argsort(-score, kind="stable")[:k]] = True
    lots = lot.loc[te].to_numpy()
    ld = ~np.isin(lots, lot.loc[A["A_dev"]].unique())
    tl = pd.Series(lots[top & ld]).value_counts()
    return dict(candidate=name, dom=cm.short(name), n_te_rows=len(te), k_top=k, n_top_ld=int((top & ld).sum()), n_ld_rows=int(ld.sum()),
                n_ld_lots=int(len(np.unique(lots[ld]))), n_top_lots=int(len(tl)), top_rows_largest_lot=int(tl.max()) if len(tl) else 0,
                top_rows_3_largest_lots=int(tl.head(3).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--disjoint-run-id", default="lot_disjoint")
    a = ap.parse_args()
    df, y = load_data()
    lot = df[config.LOT_COL]
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    names = [("challenge", n) for n in pf.candidate.unique()]
    cv = pd.read_csv(os.path.join(rf, a.disjoint_run_id, "ctrl_values.csv"))
    names += [("control", n) for n in cv.candidate.unique()]
    if sum(g == "challenge" for g, _ in names) != 7 or sum(g == "control" for g, _ in names) != 10:
        raise SystemExit(f"the number of domains is not 7 + 10: {names}")
    nstar_rec = pf[pf.regime == "ID_match"].groupby(["candidate", "repeat"]).n_fit.max() if "n_fit" in pf.columns else None
    recs, tails = [], []
    for group, name in names:
        slot, mA, mB = build_one(df, name)
        A = get_A(slot, df.index[mA], y, lot)
        bv = B_view(slot, df.index[mB], y, lot, "benchmark")
        for i in range(config.N_REPEATS_BENCHMARK):
            a_train, _ = repeat_split(A["A_dev"], y, lot, i)
            n_star, a_match, b_match = matched(a_train, bv["B_tr"], i)
            a_fit, _ = es_split(a_match, y, lot, i)
            b_fit, _ = es_split(b_match, y, lot, i)
            base = dict(group=group, candidate=name, dom=cm.short(name), repeat=i, n_star=n_star,
                        pool_lots_A=int(lot.loc[a_train].nunique()), pool_lots_B=int(lot.loc[bv["B_tr"]].nunique()))
            for rows_kind, (ra, rb) in (("match", (a_match, b_match)), ("es_fit", (a_fit, b_fit))):
                r = dict(base, rows=rows_kind)
                for side, rr in (("A", ra), ("B", rb)):
                    r.update({f"{k}_{side}": v for k, v in lot_stats(rr, lot).items()})
                r.update(shared(ra, rb, lot))
                recs.append(r)
        if group == "challenge":
            tails.append(tail_lots(name, A, bv, y, lot))
        log(f"[lot_info] {group} {cm.short(name)} done")
    d = pd.DataFrame(recs)
    if nstar_rec is not None:   # check of the recorded n_fit against the rebuilt n* (number of ID_match training rows of a model without early stopping = n*)
        m = d[(d.group == "challenge") & (d.rows == "match")].set_index(["candidate", "repeat"]).n_star
        diff = (m - nstar_rec.reindex(m.index)).abs().max()
        log(f"[lot_info] maximum difference between the rebuilt n* and n_fit of per_fit: {diff}")
        if diff != 0:
            raise SystemExit("the rebuilt n* differs from the record")
    num = [c for c in d.columns if c not in ("group", "candidate", "dom", "repeat", "rows")]
    summ = d.groupby(["group", "candidate", "dom", "rows"])[num].agg(["mean", "min", "max"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ = summ.reset_index()
    summ["lot_ratio_B_over_A_mean"] = d.assign(r=d.n_lots_B / d.n_lots_A).groupby(["group", "candidate", "dom", "rows"]).r.mean().to_numpy()
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, "lot_info_reps.csv"), index=False)
    summ.to_csv(os.path.join(out, "lot_info_domain.csv"), index=False)
    tl = pd.DataFrame(tails)
    tl.to_csv(os.path.join(out, "tail_lots.csv"), index=False)
    # rank correlation of the lot-count ratio (B_match / A_match) and the RMSE target-reference gap (median over 22 models) over the controls: descriptive statistic
    cvr = cv[cv.metric == "rmse"].set_index("candidate").tr
    cm_ = summ[(summ.group == "control") & (summ.rows == "match")].set_index("candidate")
    rho = float(pd.Series(cm_.lot_ratio_B_over_A_mean).corr(cvr.reindex(cm_.index), method="spearman"))
    log(f"[lot_info] 10 controls: Spearman between the lot-count ratio (B/A) and the RMSE target-ref gap {rho:+.2f}")
    print("\nlot composition of the tail rows (top 1%) (within the lot-disjoint rows):\n", tl.drop(columns="candidate").to_string(index=False))
    pd.set_option("display.width", 250)
    cols = ["n_star_mean", "n_lots_A_mean", "n_lots_B_mean", "lot_ratio_B_over_A_mean", "rows_per_lot_med_A_mean", "rows_per_lot_med_B_mean",
            "rows_per_lot_max_A_max", "rows_per_lot_max_B_max", "max_lot_share_A_max", "max_lot_share_B_max", "eff_lots_A_mean", "eff_lots_B_mean",
            "n_shared_lots_mean", "shared_row_share_A_mean", "shared_row_share_B_mean"]
    for rk in ("match", "es_fit"):
        print(f"\n[{rk}] repetition mean (maximum = maximum over repetitions):\n", summ[summ.rows == rk].set_index(["group", "dom"])[cols].round(3).to_string())
    log("[lot_info] done")


if __name__ == "__main__":
    main()
