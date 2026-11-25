"""Selects, for every domain, the 10 controls whose lot composition is similar (a subset of the controls for the auxiliary sensitivity analysis).
It uses only split information (gaps and predictions are not read; CPU only). 50 controls = the original 10 (seeds 9001-9010) + the additional 40 (extra_controls.PLAN).

Matching variables (for each control and challenge domain, mean over 5 repetitions; the row rule is the same as in lot_composition.py)
  log n*                                    n* = min(|A_train|, |B_tr|) (splits.matched)
  log (number of lots of B_match / number of lots of A_match)
  log (number of target test lots evaluated)   challenge = lots of the lot-disjoint test rows (lots shared with A_dev excluded), controls = lots of B_te
Standardization: the three variables are standardized with the mean and standard deviation (ddof 0) of the 50 controls, and the same reference is used for all domains.
Selection: for every domain the 10 controls with the smallest standardized Euclidean distance (ties by control name). A control may be selected for several domains.
The existing results over all controls are not changed.
Check: n* and the lot ratio of the original 10 controls and the 7 challenge domains must equal lot_composition/lot_info_domain.csv.
Output: results/analysis/<run-id>/{lot_match_features.csv, lot_match_selection.csv, lot_match_quality.csv}
Usage: python paper/analysis/lot_matched_controls.py --run-id lot_match
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as cm
from defectshift import config, extra_controls
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data
from defectshift.splits import B_view, get_A, matched, repeat_split
from defectshift.utils import log

K = 10
VARS = ("log_n_star", "log_lot_ratio", "log_test_lots")


def features(group, name, df, y, lot):
    slot, mA, mB = build_one(df, name)
    A = get_A(slot, df.index[mA], y, lot)
    bv = B_view(slot, df.index[mB], y, lot, "benchmark")
    te_lots = lot.loc[np.asarray(bv["B_te"])]
    if group == "challenge":
        te_lots = te_lots[~te_lots.isin(lot.loc[A["A_dev"]].unique())]
    elif te_lots.isin(lot.loc[A["A_dev"]].unique()).any():
        raise SystemExit(f"{name}: B_te of the control contains lots that also occur in A_dev")
    ns, ratio = [], []
    for i in range(config.N_REPEATS_BENCHMARK):
        a_train, _ = repeat_split(A["A_dev"], y, lot, i)
        n_star, a_match, b_match = matched(a_train, bv["B_tr"], i)
        ns.append(n_star)
        ratio.append(lot.loc[b_match].nunique() / lot.loc[a_match].nunique())
    return dict(group=group, candidate=name, dom=cm.short(name), n_star=float(np.mean(ns)), lot_ratio=float(np.mean(ratio)),
                test_lots=int(te_lots.nunique()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    a = ap.parse_args()
    extra_controls.register()
    df, y = load_data()
    lot = df[config.LOT_COL]
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    names = [("challenge", n) for n in pf.candidate.unique()]
    seeds = list(range(9001, 9011)) + sorted(extra_controls.seed_table())
    names += [("control", f"placebo@{s}") for s in seeds]
    if sum(g == "challenge" for g, _ in names) != 7 or sum(g == "control" for g, _ in names) != 50:
        raise SystemExit("the number of domains is not 7 + 50")
    F = pd.DataFrame([features(g, n, df, y, lot) for g, n in names])
    log("[lot_match] matching variables computed")
    li = pd.read_csv(os.path.join(rf, "lot_composition", "lot_info_domain.csv"))
    li = li[li.rows == "match"].set_index("candidate")
    chk = F[F.candidate.isin(li.index)].set_index("candidate")
    if len(chk) != 17:
        raise SystemExit(f"the number of domains to compare with lot_info is not 17: {len(chk)}")
    d1 = float((chk.n_star - li.loc[chk.index, "n_star_mean"]).abs().max())
    d2 = float((chk.lot_ratio - li.loc[chk.index, "lot_ratio_B_over_A_mean"]).abs().max())
    if d1 > 1e-9 or d2 > 1e-9:
        raise SystemExit(f"differs from lot_info: n* {d1}, lot ratio {d2}")
    F["log_n_star"], F["log_lot_ratio"], F["log_test_lots"] = np.log(F.n_star), np.log(F.lot_ratio), np.log(F.test_lots)
    ctl = F[F.group == "control"]
    mu, sd = ctl[list(VARS)].mean(), ctl[list(VARS)].std(ddof=0)
    Z = (F[list(VARS)] - mu) / sd
    sel, qual = [], []
    for i, r in F[F.group == "challenge"].iterrows():
        dist = np.sqrt(((Z.loc[ctl.index] - Z.loc[i]) ** 2).sum(1))
        t = ctl.assign(distance=dist).sort_values(["distance", "candidate"]).head(K)
        for rank, (_, c) in enumerate(t.iterrows(), 1):
            sel.append(dict(dom=r.dom, rank=rank, control=c.candidate, distance=c.distance, n_star=c.n_star, lot_ratio=c.lot_ratio,
                            test_lots=c.test_lots))
        qual.append(dict(dom=r.dom, n_star=r.n_star, lot_ratio=r.lot_ratio, test_lots=r.test_lots, dist_mean=float(t.distance.mean()),
                         dist_max=float(t.distance.max()), sel_n_star_min=t.n_star.min(), sel_n_star_max=t.n_star.max(),
                         sel_lot_ratio_min=t.lot_ratio.min(), sel_lot_ratio_max=t.lot_ratio.max(), sel_test_lots_min=int(t.test_lots.min()),
                         sel_test_lots_max=int(t.test_lots.max()), n_original_ten=int(t.candidate.isin([f"placebo@{s}" for s in range(9001, 9011)]).sum())))
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    F.to_csv(os.path.join(out, "lot_match_features.csv"), index=False)
    pd.DataFrame(sel).to_csv(os.path.join(out, "lot_match_selection.csv"), index=False)
    Q = pd.DataFrame(qual)
    Q.to_csv(os.path.join(out, "lot_match_quality.csv"), index=False)
    pd.set_option("display.width", 250)
    print(F[["group", "dom", "n_star", "lot_ratio", "test_lots"]].round(3).to_string(index=False))
    print(Q.round(3).to_string(index=False))
    log("[lot_match] done")


if __name__ == "__main__":
    main()
