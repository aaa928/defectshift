"""Size table of the additional controls (no training, CPU). For the additional randomized lot splits (extra_controls.PLAN) and the original 10 controls, only the splits are made
and, per domain, the size of the training pool n*, the number of target test rows and the number of test lots are obtained and compared with the challenge domain values. Predictions and performance are not read.

Size-matched split = n* and the number of target test rows are each within +-TOL (default 10 %) of the value of that domain (both conditions). The number of test lots is not matched; only the difference is recorded.
n* = min(|A_train|, |B_tr|) differs by repetition, so the mean over the 5 repetitions is used (the same method for the challenge values). The splits use the same functions as the benchmark
(candidates.build_one -> splits.split_A, split_B4, repeat_split, matched) but write no cache files.
Output: results/analysis/<run-id>/{control_sizes.csv, control_matching.csv, control_matching_summary.csv}
Usage: python paper/analysis/control_sizes.py --run-id control_sizes [--extra-seeds 9051:24,9052:24]
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as cm  # noqa: F401
from defectshift import config, extra_controls
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data
from defectshift.splits import matched, repeat_split, split_A, split_B4
from defectshift.utils import log

TOL = 0.10


def sizes_of(df, y, lot, name):
    """Size of one candidate: n* (mean of 5 repetitions), range of n*, number of rows and lots of B_te, number of rows of A and B."""
    slot, mA, mB = build_one(df, name)
    A, B = df.index[mA], df.index[mB]
    a = split_A(A, y, lot)
    b = split_B4(B, y, lot)
    ns = []
    for i in range(config.N_REPEATS_BENCHMARK):
        a_train, _ = repeat_split(a["A_dev"], y, lot, i)
        ns.append(matched(a_train, b["B_tr"], i)[0])
    return dict(candidate=name, n_A=len(A), n_B=len(B), n_star=float(np.mean(ns)), n_star_min=int(min(ns)), n_star_max=int(max(ns)),
                n_B_te=len(b["B_te"]), n_B_te_lots=int(lot.loc[b["B_te"]].nunique()), q_B=len(B) / len(df))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--extra-seeds", default=None, help="pairs seed:m (comma-separated). To check the size of candidates outside the plan when splits are added to domains with too few matches after the size table has been seen")
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    extra_controls.register()
    table = extra_controls.seed_table()
    if a.extra_seeds:
        for kv in a.extra_seeds.split(","):
            s, m = (int(x) for x in kv.split(":"))
            if s in table or s in config.PLACEBO_B_TARGET:
                raise SystemExit(f"seed {s} already exists")
            table[s] = ("extra", m)
            config.PLACEBO_B_TARGET[s] = int(round(m * extra_controls.N_ROWS / config.PLACEBO_N_FOLDS))
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", "benchmark_canonical", "report", "per_fit.csv"))
    pf = pf[pf.regime.isin(["ID_match", "OOD_match"])]
    ch = pf[pf.regime == "ID_match"].groupby("candidate").agg(n_star=("n_star", "mean")).reset_index()
    ch["domain"] = ch.candidate.str.split("<").str[0]
    te = []
    for c in ch.candidate:
        g = pf[pf.candidate == c]
        te.append(cm.load_domain("challenge", c, g, df, y, lot, [g.model.iloc[0]])["idx"])
    ch["n_B_te"] = [len(i) for i in te]
    ch["n_B_te_lots"] = [int(lot.loc[i].nunique()) for i in te]
    rows = [dict(group="old", **sizes_of(df, y, lot, f"placebo@{s}")) for s in range(9001, 9011)]
    rows += [dict(group=g, m=m, seed=s, **sizes_of(df, y, lot, f"placebo@{s}")) for s, (g, m) in sorted(table.items())]
    sz = pd.DataFrame(rows)
    sz["seed"] = sz.candidate.str.split("@").str[1].astype(int)
    sz.to_csv(os.path.join(out, "control_sizes.csv"), index=False)
    ch.to_csv(os.path.join(out, "challenge_sizes.csv"), index=False)
    mt = []
    for _, d in ch.iterrows():
        for _, c in sz.iterrows():
            rn, rt = abs(c.n_star - d.n_star) / d.n_star, abs(c.n_B_te - d.n_B_te) / d.n_B_te
            mt.append(dict(domain=d.domain, candidate=c.candidate, group=c.group, n_star_ctrl=c.n_star, n_star_dom=d.n_star, rel_n_star=rn,
                           n_B_te_ctrl=c.n_B_te, n_B_te_dom=d.n_B_te, rel_n_B_te=rt, lots_ctrl=c.n_B_te_lots, lots_dom=d.n_B_te_lots,
                           matched=bool(rn <= a.tol and rt <= a.tol), tol=a.tol))
    mt = pd.DataFrame(mt)
    mt.to_csv(os.path.join(out, "control_matching.csv"), index=False)
    s = mt.assign(new=mt.group != "old").groupby(["domain", "new"]).agg(n=("matched", "size"), matched=("matched", "sum")).reset_index()
    s["tol"] = a.tol
    s.to_csv(os.path.join(out, "control_matching_summary.csv"), index=False)
    p = s.pivot(index="domain", columns="new", values="matched").rename(columns={False: "among the original 10", True: "among the additional ones"})
    print(f"number of size-matched splits (n* and test rows each within +-{a.tol:.0%}):\n", p.to_string())
    print("\nsizes of the groups of the plan:\n", sz[sz.group != "old"].groupby("group")[["n_star", "n_B_te", "n_B_te_lots"]].agg(["min", "max"]).round(0).to_string())
    print("\nchallenge domain:\n", ch[["domain", "n_star", "n_B_te", "n_B_te_lots"]].round(0).to_string(index=False))
    log("[ctrl_sizes] done")


if __name__ == "__main__":
    main()
