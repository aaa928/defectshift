"""Rank report against the randomized lot splits (controls) and the verdict per model. It reads only the cell values of the existing robustness analysis (no new training or prediction).

Input: robust_cells.csv made by verdict_robustness.py (cells model x domain x test-row configuration, 22 models, 10 controls)
(1) Domain rank: for every test-row configuration, how many of the N control domain values the challenge domain value (median over models) exceeds (k / N), for RMSE and MAE separately,
    and the number of controls exceeded on both metrics. k = N is the same as the earlier "exceeds the control maximum"
(2) Verdict per model: for every model, how many of the N control cells of the same model its own cell value exceeds (k / N). The share of the 22 models with k = N (on both metrics)
    is given per domain and configuration. Because it does not go through the domain median, the dependence on the model composition is reduced
Main configuration = not_shared (test rows of shared lots excluded; the controls have no such rows by construction). The other configurations are given as well. Observed ranks, not a test.
Output: results/analysis/<run-id>/{rank_domain.csv, rank_model.csv, rank_model_share.csv}
Usage: python paper/analysis/control_rank.py --run-id control_rank
"""
import argparse
import json
import os

import pandas as pd

import common as cm
from defectshift import config
from defectshift.utils import log

PRIMARY = "not_shared"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--robust-run-id", default="robust")
    a = ap.parse_args()
    src = os.path.join(config.RESULTS_DIR, "analysis", a.robust_run_id)
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(src, "robust_meta.json")))
    c = pd.read_csv(os.path.join(src, "robust_cells.csv"))
    n_ctrl = c[c.group == "control"].candidate.nunique()
    if n_ctrl != meta["n_ctrl"] or c.model.nunique() != len(meta["models"]):
        raise SystemExit(f"the cell file and meta differ: controls {n_ctrl} versus {meta['n_ctrl']}, models {c.model.nunique()} versus {len(meta['models'])}")
    log(f"[rank] input {a.robust_run_id}: {c.model.nunique()} models, {n_ctrl} controls")
    dom = c.groupby(["group", "candidate", "variant"])[["rmse", "mae"]].median().reset_index()
    ch, ct = dom[dom.group == "challenge"], dom[dom.group == "control"]
    rows = []
    for _, x in ch.iterrows():
        cv = ct[ct.variant == x.variant]
        rows.append(dict(candidate=x.candidate, variant=x.variant, n_ctrl=len(cv), rmse=x.rmse, mae=x.mae,
                         k_rmse=int((cv.rmse < x.rmse).sum()), k_mae=int((cv.mae < x.mae).sum()),
                         k_both=int(((cv.rmse < x.rmse) & (cv.mae < x.mae)).sum())))
    rd = pd.DataFrame(rows)
    rd.to_csv(os.path.join(out, "rank_domain.csv"), index=False)
    cc, cx = c[c.group == "challenge"], c[c.group == "control"]
    mrows = []
    for (cand, var, m), x in cc.groupby(["candidate", "variant", "model"]):
        cv = cx[(cx.variant == var) & (cx.model == m)]
        r, e = float(x.rmse.iloc[0]), float(x.mae.iloc[0])
        mrows.append(dict(candidate=cand, variant=var, model=m, n_ctrl=len(cv), rmse=r, mae=e,
                          k_rmse=int((cv.rmse < r).sum()), k_mae=int((cv.mae < e).sum()),
                          k_both=int(((cv.rmse < r) & (cv.mae < e)).sum())))
    rm = pd.DataFrame(mrows)
    rm.to_csv(os.path.join(out, "rank_model.csv"), index=False)
    rm["all_rmse"] = rm.k_rmse == rm.n_ctrl
    rm["all_mae"] = rm.k_mae == rm.n_ctrl
    rm["all_both"] = rm.all_rmse & rm.all_mae
    share = rm.groupby(["candidate", "variant"])[["all_rmse", "all_mae", "all_both"]].mean().reset_index()
    share.to_csv(os.path.join(out, "rank_model_share.csv"), index=False)
    rd["dom"], share["dom"] = rd.candidate.map(cm.short), share.candidate.map(cm.short)
    order = [PRIMARY] + [v for v in rd.variant.unique() if v != PRIMARY]
    for k in ("k_rmse", "k_mae", "k_both"):
        print(f"\ndomain rank {k} (/ {n_ctrl}):\n", rd.pivot(index="dom", columns="variant", values=k)[order].to_string())
    print("\nverdict per model: share of models that exceed all their own controls on both metrics:\n",
          share.pivot(index="dom", columns="variant", values="all_both")[order].round(2).to_string())
    log("[rank] done")


if __name__ == "__main__":
    main()
