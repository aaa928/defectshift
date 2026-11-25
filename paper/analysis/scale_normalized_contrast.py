"""Direction comparison of the outcome-scale-normalized ID-OOD contrast and the target-reference gap.
It reads only saved results (no new training or prediction, CPU only).

Definitions (model m, repetition i, metric Err = RMSE or MAE; s = outcome-scale ratio, constant within a domain, the same as in lot_disjoint_gaps.py)
  1 + g_adj = (1 + ID-OOD gap) / s = q x (1 + target-reference gap)      (the identity 1 + ID-OOD = (1 + g_TR) x s x q divided by s)
  Because s is constant within a domain, g_adj is a first-order transformation of the ID-OOD gap. Whatever the order of taking the repetition mean and the model median,
  cell value = (1 + repetition-mean ID-OOD) / s - 1 and domain value = (1 + median of the domain ID-OOD) / s - 1 hold exactly (checked below).
  For RMSE, g_adj carries the same information as the R^2 comparison: (1 - R^2_T) / (1 - R^2_S) = (1 + g_adj)^2 (when both R^2 are computed with the mean and variance of their own test set).
  Therefore sign(R^2_S - R^2_T) = sign(g_adj) in every repetition, which is checked with the per-repetition R^2 of the lot-disjoint rows (output of lot_disjoint_accuracy.py).
Input: results/analysis/lot_disjoint/{disjoint_cells.csv, disjoint_domain.csv},
      results/analysis/lot_disjoint_accuracy/full_disjoint.csv (per-repetition RMSE and R^2 on the lot-disjoint rows), per_fit.csv of the benchmark (RMSE and R^2 of the source test set)
Output: results/analysis/<run-id>/{scale_adj_cells.csv, scale_adj_domain.csv, scale_adj_agree.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/scale_normalized_contrast.py --run-id scale_normalized_contrast
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as cm
from defectshift import config
from defectshift.utils import read_per_fit
from defectshift.utils import log
from defectshift.directions import same_direction


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--disjoint-run-id", default="lot_disjoint")
    ap.add_argument("--full-run-id", default="lot_disjoint_accuracy")
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    a = ap.parse_args()
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    cells = pd.read_csv(os.path.join(rf, a.disjoint_run_id, "disjoint_cells.csv"))
    dom = pd.read_csv(os.path.join(rf, a.disjoint_run_id, "disjoint_domain.csv"))
    if len(cells) != 2 * 2 * 7 * 22:
        raise SystemExit(f"the number of cells {len(cells)} != 616 (variants 2 x metrics 2 x domains 7 x models 22)")

    # 1) cell and domain values
    cells["adj"] = 100 * ((1 + cells.idood / 100) / cells.s - 1)
    alt = 100 * ((1 + cells.d / 100) / cells.s * (1 + cells.tr / 100) - 1)   # q(1 + g_TR) formed by multiplying the repetition means: an approximation, for reference
    cells["adj_product_of_means"] = alt
    keys = ["variant", "metric", "candidate"]
    g = cells.groupby(keys)
    t = g.adj.median().rename("adj_med").reset_index().merge(dom[keys + ["idood_med", "tr_med", "s", "s_pct", "q_med"]], on=keys)
    chk = 100 * ((1 + t.idood_med / 100) / t.s - 1)
    if float((chk - t.adj_med).abs().max()) > 1e-9:
        raise SystemExit("the domain value differs from (1 + ID-OOD median)/s - 1")
    t["dom"] = t.candidate.map(cm.short)
    t["adj_opposite_tr"] = ~same_direction(t.adj_med, t.tr_med)
    t["idood_opposite_tr"] = ~same_direction(t.idood_med, t.tr_med)

    # 2) check of the R^2 equivalence (lot-disjoint rows, per repetition)
    fd = pd.read_csv(os.path.join(rf, a.full_run_id, "full_disjoint.csv"))
    fd = fd[fd.regime == "ID_match"]
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    pf = pf[pf.regime == "ID_match"][["candidate", "model", "repeat", "rmse_id", "r2_id"]]
    rep = fd.merge(pf, on=["candidate", "model", "repeat"], validate="one_to_one")
    s_r = dom[(dom.variant == "disjoint") & (dom.metric == "rmse")].set_index("candidate").s
    rep["s"] = rep.candidate.map(s_r)
    rep["adj_rep"] = 100 * ((rep.rmse / rep.rmse_id) / rep.s - 1)
    lhs = (1 - rep.r2) / (1 - rep.r2_id)
    rhs = (1 + rep.adj_rep / 100) ** 2
    rel = float(((lhs - rhs) / rhs).abs().max())
    log(f"[scale_adj] (1 - R^2_T)/(1 - R^2_S) = (1 + g_adj)^2, maximum relative error {rel:.2e} ({len(rep)} repetition fits)")
    if rel > 1e-6:
        raise SystemExit("the R^2 equivalence does not hold")
    if (~same_direction(rep.r2_id - rep.r2, rep.adj_rep)).any():
        raise SystemExit("there are fits in which sign(R^2_S - R^2_T) and sign(g_adj) differ per repetition")
    rc = rep.groupby(["candidate", "model"]).agg(dr2=("r2_id", "mean"), r2t=("r2", "mean")).reset_index()
    rc["dr2"] = rc.dr2 - rc.r2t                                 # repetition-mean R^2_S - repetition-mean R^2_T
    cr = cells[(cells.variant == "disjoint") & (cells.metric == "rmse")].merge(rc, on=["candidate", "model"], validate="one_to_one")
    n_r2_mismatch = int((~same_direction(cr.dr2, cr.adj)).sum())

    # 3) per-cell direction agreement
    rows = []
    for (var, met), c in cells.groupby(["variant", "metric"]):
        ok = c[~c.collapsed]
        tt = t[(t.variant == var) & (t.metric == met)]
        row = dict(variant=var, metric=met, unit="all", key="all", n=len(c),
                   agree_adj=float(same_direction(c.adj, c.tr).mean()), agree_idood=float(same_direction(c.idood, c.tr).mean()),
                   n_ok=len(ok), agree_adj_ok=float(same_direction(ok.adj, ok.tr).mean()),
                   n_adj_pos=int((c.adj > 0).sum()), n_tr_pos=int((c.tr > 0).sum()),
                   n_dom_opposite_adj=int(tt.adj_opposite_tr.sum()), dom_opposite_adj=",".join(tt[tt.adj_opposite_tr].dom),
                   n_dom_opposite_idood=int(tt.idood_opposite_tr.sum()), dom_opposite_idood=",".join(tt[tt.idood_opposite_tr].dom),
                   max_abs_product_of_means_minus_adj=float((c.adj_product_of_means - c.adj).abs().max()))
        if var == "disjoint" and met == "rmse":
            row["n_cells_sign_mean_dR2_ne_adj"] = n_r2_mismatch
        rows.append(row)
        for k, cc in c.groupby("candidate"):
            rows.append(dict(variant=var, metric=met, unit="candidate", key=cm.short(k), n=len(cc),
                             agree_adj=float(same_direction(cc.adj, cc.tr).mean()), agree_idood=float(same_direction(cc.idood, cc.tr).mean()),
                             n_adj_pos=int((cc.adj > 0).sum()), n_tr_pos=int((cc.tr > 0).sum())))
    ag = pd.DataFrame(rows)

    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    cells.to_csv(os.path.join(out, "scale_adj_cells.csv"), index=False)
    t.drop(columns="dom").to_csv(os.path.join(out, "scale_adj_domain.csv"), index=False)
    ag.to_csv(os.path.join(out, "scale_adj_agree.csv"), index=False)
    pd.set_option("display.width", 220)
    for var in ("disjoint", "all"):
        for met in ("rmse", "mae"):
            x = t[(t.variant == var) & (t.metric == met)].set_index("dom")
            print(f"\n[{var} / {met}] domain values (%): ID-OOD, s, outcome-scale-normalized ID-OOD (g_adj), target-ref, q | opposite direction (g_adj versus g_TR, ID-OOD versus g_TR)")
            print(x[["idood_med", "s_pct", "adj_med", "tr_med", "q_med", "adj_opposite_tr", "idood_opposite_tr"]].round(1).to_string())
    print("\nper cell:\n", ag[ag.unit == "all"].drop(columns=["unit", "key"]).round(3).to_string(index=False))
    print("\ncell agreement per domain (disjoint):\n", ag[(ag.unit == "candidate") & (ag.variant == "disjoint")][["metric", "key", "agree_adj", "agree_idood", "n_adj_pos", "n_tr_pos"]].round(2).to_string(index=False))
    log(f"[scale_adj] cells whose sign of the repetition-mean delta R^2 differs from the sign of g_adj (disjoint, RMSE): {n_r2_mismatch}")
    log("[scale_adj] done")


if __name__ == "__main__":
    main()
