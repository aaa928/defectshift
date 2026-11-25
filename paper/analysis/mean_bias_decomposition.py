"""Decomposition of the mean bias (the cat_050 and x_time cases). It reads only saved predictions (no training).
Evaluation rows = lot-disjoint target test rows (B_te rows whose lot also occurs in A_dev, the source training pool, are excluded). For these two domains the number of shared rows is 0, so they equal all rows.

Definitions (computed per model and repetition, then averaged over the repetitions):
  b = mean of (prediction - truth) on the evaluation rows,  MSE = error variance (ddof 0) + b^2
  bias_ratio_S = b(source model) / mean defect rate of the evaluation rows
  share = (b_S^2 - b_R^2) / (MSE_S - MSE_R)   - defined only for models with MSE_S > MSE_R (computed from the repetition-averaged b^2 and MSE)
  b2_frac_S = b_S^2 / MSE_S,  b2_frac_R = b_R^2 / MSE_R
Domain value = median over the models. Output: results/analysis/<run-id>/{bias_cells.csv, bias_domain.csv}
Usage: python paper/analysis/mean_bias_decomposition.py --run-id bias [--only cat_050,x_time]
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as cm
from defectshift import config
from defectshift.candidates import load_data
from defectshift.utils import log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--only", default="cat_050,x_time", help="first part of the domain names (comma-separated)")
    a = ap.parse_args()
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = cm.load_challenge(a.benchmark_run_id)
    pf = pf[pf.candidate.str.startswith(tuple(a.only.split(",")))]
    models = sorted(pf.model.unique())
    cells, doms = [], []
    for name, g in pf.groupby("candidate", sort=False):
        d = cm.load_domain("challenge", name, g, df, y, lot, models)
        shared = np.isin(d["lots"], lot.loc[d["a"]["A_dev"]].unique())
        mk = ~shared
        yv = d["yv"][mk]
        yS = y.loc[d["a"]["A_heldout"]].to_numpy()
        for m in models:
            bs, br, ms, mr = [], [], [], []
            for ps, pr in d["preds"][m]:
                es, er = ps[mk] - yv, pr[mk] - yv
                bs.append(es.mean()); br.append(er.mean()); ms.append(np.mean(es ** 2)); mr.append(np.mean(er ** 2))
            b_s, b_r, mse_s, mse_r = np.mean(bs), np.mean(br), np.mean(ms), np.mean(mr)
            b2s, b2r = np.mean(np.square(bs)), np.mean(np.square(br))
            share = (b2s - b2r) / (mse_s - mse_r) if mse_s > mse_r else np.nan
            cells.append(dict(candidate=name, model=m, b_S=b_s, b_R=b_r, mse_S=mse_s, mse_R=mse_r, bias_ratio_S=b_s / yv.mean(),
                              share=share, b2_frac_S=b2s / mse_s, b2_frac_R=b2r / mse_r, source_worse=bool(mse_s > mse_r)))
        c = pd.DataFrame([r for r in cells if r["candidate"] == name])
        doms.append(dict(candidate=name, n_rows=int(mk.sum()), n_shared=int(shared.sum()), mean_y_T=float(yv.mean()), mean_y_S=float(yS.mean()),
                         target_mean_vs_source_pct=100 * (yv.mean() / yS.mean() - 1),
                         bias_ratio_S_med=float(c.bias_ratio_S.median()), n_source_worse=int(c.source_worse.sum()),
                         share_med=float(c.share.median()), b2_frac_S_med=float(c.b2_frac_S.median()), b2_frac_R_med=float(c.b2_frac_R.median())))
        log(f"[bias] {cm.short(name)}: rows {mk.sum()} (shared {shared.sum()}), median b/ybar {100 * c.bias_ratio_S.median():.1f}%, "
            f"median share {100 * c.share.median():.1f}% (models for which the source model is worse: {c.source_worse.sum()})")
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(cells).to_csv(os.path.join(out, "bias_cells.csv"), index=False)
    pd.DataFrame(doms).to_csv(os.path.join(out, "bias_domain.csv"), index=False)
    print(pd.DataFrame(doms).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
