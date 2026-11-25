"""Three evaluation criteria and the test-lot bootstrap are computed on the same target evaluation rows (all rows / test rows of lots shared with the source excluded). It reads only saved predictions.

Two row configurations: all = all B_te rows, disjoint = B_te rows without those whose lot also occurs in the source training pool (A_dev) (lot-disjoint target evaluation).
The per-combination direction agreement (share of model x domain combinations in which the ID-OOD gap and the target-reference gap have the same sign) is also given for every row configuration.
Three criteria (all 22 models, source = ID_match, mean over 5 repetitions per model -> median over models, for RMSE and MAE each)
  ID-OOD gap  = Err_T(f_S; evaluation rows) / Err_S(f_S; all of A_heldout) - 1   (the denominator is the whole source test set; only the numerator follows the evaluation rows)
  target-reference gap = Err_T(f_S; evaluation rows) / Err_T(f_T; evaluation rows) - 1
  exceedance = how many of the domain values of the 10 randomized splits (controls) in the same configuration it exceeds (the same definition as k_both of control_rank.py; reads rank_domain.csv)
Bootstrap: lots of the evaluation rows are drawn with replacement (2,000 draws, the same resampling for source and reference), giving the 2.5/97.5 percentiles of the target-reference gap, separately for every row configuration.
The random numbers are seeded by the domain name (the same rows give the same interval whatever the row configuration, independent of the order of the domains).
Output: results/analysis/<run-id>/{criteria_table.csv, criteria_boot.csv}
Usage: python paper/analysis/evaluation_criteria.py --run-id criteria
"""
import argparse
import os
import zlib

import numpy as np
import pandas as pd

import common as cm
import repetition_uncertainty as unc
from defectshift import config
from defectshift.candidates import load_data
from defectshift.utils import log
from defectshift.directions import same_direction

VARIANTS = ("all", "disjoint")


def subset(preds, mask):
    return {m: [(ps[mask], pr[mask]) for ps, pr in pairs] for m, pairs in preds.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--rank-run-id", default="control_rank")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = cm.load_challenge(a.benchmark_run_id)
    models = sorted(pf.model.unique())
    rk = pd.read_csv(os.path.join(config.RESULTS_DIR, "analysis", a.rank_run_id, "rank_domain.csv")).set_index(["candidate", "variant"])
    rows, boots = [], []
    for name, g in pf.groupby("candidate", sort=False):
        d = cm.load_domain("challenge", name, g, df, y, lot, models)
        shared = np.isin(d["lots"], lot.loc[d["a"]["A_dev"]].unique())
        masks = dict(all=np.ones(len(d["yv"]), bool), disjoint=~shared)
        # error on the source test set (A_heldout): ID_match rows of per_fit (model, repetition)
        src = g[g.regime == cm.SRC].set_index(["model", "repeat"])
        for var in VARIANTS:
            mk = masks[var]
            cells = {"rmse": [], "mae": [], "idood_rmse": [], "idood_mae": []}
            for m in models:
                tr, io = {"rmse": [], "mae": []}, {"rmse": [], "mae": []}
                for j, (ps, pr) in enumerate(d["preds"][m]):
                    es, er = ps[mk] - d["yv"][mk], pr[mk] - d["yv"][mk]
                    r_s, r_r = np.sqrt(np.mean(es ** 2)), np.sqrt(np.mean(er ** 2))
                    m_s, m_r = np.mean(np.abs(es)), np.mean(np.abs(er))
                    tr["rmse"].append(100 * (r_s / r_r - 1))
                    tr["mae"].append(100 * (m_s / m_r - 1))
                    io["rmse"].append(100 * (r_s / src.loc[(m, j), "rmse_id"] - 1))
                    io["mae"].append(100 * (m_s / src.loc[(m, j), "mae_id"] - 1))
                for k in ("rmse", "mae"):
                    cells[k].append(np.mean(tr[k]))
                    cells[f"idood_{k}"].append(np.mean(io[k]))
            row = dict(candidate=name, variant=var, n_rows=int(mk.sum()), n_lots=int(len(np.unique(d["lots"][mk]))),
                       share_dropped=float(1 - mk.mean()), **{f"{k}_med": float(np.median(v)) for k, v in cells.items()})
            row.update(agree_rmse=float(np.mean(same_direction(cells["idood_rmse"], cells["rmse"]))),
                       agree_mae=float(np.mean(same_direction(cells["idood_mae"], cells["mae"]))), n_models=len(models))
            row.update(k_rmse=int(rk.loc[(name, "not_shared" if var == "disjoint" else "all"), "k_rmse"]),
                       k_mae=int(rk.loc[(name, "not_shared" if var == "disjoint" else "all"), "k_mae"]),
                       k_both=int(rk.loc[(name, "not_shared" if var == "disjoint" else "all"), "k_both"]),
                       n_ctrl=int(rk.loc[(name, "not_shared" if var == "disjoint" else "all"), "n_ctrl"]))
            rows.append(row)
            rng = np.random.default_rng([unc.BOOT_SEED, zlib.crc32(cm.short(name).encode())])
            b = unc.boot(d["yv"][mk], d["lots"][mk], subset(d["preds"], mk), models, a.n_boot, rng)
            boots.append(dict(candidate=name, variant=var, n_boot=a.n_boot, rmse_lo=float(np.percentile(b["d_rmse"], 2.5)),
                              rmse_hi=float(np.percentile(b["d_rmse"], 97.5)), mae_lo=float(np.percentile(b["d_mae"], 2.5)),
                              mae_hi=float(np.percentile(b["d_mae"], 97.5))))
        log(f"[criteria] {cm.short(name)} done (shared rows {shared.mean():.1%})")
    t = pd.DataFrame(rows)
    t["dom"] = t.candidate.map(cm.short)
    t["idood_pos_rmse"] = t.idood_rmse_med > 0
    t["tr_pos_rmse"] = t.rmse_med > 0
    t["exceed_all10"] = t.k_both == t.n_ctrl
    t.drop(columns="dom").to_csv(os.path.join(out, "criteria_table.csv"), index=False)
    bt = pd.DataFrame(boots)
    bt.to_csv(os.path.join(out, "criteria_boot.csv"), index=False)
    for var in VARIANTS:
        pooled = t[t.variant == var][["agree_rmse", "agree_mae"]].mean()
        print(f"\n[{var}] per-combination direction agreement (mean over 22 models x 7 domains): RMSE {pooled.agree_rmse:.3f}, MAE {pooled.agree_mae:.3f}")
        s = t[t.variant == var].set_index("dom")
        print(f"\n[{var}] evaluation-row configuration: three criteria (ID-OOD>0 and target-ref>0 on RMSE / values only for MAE)")
        print(s[["n_rows", "idood_rmse_med", "rmse_med", "idood_mae_med", "mae_med", "k_rmse", "k_mae", "k_both", "n_ctrl", "idood_pos_rmse", "tr_pos_rmse", "exceed_all10"]].round(1).to_string())
        print(f"\n[{var}] bootstrap 95% (target-reference gap):")
        bb = bt[bt.variant == var].assign(dom=lambda x: x.candidate.map(cm.short)).set_index("dom")
        print(bb[["rmse_lo", "rmse_hi", "mae_lo", "mae_hi"]].round(1).to_string())
    log("[criteria] done")


if __name__ == "__main__":
    main()
