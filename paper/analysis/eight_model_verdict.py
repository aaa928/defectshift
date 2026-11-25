"""Supplementary checks (reads only saved results and predictions; no new training, CPU only). Three parts:

(1) q decomposition: the log of the scale-normalized contrast = log q + log(1 + target-reference gap) (holds per fit, cell = repetition mean of the logs).
    Input lot_disjoint/disjoint_cells.csv (variant = disjoint; log_q and log_tr of a cell are the means of the 5 repetition logs). Per domain: the median over the 22 models of
    log q and log(1 + g_TR) (converted to %: 100 (exp - 1)), the number of models in which the two terms have the same sign, and the median of the share of log q among the models with the same sign.
    The share is defined only if the two terms have the same sign (if they cancel, it exceeds 100 % or becomes negative and is hard to interpret).
(2) Canonical verdict of the 8 models used for the redraws (M8, the same as in scripts/extra/run_redrawn_target_splits.py): lot-disjoint target rows, the 8-model median of the 5-repetition cells (scale_normalized_bootstrap/sn_cells.csv,
    target point estimate) is compared with the 10 controls computed with the same 8 models (k for RMSE and MAE separately and whether both are exceeded).
(3) Verdict of the 9 models for the uniform-preprocessing variant: only the 9 models that receive the released scale as it is, the first 3 repetitions. The challenge values are
    sensitivity_lot_disjoint/pipeline_disjoint_cells.csv (arm = uniform; the same cell as the variant tr_* and the main result tr_*_main of the same repetitions).
    The variant was not run for the controls, so they are computed from the predictions of the main pipeline (9 models, first 3 repetitions, all of B_te) and the verdict of the variant is conditional on this baseline.
    q_lot_ratio.csv also keeps the rank correlation of the domain q (RMSE) and the ratio of the numbers of lots of the matched samples (7 domains, descriptive statistic).
Output: results/analysis/<run-id>/{q_share.csv, q_lot_ratio.csv, m8_verdict.csv, uniform9_verdict.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/eight_model_verdict.py --run-id eight_model_verdict
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as cm
from defectshift import config
from defectshift.candidates import load_data
from defectshift.utils import log
from defectshift.directions import same_direction

M8 = ("Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL")
NINE = ("RealMLP", "TabM", "TabNet", "TabPFN", "TabICL", "TabFM", "Kumo", "TabDPT", "EXAONE")
N_REP_PIPE = 3


def q_share(rf):
    c = pd.read_csv(os.path.join(rf, "lot_disjoint", "disjoint_cells.csv"))
    c = c[c.variant == "disjoint"].copy()
    c["dom"] = c.candidate.map(cm.short)
    gap = float((c.log_idood - c.log_s - (c.log_q + c.log_tr)).abs().max())
    if gap > 1e-9:
        raise SystemExit(f"log(1 + normalized contrast) = log q + log(1 + g_TR) error {gap}")
    rows = []
    for (met, dom), g in c.groupby(["metric", "dom"]):
        same = same_direction(g.log_q, g.log_tr)
        sh = (g.log_q / (g.log_q + g.log_tr))[same]
        rows.append(dict(metric=met, dom=dom, n_models=len(g), q_pct_med=100 * (np.exp(g.log_q.median()) - 1),
                         tr_pct_med=100 * (np.exp(g.log_tr.median()) - 1), sn_pct_med=100 * (np.exp((g.log_q + g.log_tr).median()) - 1),
                         n_same_sign=int(same.sum()), n_q_pos=int((g.log_q > 0).sum()), n_tr_pos=int((g.log_tr > 0).sum()),
                         q_share_med_same=float(sh.median()) if len(sh) else np.nan,
                         q_share_min_same=float(sh.min()) if len(sh) else np.nan, q_share_max_same=float(sh.max()) if len(sh) else np.nan))
    return pd.DataFrame(rows)


def verdict(dom_vals, ctrl_vals):
    """dom_vals: {dom: (rmse, mae)}, ctrl_vals: DataFrame(candidate, rmse, mae). Returns k and the verdict per domain."""
    rows = []
    for dom, (r, m) in dom_vals.items():
        kr, km = int((r > ctrl_vals.rmse).sum()), int((m > ctrl_vals.mae).sum())
        n = len(ctrl_vals)
        rows.append(dict(dom=dom, rmse=r, mae=m, k_rmse=kr, k_mae=km, ctrl_max_rmse=float(ctrl_vals.rmse.max()),
                         ctrl_max_mae=float(ctrl_vals.mae.max()), exceed_both=(kr == n) and (km == n)))
    return pd.DataFrame(rows)


def m8_verdict(rf):
    C = pd.read_csv(os.path.join(rf, "scale_normalized_bootstrap", "sn_cells.csv"))
    C = C[(C.scheme == "target") & C.model.isin(M8)]
    if C.model.nunique() != len(M8):
        raise SystemExit("sn_cells does not contain all of M8")
    v = C.pivot_table(index=["group", "dom"], columns="metric", values="tr", aggfunc="median").reset_index()
    ch = {r.dom: (r.rmse, r.mae) for r in v[v.group == "challenge"].itertuples()}
    return verdict(ch, v[v.group == "control"].rename(columns={"dom": "candidate"}))


def uniform9_verdict(rf, a):
    p = pd.read_csv(os.path.join(rf, "sensitivity_lot_disjoint", "pipeline_disjoint_cells.csv"))
    p = p[(p.arm == "uniform") & p.model.isin(NINE)]
    if p.model.nunique() != len(NINE) or p.candidate.nunique() != 7:
        raise SystemExit("the uniform arm does not contain all 9 models x 7 domains")
    p["dom"] = p.candidate.map(cm.short)
    var = p.groupby("dom")[["tr_rmse", "tr_mae"]].median()
    main = p.groupby("dom")[["tr_rmse_main", "tr_mae_main"]].median()
    df, y = load_data()
    lot = df[config.LOT_COL]
    ctrl, dirs = cm.load_controls(a.ctrl_run_ids, list(NINE))
    cv = []
    for name, g in ctrl.groupby("candidate", sort=False):
        d = cm.load_domain("control", name, g, df, y, lot, list(NINE), dirs)
        cells = []
        for m in NINE:
            pairs = d["preds"][m][:N_REP_PIPE]
            r = [100 * (np.sqrt(np.mean((ps - d["yv"]) ** 2)) / np.sqrt(np.mean((pr - d["yv"]) ** 2)) - 1) for ps, pr in pairs]
            e = [100 * (np.mean(np.abs(ps - d["yv"])) / np.mean(np.abs(pr - d["yv"])) - 1) for ps, pr in pairs]
            cells.append((np.mean(r), np.mean(e)))
        cells = np.array(cells)
        cv.append(dict(candidate=name, rmse=float(np.median(cells[:, 0])), mae=float(np.median(cells[:, 1]))))
    cv = pd.DataFrame(cv)
    a1 = verdict({k: (r.tr_rmse_main, r.tr_mae_main) for k, r in main.iterrows()}, cv).assign(pipeline="main")
    a2 = verdict({k: (r.tr_rmse, r.tr_mae) for k, r in var.iterrows()}, cv).assign(pipeline="uniform")
    return pd.concat([a1, a2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ctrl-run-ids", default=cm.CTRL_RUNS)
    a = ap.parse_args()
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 250)
    q = q_share(rf)
    q.to_csv(os.path.join(out, "q_share.csv"), index=False)
    print(q.round(2).to_string(index=False))
    # rank correlation of q of the domain (q_med of the outcome-scale table, RMSE) and the lot-count ratio of the matched samples (B_match / A_match, repetition mean) (7 domains, descriptive statistic)
    dd = pd.read_csv(os.path.join(rf, "lot_disjoint", "disjoint_domain.csv"))
    dd = dd[(dd.variant == "disjoint") & (dd.metric == "rmse")].assign(dom=lambda x: x.candidate.map(cm.short)).set_index("dom")
    li = pd.read_csv(os.path.join(rf, "lot_composition", "lot_info_domain.csv"))
    li = li[(li.group == "challenge") & (li.rows == "match")].set_index("dom")
    j = dd[["q_med"]].join(li[["lot_ratio_B_over_A_mean"]], how="inner")
    if len(j) != 7:
        raise SystemExit(f"the number of domains for which q and the lot ratio were matched is not 7: {len(j)}")
    rho = float(spearmanr(j.q_med, j.lot_ratio_B_over_A_mean)[0])
    j.assign(spearman=rho).to_csv(os.path.join(out, "q_lot_ratio.csv"))
    print(f"\nSpearman of q (RMSE) and the lot ratio (7 domains): {rho:.3f}")
    m8 = m8_verdict(rf)
    m8.to_csv(os.path.join(out, "m8_verdict.csv"), index=False)
    print("\nM8 canonical:\n", m8.round(2).to_string(index=False))
    u = uniform9_verdict(rf, a)
    u.to_csv(os.path.join(out, "uniform9_verdict.csv"), index=False)
    print("\n9 models, first 3 repetitions:\n", u.round(2).to_string(index=False))
    log("[eight_model_verdict] done")


if __name__ == "__main__":
    main()
