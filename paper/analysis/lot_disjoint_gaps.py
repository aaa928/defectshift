"""Recomputes the ID-OOD / target-reference comparison with lot-disjoint evaluation rows throughout. It reads only saved predictions and the source test errors of per_fit (no new training, CPU only).

Two sets of evaluation rows (R): all = all B_te rows (complementary analysis), disjoint = B_te rows without those whose lot also occurs in the source training pool (A_dev) (main analysis).
For model m, repetition i and metric Err (RMSE or MAE), with f_S = ID_match, f_T = OOD_match and f_S,full = ID_full:
  1 + ID-OOD gap        = Err_R(f_S) / Err_S(f_S)     (the denominator is the whole source test set A_heldout: rmse_id, mae_id of per_fit)
  1 + target-ref gap    = Err_R(f_S) / Err_R(f_T)
  d = Err_R(f_T) / Err_S(f_S)                         (the identity (1 + ID-OOD gap) = (1 + target-ref gap) x d is checked in every repetition)
  s (outcome-scale ratio) = Err_R(best constant prediction) / Err_S(best constant prediction)   standard-deviation ratio for RMSE, ratio of the mean absolute deviation from the median for MAE. Independent of the model
  q (error ratio without scale) = d / s               d = s x q holds exactly for every model, repetition and domain value (s is constant within a domain)
Aggregation is as before: percentage per repetition -> mean over 5 repetitions per model -> median over models is the domain value. Log terms are averaged over 5 repetitions per model.
Collapsed cells (the same rule as the diagnostics of the final benchmark): in (domain, model) the repetition-mean R^2 of ID_match or OOD_match (computed on the evaluation rows R) < -1.
Output (results/analysis/<run-id>/):
  disjoint_cells.csv   cells variant x metric x domain x model (ID-OOD, target-ref, d, f_S,full, s, q, log terms, R^2, collapsed flag)
  disjoint_domain.csv  domain values (median over models), whether the directions are opposite, whether the ID-OOD direction equals the direction of s
  disjoint_agree.csv   per-combination direction agreement (all, collapsed cells excluded, f_S,full, per domain, per family), correlations, the shares of the three terms in the variance of log(1 + ID-OOD gap)
  ctrl_values.csv      domain values of the 10 controls (median over 22 models): ID-OOD gap, target-ref gap
  ranks.csv            how many control values a challenge domain value exceeds (ID-OOD and target-ref separately), minimum and maximum of the controls
Usage: python paper/analysis/lot_disjoint_gaps.py --run-id lot_disjoint
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as cm
from defectshift import config
from defectshift.utils import read_per_fit
from defectshift.candidates import load_data
from defectshift.utils import log
from defectshift.directions import same_direction

VARIANTS = ("all", "disjoint")
COLLAPSE_R2 = -1.0
KEYS = ["candidate", "model", "repeat"]


def spread(y, met):
    """Error of the best constant prediction: standard deviation around the mean for RMSE, mean absolute deviation around the median for MAE."""
    return float(np.std(y)) if met == "rmse" else float(np.mean(np.abs(y - np.median(y))))


def err(e, met):
    return float(np.sqrt(np.mean(e ** 2))) if met == "rmse" else float(np.mean(np.abs(e)))


def challenge_cells(a, df, y, lot):
    pfa = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    fam = pfa.drop_duplicates("model").set_index("model").family
    pf = cm.load_challenge(a.benchmark_run_id)
    models = sorted(pf.model.unique()) if a.models == "all" else a.models.split(",")
    pf = pf[pf.model.isin(models)]
    if a.only:
        pf = pf[pf.candidate.str.startswith(tuple(a.only.split(",")))]
    src = pfa[pfa.regime == "ID_match"].set_index(KEYS)
    srcf = pfa[pfa.regime == "ID_full"].set_index(KEYS)
    pathf = pfa[pfa.regime == "ID_full"].set_index(KEYS).pred_path
    recs, doms = [], []
    for name, g in pf.groupby("candidate", sort=False):
        d = cm.load_domain("challenge", name, g, df, y, lot, models)
        yS = y.loc[d["a"]["A_heldout"]].to_numpy()
        shared = np.isin(d["lots"], lot.loc[d["a"]["A_dev"]].unique())
        masks = dict(all=np.ones(len(d["yv"]), bool), disjoint=~shared)
        for m in models:
            reps = sorted(g[g.model == m].repeat.unique())
            if reps != list(range(config.N_REPEATS_BENCHMARK)):
                raise SystemExit(f"{name} {m}: the repetition numbers are not 0..{config.N_REPEATS_BENCHMARK - 1} {reps}")
        for var, mk in masks.items():
            yv = d["yv"][mk]
            sst = float(np.sum((yv - yv.mean()) ** 2))
            for met in ("rmse", "mae"):
                doms.append(dict(variant=var, metric=met, candidate=name, n_rows=int(mk.sum()), n_lots=int(len(np.unique(d["lots"][mk]))),
                                 s=spread(yv, met) / spread(yS, met)))
            for m in models:
                for j, (ps, pr) in enumerate(d["preds"][m]):
                    z = np.load(pathf.loc[(name, m, j)])
                    if not np.array_equal(z["ood_idx"], d["idx"]):
                        raise SystemExit(f"{name} {m} r{j}: the B_te row order of ID_full differs")
                    es, er, ef = ps[mk] - yv, pr[mk] - yv, z["ood_pred"][mk] - yv
                    r2s, r2r = 1 - float(np.sum(es ** 2)) / sst, 1 - float(np.sum(er ** 2)) / sst
                    for met in ("rmse", "mae"):
                        s_ood, r_ood, f_ood = err(es, met), err(er, met), err(ef, met)
                        s_id, f_id = src.loc[(name, m, j), f"{met}_id"], srcf.loc[(name, m, j), f"{met}_id"]
                        idood, tr, dd = s_ood / s_id, s_ood / r_ood, r_ood / s_id
                        if abs(idood - tr * dd) > 1e-9:
                            raise SystemExit(f"identity error {name} {m} r{j} {met}")
                        recs.append(dict(variant=var, metric=met, candidate=name, model=m, repeat=j, idood=100 * (idood - 1), tr=100 * (tr - 1),
                                         d=100 * (dd - 1), idood_full=100 * (f_ood / f_id - 1), log_idood=np.log(idood), log_tr=np.log(tr),
                                         log_d=np.log(dd), r2_S=r2s, r2_R=r2r))
        log(f"[disjoint] {cm.short(name)} done: rows {len(yv)} (all {len(d['yv'])}), shared rows {shared.mean():.1%}")
    rep = pd.DataFrame(recs)
    cells = rep.groupby(["variant", "metric", "candidate", "model"]).mean(numeric_only=True).drop(columns="repeat").reset_index()
    cells["family"] = cells.model.map(fam)
    dom = pd.DataFrame(doms)
    cells = cells.merge(dom[["variant", "metric", "candidate", "s"]], on=["variant", "metric", "candidate"])
    cells["log_s"] = np.log(cells.s)
    cells["log_q"] = cells.log_d - cells.log_s
    cells["q"] = 100 * ((1 + cells.d / 100) / cells.s - 1)
    cells["collapsed"] = (cells.r2_S < COLLAPSE_R2) | (cells.r2_R < COLLAPSE_R2)
    gap = float((cells.log_idood - (cells.log_s + cells.log_q + cells.log_tr)).abs().max())
    if gap > 1e-9:
        raise SystemExit(f"log(1 + ID-OOD gap) = log s + log q + log(1 + target-ref gap) error {gap}")
    return cells, dom, models


def domain_table(cells, dom):
    g = cells.groupby(["variant", "metric", "candidate"])
    t = g[["idood", "tr", "d", "idood_full", "q"]].median().add_suffix("_med").join(g.size().rename("n_models")).reset_index()
    t = t.merge(dom, on=["variant", "metric", "candidate"])
    t["s_pct"] = 100 * (t.s - 1)
    t["sign_opposite"] = ~same_direction(t.idood_med, t.tr_med)
    t["sign_opposite_full"] = ~same_direction(t.idood_full_med, t.tr_med)
    t["idood_dir_eq_s"] = same_direction(t.idood_med, t.s_pct)
    t["dom"] = t.candidate.map(cm.short)
    return t


def agree_table(cells, t):
    rows = []
    cells = cells.assign(agree=same_direction(cells.idood, cells.tr), agree_full=same_direction(cells.idood_full, cells.tr))
    for (var, met), c in cells.groupby(["variant", "metric"]):
        ok = c[~c.collapsed]
        row = dict(variant=var, metric=met, unit="all", key="all", n=len(c), agree=c.agree.mean(), agree_full=c.agree_full.mean(),
                   n_ok=len(ok), agree_ok=ok.agree.mean(), n_collapsed=int(c.collapsed.sum()))
        for tag, cc in (("", c), ("_ok", ok)):
            row[f"pearson{tag}"] = cc.idood.corr(cc.tr)
            row[f"spearman{tag}"] = cc.idood.corr(cc.tr, method="spearman")
            L = cc.log_idood
            for term in ("log_s", "log_q", "log_tr"):
                row[f"share_{term}{tag}"] = float(np.cov(cc[term], L, ddof=0)[0, 1] / np.var(L))
        tt = t[(t.variant == var) & (t.metric == met)]
        row.update(n_opposite=int(tt.sign_opposite.sum()), opposite=",".join(tt[tt.sign_opposite].dom), n_opposite_full=int(tt.sign_opposite_full.sum()),
                   n_dir_eq_s=int(tt.idood_dir_eq_s.sum()), dir_eq_s=",".join(tt[tt.idood_dir_eq_s].dom))
        rows.append(row)
        for by in ("candidate", "family"):
            for k, cc in c.groupby(by):
                rows.append(dict(variant=var, metric=met, unit=by, key=k, n=len(cc), agree=cc.agree.mean(), agree_full=cc.agree_full.mean()))
    return pd.DataFrame(rows)


def control_values(a, models, df, y):
    """Domain values of the controls (median over models): from the B_te and A_heldout errors of ID_match (source model) and OOD_match (reference) in rows.csv."""
    ctrl, _ = cm.load_controls(a.ctrl_run_ids, models)
    ctrl = ctrl.drop_duplicates(["candidate", "model", "repeat", "regime"])
    s = ctrl[ctrl.regime == cm.SRC].set_index(KEYS)
    r = ctrl[ctrl.regime == cm.REF].set_index(KEYS).reindex(s.index)
    out = []
    for met in ("rmse", "mae"):
        idood = 100 * (s[f"{met}_ood"] / s[f"{met}_id"] - 1)
        tr = 100 * (s[f"{met}_ood"] / r[f"{met}_ood"] - 1)
        c = pd.DataFrame(dict(idood=idood, tr=tr)).groupby(["candidate", "model"]).mean()
        v = c.groupby("candidate").median()
        v["metric"] = met
        out.append(v.reset_index())
    return pd.concat(out)


def rank_table(t, cv):
    rows = []
    for _, r in t[t.variant == "disjoint"].iterrows():
        c = cv[cv.metric == r.metric]
        for q, val in (("idood", r.idood_med), ("tr", r.tr_med)):
            x = c[q]
            rows.append(dict(metric=r.metric, dom=r.dom, quantity=q, value=val, ctrl_min=x.min(), ctrl_max=x.max(), n_ctrl=len(x),
                             k_above=int((val > x).sum()), k_below=int((val < x).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--ctrl-run-ids", default=cm.CTRL_RUNS)
    ap.add_argument("--models", default="all", help="all, or a comma-separated list of models (for smoke runs)")
    ap.add_argument("--only", default=None, help="first part of the challenge-domain names (comma-separated), for smoke runs")
    ap.add_argument("--check-run-id", default="decomposition_check", help="existing decomposition run against which the all-rows results are checked (skipped if absent)")
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    cells, dom, models = challenge_cells(a, df, y, lot)
    cells.to_csv(os.path.join(out, "disjoint_cells.csv"), index=False)
    chk = os.path.join(config.RESULTS_DIR, "analysis", a.check_run_id, "decomp_cells.csv")
    if os.path.exists(chk):
        old = pd.read_csv(chk)
        m = cells[cells.variant == "all"].merge(old, on=["metric", "candidate", "model"], suffixes=("", "_old"))
        if len(m) != len(cells[cells.variant == "all"]):
            raise SystemExit(f"the cells do not match the control run: {len(m)} versus {(cells.variant == 'all').sum()}")
        diff = max(float((m[c] - m[f"{c}_old"]).abs().max()) for c in ("idood", "tr", "d", "idood_full"))
        log(f"[disjoint] whether the all-rows cells equal the existing decomposition ({a.check_run_id}): maximum difference {diff:.2e} ({len(m)} cells)")
        if diff > 1e-6:
            raise SystemExit("the all-rows results differ from the existing decomposition")
    t = domain_table(cells, dom)
    t.drop(columns="dom").to_csv(os.path.join(out, "disjoint_domain.csv"), index=False)
    ag = agree_table(cells, t)
    ag.to_csv(os.path.join(out, "disjoint_agree.csv"), index=False)
    cv = control_values(a, models, df, y)
    cv.to_csv(os.path.join(out, "ctrl_values.csv"), index=False)
    rk = rank_table(t, cv)
    rk.to_csv(os.path.join(out, "ranks.csv"), index=False)
    pd.set_option("display.width", 220)
    for var in VARIANTS:
        for met in ("rmse", "mae"):
            x = t[(t.variant == var) & (t.metric == met)].set_index("dom")
            print(f"\n[{var} / {met}] domain values (%, median over models): ID-OOD, target-ref, d, s, q, f_S,full ID-OOD | opposite direction, ID-OOD direction = s direction")
            print(x[["n_rows", "idood_med", "tr_med", "d_med", "s_pct", "q_med", "idood_full_med", "sign_opposite", "idood_dir_eq_s"]].round(1).to_string())
    top = ag[ag.unit == "all"].set_index(["variant", "metric"])
    print("\nper-combination direction agreement, correlations and variance shares:\n", top[["n", "agree", "agree_full", "agree_ok", "n_collapsed", "pearson", "pearson_ok", "spearman",
                                                   "share_log_s", "share_log_q", "share_log_tr"]].round(3).to_string())
    print("\ndomains with opposite directions:\n", top[["n_opposite", "opposite", "n_opposite_full", "n_dir_eq_s", "dir_eq_s"]].to_string())
    print("\nrank against the 10 controls (lot-disjoint rows, k_above = number of controls whose domain value is larger):\n", rk.round(1).to_string(index=False))
    log("[disjoint] done")


if __name__ == "__main__":
    main()
