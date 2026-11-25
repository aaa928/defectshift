"""The outcome-scale-normalized ID-OOD contrast and the target-reference gap are computed together on the same lot-bootstrap resamples, and the direction verdict is divided into three branches:
agree, disagree and direction not determined. The 7 challenge domains and the 10 randomized lot-split controls.
It reads only saved predictions (no new training, CPU only). The trained models and predictions are held fixed, and the uncertainty of training variation is not included.

Definitions (model m, repetition i, Err = RMSE or MAE, the same as in lot_disjoint_gaps.py and scale_normalized_contrast.py)
  ID-OOD gap      g_IO = Err_T(f_S) / Err_S(f_S) - 1    (Err_S = all rows of the source test set A_heldout)
  target-ref gap  g_TR = Err_T(f_S) / Err_T(f_T) - 1
  outcome-scale ratio  s = spread(y_T) / spread(y_S)   (RMSE: standard deviation around the mean (ddof 0), MAE: mean absolute deviation around the median)
  scale normalization  g_SN = (1 + g_IO) / s - 1       (cell = (1 + repetition-mean g_IO) / s - 1, domain = median over the 22 models)
  Evaluation rows: for the challenge domains the lot-disjoint target test rows (rows of lots shared with A_dev excluded), for the controls all B_te rows (all lot-disjoint).
Resampling (lots drawn with replacement, 2,000 draws, 2.5-97.5 percentiles). All models and repetitions within a domain use the same resampling.
  target    resamples only the target test lots. The random numbers follow the same rule as repetition_uncertainty.boot ([BOOT_SEED, crc32(domain)], multinomial), so
            the g_TR interval of the challenge domains must equal the existing criteria_boot.csv (criteria, disjoint) (the run stops otherwise).
            Err_S and spread(y_S) of the source side are held fixed.
  both      the same target resampling as above + independent resampling of the source test lots (A_heldout) ([SRC_SEED, crc32(domain)]). Err_S and spread(y_S) are
            also recomputed from the resampled rows. It is an auxiliary analysis that includes the composition variation of both test sets and is not mixed with the target version as an interval of the same meaning.
  On the resampled rows RMSE and MAE are recomputed with the row-level definition as it is (the lot values are not averaged). The median for MAE is the median of the resampled rows.
Three-branch direction verdict (for domain values and for cell values separately): if either of the two intervals contains 0 (lo <= 0 <= hi), the direction is not determined;
  otherwise agree if the signs are the same and disagree if they differ. Containing 0 does not mean there is no effect but that the direction is not determined under this resampling.
  Two comparisons: before normalization (g_IO) and after normalization (g_SN). p_agree = share of resamples in which the signs of the two values are the same (a descriptive quantity, not a probability).
The direction agreement of the point estimates (two branches) is also given: the share over all 154 challenge cells, and for the controls the median and range of the per-control share over 22 cells (the 220 cells are not pooled).
Outcome scale and mean: for every domain s and the mean ratio (mean(y_T) / mean(y_S)) are computed on the same evaluation rows (17 splits, Spearman as a descriptive statistic).
Checks: the values without resampling must equal the existing scale_adj_domain.csv (disjoint), disjoint_domain.csv (challenge) and ctrl_values.csv (controls).
      The stored Err_S (per_fit or *_id of the rows.csv of the controls) must equal the value recomputed from id_pred, and id_idx must equal A_heldout.
Output: results/analysis/<run-id>/{sn_domain.csv, sn_cells.csv, sn_agree.csv, sn_scale_mean.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/scale_normalized_bootstrap.py --run-id scale_normalized_bootstrap
"""
import argparse
import os
import zlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as cm
import repetition_uncertainty as unc
from defectshift import config
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data, slot_dir
from defectshift.utils import log
from defectshift.directions import same_direction

SRC_SEED = 20261108
COLLAPSE_R2 = -1.0
METS = ("rmse", "mae")


def wspread(yv, W, met):
    """Spread under the resampling weights W (n_boot, n_rows; integer counts). Equals np.std / mean absolute deviation around the median of the array in which every row is replicated W times."""
    n = W.sum(1)
    if met == "rmse":
        mu = W @ yv / n
        return np.sqrt((W * (yv[None, :] - mu[:, None]) ** 2).sum(1) / n)
    o = np.argsort(yv, kind="stable")
    ys, cum = yv[o], np.cumsum(W[:, o], axis=1)
    k1, k2 = (n - 1) // 2, n // 2   # the two middle positions of the replicated array (0-based). They coincide if n is odd
    v1 = ys[(cum > k1[:, None]).argmax(1)]
    v2 = ys[(cum > k2[:, None]).argmax(1)]
    med = (v1 + v2) / 2
    return (W * np.abs(yv[None, :] - med[:, None])).sum(1) / n


def lot_sums(W, codes, L, e):
    """Sum of squared errors and sum of absolute errors under the lot weights W (n_boot, L)."""
    return W @ np.bincount(codes, e ** 2, minlength=L), W @ np.bincount(codes, np.abs(e), minlength=L)


def load_src(group, name, g, models, dirs, slot, n_rep):
    """Source test set predictions of the source model (ID_match). Returns {model: [id_pred per repetition]}, id_idx."""
    out, idx0 = {}, None
    for m in models:
        lst = []
        for i in range(n_rep):
            if group == "challenge":
                path = g[(g.model == m) & (g.repeat == i) & (g.regime == cm.SRC)].pred_path.iloc[0]
            else:
                path = os.path.join(dirs[m], slot_dir(slot), f"{m}__{cm.SRC}__r{i}.pred.npz")
            z = np.load(path)
            idx0 = z["id_idx"] if idx0 is None else idx0
            if not np.array_equal(idx0, z["id_idx"]):
                raise SystemExit(f"{name} {m} r{i}: the row order of the source test set differs")
            lst.append(z["id_pred"])
        out[m] = lst
    return out, idx0


def stats(yT, codesT, LT, preds, yS, codesS, LS, spreds, models, WT, WS):
    """Cell values and domain values under the resampling weights (WT: target lots, WS: source lots or None = fixed).
    Returns dict[met] = (tr_cells, io_cells, sn_cells), each (n_models, n_boot), and s (n_boot,)."""
    nT = WT @ np.bincount(codesT, minlength=LT).astype(float)
    WTr = WT[:, codesT]
    if WS is None:
        WSr = np.ones((1, len(yS)))
        nS = np.full(1, float(len(yS)))
    else:
        WSr = WS[:, codesS]
        nS = WS @ np.bincount(codesS, minlength=LS).astype(float)
    res = {}
    for met in METS:
        s = wspread(yT, WTr, met) / wspread(yS, WSr, met)
        tr, io = [], []
        for m in models:
            a_tr, a_io = [], []
            for (ps, pr), pid in zip(preds[m], spreds[m]):
                q_s, a_s = lot_sums(WT, codesT, LT, ps - yT)
                q_r, a_r = lot_sums(WT, codesT, LT, pr - yT)
                if WS is None:
                    e = pid - yS
                    q_i, a_i = np.array([np.sum(e ** 2)]), np.array([np.sum(np.abs(e))])
                else:
                    q_i, a_i = lot_sums(WS, codesS, LS, pid - yS)
                if met == "rmse":
                    eT, eR, eS = np.sqrt(q_s / nT), np.sqrt(q_r / nT), np.sqrt(q_i / nS)
                else:
                    eT, eR, eS = a_s / nT, a_r / nT, a_i / nS
                a_tr.append(100 * (eT / eR - 1))
                a_io.append(100 * (eT / eS - 1))
            tr.append(np.mean(a_tr, axis=0))
            io.append(np.mean(a_io, axis=0))
        tr, io = np.stack(tr), np.stack(io)
        sn = 100 * ((1 + io / 100) / s[None, :] - 1)
        res[met] = (tr, io, sn, s)
    return res


def classify(lo_a, hi_a, lo_b, hi_b, pa, pb):
    """Three branches: if either of the two intervals contains 0 it is not determined; otherwise agree or disagree by the sign of the point estimate."""
    unres = ((lo_a <= 0) & (hi_a >= 0)) | ((lo_b <= 0) & (hi_b >= 0))
    return np.where(unres, "unresolved", np.where(same_direction(pa, pb), "agree", "disagree"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--ctrl-run-ids", default=cm.CTRL_RUNS)
    ap.add_argument("--criteria-run-id", default="criteria")
    ap.add_argument("--scale-run-id", default="scale_normalized_contrast")
    ap.add_argument("--disjoint-run-id", default="lot_disjoint")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--only", default="", help="comma-separated short domain names (for smoke runs)")
    a = ap.parse_args()
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = cm.load_challenge(a.benchmark_run_id)
    models = sorted(pf.model.unique())
    pfa = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    src_ch = pfa[pfa.regime == cm.SRC].set_index(["candidate", "model", "repeat"])
    ctrl, dirs = cm.load_controls(a.ctrl_run_ids, models)
    src_ct = ctrl[ctrl.regime == cm.SRC].drop_duplicates(["candidate", "model", "repeat"]).set_index(["candidate", "model", "repeat"])
    old_boot = pd.read_csv(os.path.join(rf, a.criteria_run_id, "criteria_boot.csv"))
    old_boot = old_boot[old_boot.variant == "disjoint"].set_index("candidate")
    old_sn = pd.read_csv(os.path.join(rf, a.scale_run_id, "scale_adj_domain.csv"))
    old_sn = old_sn[old_sn.variant == "disjoint"].set_index(["candidate", "metric"])
    old_cv = pd.read_csv(os.path.join(rf, a.disjoint_run_id, "ctrl_values.csv")).set_index(["candidate", "metric"])
    jobs = [("challenge", nm, g, None) for nm, g in pf.groupby("candidate", sort=False)]
    jobs += [("control", nm, g, dirs) for nm, g in ctrl.groupby("candidate", sort=False)]
    if a.only:
        jobs = [j for j in jobs if cm.short(j[1]) in a.only.split(",")]
    n_rep = config.N_REPEATS_BENCHMARK
    dom_rows, cell_rows, sm_rows = [], [], []
    for group, name, g, dd in jobs:
        dom = cm.short(name)
        d = cm.load_domain(group, name, g, df, y, lot, models, dd)
        slot = build_one(df, name)[0]
        spreds, sidx = load_src(group, name, g, models, dd, slot, n_rep)
        if not np.array_equal(sidx, np.asarray(d["a"]["A_heldout"])):
            raise SystemExit(f"{dom}: id_idx differs from A_heldout")
        mk = ~np.isin(d["lots"], lot.loc[d["a"]["A_dev"]].unique()) if group == "challenge" else np.ones(len(d["yv"]), bool)
        if group == "control" and np.isin(d["lots"], lot.loc[d["a"]["A_dev"]].unique()).any():
            raise SystemExit(f"{dom}: B_te of the control contains rows of lots that also occur in A_dev")
        yT = d["yv"][mk]
        preds = {m: [(ps[mk], pr[mk]) for ps, pr in pairs] for m, pairs in d["preds"].items()}
        ulT, codesT = np.unique(d["lots"][mk], return_inverse=True)
        yS = y.loc[sidx].to_numpy()
        ulS, codesS = np.unique(lot.loc[sidx].to_numpy(), return_inverse=True)
        LT, LS = len(ulT), len(ulS)
        # check against the stored Err_S
        ref = src_ch if group == "challenge" else src_ct
        for m in models:
            for i in range(n_rep):
                e = spreds[m][i] - yS
                for met, v in (("rmse", np.sqrt(np.mean(e ** 2))), ("mae", np.mean(np.abs(e)))):
                    rec = float(ref.loc[(name, m, i), f"{met}_id"])
                    if abs(v - rec) > 1e-9 * max(1.0, abs(rec)):
                        raise SystemExit(f"{dom} {m} r{i} {met}: Err_S recomputed from id_pred {v} != recorded {rec}")
        # point estimate (no resampling)
        point = stats(yT, codesT, LT, preds, yS, codesS, LS, spreds, models, np.ones((1, LT)), None)
        # mark the collapsed cells (repetition-mean R^2)
        sst = float(np.sum((yT - yT.mean()) ** 2))
        coll = {m: (np.mean([1 - np.sum((ps - yT) ** 2) / sst for ps, _ in preds[m]]) < COLLAPSE_R2)
                or (np.mean([1 - np.sum((pr - yT) ** 2) / sst for _, pr in preds[m]]) < COLLAPSE_R2) for m in models}
        for met in METS:
            tr, io, sn, s = point[met]
            med_tr, med_io, med_sn = float(np.median(tr)), float(np.median(io)), float(np.median(sn))
            if group == "challenge":
                o = old_sn.loc[(name, met)]
                chk = max(abs(med_tr - o.tr_med), abs(med_io - o.idood_med), abs(med_sn - o.adj_med), abs(s[0] - o.s))
            else:
                o = old_cv.loc[(name, met)]
                chk = max(abs(med_tr - o.tr), abs(med_io - o.idood))
            if chk > 1e-7:
                raise SystemExit(f"{dom} {met}: the point estimate differs from the existing output (maximum difference {chk})")
        # resampling
        crc = zlib.crc32(dom.encode())
        rng_t = np.random.default_rng([unc.BOOT_SEED, crc])
        WT = rng_t.multinomial(LT, np.full(LT, 1.0 / LT), size=a.n_boot).astype(float)
        rng_s = np.random.default_rng([SRC_SEED, crc])
        WS = rng_s.multinomial(LS, np.full(LS, 1.0 / LS), size=a.n_boot).astype(float)
        for scheme, ws in (("target", None), ("both", WS)):
            bs = stats(yT, codesT, LT, preds, yS, codesS, LS, spreds, models, WT, ws)
            for met in METS:
                tr_p, io_p, sn_p, s_p = point[met]
                tr_b, io_b, sn_b, s_b = bs[met]
                dm = {k: np.median(v, axis=0) for k, v in (("tr", tr_b), ("io", io_b), ("sn", sn_b))}
                if scheme == "target" and group == "challenge" and met == "rmse" and a.n_boot == 2000:
                    lo, hi = np.percentile(dm["tr"], [2.5, 97.5])
                    o = old_boot.loc[name]
                    if abs(lo - o.rmse_lo) > 1e-9 or abs(hi - o.rmse_hi) > 1e-9:
                        raise SystemExit(f"{dom}: the g_TR interval of the target resampling differs from criteria_boot ({lo}, {hi}) versus ({o.rmse_lo}, {o.rmse_hi})")
                row = dict(group=group, candidate=name, dom=dom, metric=met, scheme=scheme, n_lots_T=LT, n_rows_T=int(mk.sum()),
                           n_lots_S=LS, n_rows_S=len(yS), s=float(s_p[0]))
                for k, pv in (("tr", tr_p), ("io", io_p), ("sn", sn_p)):
                    row[f"{k}"] = float(np.median(pv))
                    row[f"{k}_lo"], row[f"{k}_hi"] = (float(v) for v in np.percentile(dm[k], [2.5, 97.5]))
                row["s_lo"], row["s_hi"] = (float(v) for v in np.percentile(s_b, [2.5, 97.5]))
                for k in ("io", "sn"):
                    row[f"p_agree_{k}"] = float(np.mean(same_direction(dm[k], dm["tr"])))
                    row[f"class_{k}"] = str(classify(row["tr_lo"], row["tr_hi"], row[f"{k}_lo"], row[f"{k}_hi"], row["tr"], row[k]))
                dom_rows.append(row)
                for j, m in enumerate(models):
                    c = dict(group=group, dom=dom, metric=met, scheme=scheme, model=m, collapsed=bool(coll[m]))
                    for k, pv, bv in (("tr", tr_p, tr_b), ("io", io_p, io_b), ("sn", sn_p, sn_b)):
                        c[k] = float(pv[j, 0])
                        c[f"{k}_lo"], c[f"{k}_hi"] = (float(v) for v in np.percentile(bv[j], [2.5, 97.5]))
                    for k in ("io", "sn"):
                        c[f"p_agree_{k}"] = float(np.mean(same_direction((io_b if k == "io" else sn_b)[j], tr_b[j])))
                        c[f"class_{k}"] = str(classify(c["tr_lo"], c["tr_hi"], c[f"{k}_lo"], c[f"{k}_hi"], c["tr"], c[k]))
                    cell_rows.append(c)
        for met in METS:
            sm_rows.append(dict(group=group, dom=dom, metric=met, s=float(point[met][3][0]), mean_ratio=float(yT.mean() / yS.mean())))
        log(f"[sn_boot] {group} {dom} done (target lots {LT}, source lots {LS}; point estimate, Err_S and the existing interval reproduced)")
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    D, C, SM = pd.DataFrame(dom_rows), pd.DataFrame(cell_rows), pd.DataFrame(sm_rows)
    D.to_csv(os.path.join(out, "sn_domain.csv"), index=False)
    C.to_csv(os.path.join(out, "sn_cells.csv"), index=False)
    # summary: direction agreement (two branches by point estimate, three branches by resampling), at domain level and at cell level
    ag = []
    for (group, met, scheme), dg in D.groupby(["group", "metric", "scheme"]):
        cg = C[(C.group == group) & (C.metric == met) & (C.scheme == scheme)]
        for k in ("io", "sn"):
            r = dict(group=group, metric=met, scheme=scheme, comparison=k, n_domains=len(dg),
                     dom_agree_point=int(same_direction(dg[k], dg.tr).sum()))
            for cl in ("agree", "disagree", "unresolved"):
                r[f"dom_{cl}"] = int((dg[f"class_{k}"] == cl).sum())
            r["dom_disagree_names"] = ",".join(dg[~same_direction(dg[k], dg.tr)].dom)
            r["dom_unresolved_names"] = ",".join(dg[dg[f"class_{k}"] == "unresolved"].dom)
            cg = cg.assign(pa=same_direction(cg[k], cg.tr))
            per = cg.groupby("dom").pa.mean()
            r.update(cells=len(cg), cell_agree_point=float(cg.pa.mean()), cell_agree_point_nocoll=float(cg[~cg.collapsed].pa.mean()),
                     per_domain_agree_min=float(per.min()), per_domain_agree_median=float(per.median()), per_domain_agree_max=float(per.max()))
            for cl in ("agree", "disagree", "unresolved"):
                r[f"cell_{cl}"] = float((cg[f"class_{k}"] == cl).mean())
            ag.append(r)
    AG = pd.DataFrame(ag)
    AG.to_csv(os.path.join(out, "sn_agree.csv"), index=False)
    SM["log_s"], SM["log_mean_ratio"] = np.log(SM.s), np.log(SM.mean_ratio)
    for met in METS:
        x = SM[SM.metric == met]
        for grp, xx in (("all", x), ("challenge", x[x.group == "challenge"]), ("control", x[x.group == "control"])):
            if len(xx) > 2:
                SM.loc[xx.index, f"spearman_{grp}"] = float(spearmanr(xx.s, xx.mean_ratio)[0])
    SM.to_csv(os.path.join(out, "sn_scale_mean.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["group", "dom", "metric", "scheme", "tr", "tr_lo", "tr_hi", "io", "sn", "sn_lo", "sn_hi", "class_io", "class_sn", "p_agree_sn"]
    print(D[cols].round(1).to_string(index=False))
    print(AG.round(3).to_string(index=False))
    print(SM.round(3).to_string(index=False))
    log("[sn_boot] done")


if __name__ == "__main__":
    main()
