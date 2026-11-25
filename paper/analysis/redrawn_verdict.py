"""Aggregation of the redrawn target test splits: the gaps of the 5 redraws (scripts/extra/run_redrawn_target_splits.py, run names redrawn_s<k>_a/_b) are compared with the fixed 50 controls.
It reads only saved predictions (no new training, CPU only). It aggregates only if all 7 domains x 5 redraws exist (no result from a subset of the domains).

Evaluation rows (per redrawn split): the lot-disjoint rows of B_te' of that redraw (rows of lots shared with A_dev excluded; main evaluation) and, on top of that, without the 1% tail (rest99; tail =
  the ceil(1 % |B_te'|) rows with the largest |y - median(y of B_tr')|, the same rule as verdict_robustness; B_tr' is the training pool of that redraw).
Value: gap (RMSE and MAE) averaged over the 5 repetitions per model, domain value = median over the 8 models (Ridge, kNN, XGBoost, CatBoost, MLP, TabM, TabPFN, TabICL).
Comparison: values of the same configuration of the fixed 50 controls (extra_controls_compose/compose_values.csv, not_shared and rest99). The controls are not redrawn.
  margin = gap - maximum of the controls, k = number of controls with a strictly smaller gap (ties not counted), k / 50 is a relative position and not a p-value.
  canonical column: the value computed with the same function on the canonical split (benchmark_canonical). It must equal the T0 gap of ctrl40_verdict.csv (check).
Recorded: the number of test lots and test rows, n* (rows.csv), the Jaccard index of the lot sets (canonical against each redraw, 5 x 5 among the redraws; based on all test lots).
Interpretation: the 5 redraws share the same source tuning and the same fixed controls and are redraws fixed in advance. They are a sensitivity analysis with respect to the draw of the target split and not independent repetitions or an estimate of variability.
Output: results/analysis/<run-id>/{resplit_values.csv, resplit_verdict.csv, resplit_jaccard.csv}
Usage: python paper/analysis/redrawn_verdict.py --run-id redrawn_verdict
"""
import argparse
import itertools
import json
import os


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import verdict_robustness as rb  # noqa: E402
import common as cm  # noqa: E402
from defectshift import config  # noqa: E402
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data, slot_dir  # noqa: E402
from defectshift.splits import B_view, get_A  # noqa: E402
from defectshift.splits_redraw import B_view_resplit  # noqa: E402
from defectshift.utils import log  # noqa: E402

M8 = ("Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL")
HEAD_DOMAINS = ("cat_050", "x_time<=2099-05-04 21:18:39")
SRC, REF = rb.SRC, rb.REF
K = range(1, 6)


def load_preds(pred_dir, models, regs=(SRC, REF)):
    """Predictions (source, reference) per model x repetition under pred_dir and the B_te row index."""
    preds, idx0 = {}, None
    for m in models:
        pairs = []
        for i in range(config.N_REPEATS_BENCHMARK):
            z = {r: np.load(os.path.join(pred_dir, f"{m}__{r}__r{i}.pred.npz")) for r in regs}
            assert np.array_equal(z[SRC]["ood_idx"], z[REF]["ood_idx"])
            idx0 = z[SRC]["ood_idx"] if idx0 is None else idx0
            assert np.array_equal(idx0, z[SRC]["ood_idx"]), "the B_te row order differs between models and repetitions"
            pairs.append((z[SRC]["ood_pred"], z[REF]["ood_pred"]))
        preds[m] = pairs
    return preds, idx0


def values(preds, idx, y, lot, a_dev_lots, btr_median):
    """Domain values for lot-disjoint rows and for lot-disjoint rows without the 1% tail. Returns {variant: (rmse, mae, n_rows, n_lots)}."""
    yv = y.loc[idx].to_numpy()
    lots = lot.loc[idx].to_numpy()
    score = np.abs(yv - btr_median)
    kk = max(1, int(np.ceil(rb.TOP_FRAC * len(yv))))
    top = np.zeros(len(yv), bool)
    top[np.argsort(-score, kind="stable")[:kk]] = True
    shared = np.isin(lots, a_dev_lots)
    out = {}
    for var, mk in (("not_shared", ~shared), ("rest99", ~shared & ~top)):
        cells = np.array([rb.cell(preds[m], yv, mk) for m in M8])
        out[var] = (float(np.median(cells[:, 0])), float(np.median(cells[:, 1])), int(mk.sum()), int(len(np.unique(lots[mk]))))
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--screening-run-id", default="screening_main")
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--compose-run-id", default="extra_controls_compose")
    ap.add_argument("--agg-run-id", default="extra_controls_verdict")
    ap.add_argument("--only", default="", help="comma-separated domain names (for smoke runs). If given, the check of completeness is not performed")
    ap.add_argument("--ks", default="1,2,3,4,5")
    a = ap.parse_args()
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    fin = json.load(open(os.path.join(config.RESULTS_DIR, "screening", "runs", a.screening_run_id, "selection", "final_domains.json")))
    domains = fin["categorical"] + fin["temporal"]
    if a.only:
        domains = [d for d in domains if d in a.only.split(",")]
    ks = [int(k) for k in a.ks.split(",")]
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    rows, jac = [], []
    for name in domains:
        dom = cm.short(name)
        slot, mA, mB = build_one(df, name)
        a_dev_lots = lot.loc[get_A(slot, df.index[mA], y, lot)["A_dev"]].unique()
        B_idx = df.index[mB]
        # canonical (computed with the same function and checked against T0 of ctrl40_agg)
        g = pf[(pf.candidate == name) & pf.regime.isin([SRC, REF]) & pf.model.isin(M8)]
        cpreds, cidx = {}, None
        for m in M8:
            pairs = []
            for i in range(config.N_REPEATS_BENCHMARK):
                z = {r: np.load(g[(g.model == m) & (g.repeat == i) & (g.regime == r)].pred_path.iloc[0]) for r in (SRC, REF)}
                if not np.array_equal(z[SRC]["ood_idx"], z[REF]["ood_idx"]):
                    raise SystemExit(f"canonical {dom} {m} r{i}: the B_te row order of the source and reference predictions differs")
                cidx = z[SRC]["ood_idx"] if cidx is None else cidx
                if not np.array_equal(cidx, z[SRC]["ood_idx"]):
                    raise SystemExit(f"canonical {dom} {m} r{i}: the B_te row order differs between models and repetitions")
                pairs.append((z[SRC]["ood_pred"], z[REF]["ood_pred"]))
            cpreds[m] = pairs
        cbv = B_view(slot, B_idx, y, lot, "benchmark")
        cv = values(cpreds, cidx, y, lot, a_dev_lots, float(y.loc[cbv["B_tr"]].median()))
        for var, (r_, e_, n_, l_) in cv.items():
            rows.append(dict(dom=dom, k=0, variant=var, gap_rmse=r_, gap_mae=e_, n_rows=n_, n_lots=l_, n_star=int(g.n_star.max())))
        canon_lots = set(lot.loc[cidx].unique())
        lots_k = {}
        for k in ks:
            suffix = "a" if name in HEAD_DOMAINS else "b"
            run = f"redrawn_s{k}_{suffix}"
            meta = json.load(open(os.path.join(config.RESULTS_DIR, "benchmark", "runs", run, "redrawn_split.json")))
            if meta.get("canonical", True):
                raise SystemExit(f"{run}: not marked as a redrawn-split run")
            rws = pd.read_csv(os.path.join(config.RESULTS_DIR, "benchmark", "runs", run, "rows.csv"))
            rws = rws[(rws.candidate == name) & rws.regime.isin([SRC, REF]) & rws.model.isin(M8)]
            need = len(M8) * config.N_REPEATS_BENCHMARK * 2
            if len(rws[rws.status == "ok"].drop_duplicates(["model", "repeat", "regime"])) != need:
                raise SystemExit(f"{run} / {dom}: the number of finished trainings is not {need} (the redraw is not finished or has failures)")
            pdir = os.path.join(config.CACHE_DIR, "benchmark", "runs", run, "models", slot_dir(slot))
            preds, idx = load_preds(pdir, M8)
            bv = B_view_resplit(slot, B_idx, y, lot, "benchmark", k)
            if not np.array_equal(np.sort(idx), np.sort(np.asarray(bv["B_te"]))):
                raise SystemExit(f"{run} / {dom}: the B_te rows of the predictions differ from B_te of the redrawn split")
            v = values(preds, idx, y, lot, a_dev_lots, float(y.loc[bv["B_tr"]].median()))
            for var, (r_, e_, n_, l_) in v.items():
                rows.append(dict(dom=dom, k=k, variant=var, gap_rmse=r_, gap_mae=e_, n_rows=n_, n_lots=l_, n_star=int(rws.n_star.max())))
            lots_k[k] = set(lot.loc[idx].unique())
            jac.append(dict(dom=dom, a="canonical", b=f"s{k}", jaccard=jaccard(canon_lots, lots_k[k]), n_a=len(canon_lots), n_b=len(lots_k[k])))
        for k1, k2 in itertools.combinations(ks, 2):
            jac.append(dict(dom=dom, a=f"s{k1}", b=f"s{k2}", jaccard=jaccard(lots_k[k1], lots_k[k2]), n_a=len(lots_k[k1]), n_b=len(lots_k[k2])))
        log(f"[resplit_agg] {dom} done")
    V = pd.DataFrame(rows)
    V.to_csv(os.path.join(out, "resplit_values.csv"), index=False)
    pd.DataFrame(jac).to_csv(os.path.join(out, "resplit_jaccard.csv"), index=False)
    # check: the canonical values equal the T0 gap of ctrl40_agg
    ver = pd.read_csv(os.path.join(rf, a.agg_run_id, "ctrl40_verdict.csv"))
    t0 = ver[ver.track == "T0"].set_index("dom")
    c = V[(V.k == 0) & (V.variant == "not_shared")].set_index("dom")
    for met in ("rmse", "mae"):
        d = float((c[f"gap_{met}"] - t0[f"gap_{met}"].reindex(c.index)).abs().max())
        if d > 1e-7:
            raise SystemExit(f"the canonical values differ from T0 of ctrl40_agg ({met}, maximum difference {d})")
    log("[resplit_agg] check passed: the canonical values equal those of ctrl40_agg")
    # verdict against the fixed 50 controls
    cvals = pd.read_csv(os.path.join(rf, a.compose_run_id, "compose_values.csv"))
    cvals = cvals[cvals.group == "control"]
    vr = []
    for _, r in V.iterrows():
        cc = cvals[cvals.variant == r.variant]
        row = dict(dom=r.dom, k=r.k, variant=r.variant, n_rows=r.n_rows, n_lots=r.n_lots, n_star=r.n_star, n_ctrl=len(cc))
        for met in ("rmse", "mae"):
            row[f"gap_{met}"] = r[f"gap_{met}"]
            row[f"ctrl_max_{met}"] = float(cc[met].max())
            row[f"margin_{met}"] = r[f"gap_{met}"] - row[f"ctrl_max_{met}"]
            row[f"k_{met}"] = int((cc[met] < r[f"gap_{met}"]).sum())
        row["k_both"] = int(((cc.rmse < r.gap_rmse) & (cc.mae < r.gap_mae)).sum())
        vr.append(row)
    VR = pd.DataFrame(vr)
    VR.to_csv(os.path.join(out, "resplit_verdict.csv"), index=False)
    pd.set_option("display.width", 250)
    show = ["dom", "k", "n_rows", "n_lots", "n_star", "gap_rmse", "ctrl_max_rmse", "margin_rmse", "k_rmse", "gap_mae", "ctrl_max_mae", "margin_mae", "k_mae", "k_both"]
    print(VR[VR.variant == "not_shared"][show].round(2).to_string(index=False))
    print(VR[VR.variant == "rest99"][show].round(2).to_string(index=False))
    J = pd.DataFrame(jac)
    print(J[J.a == "canonical"].groupby("dom").jaccard.agg(["min", "median", "max"]).round(2).to_string())
    log("[resplit_agg] done")


if __name__ == "__main__":
    main()
