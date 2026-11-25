"""Sensitivity of the pipeline (hyperparameters, categorical representation, preprocessing, training-row scale) recomputed on lot-disjoint target test rows. It reads only saved predictions (no training).

sensitivity_directions.py read the all-rows metrics (*_ood) stored in the run records. Here the prediction files of each arm (ood_idx and ood_pred, id_idx and id_pred) are read again
and the errors are computed on the same evaluation rows as in the main analysis (B_te rows whose lot also occurs in the source training pool A_dev are excluded). The main results are also
recomputed in the same way from the prediction files of the same repetitions (0-2) for comparison (the same rows, the same repetitions).
  ID-OOD gap = Err_T(f_S)/Err_S(f_S) - 1 (numerator = lot-disjoint target rows, denominator = all of A_heldout),  target-ref gap = Err_T(f_S)/Err_T(f_T) - 1
  cell = repetition mean (%) per model x domain. For every arm
    rerun : only the cells that were rerun; the reference is the main result of the same cell and the same repetitions
    roster: 22 models (models that were not rerun are filled in with the main result), only the domains covered by that arm
  Reported: share of retained signs of the target-ref gap (rerun cells, RMSE and MAE), direction agreement (the ID-OOD and target-ref gaps have the same sign; rerun and roster).
Prediction file paths: main result = pred_path of per_fit.csv, sensitivity = cache/sensitivity/runs/<run>/preds/<slot>/<model>__<regime>__<arm>__r<i>.pred.npz
(the run stops if one is missing). It also stops if the evaluation row order differs from the main result.
Output: results/analysis/<run-id>/{pipeline_disjoint_cells.csv, pipeline_disjoint_summary.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/sensitivity_lot_disjoint.py --run-id sensitivity_lot_disjoint
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as cm
from defectshift import config
from defectshift.candidates import build_one, load_data, slot_dir
from defectshift.splits import get_A
from defectshift.utils import log
from defectshift.directions import same_direction

SENSITIVITY_RUNS = ("sensitivity_main", "sensitivity_uniform", "sensitivity_x", "sensitivity_x_uniform", "sensitivity_x_nodeprep", "sensitivity_trainscale", "sensitivity_x_trainscale")
ARMS = ("default", "hpo_seed1999", "integer", "uniform", "node_prep_a", "trainscale")
REPS = (0, 1, 2)
SRC, REF = "ID_match", "OOD_match"


def err(e, met):
    return float(np.sqrt(np.mean(e ** 2))) if met == "rmse" else float(np.mean(np.abs(e)))


def gaps(src, ref, y, keep, id_y):
    """(ID-OOD gap, target-ref gap) in % of one repetition - RMSE and MAE."""
    out = {}
    for met in ("rmse", "mae"):
        t_s = err(src["ood_pred"][keep] - y[keep], met)
        t_r = err(ref["ood_pred"][keep] - y[keep], met)
        s_s = err(src["id_pred"] - id_y, met)
        out[f"idood_{met}"], out[f"tr_{met}"] = 100 * (t_s / s_s - 1), 100 * (t_s / t_r - 1)
    return out


def agree(c):
    return {met: float(np.mean(same_direction(c[f"idood_{met}"], c[f"tr_{met}"]))) for met in ("rmse", "mae")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    a = ap.parse_args()
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = cm.load_challenge(a.benchmark_run_id)
    pf = pf[pf.repeat.isin(REPS)].set_index(["candidate", "model", "repeat", "regime"]).pred_path
    sensitivity = pd.concat([pd.read_csv(os.path.join(config.RESULTS_DIR, "sensitivity", "runs", r, "rows.csv")).assign(run=r) for r in SENSITIVITY_RUNS])
    sensitivity = sensitivity[(sensitivity.status == "ok") & sensitivity.arm.isin(ARMS) & sensitivity.regime.isin([SRC, REF]) & sensitivity.repeat.isin(REPS)]
    if sensitivity.duplicated(["arm", "candidate", "model", "repeat", "regime"]).any():
        raise SystemExit("the same arm, cell, repetition and regime occurs twice")
    masks = {}
    for name in sensitivity.candidate.unique():
        slot, mA, _ = build_one(df, name)
        A = get_A(slot, df.index[mA], y, lot)
        masks[name] = (slot, set(lot.loc[A["A_dev"]].unique()), A["A_heldout"])

    def load(path, name):
        z = np.load(path)
        slot, dev_lots, held = masks[name]
        if not np.array_equal(z["id_idx"], np.asarray(held)):
            raise SystemExit(f"{path}: the A_heldout rows differ from the current split")
        return z

    def cell_gaps(get_pair, name):
        recs = []
        for i in REPS:
            s, r = get_pair(i)
            if not np.array_equal(s["ood_idx"], r["ood_idx"]):
                raise SystemExit(f"{name} r{i}: the B_te row order of source and reference differs")
            idx = s["ood_idx"]
            keep = ~lot.loc[idx].isin(masks[name][1]).to_numpy()
            recs.append(gaps(s, r, y.loc[idx].to_numpy(), keep, y.loc[s["id_idx"]].to_numpy()))
        return pd.DataFrame(recs).mean().to_dict(), int(keep.sum()), idx

    base = []
    for name in sensitivity.candidate.unique():
        for m in sorted(pf.loc[name].index.get_level_values(0).unique()):
            g, n_keep, idx = cell_gaps(lambda i: (load(pf.loc[(name, m, i, SRC)], name), load(pf.loc[(name, m, i, REF)], name)), name)
            base.append(dict(candidate=name, model=m, n_rows=n_keep, **g))
    base = pd.DataFrame(base)
    log(f"[pipeline_disjoint] main result (repetitions 0-2, lot-disjoint) cells {len(base)}, direction agreement {agree(base)}")
    summ, allc = [dict(arm="main", scope="roster", n_cells=len(base), **{f"agree_{k}": v for k, v in agree(base).items()})], []
    for arm in ARMS:
        ea = sensitivity[sensitivity.arm == arm]
        rows = []
        for (name, m), g in ea.groupby(["candidate", "model"]):
            path = lambda i, reg: os.path.join(config.CACHE_DIR, "sensitivity", "runs", g[(g.repeat == i) & (g.regime == reg)].run.iloc[0], "preds",
                                               slot_dir(g.slot.iloc[0]), f"{m}__{reg}__{arm}__r{i}.pred.npz")
            for i in REPS:
                for reg in (SRC, REF):
                    if len(g[(g.repeat == i) & (g.regime == reg)]) != 1 or not os.path.exists(path(i, reg)):
                        raise SystemExit(f"{arm} {name} {m} r{i} {reg}: no record or prediction file")
            gg, n_keep, _ = cell_gaps(lambda i: (load(path(i, SRC), name), load(path(i, REF), name)), name)
            rows.append(dict(candidate=name, model=m, n_rows=n_keep, **gg))
        c = pd.DataFrame(rows)
        key = c.set_index(["candidate", "model"]).index
        b = base.set_index(["candidate", "model"]).loc[key].reset_index()
        doms = c.candidate.unique()
        roster = pd.concat([base.set_index(["candidate", "model"]).drop(key), c.set_index(["candidate", "model"])]).reset_index()
        roster = roster[roster.candidate.isin(doms)]
        base_d = base[base.candidate.isin(doms)]
        keep_sign = {met: float(np.mean(same_direction(c[f"tr_{met}"].to_numpy(), b[f"tr_{met}"].to_numpy()))) for met in ("rmse", "mae")}
        for scope, x, ref in (("rerun", c, b), ("roster", roster, base_d)):
            ag, ar = agree(x), agree(ref)
            summ.append(dict(arm=arm, scope=scope, n_cells=len(x), n_domains=len(doms), agree_rmse=ag["rmse"], agree_mae=ag["mae"],
                             main_agree_rmse=ar["rmse"], main_agree_mae=ar["mae"],
                             tr_sign_kept_rmse=keep_sign["rmse"] if scope == "rerun" else np.nan,
                             tr_sign_kept_mae=keep_sign["mae"] if scope == "rerun" else np.nan))
        c = c.merge(b, on=["candidate", "model"], suffixes=("", "_main"))
        c.insert(0, "arm", arm)
        allc.append(c)
        log(f"[pipeline_disjoint] {arm}: cells {len(c)}, retained sign {keep_sign}")
    allc = pd.concat(allc)
    allc["tr_rmse_same_sign"] = same_direction(allc.tr_rmse, allc.tr_rmse_main)
    allc["tr_mae_same_sign"] = same_direction(allc.tr_mae, allc.tr_mae_main)
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    allc.to_csv(os.path.join(out, "pipeline_disjoint_cells.csv"), index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(out, "pipeline_disjoint_summary.csv"), index=False)
    pooled = {met: float(allc[f"tr_{met}_same_sign"].mean()) for met in ("rmse", "mae")}
    print(s.round(3).to_string(index=False))
    print("retained sign of the target-ref gap over all arms (rerun cells):\n", {k: round(v, 3) for k, v in pooled.items()}, "cells", len(allc))
    log("[pipeline_disjoint] done")


if __name__ == "__main__":
    main()
