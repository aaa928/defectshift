"""Tail concentration: is the transfer loss concentrated in the tail rows of the outcome? The same rule is applied to the challenge domains and to the randomized lot-split controls.
Only saved predictions are read (no retraining, CPU only).
Tail rows = the 1 % rows of B_te with the largest |y - median(y of B_tr)| (the same as the y_extreme rule of the diagnostics of the final benchmark, independent of the predictions).
Cell = model x domain (repetition mean). The model set is matched to the 8 models of the control run.
Output: results/analysis/<run-id>/tail_cells.csv, tail_summary.csv
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/tail_concentration.py --run-id tail_main
"""
import argparse
import os

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from defectshift import config  # noqa: E402
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data, slot_dir  # noqa: E402
from defectshift.splits import B_view  # noqa: E402
from defectshift.utils import log  # noqa: E402

M8 = ["Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL"]
SRC, REF = "ID_match", "OOD_match"
TOP_FRAC = 0.01


def gap_pct(s, r):
    return 100 * (s - r) / r


def cell(preds_s, preds_r, yv, mask):
    """Repetition mean of the RMSE and MAE transfer loss on the masked rows for the (source, reference) predictions of every repetition."""
    d_rmse, d_mae = [], []
    for ps, pr in zip(preds_s, preds_r):
        es, er = ps[mask] - yv[mask], pr[mask] - yv[mask]
        d_rmse.append(gap_pct(np.sqrt(np.mean(es ** 2)), np.sqrt(np.mean(er ** 2))))
        d_mae.append(gap_pct(np.mean(np.abs(es)), np.mean(np.abs(er))))
    return float(np.mean(d_rmse)), float(np.mean(d_mae))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--ctrl-run-id", default="controls_batch_01")
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    pf = pf[pf.regime.isin([SRC, REF]) & pf.model.isin(M8)]
    ctrl_dir = os.path.join(config.CACHE_DIR, "benchmark", "runs", a.ctrl_run_id, "models")
    ctrl_rows = pd.read_csv(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.ctrl_run_id, "rows.csv"))
    ctrl_rows = ctrl_rows[(ctrl_rows.status == "ok") & ctrl_rows.regime.isin([SRC, REF])]
    rows = []
    jobs = [("challenge", name, g) for name, g in pf.groupby("candidate", sort=False)] + \
           [("control", name, g) for name, g in ctrl_rows.groupby("candidate", sort=False)]
    for group, name, g in jobs:
        slot, mA, mB = build_one(df, name)
        bv = B_view(slot, df.index[mB], y, lot, "benchmark")
        med = float(y.loc[bv["B_tr"]].median())
        for m in M8:
            ps, pr, idx0 = [], [], None
            for i in sorted(g[g.model == m].repeat.unique()):
                paths = {}
                for reg in (SRC, REF):
                    if group == "challenge":
                        paths[reg] = g[(g.model == m) & (g.repeat == i) & (g.regime == reg)].pred_path.iloc[0]
                    else:
                        paths[reg] = os.path.join(ctrl_dir, slot_dir(slot), f"{m}__{reg}__r{i}.pred.npz")
                zs, zr = np.load(paths[SRC]), np.load(paths[REF])
                assert np.array_equal(zs["ood_idx"], zr["ood_idx"])
                idx0 = zs["ood_idx"] if idx0 is None else idx0
                assert np.array_equal(idx0, zs["ood_idx"])
                ps.append(zs["ood_pred"])
                pr.append(zr["ood_pred"])
            yv = y.loc[idx0].to_numpy()
            score = np.abs(yv - med)
            k = max(1, int(np.ceil(TOP_FRAC * len(yv))))   # rounded up, as in the y_extreme rule of the final-benchmark diagnostics
            top = np.zeros(len(yv), bool)
            top[np.argsort(-score, kind="stable")[:k]] = True
            all_r, all_m = cell(ps, pr, yv, np.ones(len(yv), bool))
            top_r, top_m = cell(ps, pr, yv, top)
            rest_r, rest_m = cell(ps, pr, yv, ~top)
            rows.append(dict(group=group, candidate=name, model=m, n_rows=len(yv), n_top=int(k), all_rmse=all_r, top_rmse=top_r,
                             rest_rmse=rest_r, all_mae=all_m, top_mae=top_m, rest_mae=rest_m))
        log(f"[tail] {group} {name[:14]:14s} done")
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "tail_cells.csv"), index=False)
    dom = t.groupby(["group", "candidate"])[["all_rmse", "top_rmse", "rest_rmse", "all_mae", "top_mae", "rest_mae"]].median()
    dom.to_csv(os.path.join(out, "tail_summary.csv"))
    print(dom.round(1).to_string())
    for g, d in dom.groupby("group"):
        print(f"\n[{g}] median of the domain medians:", d.median().round(1).to_dict())
        print(f"[{g}] domains whose tail loss exceeds the loss of the remaining rows: RMSE {int((d.top_rmse > d.rest_rmse).sum())}/{len(d)}, MAE {int((d.top_mae > d.rest_mae).sum())}/{len(d)}")
        print(f"[{g}] share of cells with tail loss > remaining loss: RMSE {(t[t.group == g].top_rmse > t[t.group == g].rest_rmse).mean():.3f}, "
              f"share with positive tail loss {(t[t.group == g].top_rmse > 0).mean():.3f}, share with positive remaining loss {(t[t.group == g].rest_rmse > 0).mean():.3f}")
    log("[tail] done")


if __name__ == "__main__":
    main()
