"""Target test accuracy per model x domain x training regime (ID_full, ID_match, OOD_match, OOD_match_tuned) recomputed on lot-disjoint rows (Supplementary S6-A).
It reads only saved predictions (no training). The all-rows results (Supplementary S6-B) are the *_ood values of the existing per_fit.csv.
Lot-disjoint = B_te rows whose lot also occurs in the source training pool A_dev are excluded (the same definition as lot_disjoint_gaps.py). R^2 is computed with the mean and variance of those rows.
Output: results/analysis/<run-id>/full_disjoint.csv (per fit), full_disjoint_cells.csv (repetition mean and standard deviation)
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/lot_disjoint_accuracy.py --run-id lot_disjoint_accuracy
"""
import argparse
import os

import numpy as np
import pandas as pd

from defectshift import config
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data
from defectshift.splits import get_A
from defectshift.utils import log
import common as cm

REGIMES = ("ID_full", "ID_match", "OOD_match", "OOD_match_tuned")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    a = ap.parse_args()
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    pf = pf[pf.regime.isin(REGIMES)]
    recs = []
    for name, g in pf.groupby("candidate", sort=False):
        slot, mA, _ = build_one(df, name)
        dev_lots = set(lot.loc[get_A(slot, df.index[mA], y, lot)["A_dev"]].unique())
        idx0 = None
        for r in g.itertuples():
            z = np.load(r.pred_path)
            idx0 = z["ood_idx"] if idx0 is None else idx0
            if not np.array_equal(z["ood_idx"], idx0):
                raise SystemExit(f"{name} {r.model} r{r.repeat} {r.regime}: the B_te row order differs")
            keep = ~lot.loc[idx0].isin(dev_lots).to_numpy()
            yv = y.loc[idx0].to_numpy()[keep]
            e = z["ood_pred"][keep] - yv
            all_e = z["ood_pred"] - y.loc[idx0].to_numpy()
            if abs(np.sqrt(np.mean(all_e ** 2)) - r.rmse_ood) > 1e-9:
                raise SystemExit(f"{name} {r.model} r{r.repeat} {r.regime}: the all-rows RMSE of the prediction file differs from the record")
            recs.append(dict(candidate=name, model=r.model, regime=r.regime, repeat=r.repeat, n_rows=int(keep.sum()),
                             rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                             r2=float(1 - np.sum(e ** 2) / np.sum((yv - yv.mean()) ** 2))))
        log(f"[full_disjoint] {cm.short(name)} done: evaluation rows {int(keep.sum())}/{len(keep)}")
    d = pd.DataFrame(recs)
    n = d.groupby(["candidate", "model", "regime"]).repeat.nunique()
    if not n.eq(config.N_REPEATS_BENCHMARK).all():
        raise SystemExit(f"entries with a wrong number of repetitions: {n[n != config.N_REPEATS_BENCHMARK].head().to_dict()}")
    cells = d.groupby(["candidate", "model", "regime"]).agg(n_rows=("n_rows", "first"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                                                            mae=("mae", "mean"), mae_sd=("mae", "std"), r2=("r2", "mean"), r2_sd=("r2", "std")).reset_index()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, "full_disjoint.csv"), index=False)
    cells.to_csv(os.path.join(out, "full_disjoint_cells.csv"), index=False)
    print(cells.groupby("regime").size().to_dict(), len(cells))
    log("[full_disjoint] done")


if __name__ == "__main__":
    main()
