"""Recomputes the direction agreement of the ID-OOD gap and the target-reference gap for the sensitivity runs. It reads only saved metrics.

For every arm the sensitivity runs saved ID_match and OOD_match of the rerun models for repetitions 0-2, and every row has the A_heldout error (*_id) and the B_te error (*_ood).
  ID-OOD gap = Err_T(f_S)/Err_S(f_S) - 1 (*_ood / *_id of ID_match),  target-ref gap = Err_T(f_S)/Err_T(f_T) - 1 (*_ood of ID_match / *_ood of OOD_match)
cell = repetition mean (%) per model x domain, direction agreement = the two cell values have the same sign.
Two results are given for every arm
  rerun: only the cells that were rerun. The reference is the main result of the same cell and the same repetitions (0-2)
  roster: all 22 models. Models that were not rerun are filled in with the main result (repetitions 0-2), as in the sensitivity report
Arms: default (default settings), hpo_seed1999 (seed of the tuning split, 2 domains), integer (integer category codes), uniform (uniform preprocessing), node_prep_a (NODE preprocessing)
Output: results/analysis/<run-id>/{pipeline_rq1_cells.csv, pipeline_rq1_summary.csv}
Usage: python paper/analysis/sensitivity_directions.py --run-id sensitivity_directions
"""
import argparse
import os

import numpy as np
import pandas as pd

from defectshift import config
from defectshift.utils import read_per_fit
from defectshift.utils import log
from defectshift.directions import same_direction

SENSITIVITY_RUNS = "sensitivity_main,sensitivity_uniform,sensitivity_x,sensitivity_x_uniform,sensitivity_x_nodeprep"
ARMS = ("default", "hpo_seed1999", "integer", "uniform", "node_prep_a")
REPS = (0, 1, 2)
KEYS = ["candidate", "model", "repeat"]


def cells(rows):
    """rows (ID_match and OOD_match rows) -> repetition mean (%) of the ID-OOD gap and the target-ref gap per cell (domain, model)."""
    s = rows[rows.regime == "ID_match"].set_index(KEYS)
    r = rows[rows.regime == "OOD_match"].set_index(KEYS)
    if not s.index.sort_values().equals(r.index.sort_values()):
        raise SystemExit("(domain, model, repetition) of ID_match and OOD_match differ")
    r = r.reindex(s.index)
    out = {}
    for met in ("rmse", "mae"):
        out[f"idood_{met}"] = 100 * (s[f"{met}_ood"] / s[f"{met}_id"] - 1)
        out[f"tr_{met}"] = 100 * (s[f"{met}_ood"] / r[f"{met}_ood"] - 1)
    return pd.DataFrame(out).groupby(["candidate", "model"]).mean().reset_index()


def agree(c):
    return {met: float(np.mean(same_direction(c[f"idood_{met}"], c[f"tr_{met}"]))) for met in ("rmse", "mae")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--sensitivity-run-ids", default=SENSITIVITY_RUNS)
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    main_rows = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    main_rows = main_rows[main_rows.regime.isin(["ID_match", "OOD_match"]) & main_rows.repeat.isin(REPS)]
    sensitivity = pd.concat([pd.read_csv(os.path.join(config.RESULTS_DIR, "sensitivity", "runs", r, "rows.csv")) for r in a.sensitivity_run_ids.split(",")])
    sensitivity = sensitivity[(sensitivity.status == "ok") & sensitivity.arm.isin(ARMS) & sensitivity.regime.isin(["ID_match", "OOD_match"])]
    if not set(sensitivity.repeat.unique()) <= set(REPS):
        raise SystemExit(f"the sensitivity runs contain repetitions {sorted(set(sensitivity.repeat.unique()) - set(REPS))} (only the repetitions 0-2 of the main result are compared)")
    sensitivity = sensitivity[sensitivity.repeat.isin(REPS)]
    if sensitivity.duplicated(["arm"] + KEYS + ["regime"]).any():
        raise SystemExit("the same arm, cell, repetition and regime occurs twice")
    base = cells(main_rows)
    base_all = agree(base)
    log(f"[pipeline_rq1] main result (repetitions 0-2, 22 models) direction agreement RMSE {base_all['rmse']:.3f} MAE {base_all['mae']:.3f}")
    summ, allc = [dict(arm="main", scope="roster", n_cells=len(base), **{f"agree_{k}": v for k, v in base_all.items()})], []
    for arm in ARMS:
        ea = sensitivity[sensitivity.arm == arm]
        if ea.empty:
            raise SystemExit(f"no rows for the arm {arm}")
        c = cells(ea)
        key = c.set_index(["candidate", "model"]).index
        b = base.set_index(["candidate", "model"]).loc[key].reset_index()
        roster = pd.concat([base.set_index(["candidate", "model"]).drop(key), c.set_index(["candidate", "model"])]).reset_index()
        doms = c.candidate.unique()
        roster = roster[roster.candidate.isin(doms)]
        base_d = base[base.candidate.isin(doms)]
        for scope, x, ref in (("rerun", c, b), ("roster", roster, base_d)):
            ag, ar = agree(x), agree(ref)
            summ.append(dict(arm=arm, scope=scope, n_cells=len(x), n_domains=len(doms),
                             agree_rmse=ag["rmse"], agree_mae=ag["mae"], main_agree_rmse=ar["rmse"], main_agree_mae=ar["mae"]))
        c = c.merge(b, on=["candidate", "model"], suffixes=("", "_main"))
        c.insert(0, "arm", arm)
        allc.append(c)
    pd.concat(allc).to_csv(os.path.join(out, "pipeline_rq1_cells.csv"), index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(out, "pipeline_rq1_summary.csv"), index=False)
    print(s.round(3).to_string(index=False))
    log("[pipeline_rq1] done")


if __name__ == "__main__":
    main()
