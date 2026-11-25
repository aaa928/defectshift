"""Retention of the sign for the uniform-preprocessing variant (training-row median imputation + training-row standardization added before the own preprocessing of each model), recounted
only for the 9 models that receive the released scale as it is (RealMLP, TabM, TabNet, 6 foundation models). It reads only saved results (CPU only).
Input: results/analysis/sensitivity_lot_disjoint/pipeline_disjoint_cells.csv (arm = uniform; cell = model x domain,
      repetition mean of the first 3 repetitions, lot-disjoint target test rows; the main result to compare with is the same cell and the same 3 repetitions)
      results/analysis/lot_disjoint/disjoint_cells.csv (marking of collapsed cells based on the 5 repetitions of the main result: repetition-mean R^2 < -1)
Sign retention = the sign of the target-reference gap of the variant equals that of the main result. It is checked that no cell has sign 0. The values with and without the collapsed cells are both given.
Scope of interpretation: this variant also changes the handling of missing values, so it is not an experiment that isolates the effect of the released scaling.
Output: results/analysis/<run-id>/uniform9.csv
Usage: python paper/analysis/uniform_preprocessing_signs.py --run-id uniform_preprocessing_signs
"""
import argparse
import os


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from defectshift import config  # noqa: E402
from defectshift.utils import log  # noqa: E402
from defectshift.directions import same_direction  # noqa: E402

NINE = ("RealMLP", "TabM", "TabNet", "TabPFN", "TabICL", "TabFM", "Kumo", "TabDPT", "EXAONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    c = pd.read_csv(os.path.join(rf, "sensitivity_lot_disjoint", "pipeline_disjoint_cells.csv"))
    c = c[c.arm == "uniform"]
    dc = pd.read_csv(os.path.join(rf, "lot_disjoint", "disjoint_cells.csv"))
    coll = dc[(dc.variant == "disjoint") & (dc.metric == "rmse")].set_index(["candidate", "model"]).collapsed
    c = c.assign(collapsed=[bool(coll.get((d, m), False)) for d, m in zip(c.candidate, c.model)])
    for met in ("rmse", "mae"):
        if ((c[f"tr_{met}"] == 0) | (c[f"tr_{met}_main"] == 0)).any():
            raise SystemExit(f"{met}: there are cells with sign 0")
        if not (c[f"tr_{met}_same_sign"] == (same_direction(c[f"tr_{met}"], c[f"tr_{met}_main"]))).all():
            raise SystemExit(f"{met}: the same_sign column differs from the recomputed value")
    rows = []
    for scope, t in (("14 models", c), ("9 models", c[c.model.isin(NINE)]), ("5 tree models", c[~c.model.isin(NINE)])):
        for tag, tt in (("all", t), ("no collapsed", t[~t.collapsed])):
            rows.append(dict(scope=scope, cells=tag, n_cells=len(tt), n_models=tt.model.nunique(), n_domains=tt.candidate.nunique(),
                             n_collapsed=int(tt.collapsed.sum()), sign_kept_rmse=float(tt.tr_rmse_same_sign.mean()),
                             sign_kept_mae=float(tt.tr_mae_same_sign.mean())))
    nine = c[c.model.isin(NINE)]
    if sorted(nine.model.unique()) != sorted(NINE):
        raise SystemExit(f"not all 9 models are present: {sorted(nine.model.unique())}")
    if c[~c.model.isin(NINE)].model.nunique() != 5:
        raise SystemExit(f"the models outside the 9 are not the 5 (tree) models: {sorted(c[~c.model.isin(NINE)].model.unique())}")
    R = pd.DataFrame(rows)
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    R.to_csv(os.path.join(out, "uniform9.csv"), index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))
    print("\ncells whose sign changed for the 9 models:\n", nine[~nine.tr_rmse_same_sign | ~nine.tr_mae_same_sign][
        ["candidate", "model", "tr_rmse", "tr_rmse_main", "tr_mae", "tr_mae_main", "collapsed"]].round(2).to_string(index=False))
    log("[uniform9] done")


if __name__ == "__main__":
    main()
