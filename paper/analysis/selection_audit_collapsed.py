"""Selection audit without the collapsed cells (supplementary S3, column Test* of Table S4). It reads only the saved selection audit
(scripts/03a_audit_selection.py; no retraining, CPU only).

A candidate-by-model cell is collapsed when the repetition mean of the target R^2 on B_te of the source-trained model (ID_match) or of the
reference (OOD_match) is below -1, the definition of the benchmark. Test* is the consensus score on B_te (median over the four screening
models of the RMSE target-reference gap) without the collapsed models; it is empty if no model is left. The Spearman correlation, the sign
agreement and the mean change from screening to B_te are given with all cells and without the collapsed cells, for the eligible
process-condition candidates (the same set as in 03a).
Output: results/analysis/<run-id>/{audit_collapsed.csv, audit_collapsed_summary.json}
Usage: python paper/analysis/selection_audit_collapsed.py --run-id selection_audit_collapsed [--selection-audit-run-id selection_audit_main]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from defectshift import config
from defectshift.utils import log

COLLAPSE_R2 = -1.0
MODELS = ["XGBoost", "TabPFN", "MLP", "TabM"]


def stats(frame, te_col):
    f = frame[np.isfinite(frame[te_col])]
    scr = f.consensus_gap_rmse_pct_screen
    return dict(n=int(len(f)), spearman=float(spearmanr(scr, f[te_col]).correlation),
                sign_agreement=float((np.sign(scr) == np.sign(f[te_col])).mean()),
                mean_fall_selected=float((scr - f[te_col])[f.selected].mean()),
                mean_fall_not_selected=float((scr - f[te_col])[~f.selected].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--selection-audit-run-id", default="selection_audit_main")
    a = ap.parse_args()
    base = os.path.join(config.RESULTS_DIR, "selection_audit", "runs", a.selection_audit_run_id)
    c = pd.read_csv(os.path.join(base, "report", "candidates.csv"))
    rows = pd.read_csv(os.path.join(base, "rows.csv"))
    rows = rows[rows.status == "ok"] if "status" in rows else rows
    r2 = rows.groupby(["candidate", "model", "regime"]).r2_te.mean().unstack("regime")
    collapsed = (r2[["ID_match", "OOD_match"]] < COLLAPSE_R2).any(axis=1)
    te_star, which = [], []
    for r in c.itertuples():
        bad = [m for m in MODELS if bool(collapsed.get((r.candidate, m), False))]
        ok = [getattr(r, f"{m}_gap_rmse_pct_te") for m in MODELS if m not in bad]
        te_star.append(float(np.median(ok)) if ok else np.nan)
        which.append(",".join(bad))
    c["consensus_gap_rmse_pct_te_excl_collapsed"] = te_star
    c["models_collapsed"] = which
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    keep = ["candidate", "kind", "selected", "b_te_eligible", "consensus_gap_rmse_pct_screen", "consensus_gap_rmse_pct_te",
            "consensus_gap_rmse_pct_te_excl_collapsed", "models_collapsed"]
    c[keep].to_csv(os.path.join(out, "audit_collapsed.csv"), index=False)
    elig = c[(c.kind == "cat") & c.b_te_eligible.astype(bool)]
    summ = dict(selection_audit_run_id=a.selection_audit_run_id, collapse_r2=COLLAPSE_R2,
                n_collapsed_cells_process_condition=int(sum(len(x.split(",")) for x in c[c.kind == "cat"].models_collapsed if x)),
                n_candidates_with_collapsed_cells=int((c[c.kind == "cat"].models_collapsed != "").sum()),
                all_cells=stats(elig, "consensus_gap_rmse_pct_te"),
                without_collapsed_cells=stats(elig, "consensus_gap_rmse_pct_te_excl_collapsed"))
    with open(os.path.join(out, "audit_collapsed_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))
    log("[audit_collapsed] done")


if __name__ == "__main__":
    main()
