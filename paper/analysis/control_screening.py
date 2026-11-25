"""Applies the screening verdict rule of the challenge domains to the 10 randomized lot-split controls: the number passing and their ranks,
and the change per domain in the selection audit (screening score -> rescoring on B_te). It reads only saved results (no retraining, CPU only).

Screening rule (defectshift/selection.py, using the columns of the screening summary.csv as they are; the 4 screening models = XGBoost, TabPFN, MLP, TabM)
  consistency  n_models_consistent >= 3 (per model the repetition-mean RMSE target-reference gap > 0 and positive in at least 2 of 3 repetitions)
  eligibility  b_te_eligible (number of target test rows floor(0.25 |B|) >= 500)
  overall      complete & consistency & eligibility
The controls were not subject to the selection, so passes is recorded as False for all of them in summary.csv. Here the rule is only recomputed from the same columns;
the controls are not entered as selection candidates, and the existing selection, control range and verdict criteria are not changed (a diagnostic of the selection asymmetry).
Ranking: the 16 process-condition candidates that passed and the controls that passed, in descending order of the screening consensus score (consensus_d_rmse_pct, median of the 4 models).
Audit: shrink_pct in candidates.csv of the selection audit = screening consensus score - B_te consensus score (RMSE, the same 4 models and the same saved models).
Output: results/analysis/<run-id>/{ctrl_screen.csv, ctrl_screen_rank.csv, screen_audit.csv}
Usage: python paper/analysis/control_screening.py --run-id control_screening
"""
import argparse
import os


import pandas as pd  # noqa: E402

from defectshift import config  # noqa: E402
from defectshift.utils import log  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--screening-run-id", default="screening_main")
    ap.add_argument("--selection-audit-run-id", default="selection_audit_main")
    a = ap.parse_args()
    s = pd.read_csv(os.path.join(config.RESULTS_DIR, "screening", "runs", a.screening_run_id, "selection", "summary.csv"))
    cat, ctl = s[s.kind == "cat"].copy(), s[s.slot.str.startswith("placebo")].copy()
    if len(cat) != 48 or len(ctl) != 10:
        raise SystemExit(f"the number of candidates is not 48 + 10: {len(cat)}, {len(ctl)}")
    for t in (cat, ctl):
        t["consistent"] = t.n_models_consistent >= 3
        t["eligible"] = t.b_te_eligible.astype(bool)
        t["overall"] = t.complete.astype(bool) & t.consistent & t.eligible
    if not (cat.overall == cat.passes.astype(bool)).all():
        raise SystemExit("the recomputed pass flags differ from passes in summary.csv (process-condition candidates)")
    rows = []
    for grp, t in (("process", cat), ("control", ctl)):
        rows.append(dict(group=grp, n=len(t), eligible=int(t.eligible.sum()), consistent=int(t.consistent.sum()), overall=int(t.overall.sum()),
                         passed=",".join(t[t.overall].candidate)))
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "ctrl_screen.csv"), index=False)
    pool = pd.concat([cat[cat.overall].assign(group="process"), ctl[ctl.overall].assign(group="control")])
    pool = pool.sort_values(["consensus_gap_rmse_pct", "candidate"], ascending=[False, True]).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    rk = pool[["rank", "group", "candidate", "consensus_gap_rmse_pct", "n_models_consistent", "selected"]]
    rk.to_csv(os.path.join(out, "ctrl_screen_rank.csv"), index=False)
    au = pd.read_csv(os.path.join(config.RESULTS_DIR, "selection_audit", "runs", a.selection_audit_run_id, "report", "candidates.csv"))
    au = au[(au.kind == "cat") | (au.kind == "placebo")][["candidate", "kind", "selected", "consensus_gap_rmse_pct_screen", "consensus_gap_rmse_pct_te", "shrink_pct"]]
    au.to_csv(os.path.join(out, "screen_audit.csv"), index=False)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    print(rk.round(2).to_string(index=False))
    sel = au[(au.kind == "cat") & au.selected]
    print("\nChange of the 5 selected domains from screening to B_te (RMSE, percentage points):\n", sel.round(2).to_string(index=False))
    print(f"mean decrease of the 5 selected {sel.shrink_pct.mean():.2f}, mean of the non-selected process-condition candidates {au[(au.kind == 'cat') & ~au.selected].shrink_pct.mean():.2f}, "
          f"mean of the 10 controls {au[au.kind == 'placebo'].shrink_pct.mean():.2f}")
    log("[ctrl_screen] done")


if __name__ == "__main__":
    main()
