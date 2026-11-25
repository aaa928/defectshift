"""Per-candidate summary of the results of the three-fold shift diagnosis (10c_shift_diagnostics.py), without retraining. Only successful rows
(status == "ok") are read - if a successful row with the same key follows a failed row (a resumed run), only the successful row remains, and
more than one successful row with the same key stops the run.

- Feature OT (otdd.csv): the difference to its own baseline (excess; all columns, numerical only, categorical only) as mean and standard deviation
  over the repetitions, the position among the 10 randomized lot-split controls (number of controls with a smaller value / number of controls,
  described without a verdict) and the sample limits (sample_limited, baseline_size_limited).
- DISDE (disde.csv): main quantity = Y|X term on the absolute-error basis in percent (abs_yx_shift_pct), supplement = squared-error basis
  (yx_shift_pct) and the ratio to the total difference. Per candidate the mean, the standard deviation, the number of positive repetitions
  and the position among the randomized controls. The sign label is given per component: "unstable" if the range over the repetitions
  (minimum to maximum, the empirical range of variation) contains 0, otherwise "positive" or "negative". The ratio to the total difference is
  "unstable" if the range of the total difference (total_gap) over the repetitions contains 0 and "undefined" if it is exactly 0 or contains
  NaN. Mean over the repetitions of the domain-classifier AUC and of the effective-sample-size ratio on the target side.
- Retuning comparison (--tuned-run-id): per (candidate, repetition) pair the mean difference of abs_yx_shift_pct between the default and the retuned
  evaluation model and the number of equal signs.
Output: results/shift_diag/<run>/summary/{candidates.csv, association_input.csv}. association_input.csv holds the per-candidate columns
(feature_ot_excess, _num, _cat, disde_yx_abs_pct, disde_yx_sq_pct) that 10b_shift_association.py --shift-diag-run-id reads.
Usage: python scripts/10d_shift_diag_summary.py --run-id <run> [--tuned-run-id <run>]
"""
import argparse
import os

import numpy as np
import pandas as pd

from defectshift import config  # noqa: E402
from defectshift.directions import same_direction  # noqa: E402


def ok_rows(path, key):
    t = pd.read_csv(path)
    if "status" in t:
        t = t[t.status == "ok"]
    dup = t.duplicated(key, keep=False)
    if dup.any():
        raise SystemExit(f"{path}: more than one successful row with the same key {t[dup][key].head(3).values.tolist()}")
    return t


def sign_label(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    return "unstable" if lo <= 0 <= hi else ("positive" if lo > 0 else "negative")


def placebo_pos(v, ref):
    """Number of controls with a value smaller than v / number of controls (the control itself is not left out - the control rows are described including their own value)."""
    ref = np.asarray(ref, float)
    return float((ref < v).sum() / len(ref)) if len(ref) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tuned-run-id", default=None)
    args = ap.parse_args()
    d = os.path.join(config.RESULTS_DIR, "shift_diag", args.run_id)
    o = ok_rows(os.path.join(d, "otdd.csv"), ["candidate"]).set_index("candidate")
    q = ok_rows(os.path.join(d, "disde.csv"), ["candidate", "repeat"])
    is_pl = lambda idx: pd.Index(idx).str.startswith("placebo")
    g = q.groupby("candidate")
    s = pd.DataFrame(dict(
        disde_yx_abs_pct=g.abs_yx_shift_pct.mean(), disde_yx_abs_pct_sd=g.abs_yx_shift_pct.std(),
        disde_yx_abs_pos=g.abs_yx_shift_pct.apply(lambda v: int((v > 0).sum())), disde_reps=g.size(),
        disde_yx_abs_lo=g.abs_yx_shift_pct.min(), disde_yx_abs_hi=g.abs_yx_shift_pct.max(),
        disde_yx_sq_pct=g.yx_shift_pct.mean(), disde_yx_sq_pct_sd=g.yx_shift_pct.std(),
        disde_yx_sq_lo=g.yx_shift_pct.min(), disde_yx_sq_hi=g.yx_shift_pct.max(),
        total_gap_lo=g.total_gap.min(), total_gap_hi=g.total_gap.max(), abs_total_gap_lo=g.abs_total_gap.min(),
        abs_total_gap_hi=g.abs_total_gap.max(), yx_ratio_mean=g.yx_ratio.mean(),
        prop_auc=g.prop_auc.mean(), ess_tgt_frac=g.ess_tgt_frac.mean(), ess_src_frac=g.ess_src_frac.mean()))
    s["disde_yx_abs_sign"] = [sign_label(a, b) for a, b in zip(s.disde_yx_abs_lo, s.disde_yx_abs_hi)]
    s["disde_yx_sq_sign"] = [sign_label(a, b) for a, b in zip(s.disde_yx_sq_lo, s.disde_yx_sq_hi)]
    gap_any_bad = g.total_gap.apply(lambda v: bool((~np.isfinite(v)).any() or (v == 0).any()))
    s["ratio_status"] = ["undefined" if bad else ("unstable" if lo <= 0 <= hi else "defined")
                         for bad, lo, hi in zip(gap_any_bad, s.total_gap_lo, s.total_gap_hi)]
    cols = ["n_src", "n_tgt", "n_used", "sample_limited", "n_base", "baseline_size_limited", "ot_all_mean", "base_all_mean",
            "excess_all_mean", "excess_all_sd", "excess_num_mean", "excess_cat_mean"]
    t = o[cols].rename(columns=dict(excess_all_mean="feature_ot_excess", excess_num_mean="feature_ot_excess_num",
                                    excess_cat_mean="feature_ot_excess_cat", excess_all_sd="feature_ot_excess_sd")).join(s, how="outer")
    pl = t[is_pl(t.index)]
    t["placebo"] = is_pl(t.index)
    for c in ("feature_ot_excess", "disde_yx_abs_pct", "disde_yx_sq_pct"):
        t[f"{c}_placebo_pos"] = [placebo_pos(v, pl[c]) for v in t[c]]
    if args.tuned_run_id:
        u = ok_rows(os.path.join(config.RESULTS_DIR, "shift_diag", args.tuned_run_id, "disde.csv"), ["candidate", "repeat"])
        m = q.merge(u, on=["candidate", "repeat"], suffixes=("", "_tuned"))
        for c in ("src_train_sha", "src_eval_sha", "tgt_sha", "fold_sha"):
            if not (m[c] == m[f"{c}_tuned"]).all():
                raise SystemExit(f"{c} of the default and retuned pair differs")
        mg = m.groupby("candidate")
        t = t.join(pd.DataFrame(dict(tuned_yx_abs_pct=mg.abs_yx_shift_pct_tuned.mean(),
                                     tuned_minus_default=mg.apply(lambda x: (x.abs_yx_shift_pct_tuned - x.abs_yx_shift_pct).mean(), include_groups=False),
                                     tuned_sign_agree=mg.apply(lambda x: int(same_direction(x.abs_yx_shift_pct_tuned, x.abs_yx_shift_pct).sum()),
                                                              include_groups=False),
                                     tuned_pairs=mg.size())))
    out = os.path.join(d, "summary")
    os.makedirs(out, exist_ok=True)
    t.index.name = "candidate"
    t.reset_index().to_csv(os.path.join(out, "candidates.csv"), index=False)
    t.reset_index()[["candidate", "feature_ot_excess", "feature_ot_excess_num", "feature_ot_excess_cat", "disde_yx_abs_pct",
                     "disde_yx_sq_pct"]].to_csv(os.path.join(out, "association_input.csv"), index=False)
    print(f"{len(t)} candidates ({int(t.placebo.sum())} controls) -> {out}")


if __name__ == "__main__":
    main()
