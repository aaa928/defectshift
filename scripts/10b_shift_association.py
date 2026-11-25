"""Exploratory association between the shift statistics (10a_shift_statistics) and the loss (screening consensus target-reference gap), without
new training. Only "association" is claimed; no causal wording.

Inputs: the screening selection table (results/screening/runs/<screening>/selection/summary.csv, consensus target-reference gap (RMSE) on B_screen)
and shift_stats.csv of the shift statistics. With --selection-audit-run-id the same analysis is also run, as a supplement, on the
B_te consensus target-reference gap of the selection audit (association on another evaluation set after the selection).
Analysed over all 78 candidates (using only the 7 selected would restrict the range through the selection). Methods sentence: The
association analysis was performed over all 78 candidate splits.

- Temporal splits: per axis (x_time, y_time), Spearman rho over the 15 cutoffs only (descriptive). The cutoffs define overlapping
  splits, not independent samples, so no confidence interval or p value is given.
- Process-condition splits (48): Spearman rho + bootstrap 95% interval (cluster = categorical column; one candidate per column, so
  candidates are resampled) + permutation p value, and next to them the n*-controlled partial rank correlation (correlation of the
  residuals after linearly regressing the ranks on the rank of n*) with its bootstrap interval. Effect sizes and intervals first; p
  values are not reported alone.
- Statistics using X only and oracle statistics using Y are reported as separate stages of the tables.

Output: results/shift_association/<run_id>/{assoc_categorical.csv, assoc_temporal.csv, summary.json}
Usage: python scripts/10b_shift_association.py --screening-run-id <screening> --shift-statistics-run-id <run> --run-id <run> [--selection-audit-run-id <run>]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from defectshift import config  # noqa: E402

X_ONLY = ("domain_auc", "wasserstein_mean", "mmd_rbf", "mean_shift_smd", "scale_shift", "missing_shift_mean",
          "missing_shift_max", "cat_tvd_mean", "cat_tvd_max", "n_star")
Y_ORACLE = ("dmean_y", "abs_dmean_y", "var_ratio_y", "dpos_y", "resid_shift", "rmse_ratio")
SHIFT_DIAG_X = ("feature_ot_excess", "feature_ot_excess_num", "feature_ot_excess_cat")   # only with --shift-diag-run-id
SHIFT_DIAG_Y = ("disde_yx_abs_pct", "disde_yx_sq_pct")     # DISDE uses loss information, so it enters the Y stage (it shares part of the information with the loss; stated in the interpretation)
SEED = 2027


def partial_rank_corr(x, y, z):
    """Pearson correlation of the residuals after linearly regressing the ranks of x and y on the rank of z (n*-controlled partial rank correlation)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / den) if den > 0 else np.nan


def assoc(x, y, z, B, rng, control):
    """Spearman rho, bootstrap 95% interval, permutation p (two-sided), n*-controlled partial rank correlation and its interval."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    n = len(x)
    if n < 4:
        return dict(n=n, note="fewer than 4 valid candidates")
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:      # the rank correlation of a constant quantity is undefined -> empty value with the reason
        return dict(n=n, note="constant values (rank correlation undefined)")
    rho = float(spearmanr(x, y).correlation)
    boots, pboots = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) > 1 and len(np.unique(y[i])) > 1:
            boots.append(spearmanr(x[i], y[i]).correlation)
            if control:
                pboots.append(partial_rank_corr(x[i], y[i], z[i]))
    perm = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(B)])
    ci = (lambda v, q: float(np.nanpercentile(v, q)) if len(v) else np.nan)     # empty value if no bootstrap sample varies
    out = dict(n=n, spearman=rho, ci_lo=ci(boots, 2.5), ci_hi=ci(boots, 97.5), n_boot_used=len(boots),
               perm_p=float((np.sum(np.abs(perm) >= abs(rho)) + 1) / (B + 1)))
    if control:
        out.update(partial_nstar=partial_rank_corr(x, y, z), partial_ci_lo=ci(pboots, 2.5), partial_ci_hi=ci(pboots, 97.5))
    return out


def load_targets(screening_run_id, selection_audit_run_id):
    s = pd.read_csv(os.path.join(config.RESULTS_DIR, "screening", "runs", screening_run_id, "selection", "summary.csv"))
    t = s[["candidate", "kind", "consensus_gap_rmse_pct", "b_te_eligible"]].rename(columns={"consensus_gap_rmse_pct": "target_reference_gap_screen"})
    if selection_audit_run_id:
        c = pd.read_csv(os.path.join(config.RESULTS_DIR, "selection_audit", "runs", selection_audit_run_id, "report", "candidates.csv"))
        t = t.merge(c[["candidate", "consensus_gap_rmse_pct_te"]].rename(columns={"consensus_gap_rmse_pct_te": "target_reference_gap_te"}),
                    on="candidate", how="left")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--screening-run-id", required=True)
    ap.add_argument("--shift-statistics-run-id", required=True)
    ap.add_argument("--selection-audit-run-id", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--shift-diag-run-id", default=None,
                    help="attach the three-shift diagnosis summary (association_input.csv of 10d_shift_diag_summary.py): feature OT enters the X-only stage, DISDE the Y stage")
    args = ap.parse_args()
    if args.boot < 1:
        raise SystemExit("--boot must be at least 1")
    stats = pd.read_csv(os.path.join(config.RESULTS_DIR, "shift_statistics", args.shift_statistics_run_id, "shift_stats.csv"))
    stats["abs_dmean_y"] = stats.dmean_y.abs()
    x_only, y_oracle = X_ONLY, Y_ORACLE
    if args.shift_diag_run_id:   # listed candidates only (the random-split controls are not part of the association); a missing candidate stops the run
        sd = pd.read_csv(os.path.join(config.RESULTS_DIR, "shift_diag", args.shift_diag_run_id, "summary", "association_input.csv"))
        sd = sd[~sd.candidate.str.startswith("placebo")]
        miss = sorted(set(stats.candidate) - set(sd.candidate))
        if miss:
            raise SystemExit(f"candidates missing from the shift diagnosis summary: {miss[:3]}")
        stats = stats.merge(sd, on="candidate", how="left")
        x_only = X_ONLY + SHIFT_DIAG_X
        y_oracle = Y_ORACLE + SHIFT_DIAG_Y
    tg = load_targets(args.screening_run_id, args.selection_audit_run_id)
    for name_, frame in (("screening selection summary", tg), ("shift statistics", stats)):
        dup = frame.candidate[frame.candidate.duplicated()].tolist()
        if dup:
            raise SystemExit(f"the same candidate twice in {name_}: {dup[:3]}")
    official = set(tg.candidate[tg.kind != "placebo"])        # the 10 random-split controls are not in the shift statistics (by design)
    only_screening, only_shift_statistics = sorted(official - set(stats.candidate)), sorted(set(stats.candidate) - official)
    if only_screening or only_shift_statistics:      # so that an inner merge does not silently hide missing listed candidates
        raise SystemExit(f"the listed screening candidates and the shift-statistics candidates differ (screening only {only_screening[:3]}, shift statistics only {only_shift_statistics[:3]})")
    d = tg.merge(stats, on="candidate", how="inner")
    targets = ["target_reference_gap_screen"] + (["target_reference_gap_te"] if "target_reference_gap_te" in d else [])
    rng = np.random.default_rng(SEED)
    cat_rows, time_rows = [], []
    cats = d[d.kind == "cat"]
    for tgt in targets:
        for stage, qs in (("X-only", x_only), ("Y-oracle", y_oracle)):
            for q in qs:
                r = assoc(cats[q].to_numpy(float), cats[tgt].to_numpy(float), cats.n_star.to_numpy(float), args.boot, rng,
                          control=q != "n_star")
                cat_rows.append(dict(target=tgt, stage=stage, quantity=q, **r))
                for ax in ("x_time", "y_time"):
                    t = d[d.kind == ax]
                    ok = t[[q, tgt]].notna().all(axis=1)
                    rho = float(spearmanr(t.loc[ok, q], t.loc[ok, tgt]).correlation) if ok.sum() >= 4 else np.nan
                    time_rows.append(dict(target=tgt, stage=stage, quantity=q, axis=ax, n=int(ok.sum()), spearman=rho))
    out_dir = os.path.join(config.RESULTS_DIR, "shift_association", args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(cat_rows).to_csv(os.path.join(out_dir, "assoc_categorical.csv"), index=False)
    pd.DataFrame(time_rows).to_csv(os.path.join(out_dir, "assoc_temporal.csv"), index=False)
    summary = dict(screening_run_id=args.screening_run_id, shift_statistics_run_id=args.shift_statistics_run_id, selection_audit_run_id=args.selection_audit_run_id, boot=args.boot, seed=SEED,
                   n_candidates=len(d), n_categorical=len(cats), n_temporal={ax: int((d.kind == ax).sum()) for ax in ("x_time", "y_time")},
                   note="Exploratory association. Temporal axes: descriptive statistics only (overlapping splits). Process-condition splits: "
                        "candidate-level bootstrap and permutation, with the n*-controlled partial rank correlation alongside")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
