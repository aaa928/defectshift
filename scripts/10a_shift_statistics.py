"""Shift statistics of the candidate domain splits (CPU; the only new training is the auxiliary models of the statistics).

Rows: A_dev, B_tr and B_screen (splits.B_view(..., "shift_statistics")). A_heldout and B_te are not used.
Features: the same set as the benchmark (candidates.feature_sets: without the production-lot column and the column defining a
process-condition split).
Candidates: the 78 listed (+ the 10 random-split controls with --placebo).

Statistics using X only (observable at deployment without labels)
- domain_auc: out-of-sample ROC-AUC of a domain classifier (A=0, B=1); out-of-fold predictions of 5 folds stratified and grouped by
  production lot (StratifiedGroupKFold, fixed seed). Classifier = HistGradientBoostingClassifier (fixed settings, categorical columns
  declared categorical, missing values kept). The same classifier, settings and seed for every candidate
- wasserstein_mean: mean over numerical features of the 1-Wasserstein distance (observed values only; numerical features are
  normalized to [0, 1] at the dataset level)
- mmd_rbf: RBF-kernel MMD^2 of the numerical features (median imputation and standardization over the pooled rows; unbiased estimate,
  at most 1,000 random rows per side, bandwidth = median distance)
- mean_shift_smd: mean over numerical features of |mean difference| / pooled standard deviation; scale_shift: mean of |log(sd_B / sd_A)|
  (observed values only)
- missing_shift_mean, max: mean and maximum over missing groups (numerical columns with exactly the same missing positions in the whole
  data; 152 columns -> 29 groups) of |missing rate_B - missing rate_A|
- cat_tvd_mean, max: mean and maximum over categorical features of the total variation distance between the A and B distributions
- n_star: n* of repetition 0 (the benchmark rule)
Statistics using Y (diagnostic oracle, reported separately in two parts)
- dmean_y = mean(y_B) - mean(y_A), var_ratio_y = Var(y_B)/Var(y_A), dpos_y = P_B(y>0) - P_A(y>0)
- resid_shift: mean residual on B of a HistGradientBoostingRegressor trained on the A rows minus the mean of its 5-fold lot-level
  out-of-fold residuals within A; rmse_ratio = RMSE(B) / RMSE(A out-of-fold) — a proxy for a shift of the conditional residual (same
  settings and seed)
Per-feature table (feature_stats.csv): candidate x feature Wasserstein, SMD, log sd ratio, missing rates A/B (numerical) and TVD
(categorical) — a public table complementing the anonymized features.

Output: results/shift_statistics/<run_id>/{manifest.json, shift_stats.csv, feature_stats.csv, missing_groups.csv}. Rerunning with the
same run-id skips finished candidates.
Usage: python scripts/10a_shift_statistics.py --run-id <run> [--placebo] [--threads 1]
"""
import argparse
import os
import sys

if "--threads" in sys.argv:   # fix the number of threads before importing sklearn and numpy (when sharing the CPU with a training run)
    _n = sys.argv[sys.argv.index("--threads") + 1]
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_k] = _n

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

from defectshift import config, manifest  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_candidates, load_data  # noqa: E402
from defectshift.runio import append_row  # noqa: E402
from defectshift.shift_stats import HGB, MMD_MAX_PER_SIDE, N_FOLDS, SEED, missing_groups, mmd_rbf, oof_auc, resid_shift  # noqa: E402
from defectshift.splits import B_view, get_A, matched, repeat_split  # noqa: E402
from defectshift.utils import log  # noqa: E402

FEAT_COLS = ["candidate", "feature", "kind", "miss_A", "miss_B", "wasserstein", "smd", "log_sd_ratio", "tvd"]


def candidate_stats(df, y, lot, name, mgroups):
    slot, mA, mB = build_one(df, name)
    a = get_A(slot, df.index[mA], y, lot)
    bv = B_view(slot, df.index[mB], y, lot, "shift_statistics")
    assert set(bv) == {"B_tr", "B_screen"}
    A_rows, B_rows = a["A_dev"], bv["B_tr"].append(bv["B_screen"])
    rows = A_rows.append(B_rows)
    cat_cols, num_cols = feature_sets(df, name)
    feats = cat_cols + num_cols
    X = df.loc[rows, feats]
    is_b = np.r_[np.zeros(len(A_rows), bool), np.ones(len(B_rows), bool)]
    yy = y.loc[rows].to_numpy(float)
    grp = lot.loc[rows].to_numpy()
    cat_mask = [c in cat_cols for c in feats]
    rng = np.random.default_rng(SEED)
    XA, XB = df.loc[A_rows], df.loc[B_rows]

    feat_rows, ws, smd, lsd = [], [], [], []
    for c in num_cols:
        va, vb = XA[c].dropna().to_numpy(float), XB[c].dropna().to_numpy(float)
        r = dict(candidate=name, feature=c, kind="num", miss_A=float(XA[c].isna().mean()), miss_B=float(XB[c].isna().mean()))
        if len(va) > 1 and len(vb) > 1:
            pooled = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
            r.update(wasserstein=float(wasserstein_distance(va, vb)),
                     smd=float(abs(va.mean() - vb.mean()) / pooled) if pooled > 0 else np.nan,
                     log_sd_ratio=float(abs(np.log(vb.std(ddof=1) / va.std(ddof=1)))) if va.std() > 0 and vb.std() > 0 else np.nan)
            ws.append(r["wasserstein"])
            smd.append(r["smd"])
            lsd.append(r["log_sd_ratio"])
        feat_rows.append(r)
    tvd = []
    for c in cat_cols:
        pa, pb = XA[c].value_counts(normalize=True), XB[c].value_counts(normalize=True)
        t = float(0.5 * pa.subtract(pb, fill_value=0).abs().sum())
        tvd.append(t)
        feat_rows.append(dict(candidate=name, feature=c, kind="cat", tvd=t))
    gshift = [abs(XB[cols[0]].isna().mean() - XA[cols[0]].isna().mean()) for cols in mgroups.values() if cols[0] in num_cols]

    num_all = df.loc[rows, num_cols]
    med = num_all.median()
    Z = num_all.fillna(med)
    sd = Z.std(ddof=0).replace(0, 1.0)
    Z = ((Z - Z.mean()) / sd).to_numpy(float)
    a_train, _ = repeat_split(a["A_dev"], y, lot, 0)
    n_star = matched(a_train, bv["B_tr"], 0)[0]
    rs, rr = resid_shift(X, yy, is_b, grp, cat_mask)
    ya, yb = yy[~is_b], yy[is_b]
    stats = dict(candidate=name, slot=slot, n_A_dev=len(A_rows), n_B=len(B_rows), n_star=int(n_star),
                 domain_auc=oof_auc(X, is_b.astype(int), grp, cat_mask),
                 wasserstein_mean=float(np.nanmean(ws)), mmd_rbf=mmd_rbf(Z[~is_b], Z[is_b], rng),
                 mean_shift_smd=float(np.nanmean(smd)), scale_shift=float(np.nanmean(lsd)),
                 missing_shift_mean=float(np.mean(gshift)) if gshift else 0.0, missing_shift_max=float(np.max(gshift)) if gshift else 0.0,
                 cat_tvd_mean=float(np.mean(tvd)) if tvd else 0.0, cat_tvd_max=float(np.max(tvd)) if tvd else 0.0,
                 dmean_y=float(yb.mean() - ya.mean()), var_ratio_y=float(yb.var() / ya.var()) if ya.var() > 0 else np.nan,
                 dpos_y=float((yb > 0).mean() - (ya > 0).mean()), resid_shift=rs, rmse_ratio=rr)
    return stats, feat_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated candidates (smoke runs)")
    ap.add_argument("--threads", default=None, help="number of BLAS and OpenMP threads (1 when sharing the CPU with a training run)")
    args = ap.parse_args()
    names = load_candidates(include_placebo=args.placebo)
    if args.only:
        want = args.only.split(",")
        bad = [n for n in want if n not in load_candidates(include_placebo=True)]
        if bad:
            raise SystemExit(f"not candidates: {bad}")
        names = [n for n in load_candidates(include_placebo=True) if n in want]
    run_dir = os.path.join(config.RESULTS_DIR, "shift_statistics", args.run_id)
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="shift_statistics", candidates=names, seed=SEED, n_folds=N_FOLDS, hgb=HGB, mmd_max_per_side=MMD_MAX_PER_SIDE,
        rows="A_dev + B_tr + B_screen", features="candidates.feature_sets")))
    out_stats, out_feat = os.path.join(run_dir, "shift_stats.csv"), os.path.join(run_dir, "feature_stats.csv")
    done = set(pd.read_csv(out_stats).candidate) if os.path.exists(out_stats) else set()
    if os.path.exists(out_feat):   # the feature table is written before the candidate summary -> feature rows of candidates without a summary (traces of an interruption) are removed and recomputed
        f = pd.read_csv(out_feat)
        f[f.candidate.isin(done)].to_csv(out_feat, index=False)
    df, y = load_data()
    lot = df[config.LOT_COL]
    num_all = [c for c in df.columns if c.startswith("num_")]
    mgroups = missing_groups(df, num_all)
    pd.DataFrame([dict(group=g, n_cols=len(c), cols=",".join(c)) for g, c in mgroups.items()]).to_csv(
        os.path.join(run_dir, "missing_groups.csv"), index=False)
    log(f"[shift_statistics] run {args.run_id} ({status}) {len(names)} candidates ({len(done & set(names))} done), {len(mgroups)} missing groups")
    for name in names:
        if name in done:
            continue
        stats, feat_rows = candidate_stats(df, y, lot, name, mgroups)
        pd.DataFrame(feat_rows).reindex(columns=FEAT_COLS).to_csv(out_feat, mode="a", index=False, header=not os.path.exists(out_feat))
        append_row(out_stats, stats)
        log(f"[shift_statistics] {name:30s} AUC {stats['domain_auc']:.3f} W {stats['wasserstein_mean']:.4f} MMD {stats['mmd_rbf']:.4f} "
            f"missing shift {stats['missing_shift_mean']:.3f} TVD {stats['cat_tvd_mean']:.3f} n* {stats['n_star']}")
    log("[shift_statistics] done")


if __name__ == "__main__":
    main()
