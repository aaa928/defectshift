"""Diagnostics of the final benchmark (no training): collapse cells and the largest-error rows. Inputs = report/per_fit.csv and
target_reference_gap.csv of 06c_aggregate and the saved predictions and training rows (per_fit.pred_path; for a merged run, the files of the
source run).

A. Collapse cells: (domain, model, regime ID_match or OOD_match) with repetition-mean B_te R^2 < COLLAPSE_R2 (-1).
   For every collapse cell, the TOP_ROWS rows with the largest B_te error per repetition, and for each such row the numerical feature
   farthest outside the training rows (distance outside the training-row range / training-row range; standardized distance
   |x - mean| / max(standard deviation, 1e-3) — the same formula as the benchmark's standardization floor), the training-row
   statistics of that column, and the RMSE without the top DROP_ROWS rows.
B. Sensitivity to excluding collapse cells: share of positive target-reference gaps (RMSE, MAE, R^2 difference) and the domain, family and model median gap,
   all vs excluding.
C. Error concentration:
   - per model: the share of each fit's sum of squared B_te errors taken by its top 1% rows (absolute-error order = squared-error order).
   - common rows (the same rows for every model, top 1% per domain), with three criteria depending on whether the row selection is
     independent of the predictions of the two fits compared:
     y_extreme    — independent of predictions: B_te rows with the largest |y - median of y on B_tr| (ties by row number)
     cross_repeat — the repetitions that select and the repetitions that evaluate are separated: selected by the median absolute error
                    over the even repetitions (0, 2, 4) and evaluated on the odd ones (1, 3), and the reverse, averaged
     same_fits    — descriptive: selected by the median absolute error of the very fits evaluated (all models, ID_match, OOD_match,
                    repetitions). Rows where the source model is badly wrong are likely to be chosen, which inflates the loss on the top
                    rows and deflates it on the others, so it is not used as evidence that "the loss is concentrated in a few rows"
     For those rows and the remaining rows, the RMSE of ID_match and OOD_match per model and the target-reference gap (repetition mean).
     Main criterion = y_extreme, secondary = cross_repeat; same_fits is descriptive.
Output: results/benchmark/runs/<run>/diag/{collapse_cells.csv, collapse_rows.csv, collapse_exclusion.json, error_share_per_fit.csv,
        error_rows_common.csv, error_common_by_model.csv, summary.json}
Usage: python scripts/06d_diagnostics.py --benchmark-run-id <run> [--out <folder>]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from defectshift import config, metrics  # noqa: E402
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, feature_sets, load_data  # noqa: E402
from defectshift.splits import B_view  # noqa: E402

COLLAPSE_R2 = -1.0
TOP_ROWS, DROP_ROWS = 5, 2
TOP_FRAC = 0.01
REGIMES = ("ID_match", "OOD_match")
Z_FLOOR = 1e-3


def _prefix(pred_path):
    assert pred_path.endswith(".pred.npz"), pred_path
    return pred_path[: -len(".pred.npz")]


def _rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def collapse_cells(pf):
    m = pf[pf.regime.isin(REGIMES)].groupby(["candidate", "model", "regime"]).r2_ood.mean()
    return m[m < COLLAPSE_R2].rename("r2_ood_mean").reset_index()


def collapse_rows(pf, cells, df, y):
    """Per collapse cell, the largest-error rows per repetition and the farthest-outlying feature."""
    out = []
    for c in cells.itertuples(index=False):
        num_cols = feature_sets(df, c.candidate)[1]
        g = pf[(pf.candidate == c.candidate) & (pf.model == c.model) & (pf.regime == c.regime)].sort_values("repeat")
        for r in g.itertuples(index=False):
            z = np.load(r.pred_path)
            tr = np.load(_prefix(r.pred_path) + ".npz")
            fit = np.concatenate([tr["fit_idx"], tr["es_idx"]]) if len(tr["es_idx"]) else tr["fit_idx"]
            idx, p = z["ood_idx"], z["ood_pred"]
            e = p - y.loc[idx].to_numpy()
            order = np.argsort(-np.abs(e))
            X = df.loc[fit, num_cols]
            lo, hi, mu, sd = X.min(), X.max(), X.mean(), X.std(ddof=0)
            rng = (hi - lo).where(hi > lo)
            keep = np.ones(len(e), bool)
            keep[order[:DROP_ROWS]] = False
            for k in order[:TOP_ROWS]:
                row = df.loc[idx[k], num_cols]
                outside = ((row - hi).clip(lower=0) + (lo - row).clip(lower=0)) / rng
                zz = (row - mu).abs() / sd.clip(lower=Z_FLOOR)
                col = outside.idxmax() if outside.notna().any() and outside.max() > 0 else zz.idxmax()
                out.append(dict(candidate=c.candidate, model=c.model, regime=c.regime, repeat=int(r.repeat), row=int(idx[k]),
                                y=float(y.loc[idx[k]]), pred=float(p[k]), abs_err=float(abs(e[k])), feature=col, value=float(row[col]),
                                train_min=float(lo[col]), train_max=float(hi[col]), train_mean=float(mu[col]), train_std=float(sd[col]),
                                outside_range_ratio=float(outside[col]) if pd.notna(outside[col]) else np.nan, z_floored=float(zz[col]),
                                rmse_all=_rmse(e), rmse_without_top=_rmse(e[keep])))
    return pd.DataFrame(out)


def exclusion(ds, pf, cells):
    """Sensitivity to excluding collapse cells (cell = (domain, model), repetition-mean target-reference gap)."""
    cell = ds.groupby(["candidate", "model"])[["gap_rmse_pct", "gap_mae_pct", "r2_difference"]].mean()
    bad = set(zip(cells.candidate, cells.model))
    fam = pf.drop_duplicates("model").set_index("model").family
    res = {}
    for tag, c in (("all", cell), ("excluding_collapse", cell[[k not in bad for k in cell.index]])):
        c = c.reset_index().assign(family=lambda d: d.model.map(fam))
        res[tag] = dict(n_cells=len(c), positive_share={q: float((c[q] > 0).mean()) for q in ("gap_rmse_pct", "gap_mae_pct", "r2_difference")},
                        domain_median_rmse=c.groupby("candidate").gap_rmse_pct.median().to_dict(),
                        family_median_rmse=c.groupby("family").gap_rmse_pct.median().to_dict(),
                        model_median_rmse=c.groupby("model").gap_rmse_pct.median().to_dict())
    res["excluded_cells"] = sorted(f"{a}|{b}" for a, b in bad)
    return res


def error_share_per_fit(pf, y):
    rows = []
    for r in pf[pf.regime.isin(REGIMES)].itertuples(index=False):
        z = np.load(r.pred_path)
        se = np.square(z["ood_pred"] - y.loc[z["ood_idx"]].to_numpy())
        k = max(1, int(np.ceil(TOP_FRAC * len(se))))
        rows.append(dict(candidate=r.candidate, model=r.model, regime=r.regime, repeat=int(r.repeat), n=len(se), k=k,
                         share_top=float(np.sort(se)[-k:].sum() / se.sum()) if se.sum() > 0 else np.nan))
    return pd.DataFrame(rows)


def _top_mask(score, k):
    """The top k rows by score (ties: earlier rows first)."""
    m = np.zeros(len(score), bool)
    m[np.argsort(-np.asarray(score), kind="stable")[:k]] = True
    return m


def _by_model(cand, criterion, fold, preds, yt, top, reps_eval):
    out = []
    for model in sorted({m for m, _, _ in preds}):
        reps = [i for i in sorted({i for m, _, i in preds if m == model}) if i in reps_eval]
        for part, mask in (("top", top), ("rest", ~top)):
            d = []
            for i in reps:
                ra = _rmse(preds[(model, "ID_match", i)][mask] - yt[mask])
                rb = _rmse(preds[(model, "OOD_match", i)][mask] - yt[mask])
                d.append((ra, rb, metrics.pct_change(ra, rb)))
            d = np.asarray(d)
            out.append(dict(candidate=cand, criterion=criterion, fold=fold, model=model, part=part, n_rows=int(mask.sum()),
                            repeats=",".join(map(str, reps)), rmse_id_match=float(d[:, 0].mean()), rmse_ood_match=float(d[:, 1].mean()),
                            gap_rmse_pct=float(d[:, 2].mean())))
    return out


def error_common(pf, df, y):
    """Per domain, the common top 1% rows by three criteria (y_extreme, cross_repeat, same_fits), and per model the error and target-reference gap
    on those rows and on the remaining rows."""
    rows_out, by_model = [], []
    lot = df[config.LOT_COL]
    sub = pf[pf.regime.isin(REGIMES)]
    for cand, g in sub.groupby("candidate"):
        preds, idx0 = {}, None
        for r in g.itertuples(index=False):
            z = np.load(r.pred_path)
            if idx0 is None:
                idx0 = z["ood_idx"]
            if not np.array_equal(z["ood_idx"], idx0):
                raise SystemExit(f"{cand}: B_te rows differ between fits")
            preds[(r.model, r.regime, int(r.repeat))] = z["ood_pred"]
        yt = y.loc[idx0].to_numpy()
        k = max(1, int(np.ceil(TOP_FRAC * len(yt))))
        slot, _, mB = build_one(df, cand)
        bv = B_view(slot, df.index[mB], y, lot, "benchmark")
        if not np.array_equal(np.sort(np.asarray(bv["B_te"])), np.sort(idx0)):
            raise SystemExit(f"{cand}: B_te recomputed from the split differs from the rows of the prediction files")
        y_ref = float(np.median(y.loc[bv["B_tr"]]))
        all_reps = sorted({i for _, _, i in preds})

        def med_err(reps):
            return np.median(np.abs(np.stack([v for (m, rg, i), v in preds.items() if i in reps]) - yt), axis=0)
        top = _top_mask(np.abs(yt - y_ref), k)
        by_model += _by_model(cand, "y_extreme", "", preds, yt, top, all_reps)
        rows_out += [dict(candidate=cand, criterion="y_extreme", fold="", row=int(idx0[j]), y=float(yt[j]), score=float(abs(yt[j] - y_ref)))
                     for j in np.flatnonzero(top)]
        even, odd = [i for i in all_reps if i % 2 == 0], [i for i in all_reps if i % 2 == 1]
        for fold, (sel, ev) in (("select_even", (even, odd)), ("select_odd", (odd, even))):
            if sel and ev:
                sc = med_err(sel)
                top = _top_mask(sc, k)
                by_model += _by_model(cand, "cross_repeat", fold, preds, yt, top, ev)
                rows_out += [dict(candidate=cand, criterion="cross_repeat", fold=fold, row=int(idx0[j]), y=float(yt[j]), score=float(sc[j]))
                             for j in np.flatnonzero(top)]
        sc = med_err(all_reps)
        top = _top_mask(sc, k)
        by_model += _by_model(cand, "same_fits", "", preds, yt, top, all_reps)
        rows_out += [dict(candidate=cand, criterion="same_fits", fold="", row=int(idx0[j]), y=float(yt[j]), score=float(sc[j]))
                     for j in np.flatnonzero(top)]
    return pd.DataFrame(rows_out), pd.DataFrame(by_model)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark-run-id", required=True)
    ap.add_argument("--out", default=None, help="output folder (default results/benchmark/runs/<run>/diag)")
    args = ap.parse_args()
    rep = os.path.join(config.RESULTS_DIR, "benchmark", "runs", args.benchmark_run_id, "report")
    pf = read_per_fit(os.path.join(rep, "per_fit.csv"))
    ds = pd.read_csv(os.path.join(rep, "target_reference_gap.csv"))
    df, y = load_data()
    out = args.out or os.path.join(config.RESULTS_DIR, "benchmark", "runs", args.benchmark_run_id, "diag")
    os.makedirs(out, exist_ok=True)
    cells = collapse_cells(pf)
    cells.to_csv(os.path.join(out, "collapse_cells.csv"), index=False)
    cr = collapse_rows(pf, cells, df, y)
    cr.to_csv(os.path.join(out, "collapse_rows.csv"), index=False)
    ex = exclusion(ds, pf, cells)
    json.dump(ex, open(os.path.join(out, "collapse_exclusion.json"), "w"), ensure_ascii=False, indent=1)
    es = error_share_per_fit(pf, y)
    es.to_csv(os.path.join(out, "error_share_per_fit.csv"), index=False)
    er, eb = error_common(pf, df, y)
    er.to_csv(os.path.join(out, "error_rows_common.csv"), index=False)
    eb.to_csv(os.path.join(out, "error_common_by_model.csv"), index=False)
    n_rep_collapse = int((pf[pf.regime.isin(REGIMES)].r2_ood < COLLAPSE_R2).sum())
    top_feat = (cr[cr.groupby(["candidate", "model", "regime", "repeat"]).cumcount() < DROP_ROWS]
                .groupby(["candidate", "model", "regime"]).feature.agg(lambda s: s.value_counts().to_dict()).to_dict()) if len(cr) else {}
    summ = dict(benchmark_run_id=args.benchmark_run_id, collapse_threshold_r2=COLLAPSE_R2, criterion="chosen after seeing the results",
                n_collapse_cells=len(cells), n_fits_below_threshold=n_rep_collapse,
                collapse_cells=[f"{a}|{b}|{c}" for a, b, c in zip(cells.candidate, cells.model, cells.regime)],
                top_rows_feature={"|".join(k): v for k, v in top_feat.items()},
                positive_share=dict(all=ex["all"]["positive_share"], excluding_collapse=ex["excluding_collapse"]["positive_share"]),
                share_top1pct_median=es.groupby(["candidate", "regime"]).share_top.median().unstack().round(4).to_dict(),
                common_top_rows=dict(
                    primary="y_extreme", secondary="cross_repeat", descriptive_only="same_fits",
                    rules=dict(y_extreme="top 1% of B_te rows by |y - median(y of B_tr)| (independent of predictions)",
                               cross_repeat="top 1% by median absolute error over even (odd) repeats, evaluated on odd (even) repeats",
                               same_fits="top 1% by median absolute error of the evaluated fits themselves (descriptive only; selection bias)"),
                    n_rows=er[er.criterion == "y_extreme"].groupby("candidate").size().to_dict(),
                    consensus_gap_rmse_pct={c: g.groupby(["candidate", "model", "part"]).gap_rmse_pct.mean()   # mean over the two folded directions, then the model median
                                              .groupby(level=["candidate", "part"]).median().unstack().round(3).to_dict()
                                          for c, g in eb.groupby("criterion")}))
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
