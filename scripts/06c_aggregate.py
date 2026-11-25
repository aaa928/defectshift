"""Aggregation of the benchmark results. No retraining or re-prediction: the inputs are rows.csv and the manifest of a benchmark run,
the predictions saved per fit (<prefix>.pred.npz: A_heldout and B_te row indices and predictions), and y of the data. Metrics are
recomputed from the saved predictions (the record of reference) and checked against the metrics in rows.csv. With several successful rows
of the same key, the last row is the record of reference (retraining history after broken artifacts).

Signs (for every quantity: positive = worse than the reference; only for H_tune positive = the target-side search values are better)
- target-reference gap (B_te, main measure): ID_match vs OOD_match. R^2 = R^2_OOD_match - R^2_ID_match, RMSE and MAE = (ID_match - OOD_match) / OOD_match x 100
- ID-OOD gap (for contrast): A_heldout vs B_te of the same source model (ID_full, ID_match). R^2 = R^2_A_heldout - R^2_B_te,
  RMSE and MAE = (B_te - A_heldout) / A_heldout x 100. A comparison across different test sets, so the difference in target variance
  enters -> contrast only
- R^2 difference (column ood_gap_r2) = R^2_ID - R^2_OOD, OOD ratio = RMSE_OOD / RMSE_ID (A_heldout vs B_te of the same model)
- identity: (1 - R^2_OOD)/(1 - R^2_ID) = (MSE_OOD / MSE_ID) x (Var_ID / Var_OOD), Var with ddof 0 (the same definition as sklearn r2_score)
- H_tune (supplementary, tuned models): OOD_match_tuned vs OOD_match (B_te). R^2 = R^2_tuned - R^2_match, RMSE and MAE = (match - tuned) / tuned x 100
- tail: MAE and target-reference gap (MAE) per quantile bin (0-50, 50-80, 80-90, 90-95, 95-100%) defined once per domain from the actual y on B_te
- diagnostics: share of evaluation rows with a categorical value absent from the training rows and the number of such columns
  (unseen_cat_* of per_fit; the categorical vocabulary is a fixed dataset schema)

Output (results/benchmark/runs/<run>/report/): per_fit.csv, target_reference_gap.csv, id_ood_gap.csv, h_tune.csv, tail.csv, domain_summary.csv
(statistical unit = domain: repetition mean and standard deviation), model_summary.csv (median over domains, number of domains with
positive target-reference gap, median of the per-domain ranks), robustness_input.csv (long format for the robustness analysis), report.json
(completeness: successes, failures and missing keys against the expected keys, agreement of the recomputed metrics, key aggregates).
Before a main run finishes, --partial aggregates the finished items only (left as missing in the completeness table).

Usage: python scripts/06c_aggregate.py --run-id <run> [--partial]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from defectshift import config, metrics  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_data  # noqa: E402
from defectshift.runio import check_same_inputs  # noqa: E402
from defectshift.models import extra_models  # noqa: E402
from defectshift.splits import B_view, get_A  # noqa: E402

KEY = ["candidate", "slot", "repeat", "model", "regime"]
TAIL_EDGES = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
TAIL_LABELS = ("0-50", "50-80", "80-90", "90-95", "95-100")


def _metrics(y_true, y_pred):
    m = metrics.regression(y_true, y_pred)
    return dict(r2=m["r2"], rmse=m["rmse"], mae=m["mae"], mse=m["rmse"] ** 2, var_y=float(np.var(np.asarray(y_true), ddof=0)))


def expected_keys(man, slots):
    """The (candidate, slot, repeat, model, regime) expected from the domains, repetitions, models and regimes of the manifest."""
    a = man["args"]
    out = set()
    for name in a["domains"]:
        for i in range(a["repeats"]):
            for m in a["models"]:
                for r in a["regimes"]:
                    if r == "OOD_match_tuned" and m not in extra_models.tuned_all():
                        continue
                    out.add((name, slots[name], i, m, r))
    return out


def load_fits(run_id, partial):
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "runs", run_id)
    man = json.load(open(os.path.join(run_dir, "manifest.json")))
    check_same_inputs(man, f"benchmark run {run_id}")      # the same data and candidate list as at training time (evaluation rows and recomputed metrics cannot detect feature changes)
    rows = pd.read_csv(os.path.join(run_dir, "rows.csv"))
    slots = dict(rows.drop_duplicates("candidate")[["candidate", "slot"]].values)
    missing_slot = [n for n in man["args"]["domains"] if n not in slots]
    exp = expected_keys(man, {**slots, **{n: None for n in missing_slot}})
    ok = rows[rows.status == "ok"].drop_duplicates(KEY, keep="last")
    got = set(map(tuple, ok[KEY].itertuples(index=False)))
    failed_only = set(map(tuple, rows[rows.status != "ok"][KEY].itertuples(index=False))) - got
    completeness = dict(expected=len(exp), ok=len(got & exp), failed_without_success=len(failed_only & exp),
                        missing=len(exp - got - failed_only), unexpected=len(got - exp),
                        n_failed_rows_total=int((rows.status != "ok").sum()),
                        n_duplicate_ok_rows=int(rows[rows.status == "ok"].duplicated(KEY).sum()))
    if not partial and got != exp:
        raise SystemExit(f"incomplete run (not aggregated without --partial): {completeness}")
    return run_dir, man, ok, completeness


def eval_sets(names):
    """Domain -> (A_heldout, B_te) row indices recomputed by the split code (to check that the rows of the prediction files really are those evaluation sets)."""
    df, y = load_data()
    lot = df[config.LOT_COL]
    out = {}
    for n in names:
        slot, mA, mB = build_one(df, n)
        out[n] = (np.asarray(get_A(slot, df.index[mA], y, lot)["A_heldout"]),
                  np.asarray(B_view(slot, df.index[mB], y, lot, "benchmark")["B_te"]))
    return out


def _unseen_exposure(df, cats, prefix, z):
    """Diagnostic (categorical-encoding assumption): share of evaluation rows with at least one categorical value absent from the
    training rows (fit + early-stopping rows) and the number of such categorical columns. The categorical vocabulary is a fixed dataset
    schema, so this records only how many unseen categories occurred, independently of how they were encoded."""
    f = np.load(prefix + ".npz")
    train = np.concatenate([f["fit_idx"], f["es_idx"]]).astype(np.int64)
    seen = {c: set(df.loc[train, c].unique()) for c in cats}
    out = {}
    for tag, idx in (("id", z["id_idx"]), ("ood", z["ood_idx"])):
        ev = df.loc[idx, cats]
        new = np.column_stack([~ev[c].isin(seen[c]).to_numpy() for c in cats]) if cats else np.zeros((len(ev), 0), bool)
        out[f"unseen_cat_rows_{tag}"] = float(new.any(axis=1).mean()) if len(ev) else 0.0
        out[f"unseen_cat_cols_{tag}"] = int(new.any(axis=0).sum())
    return out


def recompute(run_dir, man, ok, y):
    """Recompute the metrics from the saved predictions. The row indices of the prediction files must equal A_heldout and B_te
    recomputed by the split code (otherwise the run stops), and the predictions must have the right length and be finite. Returns the
    per_fit DataFrame and the maximum difference from the metrics in rows."""
    from defectshift import benchmark as bench
    sets = eval_sets(sorted(ok.candidate.unique()))
    df, _ = load_data()
    cat_cols = {n: feature_sets(df, n)[0] for n in ok.candidate.unique()}
    model_dir = os.path.join(config.CACHE_DIR, "benchmark", "runs", os.path.basename(run_dir), "models")   # the same location as 06_benchmark
    out, worst = [], 0.0
    for r in ok.to_dict("records"):
        prefix = bench.artifact_prefix(model_dir, r["slot"], r["model"], r["regime"], int(r["repeat"]))
        z = np.load(prefix + ".pred.npz")
        held, te = sets[r["candidate"]]
        if not (np.array_equal(z["id_idx"], held) and np.array_equal(z["ood_idx"], te)):
            raise SystemExit(f"{prefix}: the prediction rows differ from A_heldout and B_te of the split code")
        if not (len(z["id_pred"]) == len(held) and len(z["ood_pred"]) == len(te)
                and np.isfinite(z["id_pred"]).all() and np.isfinite(z["ood_pred"]).all()):
            raise SystemExit(f"{prefix}: the number of predictions differs from the evaluation rows or a value is not finite")
        mid = _metrics(y.loc[z["id_idx"]], z["id_pred"])
        mood = _metrics(y.loc[z["ood_idx"]], z["ood_pred"])
        for k in ("r2", "rmse", "mae"):
            worst = max(worst, abs(mid[k] - r[f"{k}_id"]), abs(mood[k] - r[f"{k}_ood"]))
        rec = {c: r[c] for c in KEY + ["family", "n_fit", "n_star", "seed"] if c in r}
        rec.update({f"{k}_id": v for k, v in mid.items()}, **{f"{k}_ood": v for k, v in mood.items()})
        rec.update(_unseen_exposure(df, cat_cols[r["candidate"]], prefix, z))
        rec.update(ood_gap_r2=mid["r2"] - mood["r2"], ood_ratio_rmse=metrics.safe_ratio(mood["rmse"], mid["rmse"]),
                   rq3_lhs=metrics.safe_ratio(1 - mood["r2"], 1 - mid["r2"]),
                   rq3_mse_ratio=metrics.safe_ratio(mood["mse"], mid["mse"]), rq3_var_ratio=metrics.safe_ratio(mid["var_y"], mood["var_y"]),
                   pred_path=prefix + ".pred.npz")
        out.append(rec)
    return pd.DataFrame(out), worst


def _pair(per_fit, ref_regime, cmp_regime, fn):
    """fn(reference row, comparison row) for each pair of the two regimes with the same (domain, repetition, model)."""
    a = per_fit[per_fit.regime == ref_regime].set_index(["candidate", "repeat", "model"])
    b = per_fit[per_fit.regime == cmp_regime].set_index(["candidate", "repeat", "model"])
    idx = a.index.intersection(b.index)
    return pd.DataFrame([dict(zip(["candidate", "repeat", "model"], k), **fn(a.loc[k], b.loc[k])) for k in idx])


def _d(id_row, ood_row, sfx):
    """metrics.target_reference_gap applied to two rows on the same evaluation set (sfx)."""
    m = lambda r: dict(r2=r[f"r2_{sfx}"], rmse=r[f"rmse_{sfx}"], mae=r[f"mae_{sfx}"])
    return metrics.target_reference_gap(m(id_row), m(ood_row))


def tail_table(per_fit, y):
    """Quantile bounds defined once per domain from y on B_te; MAE and target-reference gap (MAE) per bin for ID_full, ID_match and OOD_match."""
    out = []
    for cand, g in per_fit[per_fit.regime.isin(["ID_full", "ID_match", "OOD_match"])].groupby("candidate"):
        first = np.load(g.pred_path.iloc[0])
        y_te = y.loc[first["ood_idx"]].to_numpy()
        edges = np.quantile(y_te, TAIL_EDGES)
        for r in g.itertuples():
            z = np.load(r.pred_path)
            assert np.array_equal(z["ood_idx"], first["ood_idx"]), "B_te rows of the same domain differ"
            err = np.abs(z["ood_pred"] - y_te)
            for b, lab in enumerate(TAIL_LABELS):
                lo, hi = edges[b], edges[b + 1]
                mask = (y_te >= lo) & ((y_te <= hi) if b == len(TAIL_LABELS) - 1 else (y_te < hi))
                out.append(dict(candidate=cand, repeat=r.repeat, model=r.model, regime=r.regime, bin=lab, n=int(mask.sum()),
                                y_lo=float(lo), y_hi=float(hi), mae=float(err[mask].mean()) if mask.any() else np.nan))
    t = pd.DataFrame(out)
    if t.empty:
        return t
    w = t.pivot_table(index=["candidate", "repeat", "model", "bin"], columns="regime", values="mae").reset_index()
    if {"ID_match", "OOD_match"} <= set(w.columns):
        w["target_reference_gap_mae_pct"] = metrics.pct_change(w.ID_match, w.OOD_match)
    return t.merge(w[["candidate", "repeat", "model", "bin"] + (["target_reference_gap_mae_pct"] if "target_reference_gap_mae_pct" in w else [])],
                   on=["candidate", "repeat", "model", "bin"], how="left")


def summaries(per_fit, target_reference_gap, id_ood_gap, h_tune):
    """Statistical unit = domain: repetition mean and standard deviation; per model the median over domains, the number of positive
    domains and the median of the per-domain ranks."""
    long = []
    for reg in ("ID_full", "ID_match", "OOD_match", "OOD_match_tuned"):
        p = per_fit[per_fit.regime == reg]
        for q in ("r2_id", "rmse_id", "mae_id", "r2_ood", "rmse_ood", "mae_ood", "ood_gap_r2", "ood_ratio_rmse"):
            long.append(p.assign(quantity=f"{reg}:{q}", value=p[q])[["candidate", "repeat", "model", "quantity", "value"]])
    for tab, name in ((target_reference_gap, "target_reference_gap"), (h_tune, "h_tune")):
        for q in ("r2_difference", "gap_rmse_pct", "gap_mae_pct"):
            if len(tab):
                long.append(tab.assign(quantity=f"{name}:{q}", value=tab[q])[["candidate", "repeat", "model", "quantity", "value"]])
    if len(id_ood_gap):
        for reg in ("ID_full", "ID_match"):
            t = id_ood_gap[id_ood_gap.regime == reg]
            for q in ("r2_difference", "gap_rmse_pct", "gap_mae_pct"):
                long.append(t.assign(quantity=f"id_ood_gap_{reg}:{q}", value=t[q])[["candidate", "repeat", "model", "quantity", "value"]])
    long = pd.concat(long, ignore_index=True)
    dom = long.groupby(["candidate", "model", "quantity"]).value.agg(mean="mean", sd="std", n_repeats="count").reset_index()
    # within-domain rank (1 = best): errors and D smaller is better, R^2 larger is better
    higher_better = dom.quantity.str.endswith(("r2_id", "r2_ood")) | dom.quantity.str.startswith("h_tune")
    dom["rank_in_domain"] = np.nan
    for (c, q), g in dom.groupby(["candidate", "quantity"]):
        asc = not bool(higher_better.loc[g.index].iloc[0])
        dom.loc[g.index, "rank_in_domain"] = g["mean"].rank(ascending=asc, method="average")
    model = dom.groupby(["model", "quantity"]).agg(median_over_domains=("mean", "median"),
                                                   median_rank=("rank_in_domain", "median"),
                                                   n_domains=("mean", "count"),
                                                   n_domains_positive=("mean", lambda s: int((s > 0).sum())),
                                                   n_domains_nonpositive=("mean", lambda s: int((s <= 0).sum()))).reset_index()
    return dom, model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--partial", action="store_true", help="aggregate the finished items of an unfinished run")
    args = ap.parse_args()
    run_dir, man, ok, completeness = load_fits(args.run_id, args.partial)
    _, y = load_data()
    per_fit, worst = recompute(run_dir, man, ok, y)
    if worst > 1e-9:
        raise SystemExit(f"metrics recomputed from the saved predictions differ from rows.csv (maximum difference {worst:.2e})")
    target_reference_gap = _pair(per_fit, "ID_match", "OOD_match", lambda a, b: _d(a, b, "ood"))
    id_ood_gap = pd.concat([_pair(per_fit, reg, reg, lambda a, b: dict(
        r2_difference=a["r2_id"] - b["r2_ood"], gap_rmse_pct=metrics.pct_change(a["rmse_ood"], a["rmse_id"]),
        gap_mae_pct=metrics.pct_change(a["mae_ood"], a["mae_id"]))).assign(regime=reg) for reg in ("ID_full", "ID_match")],
        ignore_index=True)
    h_tune = _pair(per_fit, "OOD_match", "OOD_match_tuned", lambda a, b: _d(a, b, "ood"))
    tail = tail_table(per_fit, y)
    core = [f"{m}_{s}" for m in ("r2", "rmse", "mae") for s in ("id", "ood")]
    if not np.isfinite(per_fit[core].to_numpy(float)).all():          # no aggregation if a core metric is not finite
        raise SystemExit("a core metric (R^2, RMSE, MAE) has a non-finite value")
    undefined = {f"{name}:{c}": int(t_[c].isna().sum()) for name, t_ in (("per_fit", per_fit), ("target_reference_gap", target_reference_gap), ("id_ood_gap", id_ood_gap))
                 for c in t_.columns if c.startswith(("d_", "ood_ratio", "rq3_")) and t_[c].isna().any()}
    dom, model = summaries(per_fit, target_reference_gap, id_ood_gap, h_tune)

    out_dir = os.path.join(run_dir, "report")
    os.makedirs(out_dir, exist_ok=True)
    for name, tab in (("per_fit", per_fit), ("target_reference_gap", target_reference_gap), ("id_ood_gap", id_ood_gap), ("h_tune", h_tune), ("tail", tail),
                      ("domain_summary", dom), ("model_summary", model)):
        tab.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    dom.rename(columns={"mean": "domain_mean"}).to_csv(os.path.join(out_dir, "robustness_input.csv"), index=False)
    ds_dom = dom[dom.quantity == "target_reference_gap:gap_rmse_pct"]
    report = dict(run_id=args.run_id, partial=args.partial, completeness=completeness, metric_recompute_max_abs_diff=worst,
                  undefined_ratio_counts=undefined,     # number of ratios undefined because of a zero or non-finite denominator (0 expected)
                  n_fits=len(per_fit), domains_done=sorted(per_fit.candidate.unique()),
                  target_reference_gap_rmse_domain_means=dict(positive=int((ds_dom["mean"] > 0).sum()), nonpositive=int((ds_dom["mean"] <= 0).sum()),
                                                n=len(ds_dom)),
                  rq3_identity_max_abs_err=float(np.nanmax(np.abs(per_fit.rq3_lhs - per_fit.rq3_mse_ratio * per_fit.rq3_var_ratio)))
                  if len(per_fit) else None)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
