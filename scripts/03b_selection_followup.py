"""Follow-up analyses of the domain selection (selection sensitivity and the temporal-axis trajectories). No retraining or re-prediction: only
the screening result rows (rows.csv) and the selection result are read.

(1) Reselection leaving one model out (4 ways): the same rule as selection.py on the remaining 3 models (definition of consistently
    positive, eligibility, consensus score = median of the model scores, descending, ties by name, top N_CAT). Pass threshold =
    consistently positive in 3/3 — the original rule "at least 3 of 4" gives 3/3 for 3 models whether read as a count or as a share
    (3/4). The 2/3 threshold is computed for reference as well (not used for the verdict).
(2) Alternative ranking criteria: dRMSE (= RMSE_ID_match - RMSE_OOD_match, no normalization, sign kept), D_RMSE% (the selection
    criterion), D_R^2. All three compare the same two models on the same evaluation set (B_screen), so a single repetition has the same
    sign under each (R^2 = 1 - MSE/Var(y_screen)). The sign of the repetition mean can differ between criteria (each repetition is
    divided by a different scale), so whether the consistently-positive verdicts and the passing set agree is computed and recorded.
(3) Consensus target-reference gap (RMSE) vs n*: the 48 process-condition candidates (2 ineligible marked) + 10 random-split controls, the 5 selected
    highlighted. Rank correlations (Spearman, Kendall) over the 48 process-condition candidates and over the 58 with the controls. The
    correlation with lot exposure (share of B_screen rows whose lot appears in the ID_match training rows, repetition mean) is recorded too.
(4) Temporal trajectories: per axis, the consensus target-reference gap (RMSE) of the 15 cutoffs, the minimum and maximum of the per-repetition consensus
    (median over the 4 models within a repetition), the per-model repetition means, and the pre-specified median cutoff. x axis = position
    within the period (first row 0, last row 1, time share; the row-based A share is kept in the table). Below the x_time axis, the
    production share of cat_050 value 1 is tabulated on the same position axis (level_profile.csv).
Comparison measures: top-5 overlap and Jaccard, whether each of the 5 originally selected is retained, rank correlation of the consensus
scores (the 48 complete-case process-condition candidates, the 15 originally passing).
First, the result recomputed with 4 models, D_RMSE% and the 3/4 threshold must equal final_domains.json; otherwise the script stops.

(5) (with --selection-audit-run-id) Percentiles of the selected domains by the B_te consensus score of the selection audit, in three
    reference sets: the 46 eligible process-condition candidates (the selection was made within this set; the main reference, as in the
    audit report), the 78 candidates (with temporal ones), the 88 with the controls. Percentile = share of the set with an equal or lower
    B_te consensus (%). The audit may have fewer repetitions than screening (run argument repeats), so the consistently-positive verdict
    uses that number of repetitions too.

Output: results/screening/runs/<screening>/followup/{report.json, lomo.csv, metric_alt.csv, n_star.csv, temporal.csv, selection_audit_te.csv, level_profile.csv}
Usage: python scripts/03b_selection_followup.py --screening-run-id <screening> [--selection-audit-run-id <audit>]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from defectshift import config, selection  # noqa: E402
from defectshift.candidates import load_candidates, load_data  # noqa: E402
from defectshift.runio import load_final_domains  # noqa: E402

METRICS = ("rmse_difference", "gap_rmse_pct", "r2_difference")
PRIMARY = "gap_rmse_pct"
LEVEL_COL, LEVEL_CODE = "cat_050", 1


def per_run(rows):
    """target-reference gap under the three criteria per (candidate, model, repetition). Uses the same rows as selection.target_reference_gap_table (last successful row)."""
    ok = rows[rows.status == "ok"].drop_duplicates(["slot", "model", "repeat", "regime"], keep="last")
    p = ok.pivot_table(index=["candidate", "kind", "model", "repeat"], columns="regime",
                       values=["rmse_screen", "r2_screen"], aggfunc="first")
    p = p.dropna()
    out = pd.DataFrame(dict(
        rmse_difference=p[("rmse_screen", "ID_match")] - p[("rmse_screen", "OOD_match")],
        gap_rmse_pct=(p[("rmse_screen", "ID_match")] - p[("rmse_screen", "OOD_match")]) / p[("rmse_screen", "OOD_match")] * 100,
        r2_difference=p[("r2_screen", "OOD_match")] - p[("r2_screen", "ID_match")])).reset_index()
    ref = selection.target_reference_gap_table(rows).set_index(["candidate", "model", "repeat"])
    mine = out.set_index(["candidate", "model", "repeat"])
    assert ref.index.sort_values().equals(mine.index.sort_values())
    assert np.allclose(ref.loc[mine.index, "gap_rmse_pct"], mine.gap_rmse_pct) and np.allclose(ref.loc[mine.index, "r2_difference"], mine.r2_difference)
    return out


def summarize(dt, meta, models, n_repeats, metric):
    """One row per candidate: per-model repetition mean, consistently positive (repetition mean > 0 and > 0 in at least 2/3 of the
    repetitions), number of consistently positive models, consensus (median of the model scores)."""
    g = dt[dt.model.isin(models)]
    agg = g.groupby(["candidate", "model"])[metric].agg(mean="mean", npos=lambda s: int((s > 0).sum()), n="count")
    agg["consistent"] = (agg.n == n_repeats) & (agg["mean"] > 0) & (agg.npos * 3 >= 2 * n_repeats)
    out = []
    for c, m in meta.iterrows():
        a = agg.loc[c] if c in agg.index.get_level_values(0) else None
        complete = a is not None and len(a) == len(models) and bool((a.n == n_repeats).all())
        out.append(dict(candidate=c, kind=m.kind, b_te_eligible=bool(m.b_te_eligible), complete=complete,
                        n_models_consistent=int(a.consistent.sum()) if a is not None else 0,
                        consensus=float(np.median(a["mean"])) if complete else np.nan,
                        consistent_models=",".join(sorted(a.index[a.consistent])) if a is not None else ""))
    return pd.DataFrame(out).set_index("candidate")


def rule_select(summary, k, n_cat=selection.N_CAT):
    """Process-condition candidates: complete cases & consistently positive models >= k & eligible -> top n_cat by consensus
    (descending, ties by name). (selected, all passing)."""
    s = summary[(summary.kind == "cat") & summary.complete & (summary.n_models_consistent >= k) & summary.b_te_eligible]
    s = s.reset_index().sort_values(["consensus", "candidate"], ascending=[False, True])
    return list(s.candidate.head(n_cat)), list(s.candidate)


def rank_corr(a, b, names):
    """Spearman and Kendall of two scores over the candidates names (finite values only)."""
    x, y_ = a.reindex(names), b.reindex(names)
    ok = x.notna() & y_.notna()
    if ok.sum() < 3:
        return dict(n=int(ok.sum()), spearman=None, kendall=None)
    return dict(n=int(ok.sum()), spearman=float(spearmanr(x[ok], y_[ok]).correlation),
                kendall=float(kendalltau(x[ok], y_[ok]).correlation))


def compare(top, ref_top):
    inter = set(top) & set(ref_top)
    union = set(top) | set(ref_top)
    return dict(top=top, overlap=len(inter), jaccard=len(inter) / len(union) if union else None,
                retained={c: c in top for c in ref_top})


def lomo(dt, meta, models, n_repeats, ref, ref_passed, cats48):
    out, table = [], []
    for drop in models:
        rest = [m for m in models if m != drop]
        s = summarize(dt, meta, rest, n_repeats, PRIMARY)
        r = dict(dropped=drop, models=rest)
        for k, label in ((3, "k3_primary"), (2, "k2_reference")):
            top, passed = rule_select(s, k)
            r[label] = dict(n_passed=len(passed), **compare(top, ref["top"]))
        r["corr_consensus_cat48"] = rank_corr(ref["summary"].consensus, s.consensus, cats48)
        r["corr_consensus_passed15"] = rank_corr(ref["summary"].consensus, s.consensus, ref_passed)
        out.append(r)
        table.append(s.assign(dropped=drop).reset_index())
    return out, pd.concat(table, ignore_index=True)


def metric_alternatives(dt, meta, models, n_repeats, ref, ref_passed, cats48):
    sums = {m: summarize(dt, meta, models, n_repeats, m) for m in METRICS}
    res = {}
    for m in METRICS:
        top, passed = rule_select(sums[m], 3)
        same_consistent = bool((sums[m].consistent_models == sums[PRIMARY].consistent_models).all())
        res[m] = dict(n_passed=len(passed), passed_set_equal_primary=set(passed) == set(ref_passed),
                      consistent_models_equal_primary=same_consistent, **compare(top, ref["top"]))
    pairs = {}
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1:]:
            pairs[f"{a}__{b}"] = dict(cat48=rank_corr(sums[a].consensus, sums[b].consensus, cats48),
                                      passed15=rank_corr(sums[a].consensus, sums[b].consensus, ref_passed))
    table = pd.concat([sums[m].assign(metric=m).reset_index() for m in METRICS], ignore_index=True)
    return dict(per_metric=res, rank_corr=pairs), table


def n_star_table(rows, primary_summary, ref_top, ref_passed):
    ok = rows[rows.status == "ok"].drop_duplicates(["slot", "model", "repeat", "regime"], keep="last")
    g = ok.groupby("candidate")
    t = pd.DataFrame(dict(n_star_mean=g.n_star.mean(), n_star_r0=ok[ok.repeat == 0].groupby("candidate").n_star.first(),
                          n_A=g.n_A.first(), n_B=g.n_B.first(),
                          lot_exposure=ok[ok.regime == "ID_match"].groupby("candidate").screen_lot_in_train.mean()))
    t = t.join(primary_summary[["kind", "b_te_eligible", "n_models_consistent", "consensus"]])
    t = t[t.kind.isin(["cat", "placebo"])]

    def status(r):
        if r.kind == "placebo":
            return "placebo"
        if not r.b_te_eligible:
            return "ineligible"
        if r.name in ref_top:
            return "selected"
        return "passed" if r.name in ref_passed else "not_consistent"
    t["status"] = t.apply(status, axis=1)
    cats = t[t.kind == "cat"]
    elig = list(cats[cats.b_te_eligible].index)
    plc = t[t.kind == "placebo"]
    corr = dict(
        consensus_vs_n_star_mean=dict(cat48=rank_corr(cats.consensus, cats.n_star_mean, list(cats.index)),
                                      eligible46=rank_corr(cats.consensus, cats.n_star_mean, elig),
                                      cat48_plus_placebo10=rank_corr(t.consensus, t.n_star_mean, list(t.index))),
        consensus_vs_n_star_r0=dict(cat48=rank_corr(cats.consensus, cats.n_star_r0, list(cats.index)),
                                    eligible46=rank_corr(cats.consensus, cats.n_star_r0, elig)),
        consensus_vs_lot_exposure=dict(cat48=rank_corr(cats.consensus, cats.lot_exposure, list(cats.index)),
                                       eligible46=rank_corr(cats.consensus, cats.lot_exposure, elig)),
        status_counts=t.status.value_counts().to_dict(),
        placebo_consensus_range=[float(plc.consensus.min()), float(plc.consensus.max())],
        # the controls are not selection targets, but how many would pass the same rule (consistently positive >= 3/4) (eligibility skipped: sizes are matched)
        placebo_meeting_consistency_rule=sorted(plc[plc.n_models_consistent >= 3].index),
        selected_min_consensus=float(t[t.status == "selected"].consensus.min()),
        cat48_positive=int((cats.consensus > 0).sum()), cat48_negative=int((cats.consensus <= 0).sum()))
    return t.reset_index().rename(columns={"index": "candidate"}), corr


def time_frac(df, ax, ts):
    """Position of time ts within the period (0 = first row, 1 = last row). The manuscript uses this position instead of the anonymized times."""
    lo, hi = df[ax].min(), df[ax].max()
    return float((pd.Timestamp(ts) - lo) / (hi - lo))


def temporal_table(dt, meta, models, primary_summary, temporal_primary, df):
    """Per cutoff: consensus (median of the model repetition means = the definition of the selection table), mean of the middle two
    models (always containing the consensus), per-repetition consensus (model median within a repetition), per-model repetition means,
    position (row share a_share, period share time_frac)."""
    out = []
    for ax in ("x_time", "y_time"):
        names = sorted([c for c in meta.index if c.startswith(ax + "<=")], key=lambda n: pd.Timestamp(n.split("<=")[1]))
        for pos, c in enumerate(names, 1):
            g = dt[(dt.candidate == c) & dt.model.isin(models)]
            per_rep = g.groupby("repeat")[PRIMARY].median()
            means = np.sort([float(g[g.model == m][PRIMARY].mean()) for m in models])
            r = dict(axis=ax, cutoff_no=pos, candidate=c, a_share=meta.loc[c, "n_A"] / (meta.loc[c, "n_A"] + meta.loc[c, "n_B"]),
                     time_frac=time_frac(df, ax, c.split("<=")[1]), consensus=primary_summary.loc[c, "consensus"],
                     models_mid_lo=float(means[len(means) // 2 - 1]), models_mid_hi=float(means[len(means) // 2]),
                     repeat_consensus_min=float(per_rep.min()), repeat_consensus_max=float(per_rep.max()),
                     is_primary=c in temporal_primary, n_models_consistent=int(primary_summary.loc[c, "n_models_consistent"]))
            for i, v in per_rep.items():
                r[f"repeat{i}_consensus"] = float(v)
            for m in models:
                r[f"{m}_mean"] = float(g[g.model == m][PRIMARY].mean())
            out.append(r)
    return pd.DataFrame(out)


def level_profile(df, n_bins=50):
    """The x_time period split into n_bins equal intervals: number of rows and share of cat_050 == 1 per interval (NaN share for empty intervals)."""
    pos = ((df["x_time"] - df["x_time"].min()) / (df["x_time"].max() - df["x_time"].min())).to_numpy(float)
    bins = np.minimum((pos * n_bins).astype(int), n_bins - 1)
    is_r = pd.Series((df[LEVEL_COL].to_numpy() == LEVEL_CODE).astype(float))
    g = is_r.groupby(bins)
    n = g.size().reindex(range(n_bins), fill_value=0)
    share = g.mean().reindex(range(n_bins))
    return pd.DataFrame(dict(pos_lo=np.arange(n_bins) / n_bins, pos_hi=(np.arange(n_bins) + 1) / n_bins,
                             n_rows=n.to_numpy(), share=share.to_numpy()))


def selection_audit_percentiles(selection_audit_run_id, meta, models, final):
    """Consensus target-reference gap (RMSE) (B_te) per candidate from the selection-audit result rows (B_te metrics) and the percentiles of the selected domains."""
    run_dir = os.path.join(config.RESULTS_DIR, "selection_audit", "runs", selection_audit_run_id)
    man = json.load(open(os.path.join(run_dir, "manifest.json")))["args"]
    n_rep = int(man["repeats"])
    rows = pd.read_csv(os.path.join(run_dir, "rows.csv"))
    rows = rows[rows.repeat < n_rep].rename(columns={"r2_te": "r2_screen", "rmse_te": "rmse_screen", "mae_te": "mae_screen"})
    te = summarize(per_run(rows), meta, models, n_rep, PRIMARY)
    sets = dict(eligible46=list(te[(te.kind == "cat") & te.b_te_eligible].index),
                candidates78=list(te[te.kind != "placebo"].index), all88=list(te.index))
    assert len(sets["eligible46"]) == 46 and len(sets["candidates78"]) == 78 and len(sets["all88"]) == 88, \
        {k: len(v) for k, v in sets.items()}

    def pct(c, names):
        v = te.consensus.reindex(names)
        return float((v <= te.loc[c, "consensus"]).mean() * 100)
    out = []
    for c in final["categorical"] + final["temporal"]:
        r = dict(candidate=c, kind=te.loc[c, "kind"], consensus_te=float(te.loc[c, "consensus"]),
                 n_models_consistent_te=int(te.loc[c, "n_models_consistent"]))
        for k, names in sets.items():
            if c in names or k != "eligible46":
                r[f"percentile_{k}"] = pct(c, names)
        out.append(r)
    return dict(selection_audit_run_id=selection_audit_run_id, repeats=n_rep, sets={k: len(v) for k, v in sets.items()}, selected=out), \
        te.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--screening-run-id", required=True)
    ap.add_argument("--selection-audit-run-id", default=None, help="finished selection-audit run (gives the B_te percentiles of (5))")
    ap.add_argument("--allow-untuned-selection", action="store_true",
                    help="only for re-analysing a fixed-setting screening run for comparison")
    args = ap.parse_args()

    load_final_domains(args.screening_run_id)                         # rows and manifest unchanged since the selection (hashes)
    run_dir = os.path.join(config.RESULTS_DIR, "screening", "runs", args.screening_run_id)
    rows = pd.read_csv(os.path.join(run_dir, "rows.csv"))
    man = json.load(open(os.path.join(run_dir, "manifest.json")))
    final = json.load(open(os.path.join(run_dir, "selection", "final_domains.json")))
    base = load_candidates()
    placebo = [n for n in load_candidates(include_placebo=True) if n not in base]
    selection.validate_run(rows, man["args"], base, placebo, require_tuned=not args.allow_untuned_selection)
    models, n_rep = list(config.SCREENING_MODELS), config.N_REPEATS_SCREENING

    meta = rows.drop_duplicates("candidate").set_index("candidate")[["kind", "n_A", "n_B", "b_te_eligible"]]
    meta["b_te_eligible"] = meta.b_te_eligible.astype(bool)
    dt = per_run(rows)
    primary = summarize(dt, meta, models, n_rep, PRIMARY)
    ref_top, ref_passed = rule_select(primary, 3)
    if ref_top != final["categorical"] or len(ref_passed) != final["n_cat_passed"]:
        raise SystemExit(f"the recomputed selection {ref_top} ({len(ref_passed)} passing) differs from final_domains.json "
                         f"{final['categorical']} ({final['n_cat_passed']} passing)")
    ref = dict(top=ref_top, summary=primary)
    cats48 = list(primary[(primary.kind == "cat") & primary.complete].index)
    assert len(cats48) == 48

    out_dir = os.path.join(run_dir, "followup")
    os.makedirs(out_dir, exist_ok=True)
    lomo_res, lomo_tab = lomo(dt, meta, models, n_rep, ref, ref_passed, cats48)
    alt_res, alt_tab = metric_alternatives(dt, meta, models, n_rep, ref, ref_passed, cats48)
    nst, nst_corr = n_star_table(rows, primary, ref_top, ref_passed)
    df, _ = load_data()
    tt = temporal_table(dt, meta, models, primary, set(final["temporal"]), df)
    level = level_profile(df)
    med = final["temporal"][0]
    a_mask = df["x_time"] <= pd.Timestamp(med.split("<=")[1])
    level_ab = dict(cutoff=med, A=float((df.loc[a_mask, LEVEL_COL] == LEVEL_CODE).mean()),
                     B=float((df.loc[~a_mask, LEVEL_COL] == LEVEL_CODE).mean()))
    temporal_summary = {ax: dict(n_cutoffs=int((tt.axis == ax).sum()), n_consensus_positive=int(((tt.axis == ax) & (tt.consensus > 0)).sum()),
                                 primary_consensus=float(tt[(tt.axis == ax) & tt.is_primary].consensus.iloc[0]),
                                 max_consensus=float(tt[tt.axis == ax].consensus.max()),
                                 argmax_cutoff_no=int(tt[tt.axis == ax].sort_values("consensus").cutoff_no.iloc[-1]))
                        for ax in ("x_time", "y_time")}

    lomo_tab.to_csv(os.path.join(out_dir, "lomo.csv"), index=False)
    alt_tab.to_csv(os.path.join(out_dir, "metric_alt.csv"), index=False)
    nst.to_csv(os.path.join(out_dir, "n_star.csv"), index=False)
    tt.to_csv(os.path.join(out_dir, "temporal.csv"), index=False)
    level.to_csv(os.path.join(out_dir, "level_profile.csv"), index=False)
    selection_audit = None
    if args.selection_audit_run_id:
        selection_audit, te_tab = selection_audit_percentiles(args.selection_audit_run_id, meta, models, final)
        te_tab.to_csv(os.path.join(out_dir, "selection_audit_te.csv"), index=False)
    report = dict(screening_run_id=args.screening_run_id, reference=dict(top=ref_top, n_passed=len(ref_passed), passed=ref_passed), selection_audit=selection_audit,
                  lomo=lomo_res, metric_alternatives=alt_res, n_star=nst_corr, temporal=temporal_summary,
                  level_share_at_median_x_cutoff=level_ab,
                  notes=["LOMO pass threshold 3/3 = the value on which the count and share readings of the original rule 'at least 3 of 4' agree; 2/3 for reference",
                         "dRMSE keeps its sign (no normalization); not an absolute value",
                         "rank correlation cat48 = the 48 complete-case process-condition candidates (including ineligible ones), passed15 = candidates passing the original rule"])
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
