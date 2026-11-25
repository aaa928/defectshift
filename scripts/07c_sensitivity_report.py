"""Comparison of the sensitivity results (no retraining): per arm, whether the conclusions agree with the benchmark (the same
repetitions 0-2, ID_match and OOD_match).

Inputs: the sensitivity run (results/sensitivity/runs/<run>/rows.csv + prediction files), the benchmark aggregation
(report/per_fit.csv of 06c_aggregate.py) and the benchmark tuning records (hpo_run_id of the sensitivity manifest). The
sensitivity metrics are recomputed from the saved predictions and checked against rows.csv; the B_te and A_heldout rows of the
predictions must equal those of the same benchmark domain. The arm table is the whole roster with the benchmark values, where the
(domain, model) pairs rerun by the arm are replaced by the arm values; a rerun pair must have every repetition and regime
(otherwise the report stops; with --partial it is left out).
Quantities (domain x model, repetition mean): RMSE (B_te) of ID_match and OOD_match, the target-reference gap (RMSE) and the
target-reference gap on R^2 (B_te; paired per repetition, then averaged).
Verdicts (per domain, benchmark vs arm): Spearman and Kendall tau of the model ranks over the whole roster and over the rerun
models only (when at least 3 were rerun in that domain; the whole-roster correlation can be high because unchanged models enter
with the same values); top-3 overlap; sign agreement of the consensus target-reference gap (roster median) for the rerun models
only; cells with a changed sign (sign_flips.csv).
hpo_seed arm, in addition: per (domain, model), the share of identical hyperparameter values, the largest |log10 ratio| of the
numerical hyperparameters, the difference in the best validation score (R^2) and the Jaccard of the validation rows.
NODE preprocessing arm (node_prep_a), in addition: prediction differences between the example-default preprocessing and the main
setting on the same evaluation rows (mean and maximum absolute difference, correlation) and differences in RMSE, MAE and R^2 per
domain, repetition and regime; the same-process check (node_c_check) is reported as the maximum absolute difference from the
benchmark NODE predictions, without a verdict.
Output (results/sensitivity/runs/<run>/report/): sensitivity_per_fit.csv, arm_values.csv, arm_domain.csv, sign_flips.csv,
hpo_seed_params.csv, node_prep_diff.csv (when the arm is present), rerun_ranks.csv, arm_summary.json. For arms with fewer than 3
rerun models per domain (such as the NODE preprocessing arm) the roster rank correlation is not interpreted
(roster_rank_corr_uninformative in the summary).
Merging runs (NODE and TabNet run separately): --sensitivity-run-id <main run>,<NODE/TabNet run>; the runs must share the
selection, the number of repetitions and the benchmark tuning run, and the report stops if the same (arm, model) appears in two
runs. The output goes to report/ of the first run folder (one run) or report_with_<other runs>/ (several).
Usage: python scripts/07c_sensitivity_report.py --sensitivity-run-id <run>[,<run>] --benchmark-run-id <run> [--partial] [--smoke]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from defectshift import config, hpo, metrics  # noqa: E402
from defectshift.utils import read_per_fit
from defectshift.candidates import load_data  # noqa: E402
from defectshift.models import extra_models  # noqa: E402
from defectshift.directions import same_direction  # noqa: E402

SENSITIVITY_KEY = ["arm", "slot", "repeat", "model", "regime"]
QUANTS = ("ID_match:rmse_ood", "OOD_match:rmse_ood", "target_reference_gap:gap_rmse_pct", "target_reference_gap:gap_mae_pct", "target_reference_gap:r2_difference")


def _sensitivity_module():
    from defectshift import sensitivity
    return sensitivity


def _m(y_true, y_pred):
    m = metrics.regression(y_true, y_pred)
    return dict(r2=m["r2"], rmse=m["rmse"], mae=m["mae"])


def sensitivity_per_fit(run_id, y, benchmark_eval):
    """Metrics of the successful sensitivity rows (the last row per key) recomputed from the predictions. benchmark_eval: domain ->
    (A_heldout, B_te) rows (from the benchmark predictions)."""
    sensitivity = _sensitivity_module()
    rows = pd.read_csv(os.path.join(config.RESULTS_DIR, "sensitivity", "runs", run_id, "rows.csv"))
    ok = rows[rows.status == "ok"].drop_duplicates(SENSITIVITY_KEY, keep="last")
    out, worst = [], 0.0
    for r in ok.to_dict("records"):
        z = np.load(sensitivity.pred_prefix(run_id, r["slot"], r["model"], r["regime"], r["arm"], int(r["repeat"])) + ".pred.npz")
        held, te = benchmark_eval[r["candidate"]]
        if not (np.array_equal(z["id_idx"], held) and np.array_equal(z["ood_idx"], te)):
            raise SystemExit(f"sensitivity {r['arm']} {r['candidate']} {r['model']} {r['regime']} r{r['repeat']}: evaluation rows differ from the benchmark")
        mid, mood = _m(y.loc[z["id_idx"]], z["id_pred"]), _m(y.loc[z["ood_idx"]], z["ood_pred"])
        for k in mid:
            worst = max(worst, abs(mid[k] - r[f"{k}_id"]), abs(mood[k] - r[f"{k}_ood"]))
        out.append(dict(arm=r["arm"], candidate=r["candidate"], repeat=int(r["repeat"]), model=r["model"], regime=r["regime"],
                        **{f"{k}_id": v for k, v in mid.items()}, **{f"{k}_ood": v for k, v in mood.items()}))
    return pd.DataFrame(out), worst, ok


def domain_values(pf):
    """per_fit(candidate, repeat, model, regime, r2/rmse/mae_ood) -> repetition mean per (candidate, model, quantity)."""
    long = []
    for reg in ("ID_match", "OOD_match"):
        p = pf[pf.regime == reg]
        long.append(p.assign(quantity=f"{reg}:rmse_ood", value=p.rmse_ood)[["candidate", "repeat", "model", "quantity", "value"]])
    a = pf[pf.regime == "ID_match"].set_index(["candidate", "repeat", "model"])
    b = pf[pf.regime == "OOD_match"].set_index(["candidate", "repeat", "model"])
    for k in a.index.intersection(b.index):
        d = metrics.target_reference_gap(dict(r2=a.loc[k, "r2_ood"], rmse=a.loc[k, "rmse_ood"], mae=a.loc[k, "mae_ood"]),
                           dict(r2=b.loc[k, "r2_ood"], rmse=b.loc[k, "rmse_ood"], mae=b.loc[k, "mae_ood"]))
        for q in ("gap_rmse_pct", "gap_mae_pct", "r2_difference"):
            long.append(pd.DataFrame([dict(candidate=k[0], repeat=k[1], model=k[2], quantity=f"target_reference_gap:{q}", value=d[q])]))
    long = pd.concat(long, ignore_index=True)
    return long.groupby(["candidate", "model", "quantity"]).value.mean()


def arm_table(benchmark_pf, sensitivity_pf, arm, repeats, partial):
    """Arm table: the benchmark values with the (domain, model) pairs rerun by the arm replaced by the sensitivity values. Returns (arm
    per_fit, set of rerun (domain, model))."""
    a = sensitivity_pf[sensitivity_pf.arm == arm]
    need = {(r, g) for r in repeats for g in ("ID_match", "OOD_match")}
    rerun = set()
    for (c, m), g in a.groupby(["candidate", "model"]):
        have = set(zip(g.repeat, g.regime))
        if have >= need:
            rerun.add((c, m))
        elif not partial:
            raise SystemExit(f"{arm} {c} {m}: repetitions and regimes {sorted(need - have)} missing (--partial leaves them out)")
    doms = sorted({c for c, _ in rerun})
    base = benchmark_pf[benchmark_pf.candidate.isin(doms)].set_index(["candidate", "repeat", "model", "regime"])
    new = a[[(c, m) in rerun for c, m in zip(a.candidate, a.model)]].set_index(["candidate", "repeat", "model", "regime"])
    cols = ["r2_ood", "rmse_ood", "mae_ood"]
    out = base[cols].copy()
    out.loc[new.index, cols] = new[cols]
    return out.reset_index(), rerun


def compare(v1, v2, rerun, arm):
    """Benchmark (v1) vs arm (v2) per domain and quantity."""
    rows = []
    for (c, q), _ in v1.groupby(level=["candidate", "quantity"]):
        s1 = v1.xs((c, q), level=["candidate", "quantity"])
        s2 = v2.xs((c, q), level=["candidate", "quantity"]).reindex(s1.index)
        r = dict(arm=arm, candidate=c, quantity=q, n_models=len(s1), n_rerun=sum((c, m) in rerun for m in s1.index))
        if len(s1) >= 3:
            r["spearman"] = float(stats.spearmanr(s1, s2).statistic)
            r["kendall"] = float(stats.kendalltau(s1, s2).statistic)
            k = 3
            r["top3_overlap"] = len(set(s1.nsmallest(k).index) & set(s2.nsmallest(k).index)) / k
        rr = [m for m in s1.index if (c, m) in rerun]
        if len(rr) >= 3:   # ranks among the rerun models (so that unchanged models do not inflate the correlation)
            r["spearman_rerun"] = float(stats.spearmanr(s1[rr], s2[rr]).statistic)
            r["kendall_rerun"] = float(stats.kendalltau(s1[rr], s2[rr]).statistic)
        if q.startswith("target_reference_gap"):
            r["sign_agree_rerun"] = float(same_direction(s1[rr], s2[rr]).mean()) if rr else np.nan
            r["consensus_benchmark"], r["consensus_arm"] = float(s1.median()), float(s2.median())
            r["consensus_sign_same"] = bool(same_direction(r["consensus_benchmark"], r["consensus_arm"]))
        rows.append(r)
    return rows


def rerun_ranks(v1, v2, rerun, arm):
    """Per rerun (domain, model), its rank within the roster (smaller is better, average for ties): benchmark value vs arm value (for arms
    that change a single model, the rank correlation over the whole roster is structurally close to 1 and uninformative, so the rank change
    of that model is shown directly)."""
    out = []
    for (c, q), _ in v1.groupby(level=["candidate", "quantity"]):
        s1 = v1.xs((c, q), level=["candidate", "quantity"])
        s2 = v2.xs((c, q), level=["candidate", "quantity"]).reindex(s1.index)
        r1, r2 = s1.rank(method="average"), s2.rank(method="average")
        for m in s1.index:
            if (c, m) in rerun:
                out.append(dict(arm=arm, candidate=c, quantity=q, model=m, n_models=len(s1), rank_benchmark=float(r1[m]), rank_arm=float(r2[m]),
                                rank_change=float(r2[m] - r1[m]), value_benchmark=float(s1[m]), value_arm=float(s2[m])))
    return out


def sign_flips(av):
    """Cells of the arm_values table where the target-reference gap sign of a rerun (domain, model) differs from the benchmark. 0 counts as a sign of its
    own. Missing cells are left out."""
    cols = ["arm", "candidate", "model", "quantity", "benchmark", "arm_value"]
    if av.empty:
        return pd.DataFrame(columns=cols)
    d = av[av.rerun & av.quantity.str.startswith("target_reference_gap") & av.benchmark.notna() & av.arm_value.notna()]   # missing values left out of the sign comparison
    return d[~same_direction(d.benchmark, d.arm_value)][cols].sort_values(cols[:4]).reset_index(drop=True)


def node_prep_diff(runs, benchmark_pf, y):
    """Per fit of the NODE preprocessing arm, its predictions vs the benchmark NODE (main setting) predictions on the same evaluation rows."""
    sensitivity = _sensitivity_module()
    base = benchmark_pf[benchmark_pf.model == "NODE"].set_index(["candidate", "repeat", "regime"])
    out = []
    for run_id in runs:
        rows = pd.read_csv(os.path.join(config.RESULTS_DIR, "sensitivity", "runs", run_id, "rows.csv"))
        ok = rows[(rows.status == "ok") & (rows.arm == sensitivity.NODE_PREP_ARM)].drop_duplicates(SENSITIVITY_KEY, keep="last")
        for r in ok.to_dict("records"):
            k = (r["candidate"], int(r["repeat"]), r["regime"])
            if k not in base.index:
                raise SystemExit(f"NODE {k} not in the final benchmark")
            za = np.load(sensitivity.pred_prefix(run_id, r["slot"], "NODE", r["regime"], r["arm"], int(r["repeat"])) + ".pred.npz")
            zc = np.load(base.loc[k, "pred_path"])
            row = dict(candidate=k[0], repeat=k[1], regime=k[2])
            for part in ("id", "ood"):
                if not np.array_equal(za[f"{part}_idx"], zc[f"{part}_idx"]):
                    raise SystemExit(f"NODE {k} {part}: evaluation rows of the two preprocessings differ")
                pa, pc, yt = za[f"{part}_pred"], zc[f"{part}_pred"], y.loc[za[f"{part}_idx"]].to_numpy()
                d = np.abs(pa - pc)
                ma, mc = metrics.regression(yt, pa), metrics.regression(yt, pc)
                row.update({f"{part}_mean_abs_pred_diff": float(d.mean()), f"{part}_max_abs_pred_diff": float(d.max()),
                            f"{part}_p95_abs_pred_diff": float(np.percentile(d, 95)),
                            f"{part}_pred_corr": float(np.corrcoef(pa, pc)[0, 1]) if pa.std() > 0 and pc.std() > 0 else np.nan,
                            **{f"{part}_{q}_A_minus_C": float(ma[q] - mc[q]) for q in ("rmse", "mae", "r2")}})
            out.append(row)
    return pd.DataFrame(out)


def node_c_check(runs, benchmark_pf):
    """Predictions of the one NODE cell retrained with the main setting in the same run vs the benchmark NODE predictions (same evaluation
    rows). Differences only, no verdict."""
    sensitivity = _sensitivity_module()
    base = benchmark_pf[benchmark_pf.model == "NODE"].set_index(["candidate", "repeat", "regime"])
    out = []
    for run_id in runs:
        rows = pd.read_csv(os.path.join(config.RESULTS_DIR, "sensitivity", "runs", run_id, "rows.csv"))
        ok = rows[(rows.status == "ok") & (rows.arm == sensitivity.NODE_C_CHECK_ARM)].drop_duplicates(SENSITIVITY_KEY, keep="last")
        for r in ok.to_dict("records"):
            k = (r["candidate"], int(r["repeat"]), r["regime"])
            if k not in base.index:
                raise SystemExit(f"NODE {k} not in the final benchmark")
            za = np.load(sensitivity.pred_prefix(run_id, r["slot"], "NODE", r["regime"], r["arm"], int(r["repeat"])) + ".pred.npz")
            zc = np.load(base.loc[k, "pred_path"])
            d = dict(run=run_id, candidate=k[0], repeat=k[1], regime=k[2],   # number of fits earlier in the same process (0 = the check alone in a new process)
                     proc_pid=r.get("proc_pid"), proc_start=r.get("proc_start"), proc_prior_fits=r.get("proc_prior_fits"))
            for part in ("id", "ood"):
                if not np.array_equal(za[f"{part}_idx"], zc[f"{part}_idx"]):
                    raise SystemExit(f"NODE {k} {part}: evaluation rows differ")
                d[f"{part}_max_abs_pred_diff"] = float(np.max(np.abs(za[f"{part}_pred"] - zc[f"{part}_pred"])))
            out.append(d)
    return out


def _flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def hpo_seed_params(sensitivity_run, sensitivity_man, ok_rows):
    """Per (domain, model) of the hpo_seed arm: the benchmark's source-side search values vs the values searched on the new split."""
    sensitivity = _sensitivity_module()
    out, missing = [], []
    for arm in [a for a in sensitivity_man["arms"] if a.startswith("hpo_seed")]:
        s = int(arm[8:])
        for (slot, cand, model) in ok_rows[ok_rows.arm == arm][["slot", "candidate", "model"]].drop_duplicates().itertuples(index=False):
            p_new = sensitivity.hpo_prefix(sensitivity_run, slot, model, s)
            old_run = sensitivity_man.get("extra_hpo_run_id") if model in extra_models.EXTRA_TUNED else sensitivity_man["hpo_run_id"]
            p_old = hpo.record_prefix(old_run, slot, model, "source")
            if not (hpo.done(p_new) and hpo.done(p_old)):   # without a record, the number left out of the comparison is recorded in the summary
                missing.append(dict(arm=arm, candidate=cand, model=model, new=hpo.done(p_new), benchmark=hpo.done(p_old)))
                continue
            new, old = hpo.load(p_new), hpo.load(p_old)
            fn, fo = _flat(new["params"]), _flat(old["params"])
            keys = sorted(set(fn) | set(fo))
            same = [k for k in keys if k in fn and k in fo and fn[k] == fo[k]]
            logr = [abs(np.log10(fn[k] / fo[k])) for k in keys if k in fn and k in fo
                    and isinstance(fn[k], (int, float)) and isinstance(fo[k], (int, float)) and not isinstance(fn[k], bool)
                    and fn[k] > 0 and fo[k] > 0]
            out.append(dict(arm=arm, candidate=cand, model=model, n_params=len(keys), share_same=len(same) / len(keys) if keys else 1.0,
                            max_abs_log10_ratio=max(logr) if logr else 0.0, best_score_benchmark=old.get("best_score"),
                            best_score_new=new.get("best_score"), val_jaccard_vs_benchmark=new.get("val_jaccard_vs_benchmark"),
                            params_benchmark=json.dumps(fo, default=str), params_new=json.dumps(fn, default=str)))
    return pd.DataFrame(out), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sensitivity-run-id", required=True, help="several runs comma-separated (main run first, then the separate NODE/TabNet run)")
    ap.add_argument("--benchmark-run-id", required=True)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="also accept sensitivity runs that are not main runs (manifest full_run; smoke runs, tests); marked in the summary")
    args = ap.parse_args()
    runs = args.sensitivity_run_id.split(",")
    if len(set(runs)) != len(runs):
        raise SystemExit(f"duplicate --sensitivity-run-id: {runs}")
    sensitivity_dir = os.path.join(config.RESULTS_DIR, "sensitivity", "runs", runs[0])
    mans = {r: json.load(open(os.path.join(config.RESULTS_DIR, "sensitivity", "runs", r, "manifest.json")))["args"] for r in runs}
    man = mans[runs[0]]
    for r, m in mans.items():   # whether the runs can be merged: same selection, number of repetitions and benchmark tuning run
        for k in ("screening_final_domains_sha256", "repeats", "hpo_run_id"):
            if m.get(k) != man.get(k):
                raise SystemExit(f"{k} of sensitivity run {r} differs from {runs[0]} — cannot be merged")
    not_full = [r for r, m in mans.items() if not m.get("full_run")]
    if not_full and not args.smoke:   # the runner records the main-run verdict (arms, domains, repetitions, trials, epoch cap) in the manifest
        raise SystemExit(f"sensitivity runs that are not main runs: {not_full} (--smoke for smoke runs and tests)")
    for a_ in {a_ for m in mans.values() for a_ in m["arms"]}:   # the same arm must have the same domains in every run to be merged into one roster
        doms = {r: tuple(m["plan_domains"][a_]) for r, m in mans.items() if a_ in m["arms"]}
        if len(set(doms.values())) > 1:
            raise SystemExit(f"the domains of arm {a_} differ between runs: {doms}")
    seen = {}
    for r, m in mans.items():   # the same (arm, model) in two runs makes the value to use ambiguous -> stop
        for a_ in m["arms"]:
            for mod in m["plan"][a_]:
                if (a_, mod) in seen:
                    raise SystemExit(f"{a_} {mod} is in both sensitivity runs {seen[(a_, mod)]} and {r}")
                seen[(a_, mod)] = r
    arms = list(dict.fromkeys(a_ for r in runs for a_ in mans[r]["arms"]))
    benchmark_man = json.load(open(os.path.join(config.RESULTS_DIR, "benchmark", "runs", args.benchmark_run_id, "manifest.json")))["args"]
    if benchmark_man.get("screening_final_domains_sha256") != man["screening_final_domains_sha256"]:
        raise SystemExit("the sensitivity runs and the benchmark come from different domain selections")
    # the benchmark tuning run of each model must equal the tuning run used by the sensitivity runs. For a merged run (benchmark_merged), checked per model through hpo_run_id of the source run manifests
    if benchmark_man.get("stage") == "benchmark_merged":
        benchmark_hpo = {m: json.load(open(os.path.join(config.RESULTS_DIR, "benchmark", "runs", r, "manifest.json")))["args"].get("hpo_run_id")
                         for r, s in benchmark_man["sources"].items() for _, m in s["pairs"]}
    else:
        benchmark_hpo = {m: benchmark_man.get("hpo_run_id") for m in benchmark_man.get("models", [])}
    want = {m: (mm.get("extra_hpo_run_id") if m in extra_models.EXTRA_TUNED else mm["hpo_run_id"])
            for mm in mans.values() for a_ in mm["arms"] for m in mm["plan"][a_] if m in extra_models.tuned_all()}
    bad = sorted(m for m, r in want.items() if benchmark_hpo.get(m) != r)
    if bad:
        raise SystemExit(f"models whose sensitivity and benchmark results come from different tuning runs: {bad}")
    repeats = list(range(man["repeats"]))
    benchmark_all = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", args.benchmark_run_id, "report", "per_fit.csv"))
    benchmark_pf = benchmark_all[benchmark_all.repeat.isin(repeats) & benchmark_all.regime.isin(["ID_match", "OOD_match"])]
    benchmark_eval = {}
    for c, g in benchmark_pf.groupby("candidate"):
        z = np.load(g.pred_path.iloc[0])
        benchmark_eval[c] = (z["id_idx"], z["ood_idx"])
    _, y = load_data()
    parts = {r: sensitivity_per_fit(r, y, benchmark_eval) for r in runs}
    sensitivity_pf = pd.concat([p[0] for p in parts.values()], ignore_index=True)
    worst = max(p[1] for p in parts.values())
    # Completeness: full_run of the manifest marks the start, so a run that stopped halfway or skipped models after a failed search also
    # passes. Check that the successful rows cover the expected keys of the runner (arm x arm domains x arm models x repetitions x two
    # regimes) (skipped with --partial)
    if not args.partial:
        for r in runs:
            m = mans[r]
            have = set(zip(parts[r][2].arm, parts[r][2].candidate, parts[r][2].repeat.astype(int), parts[r][2].model, parts[r][2].regime))
            sc = m.get("arm_scope", {})   # repetitions and regimes reduced per arm (the same-process check and so on)
            exp = {(a_, n, i, mo, g) for a_ in m["arms"] for n in m["plan_domains"][a_] for mo in m["plan"][a_]
                   for i in sc.get(a_, {}).get("repeats", range(m["repeats"]))
                   for g in sc.get(a_, {}).get("regimes", ("ID_match", "OOD_match"))}
            lack = sorted(exp - have)
            if lack:
                raise SystemExit(f"sensitivity run {r}: {len(lack)} of {len(exp)} expected keys have no successful row (e.g. {lack[:3]}). "
                                 "Rerun the runner with the same --run-id, or use --partial to look at it with the gaps")
    if worst > 1e-9:
        raise SystemExit(f"metrics recomputed from the sensitivity predictions differ from rows.csv (maximum difference {worst:.2e})")

    v1_all = domain_values(benchmark_pf)
    vals, doms, rank_rows, summ = [], [], [], dict(sensitivity_run_id=runs, benchmark_run_id=args.benchmark_run_id, repeats=repeats, arms={},
                                    cuda_env={r: m.get("cuda_env") for r, m in mans.items()},
                                    full_run={r: bool(m.get("full_run")) for r, m in mans.items()}, smoke=bool(args.smoke))
    for arm in arms:
        if arm == _sensitivity_module().NODE_C_CHECK_ARM:   # a one-cell check, reported separately instead of the common comparison (node_c_check)
            summ["arms"][arm] = "same-process check of the main setting — see node_c_check in the summary"
            continue
        t, rerun = arm_table(benchmark_pf, sensitivity_pf, arm, repeats, args.partial)
        if not rerun:
            summ["arms"][arm] = "no results"
            continue
        v2 = domain_values(t)
        v1 = v1_all.loc[sorted({c for c, _ in rerun})]
        vals.append(pd.DataFrame(dict(benchmark=v1, arm_value=v2.reindex(v1.index))).reset_index().assign(
            arm=arm, rerun=lambda d: [(c, m) in rerun for c, m in zip(d.candidate, d.model)]))
        rows = compare(v1, v2, rerun, arm)
        doms += rows
        dd = pd.DataFrame(rows)
        rr = pd.DataFrame(rerun_ranks(v1, v2, rerun, arm))
        rank_rows.append(rr)
        few = int(dd.n_rerun.max()) < 3          # with fewer than 3 rerun models per domain the roster rank correlation is not interpreted
        summ["arms"][arm] = {q: dict(n_domains=len(g), spearman_median=float(g.spearman.median()), spearman_min=float(g.spearman.min()),
                                     kendall_median=float(g.kendall.median()),
                                     spearman_rerun_median=float(g.spearman_rerun.median()) if "spearman_rerun" in g and g.spearman_rerun.notna().any() else None,
                                     kendall_rerun_median=float(g.kendall_rerun.median()) if "kendall_rerun" in g and g.kendall_rerun.notna().any() else None,
                                     sign_agree_rerun_median=float(g.sign_agree_rerun.median()) if "sign_agree_rerun" in g and g.sign_agree_rerun.notna().any() else None,
                                     consensus_sign_same=int(g.consensus_sign_same.sum()) if "consensus_sign_same" in g and g.consensus_sign_same.notna().any() else None,
                                     roster_rank_corr_uninformative=few,
                                     rerun_rank_change=dict(n_cells=int((rr.quantity == q).sum()),
                                                            n_changed=int(((rr.quantity == q) & (rr.rank_change != 0)).sum()),
                                                            median_abs=float(rr[rr.quantity == q].rank_change.abs().median()),
                                                            max_abs=float(rr[rr.quantity == q].rank_change.abs().max())))
                             for q, g in dd.groupby("quantity")}
    out = os.path.join(sensitivity_dir, "report" if len(runs) == 1 else "report_with_" + "+".join(runs[1:]))
    os.makedirs(out, exist_ok=True)
    sensitivity_pf.to_csv(os.path.join(out, "sensitivity_per_fit.csv"), index=False)
    av = pd.concat(vals, ignore_index=True) if vals else pd.DataFrame()
    av.to_csv(os.path.join(out, "arm_values.csv"), index=False)
    flips = sign_flips(av)
    flips.to_csv(os.path.join(out, "sign_flips.csv"), index=False)
    (pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()).to_csv(os.path.join(out, "rerun_ranks.csv"), index=False)
    for arm, a_s in summ["arms"].items():   # per arm and quantity: number of cells with a changed sign / number of rerun cells
        if isinstance(a_s, dict):
            for q, r in a_s.items():
                if q.startswith("target_reference_gap"):
                    r["n_rerun_cells"] = int(((av.arm == arm) & (av.quantity == q) & av.rerun).sum())
                    r["n_sign_flips"] = int(((flips.arm == arm) & (flips.quantity == q)).sum())
    pd.DataFrame(doms).to_csv(os.path.join(out, "arm_domain.csv"), index=False)
    hps = [hpo_seed_params(r, mans[r], parts[r][2]) for r in runs]
    hp = pd.concat([h for h, _ in hps], ignore_index=True) if any(len(h) for h, _ in hps) else pd.DataFrame()
    hp_missing = [x for _, ms in hps for x in ms]
    hp.to_csv(os.path.join(out, "hpo_seed_params.csv"), index=False)
    if len(hp):
        summ["hpo_seed_params"] = dict(n=len(hp), share_identical_configs=float((hp.share_same == 1).mean()),
                                       median_share_same=float(hp.share_same.median()))
    summ["hpo_seed_params_missing"] = hp_missing      # (arm, domain, model) not compared because a search record is missing
    summ["metric_recompute_max_abs_diff"] = worst
    if any(_sensitivity_module().NODE_C_CHECK_ARM in mans[r]["arms"] for r in runs):
        summ["node_c_check"] = node_c_check(runs, benchmark_pf)
    if any(_sensitivity_module().NODE_PREP_ARM in mans[r]["arms"] for r in runs):
        nd = node_prep_diff(runs, benchmark_pf, y)
        nd.to_csv(os.path.join(out, "node_prep_diff.csv"), index=False)
        if len(nd):   # prediction differences and the three metric differences (example default minus main setting), per evaluation side (ID = A_heldout, OOD = B_te)
            summ["node_prep_a"] = dict(n_fits=len(nd), **{
                f"{part}_{k}": v for part in ("id", "ood") for k, v in dict(
                    mean_abs_pred_diff_median=float(nd[f"{part}_mean_abs_pred_diff"].median()),
                    p95_abs_pred_diff_median=float(nd[f"{part}_p95_abs_pred_diff"].median()),
                    max_abs_pred_diff_max=float(nd[f"{part}_max_abs_pred_diff"].max()),
                    pred_corr_median=float(nd[f"{part}_pred_corr"].median()),
                    **{f"{q}_A_minus_C_median": float(nd[f"{part}_{q}_A_minus_C"].median()) for q in ("rmse", "mae", "r2")},
                    **{f"{q}_A_minus_C_range": [float(nd[f"{part}_{q}_A_minus_C"].min()), float(nd[f"{part}_{q}_A_minus_C"].max())]
                       for q in ("rmse", "mae", "r2")}).items()})
    with open(os.path.join(out, "arm_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
