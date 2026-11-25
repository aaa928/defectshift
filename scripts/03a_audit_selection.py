"""Selection audit: evaluates B_te with the saved screening models, without retraining.

Purpose: check whether the screening scores (B_screen) hold on a test set never seen before (B_te), i.e. out-of-sample
stability. This is not a new selection. selection/final_domains.json of the screening run is only read, and these results do not
change the selection rule.

Conditions
- runs only if selection/final_domains.json exists in the screening run folder (B_te is opened only after the selection)
- the screening run must be a complete main run (selection.validate_run)
- models are obtained through regimes.load only. No parameter training: during the run, regimes.fit raises except for rebuilding
  the context of a foundation model (rebuilt from the saved context rows and seed, no parameter training), and the training entry
  points (protocol.fit_ml, fit_transformer, fit_dl2, tune_ml) always raise. If one load calls regimes.fit other than once for a
  foundation model and zero times otherwise, that row fails
- evaluation rows: B_te (`B_view(..., "selection_audit")`). As a reload check B_screen is predicted too and compared with the
  RMSE of the screening result row (that row fails if they differ)
- the saved training rows must equal the rows recomputed by the protocol: ID_match = A_match, OOD_match = B_match (within B_tr),
  and for early-stopping models the internal 80/20 split (fit, es) as well. The training rows (fit + es) must not overlap B_te,
  B_screen or A_heldout, and the lots of the OOD_match training rows must not overlap those of B_te
- result rows: B_te metrics (r2_te, rmse_te, mae_te), the B_screen reload difference, and te_lot_in_train = share of B_te rows
  whose lot appears in that regime's training set (fit + early-stopping rows) (lot exposure of the training set)

Output: results/selection_audit/runs/<run_id>/{manifest.json, rows.csv, report/{candidates.csv, report.json}}
Report: B_screen consensus score vs B_te consensus score (computed with the same selection function): rank correlation and sign
agreement, B_te rank and percentile of the five selected process-condition domains, how many stay in the B_te top 5 and top 10, the
score shrinkage (B_screen - B_te) of selected vs not selected candidates, and the B_te scores of the two pre-specified temporal
domains.

Usage
  python scripts/03a_audit_selection.py --screening-run-id <screening> --run-id audit_smoke --only cat_050,cat_001 --repeats 1   # smoke
  python scripts/03a_audit_selection.py --screening-run-id <screening> --run-id <run> --placebo                                 # main run
  python scripts/03a_audit_selection.py --screening-run-id <screening> --run-id <run> --placebo --report-only                   # recompute the report only
"""
import argparse
import contextlib
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from defectshift import screening
import torch  # noqa: E402

from defectshift import config, manifest, metrics, regimes, registry, selection  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_candidates, load_data, parse_candidate  # noqa: E402
from defectshift.models import protocol  # noqa: E402
from defectshift.splits import B_view, es_split, get_A, matched, repeat_split  # noqa: E402
from defectshift.runio import check_same_inputs  # noqa: E402
from defectshift.utils import free_memory, log, new_run_id  # noqa: E402
from defectshift.directions import same_direction  # noqa: E402

KEY = ["slot", "repeat", "model", "regime"]
SCREEN_RMSE_TOL = 1e-12   # reload check: the B_screen RMSE of the saved model must be within this of the screening result row (deterministic reproduction)
_FIT_CALLS = {"context_rebuild": 0}


def _guarded_fit(spec, *args, **kwargs):
    """No training in the selection audit. Foundation models only rebuild their context from the saved context rows and seed (no
    parameter training)."""
    if spec.route != "foundation":
        raise RuntimeError(f"training call for {spec.name} in the selection audit — only reloading saved models is allowed")
    _FIT_CALLS["context_rebuild"] += 1
    return _ORIG_FIT(spec, *args, **kwargs)


_ORIG_FIT = regimes.fit


def _blocked(name):
    def f(*args, **kwargs):
        raise RuntimeError(f"training route {name} called in the selection audit — only reloading saved models is allowed")
    return f


_TRAIN_ENTRY = ("fit_ml", "fit_transformer", "fit_dl2", "tune_ml")


@contextlib.contextmanager
def no_training():
    """Block training while running. regimes.fit passes only the context rebuild of foundation models (the fit call inside
    regimes.load goes through this function too, as a module-global name); the training entry points of protocol always raise.
    On exit the functions present on entry are restored."""
    prev_fit, prev_proto = regimes.fit, {n: getattr(protocol, n) for n in _TRAIN_ENTRY}
    regimes.fit = _guarded_fit
    for n in _TRAIN_ENTRY:
        setattr(protocol, n, _blocked(n))
    try:
        yield
    finally:
        regimes.fit = prev_fit
        for n, f in prev_proto.items():
            setattr(protocol, n, f)


def screening_paths(screening_run_id):
    run_dir = os.path.join(config.RESULTS_DIR, "screening", "runs", screening_run_id)
    return dict(run_dir=run_dir, rows=os.path.join(run_dir, "rows.csv"), manifest=os.path.join(run_dir, "manifest.json"),
                final=os.path.join(run_dir, "selection", "final_domains.json"),
                models=os.path.join(config.CACHE_DIR, "screening", "runs", screening_run_id, "models"))


def check_screening_run(p, require_tuned=True):
    """Whether the screening run is a complete main run with the selection done. Returns (screening result rows, screening
    manifest, final_domains)."""
    if not os.path.exists(p["final"]):
        raise SystemExit(f"{p['final']} does not exist; the selection audit runs only after the selection (02_select_domains.py)")
    rows = pd.read_csv(p["rows"])
    man = json.load(open(p["manifest"]))
    final = json.load(open(p["final"]))
    if final["rows_sha256"] != manifest.file_sha256(p["rows"]) or final["manifest_sha256"] != manifest.file_sha256(p["manifest"]):
        raise SystemExit("screening rows.csv or manifest.json differ from the time of the selection (hash mismatch)")
    base = load_candidates()
    placebo = [n for n in load_candidates(include_placebo=True) if n not in base]
    selection.validate_run(rows, man["args"], base, placebo, require_tuned=require_tuned)
    check_same_inputs(man, "screening run")
    return rows, man, final


def evaluate(args, screening_rows, names, models, run_id, rows_path, model_dir):
    df, y = load_data()
    lot = df[config.LOT_COL]
    screening_ok = screening_rows[screening_rows.status == "ok"].drop_duplicates(KEY, keep="last").set_index(KEY)
    done = screening.done_keys(rows_path)
    for name in names:
        kind, _ = parse_candidate(name)
        slot, mA, mB = build_one(df, name)
        A_idx, B_idx = df.index[mA], df.index[mB]
        a = get_A(slot, A_idx, y, lot)
        bte = B_view(slot, B_idx, y, lot, "selection_audit")["B_te"]
        bv = B_view(slot, B_idx, y, lot, "screening")                   # B_tr (recomputing training rows) and B_screen (reload check), the sets already used in screening
        bscr = bv["B_screen"]
        forbidden = set(bte) | set(bscr) | set(a["A_heldout"])
        te_lots = lot.loc[bte]
        cat_cols, num_cols = feature_sets(df, name)
        ctx = regimes.Ctx(df, y, cat_cols, num_cols)
        t_slot = time.time()
        for i in range(args.repeats):
            A_train, _ = repeat_split(a["A_dev"], y, lot, i)
            _, A_match, B_match = matched(A_train, bv["B_tr"], i)
            for spec in models:
                for regime in ("ID_match", "OOD_match"):
                    if (slot, str(i), spec.name, regime) in done:
                        continue
                    prefix = screening.artifact_prefix(model_dir, slot, spec.name, regime, i)
                    screening_row = screening_ok.loc[(slot, i, spec.name, regime)]
                    row = dict(run_id=run_id, screening_run_id=args.screening_run_id, slot=slot, candidate=name, kind=kind, repeat=i,
                               model=spec.name, family=spec.family, regime=regime, n_A=len(A_idx), n_B=len(B_idx),
                               n_star=int(screening_row.n_star), b_te_eligible=bool(screening_row.b_te_eligible), n_B_te=len(bte),
                               n_B_screen=len(bscr))
                    try:
                        if not screening.artifacts_ok(prefix, spec.route):
                            raise RuntimeError(f"screening artifacts missing or damaged: {prefix}")
                        if args.need_screen_pred and not os.path.exists(prefix + ".pred.npz"):   # the tuned screening always saves them
                            raise RuntimeError(f"no B_screen prediction file (no value comparison possible): {prefix}.pred.npz")
                        n_rebuild0 = _FIT_CALLS["context_rebuild"]
                        f = regimes.load(spec, ctx, prefix)
                        n_calls = _FIT_CALLS["context_rebuild"] - n_rebuild0
                        if n_calls != (1 if spec.route == "foundation" else 0):
                            raise RuntimeError(f"{n_calls} training calls during loading (must be 1 for foundation models, 0 otherwise)")
                        pool = A_match if regime == "ID_match" else B_match
                        fit_exp, es_exp = es_split(pool, y, lot, i) if spec.uses_es else (pool, None)
                        if set(f.fit_idx) != set(fit_exp) or (set() if f.es_idx is None else set(f.es_idx)) != (set() if es_exp is None else set(es_exp)):
                            raise RuntimeError("saved training rows differ from the rows recomputed by the protocol (n* draw, internal early-stopping split)")
                        used = set(f.fit_idx) | (set() if f.es_idx is None else set(f.es_idx))
                        if used & forbidden:
                            raise RuntimeError("saved training rows overlap B_te, B_screen or A_heldout")
                        if regime == "OOD_match" and te_lots.isin(set(lot.loc[list(used)])).any():
                            raise RuntimeError("lots of the target training rows overlap B_te")
                        t0 = time.time()
                        p_te = regimes.predict(f, ctx, bte)
                        p_scr = regimes.predict(f, ctx, bscr)
                        rmse_scr = metrics.regression(y.loc[bscr], p_scr)["rmse"]
                        diff = abs(rmse_scr - float(screening_row.rmse_screen))
                        if os.path.exists(prefix + ".pred.npz"):     # the tuned screening saves the B_screen predictions -> compared by value
                            zs = np.load(prefix + ".pred.npz")
                            if not np.array_equal(zs["scr_idx"], np.asarray(bscr)):
                                raise RuntimeError("saved B_screen rows differ from the current split")
                            row["screen_pred_max_abs_diff"] = regimes.check_reload(spec, zs["scr_pred"], p_scr)
                        row.update({f"{k}_te": v for k, v in metrics.regression(y.loc[bte], p_te).items()})
                        row.update(pred_sec=round(time.time() - t0, 2), n_fit=len(f.fit_idx),
                                   n_es=0 if f.es_idx is None else len(f.es_idx), seed=int(f.seed),
                                   context_rebuilt=n_calls == 1, rows_match_protocol=True,
                                   te_lot_in_train=float(te_lots.isin(set(lot.loc[list(used)])).mean()),
                                   screen_rmse_screening=float(screening_row.rmse_screen), screen_rmse_diff=diff)
                        if diff > SCREEN_RMSE_TOL:
                            raise RuntimeError(f"reload mismatch: B_screen RMSE difference {diff:.3e}")
                        row["status"] = "ok"
                        log(f"[selection_audit] {slot} r{i} {spec.name:7s} {regime:9s} RMSE_te={row['rmse_te']:.5f} "
                            f"R2_te={row['r2_te']:.3f} reload diff={diff:.1e} ({row['pred_sec']}s)")
                    except Exception as e:
                        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                        log(f"[selection_audit] {slot} r{i} {spec.name} {regime} failed\n{traceback.format_exc()}", level="ERROR")
                    screening.append_row(rows_path, row)
                    free_memory()
        log(f"[selection_audit] {slot} done: {time.time() - t_slot:.0f}s")


def _summary(rows, metric_suffix, models, n_repeats):
    """Consensus scores with the same selection function as for B_screen (selection.candidate_summary): only the metric column
    names are swapped."""
    r = rows.rename(columns={f"{m}_{metric_suffix}": f"{m}_screen" for m in ("r2", "rmse", "mae")})
    return selection.candidate_summary(r, models, n_repeats)


def _sign_agreement(a, b):
    """Share of pairs, both finite, with the same sign (None if there is no pair)."""
    ok_ = np.isfinite(a) & np.isfinite(b)
    return float(same_direction(a[ok_], b[ok_]).mean()) if ok_.any() else None


def _model_repeat_sign_agreement(screen_rows, te_rows):
    """Share of (candidate, model, repetition) whose target-reference gap (RMSE) sign is the same on B_screen and B_te (compared after aligning the keys)."""
    k = ["candidate", "model", "repeat"]
    a = selection.target_reference_gap_table(screen_rows).set_index(k).gap_rmse_pct
    b = selection.target_reference_gap_table(te_rows.rename(columns={f"{m}_te": f"{m}_screen" for m in ("r2", "rmse", "mae")})).set_index(k).gap_rmse_pct
    j = pd.concat([a, b], axis=1, keys=["screen", "te"], join="inner")
    return _sign_agreement(j.screen, j.te)


def report(args, screening_rows, final, rows_path, out_dir, names, models):
    rows = pd.read_csv(rows_path)
    ok = rows[rows.status == "ok"].drop_duplicates(KEY, keep="last")
    want = {(n, i, m.name, r) for n in names for i in range(args.repeats) for m in models for r in ("ID_match", "OOD_match")}
    have = set(zip(ok.candidate, ok.repeat, ok.model, ok.regime))
    if want - have:
        raise SystemExit(f"{len(want - have)} selection-audit keys unfinished — resume with the same --run-id before reporting")
    M = [m.name for m in models]
    meta = screening_rows.drop_duplicates("slot")[["slot", "candidate", "kind", "n_A", "n_B", "n_star", "b_te_eligible"]]
    te_rows = ok.drop(columns=["candidate", "kind", "n_A", "n_B", "n_star", "b_te_eligible"]).merge(meta, on="slot")
    # on the B_screen side too, only the candidates and repetitions evaluated on B_te (a different number of repetitions would change the complete-case verdict and the consensus score)
    scr_rows = screening_rows[screening_rows.candidate.isin(names) & (screening_rows.repeat < args.repeats)]
    scr = _summary(scr_rows, "screen", M, args.repeats)
    te = _summary(te_rows, "te", M, args.repeats)
    cols = ["consensus_gap_rmse_pct", "consensus_r2_difference", "n_models_consistent"] + [f"{m}_gap_rmse_pct" for m in M]
    c = scr[["candidate", "kind", "n_star", "b_te_eligible"] + cols].merge(te[["candidate"] + cols], on="candidate",
                                                                            suffixes=("_screen", "_te"))
    c["selected"] = c.candidate.isin(final["categorical"] + final["temporal"])
    c["shrink_pct"] = c.consensus_gap_rmse_pct_screen - c.consensus_gap_rmse_pct_te
    elig = c[(c.kind == "cat") & c.b_te_eligible.astype(bool)].copy()
    elig["rank_screen"] = elig.consensus_gap_rmse_pct_screen.rank(ascending=False, method="min")
    elig["rank_te"] = elig.consensus_gap_rmse_pct_te.rank(ascending=False, method="min")
    c = c.merge(elig[["candidate", "rank_screen", "rank_te"]], on="candidate", how="left")

    def rho(frame, a="consensus_gap_rmse_pct_screen", b="consensus_gap_rmse_pct_te"):
        f_ = frame[np.isfinite(frame[a]) & np.isfinite(frame[b])]   # finite pairs only
        return None if len(f_) < 3 else float(spearmanr(f_[a], f_[b]).correlation)

    sel = elig[elig.candidate.isin(final["categorical"])]
    out = dict(
        screening_run_id=args.screening_run_id, n_candidates=len(c), n_eligible_categorical=len(elig),
        spearman=dict(eligible_categorical=rho(elig), all_categorical=rho(c[c.kind == "cat"]),
                      all_candidates=rho(c[c.kind != "placebo"]),
                      per_model_eligible_categorical={m: rho(elig, f"{m}_gap_rmse_pct_screen", f"{m}_gap_rmse_pct_te") for m in M}),
        sign_agreement=dict(
            consensus_eligible_categorical=_sign_agreement(elig.consensus_gap_rmse_pct_screen, elig.consensus_gap_rmse_pct_te),
            model_repeat_all=_model_repeat_sign_agreement(scr_rows, te_rows)),
        selected_categorical=[dict(candidate=r.candidate, screen=r.consensus_gap_rmse_pct_screen, te=r.consensus_gap_rmse_pct_te,
                                   rank_screen=r.rank_screen, rank_te=r.rank_te,
                                   percentile_te=float((elig.consensus_gap_rmse_pct_te <= r.consensus_gap_rmse_pct_te).mean() * 100),
                                   n_models_consistent_te=int(r.n_models_consistent_te)) for r in sel.itertuples()],
        selected_in_te_top5=int((sel.rank_te <= 5).sum()), selected_in_te_top10=int((sel.rank_te <= 10).sum()),
        mean_shrink_pct=dict(selected=float(sel.shrink_pct.mean()) if len(sel) else None,
                             not_selected_eligible=float(elig[~elig.candidate.isin(final["categorical"])].shrink_pct.mean()) if len(elig) else None),
        temporal=[dict(candidate=r.candidate, screen=r.consensus_gap_rmse_pct_screen, te=r.consensus_gap_rmse_pct_te,
                       n_models_consistent_te=int(r.n_models_consistent_te)) for r in c[c.candidate.isin(final["temporal"])].itertuples()],
        placebo_te_range=[float(c[c.kind == "placebo"].consensus_gap_rmse_pct_te.min()), float(c[c.kind == "placebo"].consensus_gap_rmse_pct_te.max())]
        if (c.kind == "placebo").any() else None,
        te_lot_in_train=dict(target_max=float(ok[ok.regime == "OOD_match"].te_lot_in_train.max()),
                             source_median_eligible=float(ok[(ok.regime == "ID_match") & ok.candidate.isin(elig.candidate)]
                                                          .groupby("candidate").te_lot_in_train.mean().median()) if len(elig) else None),
        n_context_rebuilt=int(ok.context_rebuilt.sum()), screen_rmse_diff_max=float(ok.screen_rmse_diff.max()),
        rows_sha256=manifest.file_sha256(rows_path))
    os.makedirs(out_dir, exist_ok=True)
    c.to_csv(os.path.join(out_dir, "candidates.csv"), index=False)
    json.dump(out, open(os.path.join(out_dir, "report.json"), "w"), ensure_ascii=False, indent=1, default=float)
    log(f"[selection_audit] report -> {out_dir}\n{json.dumps(out, ensure_ascii=False, indent=1, default=float)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--screening-run-id", required=True, help="screening run whose selection is done")
    ap.add_argument("--run-id", default=None, help="selection-audit run folder name (default: a timestamp); an existing name resumes only if the manifest is the same")
    ap.add_argument("--only", default=None, help="comma-separated candidate names (smoke); default: the 78 listed candidates")
    ap.add_argument("--placebo", action="store_true", help="include the 10 random-split controls")
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS_SCREENING)
    ap.add_argument("--report-only", action="store_true", help="recompute the report only, without evaluation")
    ap.add_argument("--allow-untuned-selection", action="store_true",
                    help="only for re-auditing a fixed-setting screening run for comparison (the tuned screening is the default)")
    args = ap.parse_args()

    p = screening_paths(args.screening_run_id)
    screening_rows, screening_man, final = check_screening_run(p, require_tuned=not args.allow_untuned_selection)
    # a tuned screening run saves the B_screen predictions, so the value comparison is required (an RMSE comparison alone does not pass); a fixed-setting run has no such files, RMSE only
    args.need_screen_pred = bool(screening_man["args"].get("tune"))
    models = [registry.get(m) for m in screening_man["args"]["models"]]
    names = load_candidates(include_placebo=args.placebo)
    if args.only:
        want = args.only.split(",")
        unknown = [n for n in want if n not in load_candidates(include_placebo=True)]
        if unknown:
            raise SystemExit(f"not screening candidates: {unknown}")
        names = [n for n in load_candidates(include_placebo=True) if n in want]
    if not 1 <= args.repeats <= screening_man["args"]["repeats"]:
        raise SystemExit(f"--repeats must be 1 - {screening_man['args']['repeats']} (the repetitions of the screening run)")
    full_run = names == load_candidates() + ([n for n in load_candidates(include_placebo=True) if n not in load_candidates()]
                                             if args.placebo else []) and args.repeats == screening_man["args"]["repeats"]

    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(config.RESULTS_DIR, "selection_audit", "runs", run_id)
    rows_path = os.path.join(run_dir, "rows.csv")
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="selection_audit", screening_run_id=args.screening_run_id, screening_final_domains_sha256=manifest.file_sha256(p["final"]),
        screening_rows_sha256=final["rows_sha256"], models=[m.name for m in models], repeats=args.repeats, candidates=names,
        full_run=full_run, screen_rmse_tol=SCREEN_RMSE_TOL)))
    log(f"[selection_audit] run {run_id} ({status}) screening {args.screening_run_id} {len(names)} candidates x {args.repeats} repetitions x models "
        f"{[m.name for m in models]} -> {rows_path}")
    t_all = time.time()
    if not args.report_only:
        with no_training():
            evaluate(args, screening_rows, names, models, run_id, rows_path, p["models"])
        log(f"[selection_audit] evaluation {time.time() - t_all:.0f}s, {_FIT_CALLS['context_rebuild']} context rebuilds (foundation models only)")
    report(args, screening_rows, final, rows_path, os.path.join(run_dir, "report"), names, models)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
