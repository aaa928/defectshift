"""Final benchmark: selected domains x 5 training repetitions x model roster x training regimes.

Inputs (all finished runs): the domain selection results/screening/runs/<screening>/selection/final_domains.json (hash checked);
the tuning results/benchmark/hpo/<hpo>/ + cache/benchmark/hpo/<hpo>/ (the source record of each tuned model, shared by ID_full,
ID_match and OOD_match, and its target record for OOD_match_tuned); the foundation-model qualification
results/benchmark/fm_qualify/<fmq>/summary.json (of the three judged models, fm.QUALIFY_JUDGED: Kumo, TabDPT, EXAONE, only those
that passed; TabPFN, TabICL and TabFM are always in the roster, and a model that fails is excluded without replacement).
Selection, tuning and qualification must come from the same final_domains.json (hash). The roster is fixed in the manifest.

Repetition i (splits.py, production-lot level; the four B parts and A_heldout are the same as in screening): A_dev ->
A_train/A_val, n* = min(|A_train|, |B_tr|), A_match and B_match drawn with seed 42 + i (repetitions 0-2 draw the same rows as
screening).
Regimes: ID_full (A_train; early-stopping models stop on A_val); ID_match (A_match) and OOD_match (B_match), where early-stopping
models use an internal 80/20 lot-level split; OOD_match_tuned (tuned models only): the same rows and seed as OOD_match with the
hyperparameters of the target-side search (trained on B_tr, validated on B_val; B_val is not used for the final training).
Hyperparameters: tuned models use the source-side search values per domain (ID_full, ID_match, OOD_match) and the target-side
values (OOD_match_tuned); LinearRegression and the foundation models use defaults. Model seed = 42 + regime offset + i
(OOD_match_tuned as OOD_match). Categorical representation "benchmark". Foundation-model input: raw numerical values (missing
values kept) + categorical markers, handled by each model's official interface (Ctx.fm_input="raw_native").
Evaluation: every regime's model predicts both A_heldout (ID) and B_te (OOD); B_screen and B_val are not used.
Recorded per row: metrics (R^2, RMSE, MAE x id, ood), fit_sec, predict_id_sec, predict_ood_sec, n_id, n_ood, n_rounds_used and
n_rounds_unit (boosting rounds or epochs used after early stopping, as a count; only for models that report it), peak torch GPU
allocation, gpu_used_mb, cpu_rss_mb, load_sec (--verify-reload) and the lot exposure of the training rows (share of id and ood
evaluation rows whose lot appears in the training rows). Saved: models (regimes.save; context rows only for the foundation models)
and predictions (<prefix>.pred.npz).
Resuming: only keys with a successful row and intact artifacts count as done; prediction files are checked for keys, lengths and
finite values, and the stored row indices of a skipped key must equal the current A_heldout and B_te (otherwise the run stops).
The run exits normally only if every expected key of its scope succeeded; otherwise it lists them and exits with code 1.

Run folder: results/benchmark/runs/<run_id>/{manifest.json, rows.csv}; models and predictions: cache/benchmark/runs/<run_id>/models/.
Usage
  python scripts/06_benchmark.py --screening-run-id <screening> --hpo-run-id <hpo> --fmq-run-id <fmq> --run-id benchmark_smoke --only cat_050 --repeats 1 --verify-reload
  python scripts/06_benchmark.py --screening-run-id <screening> --hpo-run-id <hpo> --fmq-run-id <fmq> --run-id <run>     # main run
"""
import argparse
import os
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before importing torch

import numpy as np  # noqa: E402

from defectshift import config, hpo, manifest, metrics, regimes, registry  # noqa: E402
from defectshift.benchmark import (KEY, REGIMES, _mem, _peak_mb, _reset_peak, artifact_prefix, check_hpo_record,  # noqa: E402
                                   check_rows, in_roster, load_fmq, load_hpo, pred_ok, regime_rows, rounds_used)
from defectshift.candidates import build_one, feature_sets, load_data, parse_candidate  # noqa: E402
from defectshift.models import tabular_dl, foundation as fm, ml, transformer  # noqa: E402,F401
from defectshift.runio import append_row, check_stop, control_names, done_keys, load_final_domains  # noqa: E402
from defectshift.splits import B_view, get_A, hpo_split, matched, repeat_split  # noqa: E402
from defectshift.utils import free_memory, log, new_run_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--screening-run-id", required=True)
    ap.add_argument("--hpo-run-id", required=True)
    ap.add_argument("--fmq-run-id", default=None, help="foundation-model qualification run (required when foundation models run)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--only", default=None, help="comma-separated domains (candidate names); default: all selected domains")
    ap.add_argument("--controls", default=None,
                    help="comma-separated seeds of random-split controls (placebo@<seed>); if given, only the controls are run instead of the selected domains")
    ap.add_argument("--stop-file", default=None, help="stop before starting the next unit (domain, model, repetition, regime) once this file exists "
                                                    "(exit code 75); run the same command again to continue")
    ap.add_argument("--models", default=None, help="comma-separated models; default: the whole roster (foundation models only if qualified)")
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS_BENCHMARK)
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--verify-reload", action="store_true", help="reload right after saving and check that the predictions are the same")
    ap.add_argument("--verify-reload-routes", default="ml,transformer,tabular_dl,foundation",
                    help="training routes for --verify-reload (comma-separated). The qualification already checks reloading of the "
                         "foundation models, so the main run uses ml,transformer,tabular_dl")
    ap.add_argument("--allow-nonfinal-hpo", action="store_true",
                    help="smoke runs only: accept search records whose number of trials differs from the final budget "
                         "(config.HPO_N_TRIALS); makes the run a non-main run")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="smoke runs only: epoch cap for the deep models (MLP, FT-T, RealMLP, TabM); makes the run a non-main run")
    args = ap.parse_args()

    domains, final_sha = load_final_domains(args.screening_run_id)
    names = domains
    if args.only:
        want = args.only.split(",")
        bad = [n for n in want if n not in domains]
        if bad:
            raise SystemExit(f"not selected domains: {bad}")
        names = [n for n in domains if n in want]
    if args.controls:
        names = control_names(args.controls, args.only)
    hpo_args, hpo_sha = load_hpo(args.hpo_run_id, final_sha, need_compat=not args.allow_nonfinal_hpo)
    fm_pass, fmq_sha = load_fmq(args.fmq_run_id, final_sha) if args.fmq_run_id else ({}, None)
    candidates = list(registry.SPECS)
    roster = [n for n in candidates if in_roster(registry.get(n), fm_pass)]
    excluded = [n for n in candidates if n not in roster]
    models = [registry.get(m) for m in (args.models.split(",") if args.models else roster)]
    bad = [m.name for m in models if m.name not in roster]
    if bad:
        raise SystemExit(f"models outside the roster (foundation models only if qualified): {bad}")
    regimes_ = args.regimes.split(",")
    if not set(regimes_) <= set(REGIMES):
        raise SystemExit(f"--regimes must be among {REGIMES}")
    if not 1 <= args.repeats <= config.N_REPEATS_BENCHMARK:
        raise SystemExit(f"--repeats must be 1 - {config.N_REPEATS_BENCHMARK}")
    # Every search record of the tuned models must exist before starting
    need = [(n, m.name, side) for n in names for m in models if m.name in hpo.TUNED
            for side, regs in (("source", {"ID_full", "ID_match", "OOD_match"}), ("target", {"OOD_match_tuned"}))
            if regs & set(regimes_)]
    df, y = load_data()
    lot = df[config.LOT_COL]
    slots = {n: build_one(df, n)[0] for n in names}
    missing = [(n, m, s) for n, m, s in need if not hpo.done(hpo.record_prefix(args.hpo_run_id, slots[n], m, s))]
    if missing:
        raise SystemExit(f"{len(missing)} search records missing (e.g. {missing[:3]})")
    if hpo_args.get("cat_mode") != "benchmark":
        raise SystemExit("the categorical representation of the tuning run is not benchmark")
    full_run = (names == domains and [m.name for m in models] == roster and args.repeats == config.N_REPEATS_BENCHMARK
                and regimes_ == list(REGIMES) and not args.no_save and bool(hpo_args.get("full_run")) and args.fmq_run_id is not None
                and not args.allow_nonfinal_hpo and args.max_epochs is None
                and set(fm.FOUNDATION_MODELS) <= set(fm_pass))         # the main run needs qualification results for all six

    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "runs", run_id)
    model_dir = os.path.join(config.CACHE_DIR, "benchmark", "runs", run_id, "models")
    rows_path = os.path.join(run_dir, "rows.csv")
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="benchmark", screening_run_id=args.screening_run_id, screening_final_domains_sha256=final_sha, hpo_run_id=args.hpo_run_id,
        hpo_manifest_sha256=hpo_sha, fmq_run_id=args.fmq_run_id, fmq_summary_sha256=fmq_sha, roster=roster,
        # installed state of the foundation models (package, code commit, modified files, weight snapshot); a changed editable install refuses resuming
        foundation_versions={m.name: fm.version(m.name) for m in models if m.route == "foundation"},
        foundation_weights={m.name: fm.weights_record(m.name) for m in models if m.route == "foundation"},      # pinned weight files (commit, sha256)
        foundation_excluded=excluded, models=[m.name for m in models], domains=names, repeats=args.repeats,
        regimes=regimes_, cat_mode="benchmark", fm_input="raw_native", save_models=not args.no_save, full_run=full_run,
        hpo_trials_required=None if args.allow_nonfinal_hpo else config.HPO_N_TRIALS, max_epochs=args.max_epochs,
        verify_reload=args.verify_reload,
        verify_reload_routes=sorted(set(args.verify_reload_routes.split(","))) if args.verify_reload else None)))
    save = not args.no_save
    reload_routes = set(args.verify_reload_routes.split(","))
    if not reload_routes <= {"ml", "transformer", "tabular_dl", "foundation"}:
        raise SystemExit(f"--verify-reload-routes must be among ml, transformer, tabular_dl, foundation: {sorted(reload_routes)}")
    prefix_of = (lambda r: artifact_prefix(model_dir, r["slot"], r["model"], r["regime"], r["repeat"]))
    done = done_keys(rows_path, prefix_of if save else None, (lambda m: registry.get(m).route), KEY, need_pred=save)
    if save:   # check the content of the prediction files too (a broken file is not done and is trained again)
        done = {k for k in done if pred_ok(prefix_of(dict(zip(KEY, k))))}
    log(f"[benchmark] run {run_id} ({status}) {len(names)} domains x {args.repeats} repetitions x {len(models)} models x regimes {regimes_} "
        f"(roster {len(roster)}, excluded {excluded}) -> {rows_path} ({len(done)} rows done)")
    t_all = time.time()

    for name in names:
        kind, _ = parse_candidate(name)
        slot, mA, mB = build_one(df, name)
        A_idx, B_idx = df.index[mA], df.index[mB]
        a = get_A(slot, A_idx, y, lot)
        bv = B_view(slot, B_idx, y, lot, "benchmark")
        assert set(bv) == {"B_tr", "B_val", "B_te"}
        cat_cols, num_cols = feature_sets(df, name)
        ctx = regimes.Ctx(df, y, cat_cols, num_cols, cat_mode="benchmark", fm_input="raw_native")
        reps = {}
        for i in range(args.repeats):
            a_train, a_val = repeat_split(a["A_dev"], y, lot, i)
            n_star, a_match, b_match = matched(a_train, bv["B_tr"], i)
            reps[i] = (a_train, a_val, n_star, a_match, b_match)
        held_lots, te_lots = lot.loc[a["A_heldout"]], lot.loc[bv["B_te"]]
        t_slot = time.time()
        for spec in models:
            src = tgt = None
            need_trials = None if args.allow_nonfinal_hpo else config.HPO_N_TRIALS
            if spec.name in hpo.TUNED:
                if {"ID_full", "ID_match", "OOD_match"} & set(regimes_):
                    src = hpo.load(hpo.record_prefix(args.hpo_run_id, slot, spec.name, "source"))
                    check_hpo_record(src, spec, "source", slot, hpo_split(a["A_dev"], y, lot), need_trials)
                if "OOD_match_tuned" in regimes_:
                    tgt = hpo.load(hpo.record_prefix(args.hpo_run_id, slot, spec.name, "target"))
                    check_hpo_record(tgt, spec, "target", slot, (bv["B_tr"], bv["B_val"]), need_trials)
            for i in range(args.repeats):
                a_train, a_val, n_star, a_match, b_match = reps[i]
                for regime in regimes_:
                    if regime == "OOD_match_tuned" and spec.name not in hpo.TUNED:
                        continue
                    if (slot, str(i), spec.name, regime) in done:
                        if save:   # the stored evaluation rows must equal the current A_heldout and B_te (stop if the split changed)
                            z = np.load(artifact_prefix(model_dir, slot, spec.name, regime, i) + ".pred.npz")
                            if not (np.array_equal(z["id_idx"], np.asarray(a["A_heldout"]))
                                    and np.array_equal(z["ood_idx"], np.asarray(bv["B_te"]))):
                                raise SystemExit(f"{slot} r{i} {spec.name} {regime}: stored evaluation rows differ from the current split")
                        continue
                    check_stop(args.stop_file)
                    fit_idx, es_idx = regime_rows(regime, spec, a_train, a_val, a_match, b_match, bv, y, lot, i)
                    rec = tgt if regime == "OOD_match_tuned" else src
                    params = None if rec is None else rec["params"]
                    seed = config.RANDOM_STATE + config.SEED_OFFSET[regime] + i
                    row = dict(run_id=run_id, slot=slot, candidate=name, kind=kind, repeat=i, model=spec.name,
                               family=spec.family, regime=regime, n_A=len(A_idx), n_B=len(B_idx), n_A_train=len(a_train),
                               n_A_val=len(a_val), n_star=n_star, n_B_tr=len(bv["B_tr"]), n_B_val=len(bv["B_val"]),
                               n_fit=len(fit_idx), n_es=0 if es_idx is None else len(es_idx), seed=seed,
                               params_from=(None if rec is None else f"hpo:{rec['side']}"),
                               hpo_n_trials_complete=(None if rec is None else rec.get("n_trials_complete")),
                               n_id=len(a["A_heldout"]), n_ood=len(bv["B_te"]))
                    f = g = None
                    try:
                        check_rows(regime, fit_idx, es_idx, a, bv)
                        _reset_peak()
                        t0 = time.time()
                        f = regimes.fit(spec, ctx, fit_idx, es_idx, seed, params=params, max_epochs=args.max_epochs)
                        row["fit_sec"] = round(time.time() - t0, 3)
                        row.update(torch_peak_fit_mb=_peak_mb(), **_mem())
                        _reset_peak()
                        t0 = time.time()
                        p_id = regimes.predict(f, ctx, a["A_heldout"])
                        row["predict_id_sec"] = round(time.time() - t0, 3)
                        row["torch_peak_predict_id_mb"] = _peak_mb()
                        _reset_peak()
                        t0 = time.time()
                        p_ood = regimes.predict(f, ctx, bv["B_te"])
                        row["predict_ood_sec"] = round(time.time() - t0, 3)
                        row["torch_peak_predict_ood_mb"] = _peak_mb()
                        row.update({f"{k}_id": v for k, v in metrics.regression(y.loc[a["A_heldout"]], p_id).items()})
                        row.update({f"{k}_ood": v for k, v in metrics.regression(y.loc[bv["B_te"]], p_ood).items()})
                        train_lots = set(lot.loc[list(set(fit_idx) | (set() if es_idx is None else set(es_idx)))])
                        n_rounds, unit = rounds_used(f.config)
                        row.update(n_rounds_used=n_rounds, n_rounds_unit=unit, lot_exposure_id=float(held_lots.isin(train_lots).mean()),
                                   lot_exposure_ood=float(te_lots.isin(train_lots).mean()),
                                   n_unobserved=len(f.config.get("unobserved_num_cols", [])))
                        if save:
                            prefix = artifact_prefix(model_dir, slot, spec.name, regime, i)
                            regimes.save(f, ctx, prefix)
                            np.savez(prefix + ".pred.npz", id_idx=np.asarray(a["A_heldout"]), id_pred=p_id,
                                     ood_idx=np.asarray(bv["B_te"]), ood_pred=p_ood)
                            if args.verify_reload and spec.route in reload_routes:
                                t0 = time.time()
                                g = regimes.load(spec, ctx, prefix)
                                row["load_sec"] = round(time.time() - t0, 3)
                                row["reload_max_abs_diff"] = regimes.check_reload(spec, p_ood, regimes.predict(g, ctx, bv["B_te"]))
                        row["status"] = "ok"
                        log(f"[benchmark] {slot} r{i} {spec.name:15s} {regime:15s} R2 id={row['r2_id']:.3f} ood={row['r2_ood']:.3f} "
                            f"fit {row['fit_sec']}s pred {row['predict_id_sec']}+{row['predict_ood_sec']}s")
                    except Exception as e:
                        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                        log(f"[benchmark] {slot} r{i} {spec.name} {regime} failed\n{traceback.format_exc()}", level="ERROR")
                    append_row(rows_path, row)
                    # drop the references to this fit's model and reloaded model before the next fit (otherwise they stay in GPU and RAM during the next training)
                    f = g = None
                    free_memory()
            if spec.route == "foundation":
                fm.release(spec.name)   # unload the weights for the next model (GPU memory); f and g are already dropped above, so the memory is actually freed
                free_memory()
        log(f"[benchmark] {slot} done: {time.time() - t_slot:.0f}s")
    log(f"[benchmark] total {time.time() - t_all:.0f}s")
    # fail-closed: did every expected key of this run's scope succeed (with intact artifacts)?
    expected = {(build_one(df, n)[0], str(i), m.name, r) for n in names for i in range(args.repeats) for m in models
                for r in regimes_ if r != "OOD_match_tuned" or m.name in hpo.TUNED}
    final = done_keys(rows_path, prefix_of if save else None, (lambda m: registry.get(m).route), KEY, need_pred=save)
    if save:
        final = {k for k in final if pred_ok(prefix_of(dict(zip(KEY, k))))}
    lacking = sorted(expected - final)
    if lacking:
        log(f"[benchmark] incomplete: {len(lacking)} of {len(expected)} expected keys did not succeed (e.g. {lacking[:5]}). Rerunning with "
            f"the same --run-id trains only those", level="ERROR")
        sys.exit(1)
    log(f"[benchmark] complete: all {len(expected)} expected keys succeeded")


if __name__ == "__main__":
    main()
