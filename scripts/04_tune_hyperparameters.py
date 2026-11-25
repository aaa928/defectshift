"""Hyperparameter search of the benchmark: selected domains x 13 tuned models x (source, target).

- source: training / validation on the tuning split of A_dev (splits.hpo_split, production-lot level, seed 999). A_heldout is not used
- target: training on B_tr / validation on B_val (`B_view(..., "benchmark")`). For OOD_match_tuned only. B_te is not passed
- categorical representation "benchmark", budget config.HPO_N_TRIALS (30), global seeds fixed at the start (hpo.tune)
- output: search records cache/benchmark/hpo/<run_id>/<slot>/<model>__<side>.json, run record
  results/benchmark/hpo/<run_id>/{manifest.json, rows.csv}
- resuming: with the same --run-id and the same manifest, (slot, model, side) with a record (.json) are skipped

Usage
  python scripts/04_tune_hyperparameters.py --screening-run-id <screening> --run-id tuning_smoke --only cat_050 --models Ridge,MLP,RealMLP --trials 3   # smoke
  python scripts/04_tune_hyperparameters.py --screening-run-id <screening> --run-id <run>                                                      # main run
"""
import argparse
import os
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before importing torch

from defectshift import config, hpo, manifest, regimes, registry  # noqa: E402
from defectshift.runio import append_row, check_stop, control_names, load_final_domains, refuse_orphan_cache  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_data  # noqa: E402
from defectshift.splits import B_view, get_A, hpo_split  # noqa: E402
from defectshift.utils import free_memory, log, new_run_id  # noqa: E402
from defectshift.models import tabular_dl, ml, transformer  # noqa: E402,F401  (so that the manifest code hashes cover the code that runs)


def rows_for(side, a, bv, y, lot):
    """(training rows, validation rows). source = tuning split, target = (B_tr, B_val)."""
    if side == "source":
        return hpo_split(a["A_dev"], y, lot)
    return bv["B_tr"], bv["B_val"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--screening-run-id", required=True, help="screening run whose selection is done")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--only", default=None, help="comma-separated domains (candidate names); default: all selected domains")
    ap.add_argument("--controls", default=None,
                    help="comma-separated seeds of random-split controls (placebo@<seed>); if given, only the controls are tuned instead of the selected domains")
    ap.add_argument("--stop-file", default=None, help="stop before starting the next unit (domain, model, side) once this file exists (exit code 75); "
                                                    "run the same command again to continue")
    ap.add_argument("--models", default=",".join(hpo.TUNED))
    ap.add_argument("--sides", default=",".join(hpo.SIDES))
    ap.add_argument("--trials", type=int, default=config.HPO_N_TRIALS)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="smoke runs only: epoch cap for searching the deep models (MLP, FT-T, RealMLP, TabM); makes the run a non-main run")
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
    models = [registry.get(m) for m in args.models.split(",")]
    bad = [m.name for m in models if m.name not in hpo.TUNED]
    if bad:
        raise SystemExit(f"models not tuned: {bad}")
    sides = args.sides.split(",")
    if not set(sides) <= set(hpo.SIDES):
        raise SystemExit(f"--sides must be among {hpo.SIDES}")
    full_run = (names == domains and [m.name for m in models] == list(hpo.TUNED) and sides == list(hpo.SIDES)
                and args.trials == config.HPO_N_TRIALS and args.max_epochs is None)

    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "hpo", run_id)
    rows_path = os.path.join(run_dir, "rows.csv")
    refuse_orphan_cache(run_dir, os.path.join(config.CACHE_DIR, "benchmark", "hpo", run_id), f"tuning run {run_id}")
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="benchmark_tuning", screening_run_id=args.screening_run_id, screening_final_domains_sha256=final_sha, domains=names,
        models=[m.name for m in models], sides=sides, trials=args.trials, max_epochs=args.max_epochs, cat_mode="benchmark", seed=config.HPO_SEED,
        full_run=full_run)))
    log(f"[tuning] run {run_id} ({status}) {len(names)} domains x {len(models)} models x {sides} x {args.trials} trials -> {rows_path}")

    df, y = load_data()
    lot = df[config.LOT_COL]
    t_all = time.time()
    for name in names:
        slot, mA, mB = build_one(df, name)
        A_idx, B_idx = df.index[mA], df.index[mB]
        a = get_A(slot, A_idx, y, lot)
        bv = B_view(slot, B_idx, y, lot, "benchmark")
        assert set(bv) == {"B_tr", "B_val", "B_te"}
        bv = {k: bv[k] for k in ("B_tr", "B_val")}                 # the search does not receive B_te
        cat_cols, num_cols = feature_sets(df, name)
        ctx = regimes.Ctx(df, y, cat_cols, num_cols, cat_mode="benchmark", fm_input="raw_native")
        for spec in models:
            for side in sides:
                prefix = hpo.record_prefix(run_id, slot, spec.name, side)
                if hpo.done(prefix):
                    continue
                check_stop(args.stop_file)
                tr, va = rows_for(side, a, bv, y, lot)
                row = dict(run_id=run_id, slot=slot, candidate=name, model=spec.name, side=side, n_train=len(tr), n_val=len(va))
                try:
                    rec = hpo.tune(spec, ctx, tr, va, n_trials=args.trials, max_epochs=args.max_epochs)
                    rec.update(slot=slot, candidate=name, side=side, run_id=run_id)
                    hpo.save(rec, prefix)
                    trials = rec.get("trials") or []
                    # routes without per-trial records (pytabkit HPO returns only the best configuration, its validation score and the total time) leave the uncounted values empty
                    row.update(status="ok", hpo_sec=round(rec["hpo_sec"], 2), best_score=rec["best_score"],
                               trial_log=bool(trials),
                               n_trials_complete=sum(t["state"] == "COMPLETE" for t in trials) if trials else None,
                               n_unobserved=len(rec["unobserved_num_cols"]))
                    log(f"[tuning] {slot} {spec.name:15s} {side:6s} best R2={rec['best_score']:.4f} ({rec['hpo_sec']:.0f}s)")
                except Exception as e:
                    row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                    log(f"[tuning] {slot} {spec.name} {side} failed\n{traceback.format_exc()}", level="ERROR")
                append_row(rows_path, row)
                free_memory()
    log(f"[tuning] total {time.time() - t_all:.0f}s")
    # fail-closed: is there a search record for every (domain, model, side) of this run's scope? A failure leaves a row but no record file, so it shows here
    lacking = [(name, spec.name, side) for name in names for spec in models for side in sides
               if not hpo.done(hpo.record_prefix(run_id, build_one(df, name)[0], spec.name, side))]
    expected = len(names) * len(models) * len(sides)
    if lacking:
        log(f"[tuning] incomplete: {len(lacking)} of {len(expected)} expected search records missing (e.g. {lacking[:5]}). Rerunning with "
            f"the same --run-id searches only those", level="ERROR")
        sys.exit(1)
    log(f"[tuning] complete: all {expected} expected searches succeeded")


if __name__ == "__main__":
    main()
