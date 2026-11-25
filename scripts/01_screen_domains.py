"""Domain screening: training and evaluation for selecting the challenge domains. The selection itself is
scripts/02_select_domains.py.

Per candidate: the A split (A_heldout fixed | A_dev); B passes only the subsets allowed for screening (B_tr, B_screen; B_val and
B_te are never handed over).
Per repetition i: A_dev -> A_train, n* = min(|A_train|, |B_tr|), A_match and B_match drawn, ID_match and OOD_match trained per model.
  Early-stopping models split the n* rows 80/20 and stop on the 20% (the same rule for A and B); other models use all n* rows.
  Settings (without --tune, fixed settings): all four models fixed in advance (no per-candidate tuning):
    XGBoost  learning rate, depth and so on at the library defaults, at most 2000 rounds + 50-round early stopping, hist, CUDA (ml.build_one_model)
    MLP      transformer.DEFAULT_PARAMS["MLP"], batch size 256
    TabM     pytabkit TabM_D defaults (package-internal early stopping, patience 16)
    TabPFN   TabPFN-3.5 defaults (all n* rows as context)
  Settings (--tune, the screening used for the selection): the tunable models (XGBoost, MLP, TabM) are searched once per candidate
    on the source side and fixed across repetitions and regimes — the same function as the benchmark's source-side search
    (hpo.tune), the same rows (the tuning split of A_dev, splits.hpo_split), the same seed (config.HPO_SEED), the same budget
    (--trials, default config.HPO_N_TRIALS), the same search spaces and the same y handling. The ID_match and OOD_match models use
    the same source values. TabPFN is not tuned (defaults). Search records: cache/screening/hpo/<run_id>/<slot>/<model>__source,
    summary results/screening/runs/<run_id>/hpo_rows.csv.
Every split is at the production-lot level (splits.py). The lot column is not a feature (candidates.feature_sets).
Evaluation: B_screen only (EVAL_SETS). A_heldout is reserved for the ID evaluation of the benchmark and is not predicted here
(only its size is recorded).
Lot overlap is recorded per result row: the share of B_screen rows whose lot appears in that regime's training rows (fit + es)
(0 for OOD_match; for ID_match, as many as the lots spanning the domain boundary).
Trained models, row indices and the applied configurations are saved for the selection audit.

Run folder: results/screening/runs/<run_id>/{manifest.json, rows.csv}; models: cache/screening/runs/<run_id>/models/.
Rerunning with the same --run-id resumes only if the manifest (code, data, candidates, protocol, packages, arguments) is the same.

Usage:
  python scripts/01_screen_domains.py --run-id screening_smoke --only cat_050 --repeats 1 --verify-reload   # smoke
  python scripts/01_screen_domains.py --run-id <run> --placebo --tune                                      # main run
"""
import argparse
import os

# CUBLAS_WORKSPACE_CONFIG must be set before torch uses CUDA (needed where torch enables deterministic mode); before importing torch.
# TabM of the installed pytabkit does not enable deterministic mode and only fixes its seeds (models/tabular_dl.py header).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from defectshift import config, hpo, manifest, metrics, regimes, registry, selection  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_candidates, load_data, parse_candidate, slot_dir  # noqa: E402
from defectshift.screening import append_row, artifact_prefix, done_keys  # noqa: E402
from defectshift.splits import (B_view, b_te_eligible, check_A, es_split, get_A, hpo_split, matched,  # noqa: E402
                                repeat_split, split_A, split_B4)
from defectshift.runio import refuse_orphan_cache  # noqa: E402
from defectshift.utils import free_memory, log, new_run_id  # noqa: E402
# Import the training-route modules first, so that the code hashes of the manifest cover all code that actually runs
from defectshift.models import tabular_dl, foundation, ml, transformer  # noqa: E402,F401

KEY = ["slot", "repeat", "model", "regime"]
EVAL_SETS = ("B_screen",)   # the sets screening predicts and evaluates; no other set is predicted


def preflight(df, y, lot, names, repeats):
    """Check every candidate before training (protocol validity conditions, not a selection rule). Any violation stops the
    whole run.
    - both domain sizes >= MIN_DOMAIN (never skipped silently)
    - every split this run will make (A_heldout, tuning, A_train/A_val per repetition, the four B parts, the internal early
      stopping per repetition) can be stratified on y>0 at the lot level
    Only pure split functions without caches are used."""
    for name in names:
        _, mA, mB = build_one(df, name)
        A_idx, B_idx = df.index[mA], df.index[mB]
        assert min(len(A_idx), len(B_idx)) >= config.MIN_DOMAIN, f"{name}: domain sizes {len(A_idx)}/{len(B_idx)} < {config.MIN_DOMAIN}"
        try:
            a, b4 = split_A(A_idx, y, lot), split_B4(B_idx, y, lot)
            hpo_split(a["A_dev"], y, lot)
            for i in range(repeats):
                tr, _ = repeat_split(a["A_dev"], y, lot, i)
                _, A_m, B_m = matched(tr, b4["B_tr"], i)
                es_split(A_m, y, lot, i)
                es_split(B_m, y, lot, i)
        except RuntimeError as e:
            raise RuntimeError(f"candidate {name}: {e} — invalid under the fixed protocol (every split stratified on y>0 at the lot level)") from e


def source_params(args, run_id, slot, name, models, ctx, a, y, lot, hpo_rows_path):
    """One source-side search per tunable model on one candidate (finished records are read again). Returns {model name: params}.
    Search rows = the tuning split of A_dev (as in the benchmark's source-side search). A_heldout and B are not passed. A failed
    model is left out of the result and recorded."""
    tr, va = hpo_split(a["A_dev"], y, lot)
    check_A(a, {"hpo_train": tr, "hpo_val": va})
    out = {}
    for spec in models:
        if spec.name not in hpo.TUNED:
            continue
        prefix = os.path.join(config.CACHE_DIR, "screening", "hpo", run_id, slot_dir(slot), f"{spec.name}__source")
        if hpo.done(prefix):   # resuming: a remaining record is reused only after checking that it belongs to this domain, model, search rows, budget and epoch cap
            rec = hpo.load(prefix)
            hpo.check_record(rec, spec, "source", slot, (tr, va), args.trials, args.max_epochs)
            out[spec.name] = rec["params"]
            continue
        row = dict(run_id=run_id, slot=slot, candidate=name, model=spec.name, side="source", n_train=len(tr), n_val=len(va))
        try:
            rec = hpo.tune(spec, ctx, tr, va, n_trials=args.trials, max_epochs=args.max_epochs)
            rec.update(slot=slot, candidate=name, side="source", run_id=run_id)
            hpo.save(rec, prefix)
            out[spec.name] = rec["params"]
            trials = rec.get("trials") or []
            row.update(status="ok", hpo_sec=round(rec["hpo_sec"], 2), best_score=rec["best_score"], trial_log=bool(trials),
                       n_trials_complete=sum(t["state"] == "COMPLETE" for t in trials) if trials else None,
                       n_unobserved=len(rec["unobserved_num_cols"]))
            log(f"[screening-HPO] {slot} {spec.name:7s} best R2={rec['best_score']:.4f} ({rec['hpo_sec']:.0f}s)")
        except Exception as e:
            row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
            log(f"[screening-HPO] {slot} {spec.name} failed\n{traceback.format_exc()}", level="ERROR")
        append_row(hpo_rows_path, row)
        free_memory()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default=None, help="run folder name (default: a timestamp); an existing name resumes only if the manifest is the same")
    ap.add_argument("--only", default=None, help="comma-separated candidate names (e.g. cat_050); default: all 78 (88 with --placebo)")
    ap.add_argument("--limit", type=int, default=None, help="limit the number of candidates from the front")
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS_SCREENING)
    ap.add_argument("--models", default=",".join(config.SCREENING_MODELS))
    ap.add_argument("--placebo", action="store_true", help="include the 10 random-split control candidates")
    ap.add_argument("--no-save", action="store_true", help="do not save models (no selection audit possible)")
    ap.add_argument("--verify-reload", action="store_true", help="reload right after saving and check that the B_screen predictions are the same (smoke runs)")
    ap.add_argument("--tune", action="store_true", help="search the tunable models on the source side per candidate")
    ap.add_argument("--trials", type=int, default=config.HPO_N_TRIALS, help="number of trials for --tune")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="smoke runs only: epoch cap for searching and training the deep models (MLP, FT-T, RealMLP, TabM); makes the run a non-main run")
    args = ap.parse_args()

    models = [registry.get(m) for m in args.models.split(",")]
    names = load_candidates(include_placebo=args.placebo)
    if args.only:
        want = args.only.split(",")
        names = [n for n in names if n in want] + [n for n in want if n.startswith("placebo@") and n not in names]
    if args.limit:
        names = names[:args.limit]

    # Main run = all 88 candidates, 3 repetitions, 4 models, saved models, and the tuned-screening conditions (--tune, final budget, no
    # epoch cap). The selection (selection.validate_run) checks the same conditions again, so a fixed-setting run or a search with a
    # smoke budget cannot be used for the selection
    full_run = (names == load_candidates(include_placebo=True) and args.repeats == config.N_REPEATS_SCREENING
                and [m.name for m in models] == list(config.SCREENING_MODELS) and not args.no_save
                and args.tune and args.trials == config.HPO_N_TRIALS and args.max_epochs is None)
    if full_run and selection.N_CAT is None:
        raise SystemExit("a main run starts only after the number of selected process-condition domains (selection.N_CAT) is set")
    df, y = load_data()
    lot = df[config.LOT_COL]
    preflight(df, y, lot, names, args.repeats)

    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(config.RESULTS_DIR, "screening", "runs", run_id)
    model_dir = os.path.join(config.CACHE_DIR, "screening", "runs", run_id, "models")
    rows_path = os.path.join(run_dir, "rows.csv")
    refuse_orphan_cache(run_dir, os.path.join(config.CACHE_DIR, "screening", "hpo", run_id), f"screening run {run_id}")
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="screening", models=[m.name for m in models], repeats=args.repeats, candidates=names, save_models=not args.no_save,
        full_run=full_run, selection_n_cat=selection.N_CAT, tune=args.tune, trials=args.trials if args.tune else None,
        hpo_seed=config.HPO_SEED if args.tune else None, max_epochs=args.max_epochs,
        selection_py_sha256=selection.code_hashes()["selection_py_sha256"],
        metrics_py_sha256=selection.code_hashes()["metrics_py_sha256"])))
    done = done_keys(rows_path, None if args.no_save else model_dir)
    hpo_rows_path = os.path.join(run_dir, "hpo_rows.csv")
    log(f"[screening] run {run_id} ({status}) {len(names)} candidates x {args.repeats} repetitions x models {[m.name for m in models]} "
        f"-> {rows_path} ({len(done)} rows done)")
    t_all = time.time()

    slots = {}
    for name in names:
        kind, _ = parse_candidate(name)
        slot, mA, mB = build_one(df, name)
        slots[name] = slot
        A_idx, B_idx = df.index[mA], df.index[mB]
        a = get_A(slot, A_idx, y, lot)
        bv = B_view(slot, B_idx, y, lot, "screening")     # B_tr and B_screen only
        scr_lots = lot.loc[bv["B_screen"]]
        assert set(bv) == {"B_tr", "B_screen"}
        cat_cols, num_cols = feature_sets(df, name)
        ctx = regimes.Ctx(df, y, cat_cols, num_cols)
        t_slot = time.time()
        params = source_params(args, run_id, slot, name, models, ctx, a, y, lot, hpo_rows_path) if args.tune else {}
        for i in range(args.repeats):
            A_train, A_val = repeat_split(a["A_dev"], y, lot, i)
            n_star, A_match, B_match = matched(A_train, bv["B_tr"], i)
            for spec in models:
                for regime, pool in (("ID_match", A_match), ("OOD_match", B_match)):
                    if (slot, str(i), spec.name, regime) in done:
                        continue
                    fit_idx, es_idx = es_split(pool, y, lot, i) if spec.uses_es else (pool, None)
                    # protocol check: A_heldout is used nowhere, and the B-side training rows are within B_tr
                    if regime == "ID_match":
                        used = {"A_train": A_train, "A_val": A_val, "fit": fit_idx}
                        if es_idx is not None:
                            used["es"] = es_idx
                        check_A(a, used)
                    else:
                        assert set(fit_idx) <= set(bv["B_tr"]) and (es_idx is None or set(es_idx) <= set(bv["B_tr"]))
                    seed = config.RANDOM_STATE + config.SEED_OFFSET[regime] + i
                    row = dict(run_id=run_id, slot=slot, candidate=name, kind=kind, repeat=i, model=spec.name,
                               params_from="hpo:source" if spec.name in params else "fixed",
                               family=spec.family, regime=regime, n_A=len(A_idx), n_B=len(B_idx), n_A_dev=len(a["A_dev"]),
                               n_A_heldout=len(a["A_heldout"]), n_A_train=len(A_train), n_B_tr=len(bv["B_tr"]),
                               n_B_screen=len(bv["B_screen"]), n_star=n_star, n_fit=len(fit_idx),
                               n_es=0 if es_idx is None else len(es_idx), b_te_eligible=b_te_eligible(len(B_idx)),
                               n_features=len(ctx.feature_cols), seed=seed,
                               n_lots_A=lot.loc[A_idx].nunique(), n_lots_B=lot.loc[B_idx].nunique(),
                               screen_lot_in_train=float(scr_lots.isin(set(lot.loc[pool])).mean()))
                    f = g = None
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        if args.tune and spec.name in hpo.TUNED and spec.name not in params:
                            raise RuntimeError("source-side search failed -> not replaced by the fixed setting")
                        f = regimes.fit(spec, ctx, fit_idx, es_idx, seed, params=params.get(spec.name), max_epochs=args.max_epochs)
                        t0 = time.time()
                        preds = {s: regimes.predict(f, ctx, bv[s]) for s in EVAL_SETS}
                        row.update(fit_sec=round(f.fit_sec, 2), pred_sec=round(time.time() - t0, 2),
                                   torch_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None)
                        for s in EVAL_SETS:
                            tag = s.split("_")[1]
                            row.update({f"{k}_{tag}": v for k, v in metrics.regression(y.loc[bv[s]], preds[s]).items()})
                        if not args.no_save:
                            prefix = artifact_prefix(model_dir, slot, spec.name, regime, i)
                            regimes.save(f, ctx, prefix)
                            # the B_screen predictions are saved too: the selection audit compares the reloaded predictions by value, not by RMSE
                            np.savez(prefix + ".pred.npz", scr_idx=np.asarray(bv["B_screen"]), scr_pred=preds["B_screen"])
                            if args.verify_reload:
                                g = regimes.load(spec, ctx, prefix)
                                row["reload_max_abs_diff"] = regimes.check_reload(spec, preds["B_screen"],
                                                                                  regimes.predict(g, ctx, bv["B_screen"]))
                        row["status"] = "ok"
                        log(f"[screening] {slot} r{i} {spec.name:7s} {regime:9s} n*={n_star} fit={row['n_fit']}/es={row['n_es']} "
                            f"RMSE_scr={row['rmse_screen']:.5f} R2_scr={row['r2_screen']:.3f} ({row['fit_sec']}s+{row['pred_sec']}s)")
                    except Exception as e:
                        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                        log(f"[screening] {slot} r{i} {spec.name} {regime} failed\n{traceback.format_exc()}", level="ERROR")
                    append_row(rows_path, row)
                    f = g = None          # drop this fit's model and reloaded model before the next fit (the same lifetime policy as the benchmark)
                    free_memory()
        log(f"[screening] {slot} done: {time.time() - t_slot:.0f}s")
    log(f"[screening] total {time.time() - t_all:.0f}s")
    check_complete(args, run_id, names, slots, models, rows_path, model_dir)


def check_complete(args, run_id, names, slots, models, rows_path, model_dir):
    """End: the run exits normally only if every expected key of its scope succeeded (with intact saved artifacts) and, with
    --tune, every source-side search record of the tuned models exists for every candidate. Otherwise it records counts and
    examples and exits with code 1 (fail-closed, as the benchmark). This keeps a broken process (a CUDA error, for example) whose
    remaining candidates fail one after another from looking finished. The selection (selection.validate_run) checks
    completeness again."""
    done = done_keys(rows_path, None if args.no_save else model_dir)
    need = {(slots[n], str(i), m.name, r) for n in names for i in range(args.repeats) for m in models
            for r in ("ID_match", "OOD_match")}
    miss_fit = sorted(need - done)
    miss_hpo = []
    if args.tune:
        miss_hpo = [(n, m.name) for n in names for m in models if m.name in hpo.TUNED
                    and not hpo.done(os.path.join(config.CACHE_DIR, "screening", "hpo", run_id, slot_dir(slots[n]), f"{m.name}__source"))]
    n_hpo = sum(m.name in hpo.TUNED for m in models) * len(names) if args.tune else 0
    if miss_fit or miss_hpo:
        log(f"[screening] incomplete: fits {len(need) - len(miss_fit)}/{len(need)} (e.g. {miss_fit[:3]}), "
            f"searches {n_hpo - len(miss_hpo)}/{n_hpo} (e.g. {miss_hpo[:3]})", level="ERROR")
        sys.exit(1)
    log(f"[screening] complete: fits {len(need)}/{len(need)}, searches {n_hpo}/{n_hpo}")


if __name__ == "__main__":
    main()
