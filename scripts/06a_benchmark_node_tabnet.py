"""Additional deep models of the benchmark (NODE, TabNet): tuning (hpo) and final benchmark (bench). Same splits, repetitions,
regimes, seeds and evaluation rules as the 20-model runs (04_tune_hyperparameters, 06_benchmark); the results have the same format
(rows.csv, <prefix>.npz, .json, .pred.npz) and are merged with scripts/06b_merge_runs.py.

- Model code: models/node.py (official Qwicen/node), models/tabnet.py -> models/tabnet_tf1_worker.py (official
  google-research/tabnet, TensorFlow 1 environment). Both are early-stopping models (at most 150 epochs, stop after 17 consecutive
  epochs without improvement, restore the best epoch).
- hpo: domain x model x (source = tuning split of A_dev, target = B_tr/B_val). NODE = the 24-configuration grid of the original
  paper + the official default = 25 configurations, all evaluated (Gorishniy et al. 2021, F.6); TabNet = Optuna TPE with
  config.HPO_N_TRIALS (30) trials in the space of the original paper's appendix. Seed config.HPO_SEED. Records:
  cache/benchmark/hpo/<run>/<slot>/<model>__<side>.json (hpo.save format).
- bench: the regime_rows, check_rows and seed rules of the benchmark. Hyperparameters from the source record (ID_full, ID_match,
  OOD_match) and the target record (OOD_match_tuned). Saved: <prefix>.npz (training and early-stopping rows), <prefix>.json
  (configuration and record), weights (NODE <prefix>.pt, TabNet <prefix>.ckpt.*) and <prefix>.pred.npz (A_heldout and B_te
  predictions).
- Lineage: the manifest records the hashes of the two models' code files (EXTRA_FILES) and of the official code files (vendor) and
  the versions of the additional packages and of the TF environment; bench starts only if they match the tuning run it uses
  (shared training code through runio.check_compat, so hpo also imports ml, transformer and tabular_dl). The run record holds the
  number of trials per model (trials_per_model, NODE 25, TabNet 30) and the actual input description of the two models (inputs);
  06b_merge_runs checks the stages benchmark_extra and benchmark_tuning_extra against them.
- Location: to use the results and cache of another checkout, set DEFECTSHIFT_RESULTS_DIR and DEFECTSHIFT_CACHE_DIR.

Usage (smoke first, then the main run)
  python scripts/06a_benchmark_node_tabnet.py hpo --screening-run-id <screening> --run-id extra_hpo_smoke --only cat_050 --max-epochs 10 --trials 3 --node-configs 2
  python scripts/06a_benchmark_node_tabnet.py bench --screening-run-id <screening> --hpo-run-id extra_hpo_smoke --run-id extra_smoke --only cat_050 \\
      --repeats 1 --max-epochs 10 --allow-nonfinal-hpo --verify-reload
  python scripts/06a_benchmark_node_tabnet.py hpo --screening-run-id <screening> --run-id <hpo run>                          # main run
  python scripts/06a_benchmark_node_tabnet.py bench --screening-run-id <screening> --hpo-run-id <hpo run> --run-id <run>    # main run
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before importing torch
# CUDA allocator: against memory fragmentation in consecutive grid runs (in the same process, 2x2048 depth 8 ran out of memory after
# earlier configurations although no tensor remained). Computation must not change; identical predictions with the option on and
# off were checked beforehand. The actual value is recorded in the run record (lineage). This runner only runs NODE and TabNet, and
# TabNet runs in a TF subprocess, so it is not affected. Set only when run directly — not when the sensitivity runner imports the
# NODE/TabNet lineage, so that the allocator of that process is not changed silently.
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before importing torch

import numpy as np  # noqa: E402
import torch  # noqa: E402

from defectshift import config, hpo, manifest, metrics  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_data, parse_candidate  # noqa: E402
from defectshift.extra_runs import lineage, load_extra_hpo  # noqa: E402
from defectshift.models import extra_models, node, protocol, tabnet  # noqa: E402
from defectshift.models import tabular_dl, ml, transformer  # noqa: E402,F401  (so that the manifest code hashes cover all compatibility training code)
from defectshift.runio import append_row, check_stop, control_names, load_final_domains, refuse_orphan_cache  # noqa: E402
from defectshift.splits import B_view, get_A, hpo_split, matched, repeat_split  # noqa: E402
from defectshift.utils import free_memory, log, new_run_id, set_seed  # noqa: E402

MODELS = extra_models.EXTRA_TUNED
INPUTS = dict(NODE=node.PREPROCESS, TabNet=tabnet.PREPROCESS)   # actual input description for the run record
RELOAD_TOL = dict(NODE=1e-6, TabNet=1e-5)


def _bench_module():
    from defectshift import benchmark
    return benchmark


def budget(model, args):
    """Number of trials: NODE = number of grid configurations (--node-configs for smoke runs), TabNet = --trials."""
    return (args.node_configs or len(node.CONFIGS)) if model == "NODE" else args.trials


def rows_for(side, a, bv, y, lot):
    """(training rows, validation rows). source = tuning split, target = (B_tr, B_val). As in 04_tune_hyperparameters."""
    if side == "source":
        return hpo_split(a["A_dev"], y, lot)
    return bv["B_tr"], bv["B_val"]


def domains_of(args):
    domains, final_sha = load_final_domains(args.screening_run_id)
    names = domains
    if args.only:
        want = args.only.split(",")
        bad = [n for n in want if n not in domains]
        if bad:
            raise SystemExit(f"not selected domains: {bad}")
        names = [n for n in domains if n in want]
    if args.controls:   # only the random-split controls instead of the selected domains (the selection hash is still checked)
        names = control_names(args.controls, args.only)
    return domains, names, final_sha


def models_of(args):
    models = args.models.split(",")
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise SystemExit(f"not additional models: {bad} (possible: {MODELS})")
    return models


# ------------------------------------------------------------------ hpo
def run_hpo(args):
    domains, names, final_sha = domains_of(args)
    models = models_of(args)
    sides = args.sides.split(",")
    if not set(sides) <= set(hpo.SIDES):
        raise SystemExit(f"--sides must be among {hpo.SIDES}")
    full_run = (names == domains and models == list(MODELS) and sides == list(hpo.SIDES) and args.trials == config.HPO_N_TRIALS
                and args.node_configs is None and args.max_epochs is None)
    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "hpo", run_id)
    rows_path = os.path.join(run_dir, "rows.csv")
    refuse_orphan_cache(run_dir, os.path.join(config.CACHE_DIR, "benchmark", "hpo", run_id), f"tuning run {run_id}")
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="benchmark_tuning_extra", screening_run_id=args.screening_run_id, screening_final_domains_sha256=final_sha, domains=names, models=models, sides=sides,
        trials_per_model={m: budget(m, args) for m in models},       # actual number of trials (NODE = grid configurations, TabNet = TPE trials)
        inputs={m: INPUTS[m] for m in models}, max_epochs=args.max_epochs, seed=config.HPO_SEED, full_run=full_run,
        device=args.device, **lineage())))
    log(f"[extra-HPO] run {run_id} ({status}) {len(names)} domains x {models} x {sides} -> {rows_path}")
    df, y = load_data()
    lot = df[config.LOT_COL]
    t_all = time.time()
    for name in names:
        slot, mA, mB = build_one(df, name)
        a = get_A(slot, df.index[mA], y, lot)
        bv = B_view(slot, df.index[mB], y, lot, "benchmark")
        bv = {k: bv[k] for k in ("B_tr", "B_val")}                 # the search does not receive B_te
        cat_cols, num_cols = feature_sets(df, name)
        for m in models:
            spec = extra_models.spec_of(m)
            for side in sides:
                prefix = hpo.record_prefix(run_id, slot, m, side)
                if hpo.done(prefix):
                    continue
                check_stop(args.stop_file)
                tr, va = rows_for(side, a, bv, y, lot)
                row = dict(run_id=run_id, slot=slot, candidate=name, model=m, side=side, n_train=len(tr), n_val=len(va))
                try:
                    assert not set(tr) & set(va), "search training rows overlap the validation rows"
                    set_seed(config.HPO_SEED)
                    t0 = time.time()
                    progress = prefix + ".grid_progress.jsonl"   # NODE: per-configuration progress record (finished configurations are not trained again after an interruption)
                    params, details = extra_models.tune(m, df, y, cat_cols, num_cols, tr, va, config.HPO_SEED,
                                                        n_trials=budget(m, args), max_epochs=args.max_epochs, device=args.device,
                                                        progress_path=progress)
                    trials = details.get("trials") or []
                    rec = dict(model=m, route=spec.route, impl=spec.impl, params=params, hpo_sec=time.time() - t0,
                               n_trials=budget(m, args), seed=config.HPO_SEED, inputs=INPUTS[m], max_epochs=args.max_epochs,
                               n_train=len(tr), n_val=len(va), train_idx_sha256=hpo._idx_sha(tr), val_idx_sha256=hpo._idx_sha(va),
                               n_trials_complete=sum(t["state"] == "COMPLETE" for t in trials),
                               slot=slot, candidate=name, side=side, run_id=run_id, **details)
                    hpo.save(rec, prefix)
                    if os.path.exists(progress):                 # all trial records are in rec
                        os.remove(progress)
                    row.update(status="ok", hpo_sec=round(rec["hpo_sec"], 2), best_score=rec["best_score"], trial_log=True,
                               n_trials_complete=rec["n_trials_complete"])
                    log(f"[extra-HPO] {slot} {m:7s} {side:6s} best R2={rec['best_score']:.4f} ({rec['hpo_sec']:.0f}s, "
                        f"{rec['n_trials_complete']}/{rec['n_trials']} succeeded)")
                except Exception as e:
                    row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                    log(f"[extra-HPO] {slot} {m} {side} failed\n{traceback.format_exc()}", level="ERROR")
                append_row(rows_path, row)
                free_memory()
    log(f"[extra-HPO] total {time.time() - t_all:.0f}s")
    lacking = [(n, m, s) for n in names for m in models for s in sides
               if not hpo.done(hpo.record_prefix(run_id, build_one(df, n)[0], m, s))]
    if lacking:
        log(f"[extra-HPO] incomplete: {len(lacking)} search records missing (e.g. {lacking[:5]}). Rerunning with the same --run-id searches only those", level="ERROR")
        sys.exit(1)
    log(f"[extra-HPO] complete: all {len(names) * len(models) * len(sides)} succeeded")


# ------------------------------------------------------------------ bench
def weight_files(model, prefix):
    return [prefix + ".pt"] if model == "NODE" else [prefix + ".ckpt.index"]


def artifacts_ok(model, prefix, bench):
    need = [prefix + ".npz", prefix + ".json", *weight_files(model, prefix)]
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need):
        return False
    if model == "TabNet" and not [f for f in os.listdir(os.path.dirname(prefix))
                                  if f.startswith(os.path.basename(prefix) + ".ckpt.data-")]:
        return False
    return bench.pred_ok(prefix)


def done_keys(rows_path, prefix_of, bench):
    import pandas as pd
    if not os.path.exists(rows_path):
        return set()
    d = pd.read_csv(rows_path)
    return {tuple(str(r[k]) for k in bench.KEY) for r in d[d.status == "ok"].to_dict("records")
            if artifacts_ok(r["model"], prefix_of(r), bench)}


def run_bench(args):
    bench = _bench_module()
    domains, names, final_sha = domains_of(args)
    models = models_of(args)
    regimes_ = args.regimes.split(",")
    if not set(regimes_) <= set(bench.REGIMES):
        raise SystemExit(f"--regimes must be among {bench.REGIMES}")
    if not 1 <= args.repeats <= config.N_REPEATS_BENCHMARK:
        raise SystemExit(f"--repeats must be 1 - {config.N_REPEATS_BENCHMARK}")
    hpo_args, hpo_sha = load_extra_hpo(args.hpo_run_id, final_sha, strict=not args.allow_nonfinal_hpo)
    df, y = load_data()
    lot = df[config.LOT_COL]
    full_run = (names == domains and models == list(MODELS) and args.repeats == config.N_REPEATS_BENCHMARK and regimes_ == list(bench.REGIMES)
                and bool(hpo_args.get("full_run")) and not args.allow_nonfinal_hpo and args.max_epochs is None)
    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "runs", run_id)
    model_dir = os.path.join(config.CACHE_DIR, "benchmark", "runs", run_id, "models")
    rows_path = os.path.join(run_dir, "rows.csv")
    need_trials = {m: (None if args.allow_nonfinal_hpo else extra_models.n_trials_final(m)) for m in models}
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="benchmark_extra", screening_run_id=args.screening_run_id, screening_final_domains_sha256=final_sha, hpo_run_id=args.hpo_run_id,
        hpo_manifest_sha256=hpo_sha, roster=models, foundation_excluded=None, models=models, domains=names, repeats=args.repeats,
        regimes=regimes_, inputs={m: INPUTS[m] for m in models}, save_models=True, full_run=full_run,
        # the final per-model budget required of the search records used (NODE grid 25, TabNet 30; None for smoke runs) — checked by the lineage check of 06b_merge_runs
        hpo_trials_per_model=need_trials,
        max_epochs=args.max_epochs, verify_reload=args.verify_reload, device=args.device,
        **lineage())))
    prefix_of = (lambda r: bench.artifact_prefix(model_dir, r["slot"], r["model"], r["regime"], r["repeat"]))
    done = done_keys(rows_path, prefix_of, bench)
    log(f"[extra] run {run_id} ({status}) {len(names)} domains x {args.repeats} repetitions x {models} x {regimes_} -> {rows_path} "
        f"({len(done)} rows done)")
    t_all = time.time()
    for name in names:
        kind, _ = parse_candidate(name)
        slot, mA, mB = build_one(df, name)
        A_idx, B_idx = df.index[mA], df.index[mB]
        a = get_A(slot, A_idx, y, lot)
        bv = B_view(slot, B_idx, y, lot, "benchmark")
        cat_cols, num_cols = feature_sets(df, name)
        reps = {}
        for i in range(args.repeats):
            a_train, a_val = repeat_split(a["A_dev"], y, lot, i)
            n_star, a_match, b_match = matched(a_train, bv["B_tr"], i)
            reps[i] = (a_train, a_val, n_star, a_match, b_match)
        held_lots, te_lots = lot.loc[a["A_heldout"]], lot.loc[bv["B_te"]]
        for m in models:
            spec = extra_models.spec_of(m)
            src = tgt = None
            if {"ID_full", "ID_match", "OOD_match"} & set(regimes_):
                src = hpo.load(hpo.record_prefix(args.hpo_run_id, slot, m, "source"))
                hpo.check_record(src, spec, "source", slot, hpo_split(a["A_dev"], y, lot), need_trials[m])
            if "OOD_match_tuned" in regimes_:
                tgt = hpo.load(hpo.record_prefix(args.hpo_run_id, slot, m, "target"))
                hpo.check_record(tgt, spec, "target", slot, (bv["B_tr"], bv["B_val"]), need_trials[m])
            for i in range(args.repeats):
                a_train, a_val, n_star, a_match, b_match = reps[i]
                for regime in regimes_:
                    if (slot, str(i), m, regime) in done:
                        z = np.load(bench.artifact_prefix(model_dir, slot, m, regime, i) + ".pred.npz")
                        if not (np.array_equal(z["id_idx"], np.asarray(a["A_heldout"]))
                                and np.array_equal(z["ood_idx"], np.asarray(bv["B_te"]))):
                            raise SystemExit(f"{slot} r{i} {m} {regime}: stored evaluation rows differ from the current split")
                        continue
                    check_stop(args.stop_file)
                    fit_idx, es_idx = bench.regime_rows(regime, spec, a_train, a_val, a_match, b_match, bv, y, lot, i)
                    rec = tgt if regime == "OOD_match_tuned" else src
                    seed = config.RANDOM_STATE + config.SEED_OFFSET[regime] + i
                    prefix = bench.artifact_prefix(model_dir, slot, m, regime, i)
                    row = dict(run_id=run_id, slot=slot, candidate=name, kind=kind, repeat=i, model=m, family=spec.family,
                               regime=regime, n_A=len(A_idx), n_B=len(B_idx), n_A_train=len(a_train), n_A_val=len(a_val),
                               n_star=n_star, n_B_tr=len(bv["B_tr"]), n_B_val=len(bv["B_val"]), n_fit=len(fit_idx),
                               n_es=len(es_idx), seed=seed, params_from=f"hpo:{rec['side']}",
                               hpo_n_trials_complete=rec.get("n_trials_complete"), n_id=len(a["A_heldout"]), n_ood=len(bv["B_te"]))
                    try:
                        bench.check_rows(regime, fit_idx, es_idx, a, bv)
                        os.makedirs(os.path.dirname(prefix), exist_ok=True)
                        sets = {"id": a["A_heldout"], "ood": bv["B_te"]}
                        bench._reset_peak()
                        set_seed(seed)
                        t0 = time.time()
                        preds, info = extra_models.fit_predict(m, df, y, cat_cols, num_cols, fit_idx, es_idx, sets, rec["params"], seed,
                                                               max_epochs=args.max_epochs, save_prefix=prefix, device=args.device)
                        fit_sec = info["fit_sec"]          # fitting the input transformation + training (for TabNet including the subprocess and graph construction), without prediction
                        psec = info.get("predict_sec") or {}
                        row.update(fit_sec=round(fit_sec, 3), predict_id_sec=round(psec.get("id", float("nan")), 3),
                                   predict_ood_sec=round(psec.get("ood", float("nan")), 3), train_sec=round(info["train_sec"], 3),
                                   fit_predict_wall_sec=round(time.time() - t0, 3),
                                   torch_peak_fit_mb=bench._peak_mb() if m == "NODE" else None, **bench._mem())
                        p_id, p_ood = preds["id"], preds["ood"]
                        row.update({f"{k}_id": v for k, v in metrics.regression(y.loc[a["A_heldout"]], p_id).items()})
                        row.update({f"{k}_ood": v for k, v in metrics.regression(y.loc[bv["B_te"]], p_ood).items()})
                        train_lots = set(lot.loc[list(set(fit_idx) | set(es_idx))])
                        row.update(n_rounds_used=info["best_epoch"], n_rounds_unit="epochs", n_epochs=info["n_epochs"],
                                   batch_size=info.get("batch_size"),   # NODE: 256 only for the 4 configurations of the fixed list (node.REDUCED_BATCH)
                                   lot_exposure_id=float(held_lots.isin(train_lots).mean()),
                                   lot_exposure_ood=float(te_lots.isin(train_lots).mean()),
                                   n_unobserved=len(info.get("unobserved_num_cols", [])))
                        cfg = dict(info, model=m, route=spec.route, impl=spec.impl, seed=seed, n_fit=len(fit_idx), n_es=len(es_idx),
                                   smoke_max_epochs=args.max_epochs, max_epochs=args.max_epochs or protocol.TRANSFORMER_MAX_EPOCHS,
                                   patience=protocol.TRANSFORMER_PATIENCE)
                        cfg.pop("val_mse_hist", None)
                        np.savez(prefix + ".npz", fit_idx=np.asarray(fit_idx), es_idx=np.asarray(es_idx, dtype=np.int64))
                        with open(prefix + ".json", "w") as fh:
                            json.dump(dict(config=_jsonable(cfg), params=_jsonable(rec["params"]), fit_sec=fit_sec), fh,
                                      ensure_ascii=False, indent=1)
                        np.savez(prefix + ".pred.npz", id_idx=np.asarray(a["A_heldout"]), id_pred=p_id,
                                 ood_idx=np.asarray(bv["B_te"]), ood_pred=p_ood)
                        if args.verify_reload:
                            t1 = time.time()
                            q = extra_models.predict_saved(m, df, y, cat_cols, num_cols, fit_idx, rec["params"], seed, prefix,
                                                           {"ood": bv["B_te"]}, device=args.device)
                            d = float(np.max(np.abs(np.asarray(q["ood"]) - p_ood)))
                            row.update(load_sec=round(time.time() - t1, 3), reload_max_abs_diff=d)
                            if not (np.isfinite(d) and d <= RELOAD_TOL[m]):
                                raise RuntimeError(f"save-and-reload prediction difference {d:.3e} > tolerance {RELOAD_TOL[m]:.0e}")
                        row["status"] = "ok"
                        log(f"[extra] {slot} r{i} {m:7s} {regime:15s} R2 id={row['r2_id']:.3f} ood={row['r2_ood']:.3f} "
                            f"fit {row['fit_sec']}s (epochs {info['n_epochs']}, best {info['best_epoch']})")
                    except Exception as e:
                        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                        log(f"[extra] {slot} r{i} {m} {regime} failed\n{traceback.format_exc()}", level="ERROR")
                    append_row(rows_path, row)
                    free_memory()
    log(f"[extra] total {time.time() - t_all:.0f}s")
    expected = {(build_one(df, n)[0], str(i), m, r) for n in names for i in range(args.repeats) for m in models for r in regimes_}
    lacking = sorted(expected - done_keys(rows_path, prefix_of, bench))
    if lacking:
        log(f"[extra] incomplete: {len(lacking)} of {len(expected)} expected keys did not succeed (e.g. {lacking[:5]}). Rerun with the same --run-id", level="ERROR")
        sys.exit(1)
    log(f"[extra] complete: all {len(expected)} expected keys succeeded")


def _jsonable(v):
    from defectshift.regimes import _jsonable as j
    return j(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hpo", help="search (source, target)")
    b = sub.add_parser("bench", help="final benchmark (4 regimes)")
    for p in (h, b):
        p.add_argument("--screening-run-id", required=True)
        p.add_argument("--run-id", default=None)
        p.add_argument("--only", default=None, help="comma-separated domains (candidate names); default: all selected domains")
        p.add_argument("--controls", default=None,
                       help="comma-separated seeds of random-split controls (placebo@<seed>); if given, only the controls are run instead of the selected domains")
        p.add_argument("--stop-file", default=None, help="stop before starting the next unit once this file exists (exit code 75); run the same command again to continue")
        p.add_argument("--models", default=",".join(MODELS))
        p.add_argument("--max-epochs", type=int, default=None, help="smoke runs only: epoch cap; makes the run a non-main run")
        p.add_argument("--device", default="gpu", choices=("gpu", "cpu"), help="device of the TabNet worker (NODE uses the GPU if torch sees one)")
    h.add_argument("--sides", default=",".join(hpo.SIDES))
    h.add_argument("--trials", type=int, default=config.HPO_N_TRIALS, help="number of TabNet trials (reduced for smoke runs only)")
    h.add_argument("--node-configs", type=int, default=None, help="smoke runs only: only the first N NODE grid configurations")
    b.add_argument("--hpo-run-id", required=True)
    b.add_argument("--repeats", type=int, default=config.N_REPEATS_BENCHMARK)
    b.add_argument("--regimes", default="ID_full,ID_match,OOD_match,OOD_match_tuned")
    b.add_argument("--verify-reload", action="store_true", help="reload the weights right after saving and check that the B_te predictions are the same")
    b.add_argument("--allow-nonfinal-hpo", action="store_true", help="smoke runs only: accept search records without the final budget")
    args = ap.parse_args()
    run_hpo(args) if args.cmd == "hpo" else run_bench(args)


if __name__ == "__main__":
    main()
