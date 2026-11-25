"""Sensitivity analyses of hyperparameters, categorical representation and preprocessing. Training only; the comparison with the
benchmark is scripts/07c_sensitivity_report.py.

One factor is changed at a time, on the same rows, model seeds and regime rules as the benchmark (benchmark.regime_rows and
check_rows as they are). Scale: domains x repetitions 0-2 (the same draws as screening) x ID_match, OOD_match; A_heldout and B_te
are predicted as in the benchmark. Models that an arm does not rerun have the same condition as in the benchmark (LinearRegression
and the foundation models use defaults there too), so the benchmark values are used in the comparison.
Arms
- default: the 13 tuned models (hpo.TUNED) with default hyperparameters (params=None), categorical representation "benchmark".
- integer: only the categorical representation becomes "integer" (the integer codes of screening); hyperparameters stay the
  benchmark's source-side search values. Only the ML models whose representation changes run (LinearRegression, Ridge, kNN, SVR,
  XGBoost, LightGBM, CatBoost); DT, RF, GBR, the deep models and the foundation models do not depend on it.
- hpo_seed<s>: only the domains of --hpo-seed-domains. The split seed of the source-side search becomes s (A_dev 80/20, the same
  lot-level y>0 stratification); the search is rerun with the same budget (config.HPO_N_TRIALS) and the tuned models are trained
  with its values. The hpo.tune seed stays config.HPO_SEED. The run stops if the new split equals the benchmark's tuning split.
- uniform: training-row median imputation + standardization of the numerical inputs first (defectshift/uniform_prep.py), fitted on
  each fit's training rows (fit_idx), only for the models whose input changes (uniform_prep.affected = DT, RF, XGBoost, LightGBM,
  CatBoost, RealMLP, TabM, the six foundation models, and TabNet with --extra-hpo-run-id; NODE standardizes itself); benchmark
  hyperparameters. Runs separately with --arms uniform.
- trainscale: the released numerical values re-min-max scaled with the extremes of each fit's rows
  (defectshift/trainscale_prep.py; missing values kept), only for the models whose internal preprocessing is not confirmed to be
  affine-invariant (trainscale_prep.affected = RealMLP, TabM, TabNet, the six foundation models). Runs separately with --arms
  trainscale (chain script: scripts/extra/run_trainscale.sh).
- NODE and TabNet: run separately with --extra-hpo-run-id (a run of 06a_benchmark_node_tabnet.py hpo) and --models NODE,TabNet,
  under the allocator policy alloc_policy (NODE and TabNet with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, other models
  with the default; mixing them stops the run); 07c_sensitivity_report.py merges the runs (--sensitivity-run-id <main
  run>,<NODE/TabNet run>). default arm = node.DEFAULT / tabnet.DEFAULT; uniform arm = the source-side search values of that run;
  hpo_seed arm = searched again on the new split (the whole NODE grid, TabNet TPE). The integer arm does not apply.
- node_prep_a: NODE only, with the benchmark search values and the input preprocessing changed to the official example default (no
  standardization before the quantile transform). node.NORMALIZE is False only while this arm trains (node_preprocess_a), so the
  NODE code is unchanged; a fit whose record has normalize other than False fails. Only in runs with --extra-hpo-run-id and
  --models NODE(,TabNet). Reference = the benchmark's NODE values.
- node_c_check: within the same run and process NODE is trained once more with the main setting for one cell (first selected
  domain x repetition 0 x ID_match) and recorded against the benchmark's NODE prediction, as a basis for reading node_prep_a as a
  preprocessing effect. Only together with and after node_prep_a; the difference is recorded without a verdict.
Model files are not saved; only predictions (<prefix>.pred.npz: row indices and predictions).

Run folder: results/sensitivity/runs/<run>/{manifest.json, rows.csv, hpo_rows.csv}; predictions: cache/sensitivity/runs/<run>/preds/;
search records: cache/sensitivity/runs/<run>/hpo/<slot>/<model>__source__split<s>.json
Resuming: keys (arm, slot, repetition, model, regime) with a successful row and an intact prediction file, and searches with a record,
are skipped.
End: exit code 0 only if every expected key succeeded (fail-closed).
Usage (after the benchmark, without sharing the GPU)
  python scripts/07a_sensitivity.py --screening-run-id <screening> --hpo-run-id <hpo run> --run-id <run>
"""
import argparse
import contextlib
import os
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before importing torch

import numpy as np  # noqa: E402

from defectshift import config, hpo, manifest, metrics, regimes, registry, trainscale_prep, uniform_prep  # noqa: E402
from defectshift import benchmark as bench  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_data, parse_candidate  # noqa: E402
from defectshift.models import tabular_dl, extra_models, foundation as fm, ml, protocol, transformer  # noqa: E402,F401
from defectshift.runio import append_row, load_final_domains  # noqa: E402
from defectshift.sensitivity import NODE_C_CHECK_ARM, NODE_PREP_ARM, hpo_prefix, pred_prefix  # noqa: E402
from defectshift.splits import B_view, _strat_split, get_A, hpo_split, matched, repeat_split  # noqa: E402
from defectshift.utils import free_memory, log, new_run_id, set_seed  # noqa: E402

REGIMES = ("ID_match", "OOD_match")
KEY = ("arm", "slot", "repeat", "model", "regime")
N_REPEATS = 3
DEFAULT_SPLIT_SEED = 1999
SRC_PARAM_ARMS = ("integer", "uniform", "trainscale", NODE_PREP_ARM, NODE_C_CHECK_ARM)   # arms that use the benchmark's source-side search values as they are
ARM_SCOPE = {NODE_C_CHECK_ARM: dict(repeats=[0], regimes=["ID_match"], first_domain_only=True)}   # reduced scope per arm


def arm_repeats(arm, n):
    return [i for i in ARM_SCOPE.get(arm, {}).get("repeats", range(n)) if i < n]


def arm_regimes(arm):
    return list(ARM_SCOPE.get(arm, {}).get("regimes", REGIMES))


# Training history of this process: shows from the rows alone whether the same-process check actually ran after other fits
_PROC = dict(pid=os.getpid(), start=time.strftime("%Y-%m-%d %H:%M:%S"), fits=0)


def integer_affected(spec):
    """Whether switching the categorical mode to "integer" actually changes the model's input representation."""
    return spec.route == "ml" and protocol.ml_cat_representation(spec.impl, "benchmark") != "integer"


def uniform_affected(spec):
    """Models whose input actually changes under the uniform preprocessing (uniform arm, uniform_prep.affected)."""
    return uniform_prep.affected(spec)


def arm_models(arm, candidates):
    if arm == "default" or arm.startswith("hpo_seed"):
        return [m for m in candidates if m in extra_models.tuned_all()]
    if arm == "integer":
        return [m for m in candidates if integer_affected(extra_models.spec_of(m))]
    if arm == "uniform":
        return [m for m in candidates if uniform_affected(extra_models.spec_of(m))]
    if arm == "trainscale":   # training-row min-max scale sensitivity: only the models whose internal preprocessing is not affine-invariant
        return [m for m in candidates if trainscale_prep.affected(extra_models.spec_of(m))]
    if arm in (NODE_PREP_ARM, NODE_C_CHECK_ARM):
        return [m for m in candidates if m == "NODE"]
    raise ValueError(arm)


@contextlib.contextmanager
def node_preprocess_a():
    """NODE preprocessing arm: node.NORMALIZE = False only meanwhile (Prep reads the module constant when no normalize argument is
    given), restored afterwards."""
    from defectshift.models import node
    old = node.NORMALIZE
    node.NORMALIZE = False
    try:
        yield
    finally:
        node.NORMALIZE = old


def src_run(model, args):
    """Tuning run to read the source-side search record from: --extra-hpo-run-id for the additional models, --hpo-run-id otherwise."""
    return args.extra_hpo_run_id if model in extra_models.EXTRA_TUNED else args.hpo_run_id


ALLOC_ON = "expandable_segments:True"


def alloc_policy(pool, alloc_conf):
    """CUDA allocator policy: NODE and TabNet with the allocator option on (expandable_segments, the same condition as the NODE/TabNet
    tuning — against fragmentation in consecutive runs of the 2x2048 depth-8 grid configurations), other models with the default (off,
    the same condition as the 20 benchmark models). The two are not mixed in one run (the allocator is set when the process starts).
    Returns a reason string if the condition is violated, None otherwise."""
    xs = [m for m in pool if m in extra_models.EXTRA_TUNED]
    on = ALLOC_ON in (alloc_conf or "")
    if xs and len(xs) != len(pool):
        return f"NODE and TabNet ({xs}) run separately from the other models (--models {','.join(xs)}): the allocator conditions differ"
    if xs and not on:
        return f"NODE/TabNet runs start with PYTORCH_CUDA_ALLOC_CONF={ALLOC_ON} (the same condition as the NODE/TabNet tuning)"
    if not xs and on:
        return "models other than NODE and TabNet run with the default allocator (no PYTORCH_CUDA_ALLOC_CONF) — the same condition as the final benchmark"
    return None


def tune_extra(spec, ctx, tr, va, args, progress_path=None):
    """Search of an additional model for the hpo_seed arm (the same record fields as hpo.tune). The seed is config.HPO_SEED as in the benchmark."""
    t0 = time.time()
    n = args.trials if args.trials != config.HPO_N_TRIALS else extra_models.n_trials_final(spec.name)
    params, details = extra_models.tune(spec.name, ctx.df, ctx.y, ctx.cat_cols, ctx.num_cols, tr, va, config.HPO_SEED,
                                        n_trials=n, max_epochs=args.max_epochs, progress_path=progress_path)
    trials = details.get("trials") or []
    return dict(model=spec.name, route=spec.route, impl=spec.impl, params=params, hpo_sec=time.time() - t0, n_trials=n,
                seed=config.HPO_SEED, cat_mode=ctx.cat_mode, max_epochs=args.max_epochs, n_train=len(tr), n_val=len(va),
                train_idx_sha256=hpo._idx_sha(tr), val_idx_sha256=hpo._idx_sha(va),
                n_trials_complete=sum(t["state"] == "COMPLETE" for t in trials), **details)


def _done(rows_path, run_id):
    """Keys (string tuples) with a successful row and an intact prediction file."""
    import pandas as pd
    if not os.path.exists(rows_path):
        return set()
    d = pd.read_csv(rows_path)
    d = d[d.status == "ok"]
    return {tuple(str(r[k]) for k in KEY) for r in d.to_dict("records")
            if bench.pred_ok(pred_prefix(run_id, r["slot"], r["model"], r["regime"], r["arm"], int(r["repeat"])))}


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


HPO_SEED_CATEGORICAL = "cat_050"     # process-condition domain of the tuning-seed sensitivity analysis (one temporal and one process-condition domain)


def _no_dup(**lists):
    """Stop if a comma-separated argument has the same value twice (the same key would be trained twice but fold into one in the set of
    expected keys, so the run would look complete)."""
    for k, v in lists.items():
        if len(v) != len(set(v)):
            raise SystemExit(f"--{k} has duplicate values: {sorted(x for x in set(v) if v.count(x) > 1)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--screening-run-id", required=True)
    ap.add_argument("--hpo-run-id", required=True, help="benchmark tuning run (hyperparameters of the integer arm, reference of the hpo_seed arm)")
    ap.add_argument("--extra-hpo-run-id", default=None,
                    help="NODE/TabNet tuning run (06a_benchmark_node_tabnet.py hpo); if given, the two models become candidates")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--arms", default=None, help=f"comma-separated; default default,integer,hpo_seed{DEFAULT_SPLIT_SEED}. "
                                                 f"Uniform preprocessing: --arms uniform; training-row scale: --arms trainscale; NODE preprocessing: --arms {NODE_PREP_ARM} (NODE/TabNet run)")
    ap.add_argument("--only", default=None, help="comma-separated domains (default and integer arms); default: all selected domains")
    ap.add_argument("--hpo-seed-domains", default=None,
                    help="domains of the hpo_seed arm (comma-separated). Default = the x_time median cutoff + cat_050. If cat_050 is not "
                         "selected, the run stops without a default (specify this argument)")
    ap.add_argument("--models", default=None, help="comma-separated (testing); default: all models defined for each arm")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--trials", type=int, default=config.HPO_N_TRIALS)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="smoke runs only: epoch cap for searching and training the deep models (MLP, FT-T, RealMLP, TabM); makes the run a non-main run")
    args = ap.parse_args()

    domains, final_sha = load_final_domains(args.screening_run_id)
    names = domains
    if args.only:
        want = args.only.split(",")
        bad = [n for n in want if n not in domains]
        if bad:
            raise SystemExit(f"not selected domains: {bad}")
        names = [n for n in domains if n in want]
    if args.hpo_seed_domains:
        hs_domains = args.hpo_seed_domains.split(",")
        bad = [n for n in hs_domains if n not in domains]
        if bad:
            raise SystemExit(f"not selected domains: {bad}")
    else:   # the x_time median cutoff + cat_050; never replaced automatically by other domains
        x_time = [n for n in domains if parse_candidate(n)[0] == "x_time"]
        if len(x_time) != 1 or HPO_SEED_CATEGORICAL not in domains:
            raise SystemExit(f"the default hpo_seed domains need exactly one x_time median cutoff and {HPO_SEED_CATEGORICAL} among the "
                             "selected domains; otherwise specify --hpo-seed-domains")
        hs_domains = [x_time[0], HPO_SEED_CATEGORICAL]
    arms = (args.arms or f"default,integer,hpo_seed{DEFAULT_SPLIT_SEED}").split(",")
    for a_ in arms:
        if a_ not in ("default", "integer", "uniform", "trainscale", NODE_PREP_ARM, NODE_C_CHECK_ARM) and not (a_.startswith("hpo_seed") and a_[8:].isdigit()):
            raise SystemExit(f"unknown arm {a_} (default, integer, uniform, trainscale, {NODE_PREP_ARM}, {NODE_C_CHECK_ARM}, hpo_seed<integer>)")
        if a_.startswith("hpo_seed") and int(a_[8:]) == config.HPO_SEED:
            raise SystemExit(f"{a_}: the same as the benchmark's tuning split seed")
    if NODE_C_CHECK_ARM in arms and (NODE_PREP_ARM not in arms or arms.index(NODE_C_CHECK_ARM) < arms.index(NODE_PREP_ARM)):
        raise SystemExit(f"{NODE_C_CHECK_ARM} must come after {NODE_PREP_ARM} in the same run (same-process check)")
    if not 1 <= args.repeats <= config.N_REPEATS_BENCHMARK:
        raise SystemExit(f"--repeats must be 1 - {config.N_REPEATS_BENCHMARK}")
    hpo_args, hpo_sha = bench.load_hpo(args.hpo_run_id, final_sha)
    if hpo_args.get("cat_mode") != "benchmark":
        raise SystemExit("the categorical representation of the tuning run is not benchmark")
    extras = list(extra_models.EXTRA_TUNED) if args.extra_hpo_run_id else []
    pool = args.models.split(",") if args.models else list(registry.SPECS)   # NODE and TabNet separately through --models (allocator condition, alloc_policy)
    why = alloc_policy(pool, os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    if why:
        raise SystemExit(why)
    _no_dup(arms=arms, models=pool, **({"hpo-seed-domains": hs_domains} if args.hpo_seed_domains else {}),
            **({"only": args.only.split(",")} if args.only else {}))
    unknown = [m for m in pool if m not in registry.SPECS and m not in extras]
    if unknown:
        raise SystemExit(f"unknown models: {unknown} (NODE and TabNet together with --extra-hpo-run-id)")
    if args.extra_hpo_run_id:   # NODE/TabNet tuning run: same selection and inputs, compatible with the current code and environment (extra_runs.load_extra_hpo)
        from defectshift import extra_runs
        extra_hpo_args, extra_hpo_sha = extra_runs.load_extra_hpo(args.extra_hpo_run_id, final_sha, strict=True)
    plan = {a_: arm_models(a_, pool) for a_ in arms}
    plan_domains = {a_: (hs_domains if a_.startswith("hpo_seed") else names[:1] if ARM_SCOPE.get(a_, {}).get("first_domain_only")
                         else names) for a_ in arms}

    df, y = load_data()
    lot = df[config.LOT_COL]
    slots = {n: build_one(df, n)[0] for n in domains}
    if NODE_PREP_ARM in arms and not plan[NODE_PREP_ARM]:
        raise SystemExit(f"the {NODE_PREP_ARM} arm runs only with --extra-hpo-run-id and --models NODE(,TabNet)")
    need_src = [(n, m) for a_ in arms if a_ in SRC_PARAM_ARMS for n in plan_domains[a_] for m in plan[a_]
                if m in extra_models.tuned_all()]
    missing = [(n, m) for n, m in need_src if not hpo.done(hpo.record_prefix(src_run(m, args), slots[n], m, "source"))]
    if missing:
        raise SystemExit(f"{len(missing)} source-side benchmark search records missing (e.g. {missing[:3]})")
    # the model list is standard when none is given (main run) or for the separate NODE/TabNet run under the allocator policy (together with the NODE/TabNet tuning run)
    extra_only = bool(args.extra_hpo_run_id) and sorted(pool) == sorted(extra_models.EXTRA_TUNED)
    standard = (not args.only and (not args.models or extra_only) and args.repeats == N_REPEATS and args.trials == config.HPO_N_TRIALS
                and bool(hpo_args.get("full_run")) and args.max_epochs is None
                and (not args.extra_hpo_run_id or bool(extra_hpo_args.get("full_run"))))
    # main run = the three arms run together (the separate NODE/TabNet run has no categorical-representation arm: default and tuning-seed arms), or the uniform preprocessing alone
    full_run = standard and (({"default"} <= set(arms) and ("integer" in arms or extra_only) and any(a_.startswith("hpo_seed") for a_ in arms))
                             or set(arms) == {"uniform"} or set(arms) == {"trainscale"}
                             or (extra_only and NODE_PREP_ARM in arms and set(arms) <= {NODE_PREP_ARM, NODE_C_CHECK_ARM}))

    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(config.RESULTS_DIR, "sensitivity", "runs", run_id)
    rows_path = os.path.join(run_dir, "rows.csv")
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="sensitivity", screening_run_id=args.screening_run_id, screening_final_domains_sha256=final_sha, hpo_run_id=args.hpo_run_id,
        hpo_manifest_sha256=hpo_sha, arms=arms, plan=plan, plan_domains=plan_domains, repeats=args.repeats, regimes=list(REGIMES),
        **({"extra_hpo_run_id": args.extra_hpo_run_id, "extra_hpo_manifest_sha256": extra_hpo_sha,
            "extra_defaults": {"NODE": "node.DEFAULT", "TabNet": "tabnet.DEFAULT"}} if args.extra_hpo_run_id else {}),
        trials=args.trials, max_epochs=args.max_epochs, hpo_tune_seed=config.HPO_SEED, fm_input="raw_native", full_run=full_run,
        arm_scope={a_: dict(repeats=arm_repeats(a_, args.repeats), regimes=arm_regimes(a_)) for a_ in arms},
        cuda_env=dict(PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                      CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
        **({"uniform_preprocess": uniform_prep.DESCRIPTION} if "uniform" in arms else {}),
        **({"trainscale_preprocess": trainscale_prep.DESCRIPTION} if "trainscale" in arms else {}))))
    done = _done(rows_path, run_id)
    log(f"[sensitivity] run {run_id} ({status}) arms {arms} repetitions {args.repeats} -> {rows_path} ({len(done)} rows done)")
    for a_ in arms:
        log(f"[sensitivity]   {a_}: {len(plan_domains[a_])} domains x models {plan[a_]}")
    t_all = time.time()

    for name in domains:
        todo = [a_ for a_ in arms if name in plan_domains[a_]]
        if not todo:
            continue
        kind, _ = parse_candidate(name)
        slot, mA, mB = build_one(df, name)
        A_idx, B_idx = df.index[mA], df.index[mB]
        a = get_A(slot, A_idx, y, lot)
        bv = B_view(slot, B_idx, y, lot, "benchmark")
        cat_cols, num_cols = feature_sets(df, name)
        ctx_of = {m: regimes.Ctx(df, y, cat_cols, num_cols, cat_mode=m, fm_input="raw_native") for m in ("benchmark", "integer")}
        reps = {}
        for i in range(args.repeats):
            a_train, a_val = repeat_split(a["A_dev"], y, lot, i)
            n_star, a_match, b_match = matched(a_train, bv["B_tr"], i)
            reps[i] = (a_train, a_val, n_star, a_match, b_match)
        t_slot = time.time()
        for arm in todo:
            cat_mode = "integer" if arm == "integer" else "benchmark"
            ctx = ctx_of[cat_mode]
            hs_split = None
            if arm.startswith("hpo_seed"):
                s = int(arm[8:])
                tr, va = _strat_split(a["A_dev"], y, lot, config.HPO_VAL_FRAC, s, f"sensitivity tuning split (seed {s})")
                benchmark_tr, benchmark_va = hpo_split(a["A_dev"], y, lot)
                if set(va) == set(benchmark_va):
                    raise SystemExit(f"{slot}: the tuning split of seed {s} equals the benchmark's")
                hs_split = (s, tr, va, _jaccard(va, benchmark_va))
            for mname in plan[arm]:
                spec = extra_models.spec_of(mname)
                params, params_from = None, "default"
                try:
                    if arm in SRC_PARAM_ARMS and mname in extra_models.tuned_all():
                        rec_src = hpo.load(hpo.record_prefix(src_run(mname, args), slot, mname, "source"))
                        # the record must belong to this model, side, domain, search rows (recomputed from the current split) and final
                        # budget (the same check as the benchmark); the budget is checked only for a main tuning run (skipped for a smoke tuning run)
                        full_src = bool((extra_hpo_args if mname in extra_models.EXTRA_TUNED else hpo_args).get("full_run"))
                        need = ((extra_models.n_trials_final(mname) if mname in extra_models.EXTRA_TUNED else config.HPO_N_TRIALS)
                                if full_src else None)
                        hpo.check_record(rec_src, spec, "source", slot, hpo_split(a["A_dev"], y, lot), need)
                        params = rec_src["params"]
                        params_from = "hpo:source(benchmark)"
                    elif hs_split is not None:
                        s, tr, va, jac = hs_split
                        hp = hpo_prefix(run_id, slot, mname, s)
                        if not hpo.done(hp):
                            hrow = dict(run_id=run_id, slot=slot, candidate=name, model=mname, split_seed=s, n_train=len(tr),
                                        n_val=len(va), val_jaccard_vs_benchmark=jac)
                            try:
                                progress = hp + ".grid_progress.jsonl"   # NODE per-configuration grid progress record
                                rec = (tune_extra(spec, ctx, tr, va, args, progress_path=progress) if mname in extra_models.EXTRA_TUNED
                                       else hpo.tune(spec, ctx, tr, va, n_trials=args.trials, max_epochs=args.max_epochs))
                                rec.update(slot=slot, candidate=name, side="source", split_seed=s, run_id=run_id, val_jaccard_vs_benchmark=jac)
                                hpo.save(rec, hp)
                                if os.path.exists(progress):
                                    os.remove(progress)
                                hrow.update(status="ok", hpo_sec=round(rec["hpo_sec"], 2), best_score=rec["best_score"])
                                log(f"[sensitivity-HPO] {slot} {mname:15s} split seed {s} best R2={rec['best_score']:.4f} ({rec['hpo_sec']:.0f}s)")
                            except Exception as e:
                                hrow.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                                log(f"[sensitivity-HPO] {slot} {mname} failed\n{traceback.format_exc()}", level="ERROR")
                            append_row(os.path.join(run_dir, "hpo_rows.csv"), hrow)
                            free_memory()
                        if not hpo.done(hp):
                            continue                       # search failed: skip without training rows; left incomplete at the end
                        rec_hs = hpo.load(hp)
                        need_hs = (args.trials if args.trials != config.HPO_N_TRIALS else extra_models.n_trials_final(mname)) \
                            if mname in extra_models.EXTRA_TUNED else args.trials
                        # a record left from resuming must also belong to this model, domain, split rows, budget and epoch cap
                        hpo.check_record(rec_hs, spec, "source", slot, (tr, va), need_hs, args.max_epochs)
                        params, params_from = rec_hs["params"], f"hpo:source(split {s})"
                except SystemExit:
                    raise
                except Exception as e:
                    log(f"[sensitivity] {slot} {arm} {mname} hyperparameter preparation failed: {e}", level="ERROR")
                    continue
                for i in arm_repeats(arm, args.repeats):
                    a_train, a_val, n_star, a_match, b_match = reps[i]
                    for regime in arm_regimes(arm):
                        if (arm, slot, str(i), mname, regime) in done:
                            z = np.load(pred_prefix(run_id, slot, mname, regime, arm, i) + ".pred.npz")
                            if not (np.array_equal(z["id_idx"], np.asarray(a["A_heldout"]))
                                    and np.array_equal(z["ood_idx"], np.asarray(bv["B_te"]))):
                                raise SystemExit(f"{slot} r{i} {mname} {regime} {arm}: stored evaluation rows differ from the current split")
                            continue
                        fit_idx, es_idx = bench.regime_rows(regime, spec, a_train, a_val, a_match, b_match, bv, y, lot, i)
                        seed = config.RANDOM_STATE + config.SEED_OFFSET[regime] + i       # the same seed as the benchmark
                        row = dict(run_id=run_id, arm=arm, slot=slot, candidate=name, kind=kind, repeat=i, model=mname,
                                   family=spec.family, regime=regime, cat_mode=cat_mode, params_from=params_from,
                                   n_star=n_star, n_fit=len(fit_idx), n_es=0 if es_idx is None else len(es_idx), seed=seed,
                                   n_id=len(a["A_heldout"]), n_ood=len(bv["B_te"]),
                                   proc_pid=_PROC["pid"], proc_start=_PROC["start"], proc_prior_fits=_PROC["fits"])
                        try:
                            bench.check_rows(regime, fit_idx, es_idx, a, bv)
                            ctx_fit = ctx
                            if arm == "uniform":   # numerical imputation and standardization fitted on each fit's training rows, applied first
                                ctx_fit, uinfo = uniform_prep.uniform_ctx(ctx, fit_idx)
                                row.update(preprocess="uniform", n_uniform_unobserved=len(uinfo["uniform_unobserved_num_cols"]),
                                           uniform_change=uniform_prep.change(spec))   # what actually changes in this model's input
                            elif arm == "trainscale":   # numerical features re-min-max scaled with the extremes of each training's fit rows (missing values and imputation rules unchanged)
                                ctx_fit, tinfo = trainscale_prep.trainscale_ctx(ctx, fit_idx)
                                row.update(preprocess="trainscale", n_trainscale_unobserved=len(tinfo["trainscale_unobserved_num_cols"]),
                                           n_trainscale_constant=len(tinfo["trainscale_constant_num_cols"]),
                                           trainscale_change=trainscale_prep.change(spec))
                            t0 = time.time()
                            if mname in extra_models.EXTRA_TUNED:   # official code path (input transformation in each module; uniform applied before it)
                                set_seed(seed)
                                with (node_preprocess_a() if arm == NODE_PREP_ARM else contextlib.nullcontext()):
                                    preds, xinfo = extra_models.fit_predict(mname, ctx_fit.df, y, cat_cols, num_cols, fit_idx, es_idx,
                                                                            {"id": a["A_heldout"], "ood": bv["B_te"]}, params, seed,
                                                                            max_epochs=args.max_epochs)
                                row["fit_sec"] = round(time.time() - t0, 3)
                                if mname == "NODE":   # the preprocessing actually used (example default = normalize False, main setting = True)
                                    row["normalize"] = xinfo.get("normalize")
                                    if xinfo.get("normalize") is not (arm != NODE_PREP_ARM):
                                        raise RuntimeError(f"NODE preprocessing differs from the arm: arm={arm}, normalize={xinfo.get('normalize')}")
                                    if arm == NODE_PREP_ARM:
                                        row["preprocess"] = "node_A"
                                p_id, p_ood = preds["id"], preds["ood"]
                                n_rounds, unit, cat_rep = xinfo["best_epoch"], "epochs", None
                                row["batch_size"] = xinfo.get("batch_size")   # NODE: 256 only for the 4 configurations of the fixed list (node.REDUCED_BATCH)
                            else:
                                f = regimes.fit(spec, ctx_fit, fit_idx, es_idx, seed, params=params, max_epochs=args.max_epochs)
                                row["fit_sec"] = round(time.time() - t0, 3)
                                p_id = regimes.predict(f, ctx_fit, a["A_heldout"])
                                p_ood = regimes.predict(f, ctx_fit, bv["B_te"])
                                (n_rounds, unit), cat_rep = bench.rounds_used(f.config), f.config.get("cat_representation")
                            row.update({f"{k}_id": v for k, v in metrics.regression(y.loc[a["A_heldout"]], p_id).items()})
                            row.update({f"{k}_ood": v for k, v in metrics.regression(y.loc[bv["B_te"]], p_ood).items()})
                            row.update(n_rounds_used=n_rounds, n_rounds_unit=unit, cat_representation=cat_rep)
                            prefix = pred_prefix(run_id, slot, mname, regime, arm, i)
                            os.makedirs(os.path.dirname(prefix), exist_ok=True)
                            np.savez(prefix + ".pred.npz", id_idx=np.asarray(a["A_heldout"]), id_pred=p_id,
                                     ood_idx=np.asarray(bv["B_te"]), ood_pred=p_ood)
                            row["status"] = "ok"
                            log(f"[sensitivity] {slot} {arm:12s} r{i} {mname:15s} {regime:9s} R2 id={row['r2_id']:.3f} ood={row['r2_ood']:.3f}")
                        except Exception as e:
                            row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                            log(f"[sensitivity] {slot} {arm} r{i} {mname} {regime} failed\n{traceback.format_exc()}", level="ERROR")
                        append_row(rows_path, row)
                        _PROC["fits"] += 1                 # successes and failures both count as evaluation fits attempted in this process (search trials excluded)
                        free_memory()
                # foundation models unload their weights after each domain and arm (as in the benchmark); otherwise later models
                # (NODE and TabNet of the uniform arm, for example) would run on a GPU still holding the weights of the six foundation models
                f = None
                if spec.route == "foundation":
                    fm.release(spec.name)
                    free_memory()
        log(f"[sensitivity] {slot} done: {time.time() - t_slot:.0f}s")
    log(f"[sensitivity] total {time.time() - t_all:.0f}s")
    expected = {(a_, slots[n], str(i), m, r) for a_ in arms for n in plan_domains[a_] for m in plan[a_]
                for i in arm_repeats(a_, args.repeats) for r in arm_regimes(a_)}
    lacking = sorted(expected - _done(rows_path, run_id))
    if lacking:
        log(f"[sensitivity] incomplete: {len(lacking)} of {len(expected)} expected keys did not succeed (e.g. {lacking[:5]}). Rerunning with "
            f"the same --run-id trains only those", level="ERROR")
        sys.exit(1)
    log(f"[sensitivity] complete: all {len(expected)} expected keys succeeded")


if __name__ == "__main__":
    main()
