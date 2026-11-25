"""Foundation-model qualification of the benchmark: checks only whether a model can run under the protocol, not its performance.

Data: only A_dev of the selected process-condition domain with the largest A (cat_050) (A_heldout and B are not used; OOD
performance is not looked at). Context sizes 2,000 and 8,000 rows (drawn from A_dev with a fixed seed), query = A_dev rows not in
the context (at most 2,000 rows). Every call goes through the same path as the benchmark (regimes.fit / predict / save / load,
cat_mode "benchmark", fm_input "raw_native" = raw numerical values with missing values kept).

Pass criteria (constants below):
- output: one-dimensional real array, as many values as query rows, no NaN or Inf, not constant
- context 2,000 and 8,000 both finish without an exception
- time: training + prediction <= TIME_LIMIT_S at context 8,000
- memory: peak torch GPU allocation <= MEM_LIMIT_GIB (reset and measured per step)
- reproducibility: maximum prediction difference between two trainings and predictions with the same seed <= REPRO_TOL (same object reused)
- fresh instance: maximum prediction difference after unloading the weights and training and predicting with a newly loaded object
  and the same seed <= REPRO_TOL
- state isolation: after one training and prediction with another context (same size, other rows and seed), training and predicting
  again with the original context differs by at most REPRO_TOL (no context or cache of an earlier training remains in the next one —
  the benchmark reuses the same object across repetitions and regimes)
- protocol: maximum prediction difference after regimes.save -> load (context rebuilt from the saved context rows and seed) <= REPRO_TOL
- recorded for reference (not used for the verdict): query R^2, training and prediction times, memory, versions, commits, weight
  revisions, the configuration reported by the model
Judged models: the three additional ones, fm.QUALIFY_JUDGED (Kumo, TabDPT, EXAONE) — a model that fails is excluded from the
benchmark without replacement. TabPFN, TabICL and TabFM run the same test and are only reported (always in the benchmark roster);
the summary marks them by judged.
Resuming: only successful rows count as done (temporary failures are retried). With several rows of the same (model, context), the
last row is the record of reference.

Output: results/benchmark/fm_qualify/<run_id>/{manifest.json, rows.csv, summary.json}
Usage: python scripts/05_qualify_foundation.py --screening-run-id <screening> --run-id <run>     # all six (needed for the benchmark main run)
"""
import argparse
import json
import os
import tempfile
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from defectshift import config, manifest, metrics, regimes, registry  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_data, subsample  # noqa: E402
from defectshift.models import foundation as fm  # noqa: E402
from defectshift.runio import append_row  # noqa: E402
from defectshift.splits import get_A  # noqa: E402
from defectshift.utils import free_memory, log, new_run_id  # noqa: E402

DOMAIN = "cat_050"
CONTEXT_SIZES = (2000, 8000)
QUERY_MAX = 2000
TIME_LIMIT_S = 600.0
MEM_LIMIT_GIB = 20.0
REPRO_TOL = 1e-5
SEED = 7


def _load_final(screening_run_id):
    path = os.path.join(config.RESULTS_DIR, "screening", "runs", screening_run_id, "selection", "final_domains.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist")
    # DOMAIN (cat_050) is fixed regardless of the selection result. The test uses only the A side to check supported range, memory,
    # time and reproducibility, so it need not be a final domain (for every selected domain the source training rows are within the
    # 8,000-row context range of the test). The result is tied to the hash of this selection's final_domains.json, so it is rerun after
    # a new selection.
    return manifest.file_sha256(path)


def _peak_gib():
    """Peak torch allocation (GiB). Without a GPU it cannot be measured, so None (0 would look as if the memory limit passed)."""
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def run_one(spec, ctx, context_idx, query_idx, y, pool):
    """One training and prediction + a rerun with the same seed + save and reload. Dict of measurements."""
    out = dict(n_context=len(context_idx), n_query=len(query_idx))
    _reset_peak()
    t0 = time.time()
    f = regimes.fit(spec, ctx, context_idx, None, SEED)
    out["fit_sec"] = time.time() - t0
    t0 = time.time()
    p = regimes.predict(f, ctx, query_idx)
    out["predict_sec"] = time.time() - t0
    out["peak_gib"] = _peak_gib()
    p = np.asarray(p)
    out.update(shape_ok=bool(p.ndim == 1 and len(p) == len(query_idx)), finite=bool(np.isfinite(p).all()),
               nonconstant=bool(np.nanstd(p) > 0), pred_std=float(np.nanstd(p)))
    out["r2_query_info"] = metrics.regression(y.loc[query_idx], p)["r2"] if out["finite"] else None
    f2 = regimes.fit(spec, ctx, context_idx, None, SEED)
    out["repro_max_abs_diff"] = float(np.max(np.abs(np.asarray(regimes.predict(f2, ctx, query_idx)) - p)))
    # another context: drawn with another seed from the A_dev rows (pool) without the query (if rows are short, the same rows with another seed)
    other = subsample(pool.difference(query_idx), len(context_idx), config.RANDOM_STATE + 2)
    regimes.predict(regimes.fit(spec, ctx, other, None, SEED + 1), ctx, query_idx)      # once with another context
    f3 = regimes.fit(spec, ctx, context_idx, None, SEED)
    out["isolation_max_abs_diff"] = float(np.max(np.abs(np.asarray(regimes.predict(f3, ctx, query_idx)) - p)))
    fm.release(spec.name)                                                                 # fresh instance
    f4 = regimes.fit(spec, ctx, context_idx, None, SEED)
    out["fresh_max_abs_diff"] = float(np.max(np.abs(np.asarray(regimes.predict(f4, ctx, query_idx)) - p)))
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, spec.name)
        regimes.save(f2, ctx, prefix)
        g = regimes.load(spec, ctx, prefix)
        out["reload_max_abs_diff"] = float(np.max(np.abs(np.asarray(regimes.predict(g, ctx, query_idx)) - p)))
    out["config_get_params"] = json.dumps(regimes._jsonable(f.config.get("get_params")), ensure_ascii=False)[:2000]
    out["limits"] = json.dumps(regimes._jsonable(fm.limits(spec.name, f.model)), ensure_ascii=False)
    out["effective"] = json.dumps(regimes._jsonable(f.config.get("effective")), ensure_ascii=False)
    out["n_features_in"] = len(ctx.feature_cols)
    return out


def judge(rows):
    """Verdict and reasons for one model from its rows (one per context size)."""
    reasons = []
    by = {r["n_context_target"]: r for r in rows}
    for n in CONTEXT_SIZES:
        r = by.get(n)
        if r is None or r.get("status") != "ok":
            reasons.append(f"context {n}: failed ({None if r is None else r.get('error')})")
            continue
        if not (r["shape_ok"] and r["finite"] and r["nonconstant"]):
            reasons.append(f"context {n}: invalid output (shape {r['shape_ok']}, finite {r['finite']}, nonconstant {r['nonconstant']})")
        if r["peak_gib"] is None:
            reasons.append(f"context {n}: GPU memory not measured (run without a GPU) — cannot be judged, so failed")
        elif r["peak_gib"] > MEM_LIMIT_GIB:
            reasons.append(f"context {n}: memory {r['peak_gib']:.1f} GiB > {MEM_LIMIT_GIB}")
        if r["repro_max_abs_diff"] > REPRO_TOL:
            reasons.append(f"context {n}: reproduction difference {r['repro_max_abs_diff']:.2e} > {REPRO_TOL}")
        for key, label in (("isolation_max_abs_diff", "state isolation"), ("fresh_max_abs_diff", "fresh instance"),
                           ("reload_max_abs_diff", "save and reload")):
            if r[key] > REPRO_TOL:
                reasons.append(f"context {n}: {label} difference {r[key]:.2e} > {REPRO_TOL}")
    big = by.get(max(CONTEXT_SIZES))
    if big is not None and big.get("status") == "ok" and big["fit_sec"] + big["predict_sec"] > TIME_LIMIT_S:
        reasons.append(f"context {max(CONTEXT_SIZES)}: time {big['fit_sec'] + big['predict_sec']:.0f}s > {TIME_LIMIT_S}")
    return dict(passed=not reasons, reasons=reasons)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--screening-run-id", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--models", default=",".join(fm.FOUNDATION_MODELS))
    args = ap.parse_args()

    final_sha = _load_final(args.screening_run_id)
    models = [registry.get(m) for m in args.models.split(",")]
    bad = [m.name for m in models if m.route != "foundation"]
    if bad:
        raise SystemExit(f"not foundation models: {bad}")
    run_id = args.run_id or new_run_id()
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "fm_qualify", run_id)
    rows_path = os.path.join(run_dir, "rows.csv")
    criteria = dict(domain=DOMAIN, context_sizes=CONTEXT_SIZES, query_max=QUERY_MAX, time_limit_s=TIME_LIMIT_S,
                    mem_limit_gib=MEM_LIMIT_GIB, repro_tol=REPRO_TOL, seed=SEED, fm_input="raw_native")
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="benchmark_fm_qualify", screening_run_id=args.screening_run_id, screening_final_domains_sha256=final_sha,
        models=[m.name for m in models], criteria=criteria,
        foundation_weights={m.name: fm.weights_record(m.name) for m in models})))      # pinned weight files (commit, sha256); checked by benchmark.load_fmq
    log(f"[FMQ] run {run_id} ({status}) models {[m.name for m in models]} criteria {criteria}")

    df, y = load_data()
    lot = df[config.LOT_COL]
    slot, mA, _ = build_one(df, DOMAIN)
    a = get_A(slot, df.index[mA], y, lot)
    A_dev = a["A_dev"]
    cat_cols, num_cols = feature_sets(df, DOMAIN)
    ctx = regimes.Ctx(df, y, cat_cols, num_cols, cat_mode="benchmark", fm_input="raw_native")
    done = set()
    if os.path.exists(rows_path):
        import pandas as pd
        d = pd.read_csv(rows_path)
        d = d[d.status == "ok"]
        done = set(zip(d.model, d.n_context_target))
    summary = {}
    for spec in models:
        for n in CONTEXT_SIZES:
            if (spec.name, n) in done:
                continue
            context_idx = subsample(A_dev, min(n, len(A_dev)), config.RANDOM_STATE)
            rest = A_dev.difference(context_idx)
            query_idx = subsample(rest, min(QUERY_MAX, len(rest)), config.RANDOM_STATE + 1)
            row = dict(run_id=run_id, model=spec.name, version=fm.version(spec.name), n_context_target=n)
            try:
                row.update(run_one(spec, ctx, context_idx, query_idx, y, A_dev), status="ok")
                log(f"[FMQ] {spec.name:7s} context {n}: fit {row['fit_sec']:.1f}s predict {row['predict_sec']:.1f}s "
                    f"peak {row['peak_gib']:.2f} GiB repro {row['repro_max_abs_diff']:.1e} isolation {row['isolation_max_abs_diff']:.1e} "
                    f"fresh {row['fresh_max_abs_diff']:.1e} reload {row['reload_max_abs_diff']:.1e} "
                    f"R2 (reference) {row['r2_query_info']}")
            except Exception as e:
                row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
                log(f"[FMQ] {spec.name} context {n} failed\n{traceback.format_exc()}", level="ERROR")
            append_row(rows_path, row)
            free_memory()
    import pandas as pd
    d = pd.read_csv(rows_path).drop_duplicates(["model", "n_context_target"], keep="last")   # the retried row is the record of reference
    for spec in models:
        rows = d[d.model == spec.name].to_dict("records")
        summary[spec.name] = dict(version=fm.version(spec.name), judged=spec.name in fm.QUALIFY_JUDGED, **judge(rows))
    json.dump(dict(criteria=criteria, results=summary), open(os.path.join(run_dir, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    log(f"[FMQ] verdicts: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
