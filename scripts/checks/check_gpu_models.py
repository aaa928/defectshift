"""GPU model check (run after the first benchmark): under the actual benchmark row rules (cat_050, repetition 0, the n* rows of
ID_match with the internal 80/20 early stopping, evaluation on A_heldout), the following is measured per model and saved to
results/gpu_check/<run_id>.json. No verdict, numbers only (for review).

1) maximum prediction difference between two trainings from scratch with the same seed (retraining determinism)  2) maximum
   prediction difference after save and reload (compared with the tolerance of regimes.check_reload)
3) the three boosting models: early-stopping rounds and prediction difference on the original and the standardized scale
   (measured, not assumed to be equal). For CatBoost the early-stopping settings actually applied (od_type, od_pval, od_wait), the
   best iteration, the iteration actually stopped at, and the final number of trees (= best iteration + 1)
4) LightGBM: training on CPU and GPU (device="gpu" of the official distribution; device="cuda" tried too) with the same data and seed
   — rounds, prediction difference, time. If the GPU does not work, the error string is recorded (the installed build does not
   support the GPU)  5) training time
Measured with default hyperparameters (library defaults, DEFAULT_PARAMS). Usage: python scripts/checks/check_gpu_models.py --run-id gpu_check_1
Record: status = "measured" (measurement finished, not a pass) / "failed" (exception). Passing is given by checks (per item true or
false) and all_checks_pass.
Evidence of GPU use: change in the device memory in use before and after training (gpu_mem_delta_mb; a snapshot including other
processes, evidence only, not a peak) and the process RSS (rss_mb). Model objects are released right after their measurement so that
they do not overlap the next one.
"""
import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from defectshift import config, regimes, registry  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_data  # noqa: E402
from defectshift.models import ml  # noqa: E402
from defectshift.splits import es_split, get_A, matched, repeat_split, B_view  # noqa: E402
from defectshift.utils import free_memory, log  # noqa: E402

MODELS = ("XGBoost", "LightGBM", "CatBoost", "MLP", "FT-Transformer", "RealMLP", "TabM")
BOOSTING = ("XGBoost", "LightGBM", "CatBoost")


def rows(domain):
    df, y = load_data()
    lot = df[config.LOT_COL]
    slot, mA, mB = build_one(df, domain)
    a = get_A(slot, df.index[mA], y, lot)
    bv = B_view(slot, df.index[mB], y, lot, "benchmark")
    a_train, _ = repeat_split(a["A_dev"], y, lot, 0)
    _, a_match, _ = matched(a_train, bv["B_tr"], 0)
    fit_idx, es_idx = es_split(a_match, y, lot, 0)
    cat_cols, num_cols = feature_sets(df, domain)
    return regimes.Ctx(df, y, cat_cols, num_cols, cat_mode="benchmark", fm_input="raw_native"), fit_idx, es_idx, a["A_heldout"]


def _free_mb():
    return torch.cuda.mem_get_info()[0] / 2**20 if torch.cuda.is_available() else None


def fit_pred(spec, ctx, fit_idx, es_idx, ev, seed=1, mem=None):
    """Training and prediction. With mem (dict), fills in the change of device memory in use before and after training and the RSS right after training."""
    free0 = _free_mb()
    t0 = time.time()
    f = regimes.fit(spec, ctx, fit_idx, es_idx, seed)
    sec = time.time() - t0
    if mem is not None:
        free1 = _free_mb()
        mem.update(gpu_mem_delta_mb=None if free0 is None else round(free0 - free1),
                   rss_mb=round(psutil.Process().memory_info().rss / 2**20))
    return f, regimes.predict(f, ctx, ev), sec


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def lightgbm_devices(ctx, fit_idx, es_idx, ev):
    """LightGBM trained with the same data and seed, only the device changed. ml.LIGHTGBM_DEVICE is changed temporarily so that the regimes path is used as is."""
    spec, out, base = registry.get("LightGBM"), {}, None
    for dev in ("cpu", "gpu", "cuda"):
        old = ml.LIGHTGBM_DEVICE
        ml.LIGHTGBM_DEVICE = dev
        try:
            f, p, sec = fit_pred(spec, ctx, fit_idx, es_idx, ev)
            out[dev] = dict(ok=True, fit_sec=round(sec, 2), best_round=ml._round_of(f.model),
                            device_param=f.model.get_params().get("device"))
            if dev == "cpu":
                base = p
            else:
                out[dev]["max_abs_diff_vs_cpu"] = maxdiff(p, base)
        except Exception as e:
            out[dev] = dict(ok=False, error=f"{type(e).__name__}: {e}"[:300])
        finally:
            ml.LIGHTGBM_DEVICE = old
            free_memory()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--domain", default="cat_050")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    ctx, fit_idx, es_idx, ev = rows(args.domain)
    out = dict(domain=args.domain, n_fit=len(fit_idx), n_es=len(es_idx), n_eval=len(ev),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, models={})
    tmp = tempfile.mkdtemp(prefix="gpu_check_")
    for name in args.models.split(","):
        spec = registry.get(name)
        r, mem = dict(), dict()
        f1 = f2 = f_raw = None
        try:
            f1, p1, s1 = fit_pred(spec, ctx, fit_idx, es_idx, ev, mem=mem)
            r.update(mem)
            f2, p2, s2 = fit_pred(spec, ctx, fit_idx, es_idx, ev)
            f2 = None                 # only the predictions are needed for the retraining comparison
            r.update(fit_sec=[round(s1, 2), round(s2, 2)], refit_max_abs_diff=maxdiff(p1, p2),
                     pred_std=float(np.std(p1)), y_std_eval=float(ctx.y.loc[ev].std()))
            prefix = os.path.join(tmp, name)
            regimes.save(f1, ctx, prefix)
            p3 = regimes.predict(regimes.load(spec, ctx, prefix), ctx, ev)
            r.update(reload_max_abs_diff=maxdiff(p1, p3), reload_tol=regimes.RELOAD_TOL[spec.route])
            if name in BOOSTING:
                r["best_round_std"] = ml._round_of(f1.model)
                old = regimes.Y_STANDARDIZED_ROUTES
                regimes.Y_STANDARDIZED_ROUTES = ()
                try:
                    f_raw, p_raw, _ = fit_pred(spec, ctx, fit_idx, es_idx, ev)
                finally:
                    regimes.Y_STANDARDIZED_ROUTES = old
                r.update(best_round_raw=ml._round_of(f_raw.model), raw_vs_std_max_abs_diff=maxdiff(p1, p_raw))
                f_raw = None
            if name == "CatBoost":    # whether the settings were actually applied + the best iteration and the iteration actually stopped at, separately (final model = best iteration)
                allp = f1.model.get_all_params()
                hist = f1.model.get_evals_result().get("validation", {})
                n_trained = len(next(iter(hist.values()))) if hist else None
                best = f1.model.get_best_iteration()
                r["overfitting_detector"] = {k: allp.get(k) for k in ("od_type", "od_pval", "od_wait")}
                r.update(catboost_best_iteration=best, catboost_iterations_trained=n_trained,
                         catboost_extra_after_best=None if n_trained is None or best is None else n_trained - (best + 1),
                         catboost_tree_count=f1.model.tree_count_, catboost_restored_to_best=f1.model.tree_count_ == best + 1)
            if name == "LightGBM":
                r["devices"] = lightgbm_devices(ctx, fit_idx, es_idx, ev)
            checks = dict(reload_within_tol=r["reload_max_abs_diff"] <= r["reload_tol"],
                          same_seed_refit_equal=r["refit_max_abs_diff"] <= r["reload_tol"],
                          prediction_not_constant=r["pred_std"] > 0)
            if name == "CatBoost":
                od = r["overfitting_detector"]
                checks.update(catboost_detector_as_official=(od.get("od_type") == "IncToDec" and abs((od.get("od_pval") or 0) - 0.001) < 1e-6
                                                             and int(od.get("od_wait") or 0) == 50),
                              catboost_restored_to_best=bool(r["catboost_restored_to_best"]))
            if spec.route != "ml" or name in ("XGBoost", "CatBoost"):   # models recorded as training on the GPU: whether the device memory actually grew
                checks["gpu_memory_used_during_fit"] = (r.get("gpu_mem_delta_mb") or 0) > 0
            r.update(checks=checks, all_checks_pass=all(checks.values()), status="measured")
        except Exception as e:
            r.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
            log(f"[gpu-check] {name} failed\n{traceback.format_exc()}", level="ERROR")
        out["models"][name] = r
        log(f"[gpu-check] {name}: {json.dumps(r, ensure_ascii=False, default=str)[:400]}")
        f1 = f2 = f_raw = None
        free_memory()
    path = os.path.join(config.RESULTS_DIR, "gpu_check", f"{args.run_id}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print(path)


if __name__ == "__main__":
    main()
