"""LightGBM device comparison (on cat_050): the same configurations as the actual source-side search (same rows, preprocessing, y
standardization and search space) trained on CPU / GPU (official build device="gpu", OpenCL, single precision by default) / GPU
double precision (gpu_use_dp=True), measuring time, rounds and prediction difference. No verdict; numbers only, saved to
results/gpu_check/<run_id>.json.

- rows: the search training and validation rows of hpo_split(A_dev) (as the source side of 04_tune_hyperparameters), categorical
  representation benchmark, y standardized on the search training rows (as in hpo.tune)
- configurations: the first n of the actual search sampler (TPESampler(seed=ml.RANDOM_STATE)). The first 10 TPE trials are random draws
  independent of the scores, so they are the same configurations as the first n trials of the actual search
- each device and configuration is trained twice (retraining difference with the same seed). Before the first GPU training, one 1-round
  warm-up training is run and its initialization time recorded separately
- time = wall clock of training (including early stopping), prediction time separately. Prediction difference = maximum absolute difference
  of the predictions on the search validation rows against CPU (original y scale)
Usage: python scripts/checks/check_lgbm_device.py --run-id lgbm_device_1
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import optuna
import torch
from sklearn.metrics import r2_score

from defectshift import config, regimes, registry  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_data  # noqa: E402
from defectshift.models import ml, protocol  # noqa: E402
from defectshift.splits import get_A, hpo_split  # noqa: E402
from defectshift.utils import free_memory, log  # noqa: E402

DEVICES = {"cpu": ("cpu", {}), "gpu": ("gpu", {}), "gpu_dp": ("gpu", {"gpu_use_dp": True})}


def source_rows(domain):
    """Search rows and matrices (in the order of the ML path of hpo.tune: handling of unobserved numerical columns -> y standardization -> fitting the preprocessing)."""
    df, y = load_data()
    lot = df[config.LOT_COL]
    slot, mA, _ = build_one(df, domain)
    a = get_A(slot, df.index[mA], y, lot)
    tr, va = hpo_split(a["A_dev"], y, lot)
    cat_cols, num_cols = feature_sets(df, domain)
    ctx = regimes.Ctx(df, y, cat_cols, num_cols, cat_mode="benchmark", fm_input="raw_native")
    ctx = regimes._neutralized(ctx, regimes._unobserved(ctx, tr))
    spec = registry.get("LightGBM")
    yz, ystd = regimes.standardized_target(spec, ctx.y, tr)
    tf = protocol.ml_transform(ctx.df, ctx.feature_cols, spec.impl, tr, ctx.cat_mode)
    Xtr, Xva = tf(ctx.df.loc[tr, ctx.feature_cols]), tf(ctx.df.loc[va, ctx.feature_cols])
    extra = protocol.ml_model_extra(spec.impl, ctx.feature_cols, ctx.cat_mode)
    return Xtr, yz.loc[tr], Xva, yz.loc[va], ystd, extra


def first_configs(n):
    """The first n configurations of the actual search sampler (scores not used)."""
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=ml.RANDOM_STATE))
    out = []
    for _ in range(n):
        t = study.ask()
        out.append(ml._suggest_params("reg", "LightGBM", t))
        study.tell(t, 0.0)
    return out


def fit_once(params, dev, dev_extra, Xtr, ytr, Xva, yva):
    old = ml.LIGHTGBM_DEVICE
    ml.LIGHTGBM_DEVICE = dev
    try:
        model = ml.build_one_model("reg", "LightGBM", {**params, **dev_extra})
        t0 = time.time()
        ml.fit_with_val("reg", "LightGBM", model, Xtr, ytr, Xva, yva)
        fit_sec = time.time() - t0
        t0 = time.time()
        p = model.predict(Xva)
        pred_sec = time.time() - t0
        return dict(fit_sec=round(fit_sec, 2), predict_sec=round(pred_sec, 3), best_round=ml._round_of(model),
                    val_r2=float(r2_score(yva, p)), device_param=model.get_params().get("device")), p
    finally:
        ml.LIGHTGBM_DEVICE = old
        free_memory()


def gpu_warmup(Xtr, ytr):
    t0 = time.time()
    lgb.LGBMRegressor(n_estimators=1, device="gpu", verbosity=-1).fit(Xtr, ytr)
    return round(time.time() - t0, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--domain", default="cat_050")
    ap.add_argument("--n-configs", type=int, default=3)
    args = ap.parse_args()
    Xtr, ytr, Xva, yva, ystd, extra = source_rows(args.domain)
    configs = first_configs(args.n_configs)
    out = dict(domain=args.domain, n_train=len(Xtr), n_val=len(Xva), n_features=Xtr.shape[1], y_std=ystd["std"],
               lightgbm=lgb.__version__, cpu_threads=os.cpu_count(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, configs=[])
    log(f"[lgbm-device] {args.domain} search training {len(Xtr)} rows, validation {len(Xva)} rows, {Xtr.shape[1]} features, {len(configs)} configurations")
    out["gpu_warmup_sec"] = gpu_warmup(Xtr, ytr)
    for k, params in enumerate(configs, 1):
        rec = dict(trial=k, params=params, devices={})
        base = None
        for name, (dev, dev_extra) in DEVICES.items():
            r1, p1 = fit_once({**params, **extra}, dev, dev_extra, Xtr, ytr, Xva, yva)
            r2, p2 = fit_once({**params, **extra}, dev, dev_extra, Xtr, ytr, Xva, yva)
            r = dict(fit_sec=[r1["fit_sec"], r2["fit_sec"]], predict_sec=[r1["predict_sec"], r2["predict_sec"]],
                     best_round=[r1["best_round"], r2["best_round"]], val_r2=r1["val_r2"], device_param=r1["device_param"],
                     refit_max_abs_diff=float(np.max(np.abs(p1 - p2))) * ystd["std"])
            if base is None:
                base = p1
            else:
                d = float(np.max(np.abs(p1 - base))) * ystd["std"]
                r.update(max_abs_diff_vs_cpu=d, max_abs_diff_vs_cpu_over_y_std=d / ystd["std"])
            rec["devices"][name] = r
            log(f"[lgbm-device] configuration {k} {name}: {json.dumps(r, ensure_ascii=False)}")
        out["configs"].append(rec)
    path = os.path.join(config.RESULTS_DIR, "gpu_check", f"{args.run_id}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print(path)


if __name__ == "__main__":
    main()
