"""Hyperparameter search of the benchmark. For every domain and tuned model: one source-side search (tuning split of A_dev)
and one target-side search (B_tr / B_val, for OOD_match_tuned). The chosen values are fixed across repetitions and regimes.

- Budget: config.HPO_N_TRIALS (30). ML models, MLP and FT-Transformer: 30 Optuna TPE trials (sampler seed fixed to 42);
  RealMLP and TabM: 30 steps of the official pytabkit HPO (30 random configurations, random_state = search seed).
- The global seed (config.HPO_SEED) is fixed at the start. Preprocessing statistics come from the tuning training rows only;
  the categorical representation is ctx.cat_mode ("benchmark" in the main benchmark).
- Target standardization follows the rule of regimes: the ml and transformer routes standardize y with the mean and
  standard deviation of the tuning training rows (validation y transformed with the same statistics). The validation R^2
  applies the same affine map to y and predictions and therefore equals the R^2 on the original scale.
- Numerical columns without any observed value in the tuning training rows are set to a constant, as in regimes, and recorded.
- Not tuned: LinearRegression (no hyperparameters) and the foundation models (default settings, in-context).
- Records: prefix.json (the values used for training and the summary), with search time, trial log and best validation score
  (R^2). Only plain values (numbers, strings, booleans, lists, dictionaries) are stored; a record is never pickled.
"""
import hashlib
import json
import os
import time

import numpy as np

from defectshift import config, regimes
from defectshift.models import protocol

TUNED = ("Ridge", "kNN", "SVR", "DecisionTree", "RandomForest", "GBR", "XGBoost", "LightGBM", "CatBoost",
         "MLP", "FT-Transformer", "RealMLP", "TabM")
SIDES = ("source", "target")


def record_prefix(run_id, slot, model, side):
    """Path of a search record (without extension): cache/benchmark/hpo/<run_id>/<slot>/<model>__<side>."""
    from defectshift.candidates import slot_dir
    return os.path.join(config.CACHE_DIR, "benchmark", "hpo", run_id, slot_dir(slot), f"{model}__{side}")


def _idx_sha(idx):
    return hashlib.sha256(np.sort(np.asarray(idx, dtype=np.int64)).tobytes()).hexdigest()


def check_record(rec, spec, side, slot, rows, need_trials, max_epochs=None):
    """Whether a search record used for training belongs to this domain, model and side, was obtained on the search rows
    recomputed with the current protocol (hashes of the training and validation rows), with need_trials trials and the
    smoke-test epoch cap max_epochs (None = protocol value). With need_trials None, the trial count and epoch cap are not
    checked (benchmark smoke tests). Stops before training on a mismatch. Used by the benchmark and by screening."""
    tr, va = rows
    bad = []
    if rec.get("model") != spec.name or rec.get("side") != side or rec.get("slot") != slot:
        bad.append(f"model, side and domain of the record {rec.get('model')}/{rec.get('side')}/{rec.get('slot')}")
    if need_trials is not None and rec.get("n_trials") != need_trials:
        bad.append(f"trials {rec.get('n_trials')} != required trials {need_trials} (full runs use the final budget config.HPO_N_TRIALS)")
    if need_trials is not None and rec.get("max_epochs") != max_epochs:
        bad.append(f"epoch cap {rec.get('max_epochs')} != {max_epochs} (None = protocol value)")
    if rec.get("train_idx_sha256") != _idx_sha(tr) or rec.get("val_idx_sha256") != _idx_sha(va):
        bad.append("the search training and validation rows differ from those recomputed with the current splits")
    if bad:
        raise SystemExit(f"{slot} {spec.name} {side}: search record cannot be used: " + "; ".join(bad))


def tune(spec, ctx, train_idx, val_idx, n_trials=config.HPO_N_TRIALS, seed=config.HPO_SEED, max_epochs=None):
    """Search for one model: train on train_idx, evaluate on val_idx. Returns the record dict (params = the values passed
    to regimes.fit). max_epochs: epoch cap for smoke tests only (MLP, FT-T, RealMLP, TabM); None = protocol value. It is
    recorded, and the benchmark uses only records with None."""
    assert spec.name in TUNED, f"{spec.name} is not a tuned model"
    assert not set(train_idx) & set(val_idx), "search training and validation rows overlap"
    unobserved = regimes._unobserved(ctx, train_idx)
    ctx = regimes._neutralized(ctx, unobserved)
    details = {}
    y, ystd = regimes.standardized_target(spec, ctx.y, train_idx)
    t0 = time.time()
    if spec.route == "ml":
        params = protocol.tune_ml(config.TASK, spec.impl, ctx.df, ctx.feature_cols, y, train_idx, val_idx, n_trials,
                                  seed=seed, cat_mode=ctx.cat_mode, details=details)
    elif spec.route == "transformer":
        params = protocol.tune_transformer(config.TASK, spec.impl, ctx.df, ctx.cat_cols, ctx.num_cols, y, train_idx,
                                           val_idx, n_trials, seed=seed, details=details,
                                           max_epochs=max_epochs or protocol.TRANSFORMER_MAX_EPOCHS)
    elif spec.route == "tabular_dl":
        params = protocol.tune_dl2(config.TASK, spec.impl, ctx.df, ctx.feature_cols, ctx.cat_cols, ctx.y, train_idx,
                                   val_idx, n_trials, seed=seed, details=details, n_epochs=max_epochs)
    else:
        raise ValueError(spec.route)
    trials = details.get("trials") or []
    details["n_trials_complete"] = sum(t["state"] == "COMPLETE" for t in trials) if trials else None   # pytabkit logs no trials
    return dict(model=spec.name, route=spec.route, impl=spec.impl, params=params, hpo_sec=time.time() - t0,
                n_trials=n_trials, seed=seed, cat_mode=ctx.cat_mode, unobserved_num_cols=unobserved, y_standardization=ystd,
                max_epochs=max_epochs,
                n_train=len(train_idx), n_val=len(val_idx), train_idx_sha256=_idx_sha(train_idx),
                val_idx_sha256=_idx_sha(val_idx), **details)


def _plain(v, path="record"):
    """The record as plain Python values that JSON stores without loss: numpy scalars and arrays become Python numbers and
    lists, tuples become lists, dictionary keys become strings. Any other type is refused, so that nothing is stored as text."""
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return _plain(v.tolist(), path)
    if isinstance(v, dict):
        return {str(k): _plain(x, f"{path}.{k}") for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x, f"{path}[{i}]") for i, x in enumerate(v)]
    raise TypeError(f"a search record can hold only plain values, not {type(v).__name__} (at {path})")


def save(record, prefix):
    """Write the record to prefix.json. The file is written under a temporary name and renamed, so that an interrupted write
    never leaves a record that done() accepts."""
    plain = _plain(record)  # evaluated first: an unsupported value raises before any file is created
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    tmp = prefix + ".json.tmp"
    with open(tmp, "w") as f:
        json.dump(plain, f, ensure_ascii=False, indent=1)
    os.replace(tmp, prefix + ".json")


def load(prefix):
    with open(prefix + ".json") as f:
        return json.load(f)


def done(prefix):
    return os.path.isfile(prefix + ".json")
