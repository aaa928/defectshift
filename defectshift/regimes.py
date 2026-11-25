"""Train one model on given training rows, predict given rows, and save and restore it (common entry point of the four
training routes).

Callers pass only the training rows (fit_idx) and the early-stopping rows (es_idx). Which rows to use (A_match / B_match,
internal 80/20) is decided by the caller with the rules of splits.py; this module never looks outside the rows it receives.
- Early-stopping models (spec.uses_es) must receive es_idx, other models must not.
- Preprocessing statistics (median, standardization) are computed on fit_idx only.
- Numerical columns without any observed value in fit_idx carry no information in that fit and are set to the constant 0
  in all training and evaluation rows (equivalent to dropping the column while keeping the number of input columns, so the
  save and restore paths are unchanged). The list is recorded as unobserved_num_cols in the settings JSON. This avoids an
  undefined training-row median, which leaves NaN after imputation when missing-value groups overlap in a small n*.
- Target standardization: the ml and transformer routes (OLS, Ridge, kNN, SVR, DT, RF, GBR, XGBoost, LightGBM, CatBoost,
  MLP, FT-Transformer) standardize y with the mean and standard deviation of the training rows (fit_idx, without the
  early-stopping rows) and transform the predictions back to the original scale (Gorishniy et al. 2021: "We apply
  standardization to regression targets for all algorithms"). The y of the early-stopping rows is transformed with the same
  statistics. tabular_dl (RealMLP, TabM) is excluded because the official pytabkit implementation does the same, and the
  foundation models receive the original y through their official interfaces. The statistics are recorded as
  y_standardization in the settings JSON, and predict and load use them.
"""
import importlib.metadata as _md
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from defectshift import config
from defectshift.models import protocol
from defectshift.utils import fit_cat_cont_transform, prep_features_cat_cont, set_seed, cat_cardinalities


@dataclass
class Ctx:
    """Data context of one candidate. feature_cols = cat_cols + num_cols (the column defining the split is already
    removed). cat_mode = categorical representation of the ML models (protocol.CAT_MODES): "integer" (screening,
    selection audit, integer-coding sensitivity analysis) / "benchmark" (main benchmark). fm_input = numerical input of the
    foundation models: "impute_standardize" (training-row median imputation and standardization; screening and selection
    audit) / "raw_native" (main benchmark: raw numerical values with missing entries, no external imputation or
    standardization; missing values are handled by each model's official interface)."""
    df: object
    y: object
    cat_cols: list
    num_cols: list
    cat_mode: str = "integer"
    fm_input: str = "impute_standardize"

    @property
    def feature_cols(self):
        return self.cat_cols + self.num_cols


@dataclass
class Fitted:
    spec: object
    model: object
    fit_idx: object
    es_idx: object
    seed: int
    params: dict
    fit_sec: float
    config: dict = field(default_factory=dict)


Y_STANDARDIZED_ROUTES = ("ml", "transformer")


def y_stats(y, rows):
    """Mean and standard deviation (ddof=0) of y over rows. Stops without training if the standard deviation is zero or
    not finite."""
    v = np.asarray(y.loc[rows], dtype=float)
    mean, std = float(v.mean()), float(v.std())
    if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
        raise ValueError(f"cannot standardize y: {len(v)} training rows with mean {mean} and standard deviation {std}")
    return mean, std


def standardized_target(spec, y, rows):
    """(y standardized with the statistics of rows, record dict) if spec standardizes its target, else (y, None)."""
    if spec.route not in Y_STANDARDIZED_ROUTES:
        return y, None
    mean, std = y_stats(y, rows)
    return (y - mean) / std, dict(mean=mean, std=std, ddof=0, n_rows=len(rows))


def _unobserved(ctx, fit_idx):
    """Numerical columns without any observed value in fit_idx (training rows only)."""
    return [c for c in ctx.num_cols if ctx.df.loc[fit_idx, c].isna().all()]


def _neutralized(ctx, cols):
    """Context with cols set to the constant 0 in all rows (unchanged if cols is empty)."""
    if not cols:
        return ctx
    df = ctx.df.copy()
    df[list(cols)] = 0.0
    return Ctx(df, ctx.y, ctx.cat_cols, ctx.num_cols, ctx.cat_mode, ctx.fm_input)


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return repr(v)


def _pkg(name):
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _resolved_ml(impl, model):
    """Settings the library actually applied internally; recorded separately because get_params sometimes returns only
    None for defaults."""
    try:
        if impl == "XGBoost":
            # Training uses device=cuda (ml.build_one_model). ml.fit_with_val switches to device=cpu for prediction right
            # after training, so the device in the booster configuration below is the prediction-time value (cpu); all
            # other values are those of training.
            return dict(train_device="cuda", predict_device="cpu",
                        booster_config_at_predict=json.loads(model.get_booster().save_config()),
                        n_trees_trained=model.get_booster().num_boosted_rounds())
        if impl == "LightGBM":
            return dict(booster_params=model.booster_.params)
        if impl == "CatBoost":
            return dict(all_params=model.get_all_params())
    except Exception as e:  # a failed record must not block training
        return dict(error=repr(e))
    return None   # scikit-learn estimators: get_params is already complete


def _fm_input(ctx, fit_idx, mode, idx):
    """Foundation-model input (X_cat integer category ids, X_cont numerical values) for the idx rows. impute_standardize =
    training-row (fit_idx) median imputation and standardization; raw_native = raw numerical values (with missing entries)."""
    if mode == "impute_standardize":
        return prep_features_cat_cont(ctx.df, ctx.cat_cols, ctx.num_cols, fit_idx, idx)[1]
    assert mode == "raw_native", mode
    return (ctx.df.loc[idx, ctx.cat_cols].to_numpy(dtype="int64"), ctx.df.loc[idx, ctx.num_cols].to_numpy(dtype="float32"))


def _device(spec):
    """Device on which the model actually trains (for the record): CPU for scikit-learn models, ml.device_of for the
    boosted trees, the GPU for PyTorch and foundation models when one is available."""
    if spec.route == "ml":
        from defectshift.models.ml import device_of
        return device_of(spec.impl)
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit(spec, ctx, fit_idx, es_idx, seed, params=None, max_epochs=None):
    """Train the spec model on fit_idx (early-stopping models stop on es_idx). params=None means the fixed default
    settings. max_epochs: epoch cap for smoke tests only (MLP, FT-T, RealMLP, TabM), None = protocol value; recorded as
    smoke_max_epochs in the settings."""
    assert (es_idx is not None) == spec.uses_es, f"{spec.name}: es_idx does not match the early-stopping setting ({spec.uses_es})"
    if es_idx is not None:
        assert not set(fit_idx) & set(es_idx), "training rows and early-stopping rows overlap"
    unobserved = _unobserved(ctx, fit_idx)
    ctx = _neutralized(ctx, unobserved)
    set_seed(seed)
    t0 = time.time()
    df = ctx.df
    y, ystd = standardized_target(spec, ctx.y, fit_idx)   # the original y if the route does not standardize
    if spec.route == "ml":
        mode = protocol.ml_needs_scale(spec.impl)
        rep = protocol.ml_cat_representation(spec.impl, ctx.cat_mode)
        model, _, _, eval_tf = protocol.fit_ml(config.TASK, spec.impl, df, ctx.feature_cols, y, fit_idx, es_idx,
                                               params or {}, seed, cat_mode=ctx.cat_mode)
        cfg = dict(preprocess={"integer": {True: "train-median impute + standardize", "impute": "train-median impute",
                                           False: "none (native missing handling)"}[mode],
                               "onehot": "num: train-median impute + standardize (scale floor); cat: one-hot fitted on "
                                         "training rows, unknown -> all zeros, dummies not standardized",
                               "native": "num: none (native missing handling); cat: native categorical "
                                         "(dataset-fixed vocabulary)"}[rep],
                   cat_representation=rep,
                   estimator=type(model).__module__ + "." + type(model).__name__, get_params=model.get_params())
        for attr in ("best_iteration", "best_iteration_"):
            if hasattr(model, attr):
                try:
                    cfg[attr] = getattr(model, attr)
                except Exception:
                    pass
        cfg["resolved"] = _resolved_ml(spec.impl, model)
        used = params or {}
    elif spec.route == "transformer":
        model, used, _, _ = protocol.fit_transformer(config.TASK, spec.impl, df, ctx.cat_cols, ctx.num_cols, y,
                                                     fit_idx, es_idx, params, seed,
                                                     max_epochs=max_epochs or protocol.TRANSFORMER_MAX_EPOCHS)
        eval_tf = fit_cat_cont_transform(df, ctx.cat_cols, ctx.num_cols, fit_idx)   # same statistics as training (training rows)
        from defectshift.models.transformer import _LEGACY
        cfg = dict(preprocess="cat: label ids -> embeddings; num: train-median impute + standardize (scale floor)",
                   optimizer=f"AdamW(weight_decay={(used or {}).get('weight_decay', _LEGACY['weight_decay'])}"
                             + ("; no decay on tokenizer, normalization, bias)" if spec.impl == "FT_Transformer" else ")"),
                   max_epochs=max_epochs or protocol.TRANSFORMER_MAX_EPOCHS,
                   patience=protocol.TRANSFORMER_PATIENCE, batch_size=protocol.TRANSFORMER_BATCH_SIZE,
                   early_stopping="val MSE (standardized y) on es_idx, any decrease = improvement, stop after "
                                  "patience + 1 consecutive non-improving epochs, restore best state",
                   best_epoch=getattr(model, "_best_epoch", None),   # restored best epoch (from 1)
                   n_epochs=getattr(model, "_n_epochs", None))       # epochs trained (to see whether max_epochs was reached)
    elif spec.route == "tabular_dl":
        model, used, _, _, _, _ = protocol.fit_dl2(config.TASK, spec.impl, df, ctx.feature_cols, ctx.cat_cols, y,
                                                   fit_idx, es_idx, seed, params=params, n_epochs=max_epochs)
        cfg = dict(preprocess="num: train-median impute (pytabkit internal preprocessing after)",
                   estimator=type(model).__module__ + "." + type(model).__name__, get_params=model.get_params(),
                   early_stopping="pytabkit default on X_val=es_idx",
                   # None in get_params = package default; the applied values = defaults overridden by the non-None arguments
                   resolved={**model._get_default_params(),
                             **{k: v for k, v in model.get_params().items() if v is not None}})
        if params and "hpo_fit_params" in params:   # configuration chosen by the HPO, and the configuration applied after training
            # pytabkit merges the configuration of the HPO (random_state = search seed) into fit_params_. The training
            # seed is the estimator's random_state (= the seed of this fit), so that key is recorded under its own name.
            after = dict(getattr(model, "fit_params_", None) or {})
            cfg.update(hpo_fit_params=params["hpo_fit_params"],
                       fit_params_after={k: v for k, v in after.items() if k != "random_state"},
                       hpo_search_random_state=after.get("random_state"), training_random_state=model.random_state)
    elif spec.route == "foundation":
        from defectshift.models import foundation as fm
        Xc, cat_idx = fm.combine(_fm_input(ctx, fit_idx, ctx.fm_input, fit_idx))
        model = fm.build(spec.impl, cat_idx, seed)
        model.fit(Xc, y.loc[fit_idx].to_numpy())
        used = {}
        raw = ctx.fm_input == "raw_native"
        cfg = dict(preprocess=("cat: label ids (declared categorical); num: raw values with missing values, no external "
                               "imputation or standardization" if raw else
                               "cat: label ids (declared categorical); num: train-median impute + standardize"),
                   fm_input=ctx.fm_input, external_imputation=not raw, external_standardization=not raw,
                   missing_value_handling=fm.MISSING_HANDLING[spec.impl] if raw else "external train-median imputation",
                   # numerical columns without any observed training value are set to 0 in all rows by a separate rule
                   # (unobserved_num_cols), independent of the missing-value handling
                   context_rows=len(fit_idx), get_params=model.get_params(), version=fm.version(spec.impl),
                   # values that 'auto' in get_params resolved to after fitting
                   resolved={k: getattr(model, k) for k in ("n_estimators_", "ensemble_softmax_temperature_",
                                                            "n_train_samples_", "eval_metric_") if hasattr(model, k)},
                   # context rows and features the model actually uses, relative to the input (model-specific reduction rules)
                   effective=fm.effective_shape(spec.impl, model, len(fit_idx), Xc.shape[1]))
    else:
        raise ValueError(spec.route)
    fit_sec = time.time() - t0
    cfg.update(model=spec.name, route=spec.route, impl=spec.impl, seed=seed, n_fit=len(fit_idx), unobserved_num_cols=unobserved,
               cat_mode=ctx.cat_mode, y_standardization=ystd, smoke_max_epochs=max_epochs,
               n_es=0 if es_idx is None else len(es_idx), device=_device(spec),
               packages={p: _pkg(p) for p in ("xgboost", "scikit-learn", "torch", "pytabkit", "tabpfn", "rtdl")})
    out = Fitted(spec, model, fit_idx, es_idx, seed, used, fit_sec, _jsonable(cfg))
    if spec.route in ("ml", "transformer"):
        out._eval_tf = eval_tf      # reused at prediction time (in-memory only; a restored model builds it once on first prediction)
    return out


def _eval_transform(f, ctx):
    """Evaluation transform fitted on the training rows (ml, transformer). Uses the one built at training time, or builds
    it once for a restored model; the transform is determined by the training rows alone, so rebuilding gives the same values."""
    tf = getattr(f, "_eval_tf", None)
    if tf is None:
        if f.spec.route == "ml":   # with the categorical mode recorded at training time (records without it = integer)
            tf = protocol.ml_transform(ctx.df, ctx.feature_cols, f.spec.impl, f.fit_idx, f.config.get("cat_mode", "integer"))
        else:
            tf = fit_cat_cont_transform(ctx.df, ctx.cat_cols, ctx.num_cols, f.fit_idx)
        f._eval_tf = tf
    return tf


def predict(f, ctx, idx):
    """Predict the idx rows with a trained model (original scale). The preprocessing is refitted deterministically on
    f.fit_idx. Models trained on a standardized target are transformed back with the recorded training-row statistics
    (records without them = trained on the original scale)."""
    raw = _predict_model_scale(f, ctx, idx)
    ys = f.config.get("y_standardization")
    return raw if ys is None else raw * ys["std"] + ys["mean"]


def _predict_model_scale(f, ctx, idx):
    """Prediction on the scale the model was trained on."""
    ctx = _neutralized(ctx, f.config.get("unobserved_num_cols", []))
    df, route = ctx.df, f.spec.route
    if route == "ml":
        tf = _eval_transform(f, ctx)
        return np.asarray(f.model.predict(tf(df.loc[idx, ctx.feature_cols])), float)
    if route == "transformer":
        from defectshift.models.transformer import predict_final
        Xev = _eval_transform(f, ctx)(idx, df)
        return np.asarray(predict_final(config.TASK, f.spec.impl, f.model, Xev)[0], float)
    if route == "tabular_dl":
        from defectshift.models.tabular_dl import predict_final
        return np.asarray(predict_final(config.TASK, f.spec.impl, f.model, df.loc[idx, ctx.feature_cols])[0], float)
    if route == "foundation":   # with the input mode recorded at training time (records without it = impute_standardize)
        from defectshift.models import foundation as fm
        Xev = _fm_input(ctx, f.fit_idx, f.config.get("fm_input", "impute_standardize"), idx)
        return np.asarray(f.model.predict(fm.combine(Xev)[0]), float)
    raise ValueError(route)


# Tolerated difference between predictions before saving and after restoring (absolute, original scale; the standard
# deviation of y is about 0.06). ML models reload the same model file, so machine precision (1e-12; random forests with
# n_jobs=-1 can differ by about 3e-17 because of the summation order of parallel prediction); PyTorch and pytabkit
# recompute on the same device (1e-6); foundation models rebuild their context (1e-5, the qualification threshold).
RELOAD_TOL = dict(ml=1e-12, transformer=1e-6, tabular_dl=1e-6, foundation=1e-5)


def check_reload(spec, p_saved, p_reloaded):
    """Check that the predictions before saving and after restoring agree (length, finiteness, tolerance). Raises
    RuntimeError on a mismatch (the fit is recorded as failed), otherwise returns the maximum difference."""
    a, b = np.asarray(p_saved, float), np.asarray(p_reloaded, float)
    if a.shape != b.shape:
        raise RuntimeError(f"saved and restored predictions differ in length {a.shape} != {b.shape}")
    if not np.isfinite(b).all():
        raise RuntimeError("NaN or Inf in the predictions after restoring")
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    if d > RELOAD_TOL[spec.route]:
        raise RuntimeError(f"saved and restored predictions differ by {d:.3e} > tolerance {RELOAD_TOL[spec.route]:.0e} ({spec.route})")
    return d


# ------------------------------------------------------------------ save and restore (the selection audit predicts without retraining)
def save(f, ctx, prefix):
    """prefix.{joblib|pt} model, prefix.npz row indices, prefix.json applied settings.
    Foundation models (in-context) save no model file, only the context rows (fit_idx). Restoring rebuilds the context from
    the same context rows, seed and pretrained weights, without parameter training (regimes.load)."""
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    route = f.spec.route
    if route == "ml":
        from defectshift.models.ml import save_model
        save_model(f.model, prefix + ".joblib")
    elif route == "transformer":
        from defectshift.models.transformer import save_model
        save_model(f.spec.impl, f.model, f.params, cat_cardinalities(ctx.df, ctx.cat_cols), len(ctx.num_cols),
                   prefix + ".pt")
    elif route == "tabular_dl":
        from defectshift.models.tabular_dl import save_model
        save_model(f.model, prefix + ".joblib")
    np.savez(prefix + ".npz", fit_idx=np.asarray(f.fit_idx),
             es_idx=np.asarray([] if f.es_idx is None else f.es_idx, dtype=np.int64))
    with open(prefix + ".json", "w") as fh:
        json.dump(dict(config=f.config, params=_jsonable(f.params), fit_sec=f.fit_sec), fh, ensure_ascii=False, indent=1)


def load(spec, ctx, prefix):
    import pandas as pd
    z = np.load(prefix + ".npz")
    fit_idx = pd.Index(z["fit_idx"])
    es_idx = pd.Index(z["es_idx"]) if len(z["es_idx"]) else None
    meta = json.load(open(prefix + ".json"))
    seed = meta["config"]["seed"]
    if spec.route == "ml":
        from defectshift.models.ml import load_model
        model = load_model(prefix + ".joblib")
    elif spec.route == "transformer":
        from defectshift.models.transformer import load_model
        model, _ = load_model(prefix + ".pt")
    elif spec.route == "tabular_dl":
        from defectshift.models.tabular_dl import load_model
        model = load_model(prefix + ".joblib")
    else:   # foundation models: rebuild the context from the saved context rows and seed, without parameter training
        return fit(spec, ctx, fit_idx, es_idx, seed)
    return Fitted(spec, model, fit_idx, es_idx, seed, meta["params"], meta["fit_sec"], meta["config"])
