"""
RealMLP and TabM of pytabkit: scikit-learn-style estimators with their own training loops (regression only).

  - Training with fixed settings: RealMLP_TD_Regressor, TabM_D_Regressor (the authors' "tuned defaults").
  - Benchmark HPO: RealMLP_HPO_Regressor and TabM_HPO_Regressor (the official pytabkit random search), then retraining
    with the one chosen configuration (section below).

Input convention: one pandas DataFrame with mixed cat_ and num_ columns (feature_cols) and a categorical indicator
(cat_indicator).
Missing values: pytabkit rejects NaN in numerical columns ("NaN values in continuous columns are currently not
allowed!"), so they are filled with the training-row medians, which are attached to the model object and reused at
prediction time.

GPU: used when CUDA is available (PyTorch).

CUBLAS: CUBLAS_WORKSPACE_CONFIG takes effect only if it is set before PyTorch creates its CUDA context, so it is placed in
os.environ when this module is imported (before any other PyTorch-related import). In the installed pytabkit only the TabR
interface enables torch.use_deterministic_algorithms; TabM fixes its seeds (torch, numpy, random) but does not force
deterministic algorithms. The retraining reproducibility of TabM is therefore stated only as "identical predictions when
retrained with the same seed on the same GPU" (scripts/checks/check_gpu_models.py).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor

from defectshift.utils import log, free_memory

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The constructors take device= and random_state=; training is fit(X, y, X_val=, y_val=, cat_indicator=).
_PYTABKIT_MODELS = {"RealMLP": RealMLP_TD_Regressor, "TabM": TabM_D_Regressor}

MODEL_NAMES = list(_PYTABKIT_MODELS)


def build_one_model(task: str, name: str, verbose: bool = False, random_state: int = RANDOM_STATE, params: dict = None,
                    n_epochs=None):
    """One untrained estimator. With a configuration chosen by the HPO in params (hpo_fit_params), exactly that
    configuration; otherwise the package's tuned defaults. n_epochs: epoch cap for smoke tests only (None = package value)."""
    assert task == "reg", "regression only"
    if params and "hpo_fit_params" in params:   # exactly the configuration chosen by the HPO
        return build_fixed_pytabkit(name, params["hpo_fit_params"], params["hpo_n_steps"], random_state, verbose, n_epochs)
    return _PYTABKIT_MODELS[name](device=DEVICE, random_state=random_state,  # the model seed is passed through
                                  verbosity=(1 if verbose else 0), **({} if n_epochs is None else {"n_epochs": n_epochs}))


def fit_with_val(task: str, name: str, model, Xtr: pd.DataFrame, ytr, Xval: pd.DataFrame, yval, cat_cols: list):
    """Xtr/Xval are DataFrames with mixed cat_ and num_ columns. Missing numerical values are filled with the
    training-row medians (attached to the model for reuse at prediction time), and cat_cols marks the categorical
    columns. Early stopping = pytabkit default (on X_val)."""
    med = Xtr.median(numeric_only=True)
    Xtr, Xval = Xtr.fillna(med), Xval.fillna(med)
    cat_indicator = [c in cat_cols for c in Xtr.columns]
    model.fit(Xtr, np.asarray(ytr, dtype="float32"), X_val=Xval, y_val=np.asarray(yval, dtype="float32"),
              cat_indicator=cat_indicator)
    model._dl2_impute_median = med
    return model


def _prepare_X(model, X: pd.DataFrame):
    """The missing-value handling of training (training-row medians) at prediction time."""
    med = getattr(model, "_dl2_impute_median", None)
    return X.fillna(med) if med is not None else X


def _flatten(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return arr.reshape(-1) if arr.ndim > 1 and arr.shape[1] == 1 else arr


def predict_scores(name: str, model, X: pd.DataFrame):
    """Raw regression predictions."""
    return _flatten(model.predict(_prepare_X(model, X)))


def predict_final(task: str, name: str, model, X: pd.DataFrame):
    """(pred, pred), the same return shape as predict_final of the other model families."""
    assert task == "reg", "regression only"
    pred = predict_scores(name, model, X)
    return pred, pred


def fit_model(task: str, name: str, Xtr, ytr, Xval, yval, cat_cols: list,
              verbose: bool = False, random_state: int = RANDOM_STATE, params: dict = None, n_epochs=None):
    """One training run. Returns (model, params, val_score (R^2), elapsed); params as received (empty dict if none).
    n_epochs: epoch cap for smoke tests only (None = package value)."""
    try:
        model = build_one_model(task, name, verbose=verbose, random_state=random_state, params=params,  # seed and HPO configuration
                                n_epochs=n_epochs)
        t0 = time.time()
        fit_with_val(task, name, model, Xtr, ytr, Xval, yval, cat_cols)
        elapsed = time.time() - t0
        score = r2_score(yval, predict_scores(name, model, Xval))
        return model, (params or {}), score, elapsed
    finally:
        free_memory()


# ---------------------------------------------------------------- official pytabkit HPO (RealMLP, TabM)
# The HPO estimator trains n random configurations (RandomParams*AlgInterface, seed = combine_seeds(split_seed,
# model_idx)), keeps the one with the lowest validation loss and stores fit_params_ = {best_alg_idx, sub_fit_params (the
# full chosen configuration: defaults + sample + overrides)}. To retrain the chosen configuration on other rows, the full
# configuration is passed as fit_params to the same RandomParams*AlgInterface of the same HPO class, which then skips
# sampling and uses it as is. Passing only the sampled values to the fixed-setting models (*_TD / *_D) is avoided: the
# unsampled settings would revert to the TD defaults and differ from those of the HPO.
def _hpo_classes():
    from pytabkit import RealMLP_HPO_Regressor, TabM_HPO_Regressor
    from pytabkit.models.alg_interfaces.nn_interfaces import RandomParamsNNAlgInterface
    from pytabkit.models.alg_interfaces.tabm_interface import RandomParamsTabMAlgInterface
    return {"RealMLP": (RealMLP_HPO_Regressor, RandomParamsNNAlgInterface),
            "TabM": (TabM_HPO_Regressor, RandomParamsTabMAlgInterface)}


class _FixedParamsFactory:
    """Replaces _create_alg_interface of the HPO estimator and builds a RandomParams*AlgInterface that trains the one
    chosen configuration. As in the HPO, the config comes from the estimator's get_config() at training time. A
    module-level class, so that the estimator can be saved with joblib."""

    def __init__(self, name, fit_params, est):
        self.name, self.fit_params, self.est = name, fit_params, est

    def __call__(self, n_cv):
        import copy
        iface = _hpo_classes()[self.name][1]
        return iface(model_idx=0, fit_params=[copy.deepcopy(self.fit_params)], **self.est.get_config())


def _strip_fit_params(sub_fit_params):
    """Keep only the training configuration from sub_fit_params of the HPO result, without the entries added after
    training (sub_fit_params: early-stopping epochs and so on)."""
    return {k: v for k, v in sub_fit_params.items() if k != "sub_fit_params"}


def build_fixed_pytabkit(name, fit_params, n_steps, random_state, verbose=False, n_epochs=None):
    """Build with the class and defaults of the HPO and train only the one chosen configuration. n_epochs: epoch cap for
    smoke tests only (passed to the constructor; an epoch count inside the chosen configuration is capped as well)."""
    hpo_cls = _hpo_classes()[name][0]
    est = hpo_cls(n_hyperopt_steps=n_steps, device=DEVICE, random_state=random_state, verbosity=(1 if verbose else 0),
                  **({} if n_epochs is None else {"n_epochs": n_epochs}))
    fp = _strip_fit_params(fit_params)
    if n_epochs is not None and "n_epochs" in fp:
        fp = dict(fp, n_epochs=n_epochs)
    est._create_alg_interface = _FixedParamsFactory(name, fp, est)
    return est


def hpo_pytabkit(name, Xtr, ytr, Xval, yval, cat_cols, n_steps, random_state, verbose=False, n_epochs=None):
    """Official pytabkit HPO (n_steps random configurations, validation = Xval). Preprocessing as in fit_with_val
    (training-row median imputation). Returns dict(fit_params = the full chosen configuration, best_alg_idx, val_r2, seconds)."""
    hpo_cls = _hpo_classes()[name][0]
    med = Xtr.median(numeric_only=True)
    Xtr_f, Xval_f = Xtr.fillna(med), Xval.fillna(med)
    # n_epochs: epoch cap for smoke tests only (None = pytabkit default)
    est = hpo_cls(n_hyperopt_steps=n_steps, device=DEVICE, random_state=random_state, verbosity=(1 if verbose else 0),
                  **({} if n_epochs is None else {"n_epochs": n_epochs}))
    t0 = time.time()
    est.fit(Xtr_f, np.asarray(ytr, dtype="float32"), X_val=Xval_f, y_val=np.asarray(yval, dtype="float32"),
            cat_indicator=[c in cat_cols for c in Xtr.columns])
    seconds = time.time() - t0
    fp = est.fit_params_
    val_r2 = r2_score(yval, _flatten(est.predict(Xval_f)))
    out = dict(fit_params=fp["sub_fit_params"], best_alg_idx=int(fp["best_alg_idx"]), val_r2=float(val_r2),
               seconds=seconds, val_pred=_flatten(est.predict(Xval_f)))
    del est
    free_memory()
    return out


def save_model(model, path: str):
    """Small models without pretrained weights: the whole object is saved with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log(f"[Model saved] -> {path}")


def load_model(path: str):
    return joblib.load(path)
