"""TabNet (Arik & Pfister, AAAI 2021) bridge on the main-environment side. The official code google-research/tabnet (commit
0a1b0fe, Apache-2.0) runs on TensorFlow 1 only, so models/tabnet_tf1_worker.py is called as a subprocess in a separate environment
(environment/requirements-tabnet-tf1-lock.txt; its Python is given by DEFECTSHIFT_TABNET_PYTHON). This module builds the input
arrays (training-row statistics only), maps the worker results back to the original scale and runs the hyperparameter search
(Optuna). The deviations from the official code are listed in docs/reproducing.md ("NODE and TabNet implementation notes").

Choices and their basis (original paper arXiv 1908.07442):
- Input: "categorical inputs are mapped to a single-dimensional trainable scalar with a learnable embedding and numerical columns
  are input without any preprocessing" (experiments section); numerical values are not transformed (the official encoder applies
  batch normalization to its input). Categories are integer ids with the dataset-fixed cardinality (utils.cat_cardinalities). The
  official code has no missing-value handling, so the protocol rule applies (training-row median imputation; columns with no
  observed value in the training rows are 0).
- Target: standardized with the training-row mean and standard deviation (protocol rule; Gorishniy et al. 2021: "standardization
  to regression targets for all algorithms"); predictions are mapped back to the original scale.
- Search space: the common space of the appendix "Experiment hyperparameters": Nd, Na in {8, 16, 24, 32, 64, 128}, Nsteps in
  {3..10}, gamma in {1.0, 1.2, 1.5, 2.0}, lambda_sparse in {0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1}, learning rate in {0.005, 0.01, 0.02,
  0.025}, decay rate in {0.4, 0.8, 0.9, 0.95}, decay iterations in {0.5k, 2k, 8k, 10k, 20k}, mB in {0.6, 0.7, 0.8, 0.9, 0.95,
  0.98}. In the official code feature_dim = Nd + Na and output_dim = Nd.
- Batch size B = BV = 512, fixed (appendix, Forest Cover self-supervised experiment: "For supervised fine-tuning, we use the batch
  size B=BV as the training datasets are small"); gradient clipping 2000 (GRADIENT_THRESH of the official
  experiment_covertype.py).
- Default configuration (default-hyperparameter sensitivity analysis): the values of the official experiment_covertype.py
  (feature_dim 128, output_dim 64, Nsteps 6, gamma 1.5, mB 0.7, lambda_sparse 1e-4, lr 0.02, decay 0.95/500) with the fixed batch
  size.
- Early stopping: the common protocol rule (at most 150 epochs, stop after 17 consecutive epochs without improvement, restore the
  best epoch); see the worker documentation."""
import json
import os
import subprocess
import tempfile
import time

import numpy as np

from defectshift import config
from defectshift.models import protocol
from defectshift.utils import cat_cardinalities

TABNET_DIR = os.environ.get("DEFECTSHIFT_TABNET_DIR", os.path.join(config.REPO_ROOT, "vendor", "google-research", "tabnet"))
TABNET_COMMIT = "0a1b0fe7298a58bca6dcb47112379d829bc0c0a5"
TF_PYTHON = os.environ.get("DEFECTSHIFT_TABNET_PYTHON", "")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabnet_tf1_worker.py")

FIXED = dict(batch_size=512, virtual_batch_size=512, grad_thresh=2000.0)
SPACE = dict(n_d=(8, 16, 24, 32, 64, 128), n_a=(8, 16, 24, 32, 64, 128), n_steps=tuple(range(3, 11)),
             gamma=(1.0, 1.2, 1.5, 2.0), lambda_sparse=(0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1),
             learning_rate=(0.005, 0.01, 0.02, 0.025), decay_rate=(0.4, 0.8, 0.9, 0.95),
             decay_steps=(500, 2000, 8000, 10000, 20000), batch_momentum=(0.6, 0.7, 0.8, 0.9, 0.95, 0.98))
DEFAULT = dict(n_d=64, n_a=64, n_steps=6, gamma=1.5, lambda_sparse=1e-4, learning_rate=0.02, decay_rate=0.95,
               decay_steps=500, batch_momentum=0.7)
PREPROCESS = ("num: train-median impute, no transform (official encoder applies batch norm to inputs); cat: integer ids -> "
              "1-dimensional embedding_column(categorical_column_with_identity, dataset-fixed cardinality)")


def tmp_dir():
    """Temporary folder for the subprocess input/output and checkpoints: DEFECTSHIFT_TABNET_TMP, otherwise under the repository
    cache (config.CACHE_DIR/tmp_tabnet). The system /tmp is not used by default because it is often small."""
    d = os.environ.get("DEFECTSHIFT_TABNET_TMP") or os.path.join(config.CACHE_DIR, "tmp_tabnet")
    os.makedirs(d, exist_ok=True)
    return d


def full_params(p):
    return {**FIXED, **(p or DEFAULT)}


class Prep:
    """TabNet input decided on the training rows (numerical median imputation, 0 for columns unobserved in the training rows;
    categorical integer ids as they are)."""

    def __init__(self, df, cat_cols, num_cols, fit_idx):
        self.cat_cols, self.num_cols = list(cat_cols), list(num_cols)
        Xn = df.loc[fit_idx, self.num_cols]
        self.unobserved = [c for c in self.num_cols if Xn[c].isna().all()]
        self.median = Xn.median()
        self.cards = cat_cardinalities(df, self.cat_cols)

    def arrays(self, df, idx):
        num = df.loc[idx, self.num_cols].fillna(self.median)
        if self.unobserved:
            num[self.unobserved] = 0.0
        cat = df.loc[idx, self.cat_cols]
        if cat.isna().any().any():
            raise ValueError("missing values in categorical columns")
        return num.to_numpy(dtype=np.float32), cat.to_numpy(dtype=np.int64)


def _run_worker(arrays, job, tmp):
    if not TF_PYTHON:
        raise RuntimeError("DEFECTSHIFT_TABNET_PYTHON is not set: point it to the Python of the TensorFlow 1 environment "
                           "(docs/reproducing.md, paragraph 'TabNet environment')")
    data, out = os.path.join(tmp, "data.npz"), os.path.join(tmp, "out.npz")
    np.savez(data, **arrays)
    job = dict(job, data=data, out=out, tabnet_dir=TABNET_DIR, tmp_dir=tmp)
    with open(os.path.join(tmp, "job.json"), "w") as f:
        json.dump(job, f)
    r = subprocess.run([TF_PYTHON, WORKER, os.path.join(tmp, "job.json")], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"TabNet worker failed (code {r.returncode}): {r.stderr[-1500:]}")
    z = np.load(out)
    return {k[5:]: z[k] for k in z.files}, json.load(open(os.path.join(tmp, "out.json")))


def fit_predict(df, y, cat_cols, num_cols, fit_idx, es_idx, pred_sets, params, seed, max_epochs=None, ckpt_prefix=None,
                device=None):
    """Training on fit_idx with early stopping on es_idx; predictions for pred_sets ({name: rows}) on the original scale. With
    ckpt_prefix the best-epoch weights are saved there (TF checkpoint: <prefix>.index, .data-*). Returns (prediction dict,
    record dict)."""
    t0 = time.time()
    y_fit = y.loc[fit_idx]
    mean, std = float(y_fit.mean()), float(y_fit.std(ddof=0))
    if not (np.isfinite(std) and std > 0):
        raise ValueError(f"cannot standardize y: standard deviation {std} over {len(fit_idx)} training rows")
    prep = Prep(df, cat_cols, num_cols, fit_idx)
    arrays = {}
    for tag, idx in (("fit", fit_idx), ("es", es_idx), *pred_sets.items()):
        arrays[f"num_{tag}"], arrays[f"cat_{tag}"] = prep.arrays(df, idx)
    arrays["y_fit"] = ((y_fit - mean) / std).to_numpy(np.float32)
    arrays["y_es"] = ((y.loc[es_idx] - mean) / std).to_numpy(np.float64)
    p = full_params(params)
    job = dict(pred_sets=list(pred_sets), params=p, cat_cardinalities=prep.cards, seed=int(seed),
               max_epochs=int(max_epochs or protocol.TRANSFORMER_MAX_EPOCHS), patience=protocol.TRANSFORMER_PATIENCE,
               ckpt_prefix=ckpt_prefix, device=device or "gpu")
    with tempfile.TemporaryDirectory(prefix="tabnet_", dir=tmp_dir()) as tmp:
        preds, winfo = _run_worker(arrays, job, tmp)
    wall = time.time() - t0
    # fit_sec = input preparation, subprocess, graph construction and training minus prediction time (the whole training side, as fit_sec of the other models)
    info = dict(winfo, wall_sec=wall, fit_sec=wall - sum((winfo.get("predict_sec") or {}).values()),
                y_standardization=dict(mean=mean, std=std, ddof=0, n_rows=len(fit_idx)),
                unobserved_num_cols=prep.unobserved, cat_cardinalities=prep.cards, tabnet_commit=TABNET_COMMIT,
                preprocess=PREPROCESS, params_full=p, batch_size=p["batch_size"],
                early_stopping=f"val MSE (standardized y) each epoch, stop after {protocol.TRANSFORMER_PATIENCE + 1} non-improving "
                               f"epochs, max {job['max_epochs']}, restore best")
    return {k: v * std + mean for k, v in preds.items()}, info


def predict_saved(df, y, cat_cols, num_cols, fit_idx, params, seed, ckpt_prefix, pred_sets, device=None):
    """Restore the saved weights (ckpt_prefix) without training and predict pred_sets on the original scale (save-and-reload
    check). The input transformation is decided again from fit_idx."""
    y_fit = y.loc[fit_idx]
    mean, std = float(y_fit.mean()), float(y_fit.std(ddof=0))
    prep = Prep(df, cat_cols, num_cols, fit_idx)
    arrays = {}
    for tag, idx in (("fit", fit_idx[:1]), *pred_sets.items()):
        arrays[f"num_{tag}"], arrays[f"cat_{tag}"] = prep.arrays(df, idx)
    job = dict(pred_sets=list(pred_sets), params=full_params(params), cat_cardinalities=prep.cards, seed=int(seed),
               restore=ckpt_prefix, device=device or "gpu")
    with tempfile.TemporaryDirectory(prefix="tabnet_", dir=tmp_dir()) as tmp:
        preds, _ = _run_worker(arrays, job, tmp)
    return {k: v * std + mean for k, v in preds.items()}


def suggest(trial):
    """The common search space of the paper appendix (all discrete sets -> categorical)."""
    return {k: trial.suggest_categorical(k, list(v)) for k, v in SPACE.items()}


def tune(df, y, cat_cols, num_cols, train_idx, val_idx, n_trials, seed=config.HPO_SEED, max_epochs=None, device=None):
    """Optuna TPE (sampler seed config.RANDOM_STATE, as for the other Optuna-tuned models), n_trials trials. Each trial trains on
    train_idx with early stopping on val_idx and maximizes the validation R^2 (original scale). Returns (chosen params, record
    dict)."""
    import optuna
    from defectshift import metrics
    from defectshift.models.ml import _trial_records

    def objective(trial):
        params = suggest(trial)
        preds, info = fit_predict(df, y, cat_cols, num_cols, train_idx, val_idx, {"val": val_idx}, params, seed,
                                  max_epochs=max_epochs, device=device)
        trial.set_user_attr("n_epochs", info["n_epochs"])
        trial.set_user_attr("best_epoch", info["best_epoch"])
        return metrics.regression(y.loc[val_idx], preds["val"])["r2"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError, FloatingPointError, ValueError))
    trials = _trial_records(study)
    if not any(t["state"] == "COMPLETE" for t in trials):
        raise RuntimeError("TabNet search: every trial failed")
    return dict(study.best_params), dict(best_score=study.best_value, trials=trials,
                                         sampler=f"optuna.TPESampler(seed={config.RANDOM_STATE})", space={k: list(v) for k, v in SPACE.items()},
                                         fixed=FIXED)
