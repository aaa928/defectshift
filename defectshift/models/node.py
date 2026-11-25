"""NODE (Popov, Morozov & Babenko, ICLR 2020): training and prediction with the lib of the official code Qwicen/node (commit
3bae6a8, MIT), imported as is.

Official code location: environment variable DEFECTSHIFT_NODE_DIR (default vendor/node in the repository). It needs
category_encoders (imported by lib/__init__), tensorboardX (trainer.py) and qhoptim (QHAdam). The deviations from the official
examples are listed in docs/reproducing.md ("NODE and TabNet implementation notes").

Choices and their basis (original paper arXiv 1909.06312; official notebooks/year_node_shallow.ipynb and lib/data.py):
- Input: the quantile transform of the official lib/data.py ("we transform each data feature to follow a normal distribution via
  quantile transform", section 3.3): QuantileTransformer (output 'normal', random_state = seed) fitted on the training rows with
  added Gaussian noise of scale quantile_noise / max(std, quantile_noise) (quantile_noise = 1e-3; the formula of the official code
  and of the RTDL code of Gorishniy et al.), applied without noise to every row. The anonymization scales the numerical columns to
  [0, 1], so the features are first standardized with the training-row mean and standard deviation (ddof 0; the normalize=True
  option of the official Dataset, which the example notebooks leave off) and the noise does not depend on the anonymized scale.
  NORMALIZE = False reproduces the example default (used by the preprocessing sensitivity analysis). Order: imputation and
  categorical encoding -> standardization (training-row statistics) -> noise (fit only) -> quantile transform. Columns with
  standard deviation 0 are divided by 1.
- Categorical: category_encoders.LeaveOneOutEncoder fitted on the training rows (X, y) and applied without y, as in the official
  lib/data.py.
- Missing values: the official code has no handling, so the protocol rule applies first (training-row median imputation; columns
  with no observed value in the training rows are 0).
- Target: standardized with the training-row mean and standard deviation, as in the official regression notebook; predictions are
  mapped back to the original scale.
- Architecture and optimization: the official notebook's DenseBlock(..., choice_function=entmax15, bin_function=entmoid15) with
  QHAdam (nus (0.7, 1.0), betas (0.95, 0.998)), learning rate 1e-3 (paper, appendix A.2.3), batch size 512 as in the official
  regression notebook (256 for the four grid configurations that exceed the GPU memory, REDUCED_BATCH). The data-aware
  initialization uses the first 1000 rows after shuffling the training rows with a fixed seed (paper, section 3.3: "thresholds b
  are initialized with random feature values f_i(x) observed in the first data batch").
- Validation and early stopping (common protocol rule; Gorishniy et al. 2021 also use a common patience rule for NODE): the
  validation loop of the official notebook (average of the last 5 checkpoints, 'best_mse' restored at the end) runs once per epoch
  instead of every 100 steps, for at most protocol.TRANSFORMER_MAX_EPOCHS (150) epochs, stopping after
  protocol.TRANSFORMER_PATIENCE + 1 (17) consecutive epochs without improvement (the official notebook stops after 5000 steps
  without improvement).
- Tuning grid: paper appendix A.2.3, num layers {2, 4, 8} x total tree count {1024, 2048} x depth {6, 8} x tree output dim {2, 3}
  = 24, plus the official notebook default (1 layer, 2048, depth 6, output 3) = 25, all evaluated (Gorishniy F.6); layer_dim =
  total tree count / num layers."""
import errno
import json
import os
import random
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np

from defectshift.config import REPO_ROOT
from defectshift.models import protocol

NODE_DIR = os.environ.get("DEFECTSHIFT_NODE_DIR", os.path.join(REPO_ROOT, "vendor", "node"))
NODE_COMMIT = "3bae6a8a63f0205683270b6d566d9cfa659403e4"

DEFAULT = dict(num_layers=1, total_trees=2048, depth=6, tree_dim=3)
GRID = [dict(num_layers=nl, total_trees=t, depth=d, tree_dim=o)
        for nl in (2, 4, 8) for t in (1024, 2048) for d in (6, 8) for o in (2, 3)]
CONFIGS = [DEFAULT] + GRID            # 25 configurations; tuning evaluates all of them (no Optuna sampling)
NORMALIZE = True     # standardize before the quantile transform (official Dataset normalize=True). False = official example default
PREPROCESS = ("num: train-median impute; cat: LeaveOneOutEncoder fitted on training rows; all: standardize with training-row mean/std "
              "(official Dataset normalize=True; std 0 -> 1), then QuantileTransformer (normal; Gaussian noise with std "
              "1e-3/max(col std, 1e-3) on the fit rows only) fitted on training rows")
PREPROCESS_A = ("num: train-median impute; cat: LeaveOneOutEncoder fitted on training rows; all: QuantileTransformer (normal; Gaussian "
                "noise with std 1e-3/max(col std, 1e-3) on the fit rows only) fitted on training rows; no standardization (official example default)")
# Batch size: 512 by default (official regression notebook). The 4 grid configurations that exceeded the training memory at 512 on
# the benchmark GPU (NVIDIA L4 24 GB, 22.05 GiB available to PyTorch) use half of it, 256. This applies the memory principle of the
# original paper per configuration (appendix A.2.3: "For each dataset, we use the maximal batch size, which fits in the GPU memory.")
# without searching for the largest possible batch (intermediate sizes such as 384 were not tried). The list is fixed (no automatic
# reduction), so that another GPU or temporary memory use cannot change the setting silently; if memory still runs out with the
# fixed list, tune_grid stops instead of recording a failed trial.
# The 256 configurations can still run out of memory in consecutive runs through allocator fragmentation, so the CUDA allocator
# option expandable_segments is enabled (set by the runner 06a_benchmark_node_tabnet; predictions were identical with the option
# on and off for configurations 1 and 15, and configuration 8 could not be compared because it ran out of memory with the option
# off). With the option on, 2x2048 depth 8 output 3 peaked at 19.34 GiB allocated and 21.74 GiB reserved over 150 epochs, and the
# 25 configurations in one process peaked at 19.68 GiB allocated (22.05 GiB available).
BATCH_SIZE = 512
REDUCED_BATCH = {(2, 2048, 8, 2): 256, (2, 2048, 8, 3): 256, (4, 2048, 8, 2): 256, (4, 2048, 8, 3): 256}
CFG_KEYS = ("num_layers", "total_trees", "depth", "tree_dim")
EVAL_BATCH_SIZE = 1024
INIT_ROWS = 1000
QHADAM_PARAMS = dict(nus=(0.7, 1.0), betas=(0.95, 0.998))
N_LAST_CHECKPOINTS = 5
QUANTILE_NOISE = 1e-3
_TMP_CHECKED = False



def batch_size_for(cfg):
    """Training batch size of a grid (CONFIGS) configuration. Stops for a configuration outside the grid (tests and smoke runs
    pass batch_size to train/fit_predict directly)."""
    key = tuple(int(cfg[k]) for k in CFG_KEYS)
    if key not in {tuple(c[k] for k in CFG_KEYS) for c in CONFIGS}:
        raise ValueError(f"NODE configuration outside the grid {dict(zip(CFG_KEYS, key))}: no batch size defined (pass batch_size)")
    return REDUCED_BATCH.get(key, BATCH_SIZE)


def _release_cache(torch):
    """Return unused cache blocks only (computation unchanged), so that the large blocks left by initialization (1000 rows) and
    validation (1024-row batches) do not fragment the allocation of training batches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lib():
    if NODE_DIR not in sys.path:
        sys.path.insert(0, NODE_DIR)
    import lib   # noqa: E402  the official package name as is
    return lib


class Prep:
    """NODE input transformation fitted on the training rows (imputation -> categorical LeaveOneOut -> [standardization] ->
    quantile transform). transform(df.loc[idx]) -> float32 matrix. normalize: None means the module constant NORMALIZE. The
    standardization statistics come from the training rows only and are reused for prediction rows."""

    def __init__(self, df, cat_cols, num_cols, fit_idx, y_fit, seed, normalize=None):
        from category_encoders import LeaveOneOutEncoder
        from sklearn.preprocessing import QuantileTransformer
        self.cat_cols, self.num_cols = list(cat_cols), list(num_cols)
        Xn = df.loc[fit_idx, self.num_cols]
        self.unobserved = [c for c in self.num_cols if Xn[c].isna().all()]
        self.median = Xn.median()
        self.loo = (LeaveOneOutEncoder(cols=self.cat_cols).fit(df.loc[fit_idx, self.cat_cols].astype("int64"), np.asarray(y_fit))
                    if self.cat_cols else None)
        X = self._raw(df.loc[fit_idx])
        self.normalize = NORMALIZE if normalize is None else bool(normalize)
        if self.normalize:                                # normalize of the official lib/data.py: (X - mean) / std, training-row statistics
            self.mu = X.mean(axis=0)
            sd = X.std(axis=0)
            self.n_const = int((sd == 0).sum())
            self.sd = np.where(sd == 0, 1.0, sd)          # constant columns: divide by 1 instead of 0 (they become 0 on the training rows)
            X = (X - self.mu) / self.sd
        quantile_train = np.copy(X)                       # quantile transform of the official lib/data.py (noise for fitting only)
        rng = np.random.RandomState(seed)
        stds = np.std(quantile_train, axis=0, keepdims=True)
        noise_std = QUANTILE_NOISE / np.maximum(stds, QUANTILE_NOISE)
        quantile_train += noise_std * rng.randn(*quantile_train.shape)
        self.qt = QuantileTransformer(random_state=seed, output_distribution="normal").fit(quantile_train)

    def _raw(self, part):
        num = part[self.num_cols].fillna(self.median)
        if self.unobserved:
            num[self.unobserved] = 0.0
        cols = [num.to_numpy(dtype="float64")]
        if self.loo is not None:
            cols.append(self.loo.transform(part[self.cat_cols].astype("int64")).to_numpy(dtype="float64"))
        return np.concatenate(cols, axis=1)

    def transform(self, part):
        X = self._raw(part)
        if self.normalize:
            X = (X - self.mu) / self.sd
        return self.qt.transform(X).astype(np.float32)


def _build(lib, in_features, cfg, device):
    import torch.nn as nn
    layer_dim = cfg["total_trees"] // cfg["num_layers"]
    return nn.Sequential(
        lib.DenseBlock(in_features, layer_dim, num_layers=cfg["num_layers"], tree_dim=cfg["tree_dim"], depth=cfg["depth"],
                       flatten_output=False, choice_function=lib.entmax15, bin_function=lib.entmoid15),
        lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),   # official notebook: average first channels of every tree
    ).to(device)


def train(X_tr, y_tr, X_es, y_es, cfg, seed, max_epochs=None, device=None, batch_size=None):
    """Training on standardized y. Returns (model with the best weights loaded, record dict). Checkpoints are written to a
    temporary folder and removed. batch_size: if absent, batch_size_for(cfg) (grid configurations only)."""
    import torch
    import torch.nn.functional as F
    from qhoptim.pyt import QHAdam
    lib = _lib()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    max_epochs = max_epochs or protocol.TRANSFORMER_MAX_EPOCHS
    batch_size = batch_size or batch_size_for(cfg)
    model = _build(lib, X_tr.shape[1], cfg, device)
    init_rows = np.random.RandomState(seed).permutation(len(X_tr))[:INIT_ROWS]   # first batch = shuffled training rows (module documentation)
    with torch.no_grad():                               # official notebook: trigger data-aware init
        model(torch.as_tensor(X_tr[init_rows], device=device))
    _release_cache(torch)
    work = tempfile.mkdtemp(prefix="node_", dir=tmp_dir())
    try:
        # Trainer uses os.path.join('logs/', experiment_name), so an absolute path is used as is
        trainer = lib.Trainer(model=model, loss_function=F.mse_loss, experiment_name=os.path.join(work, "exp"), warm_start=False,
                              Optimizer=QHAdam, optimizer_params=QHADAM_PARAMS, verbose=False, n_last_checkpoints=N_LAST_CHECKPOINTS)
        best_mse, best_epoch, epoch, t0 = float("inf"), 0, 0, time.time()
        y_tr32, y_es32 = np.asarray(y_tr, np.float32), np.asarray(y_es, np.float32)
        for epoch in range(1, max_epochs + 1):
            for batch in lib.iterate_minibatches(X_tr, y_tr32, batch_size=batch_size, shuffle=True, epochs=1):
                trainer.train_on_batch(*batch, device=device)
            trainer.save_checkpoint()
            trainer.average_checkpoints(out_tag="avg")   # official code: reads the last 5 checkpoints to the GPU and averages them (hundreds of MiB per configuration)
            trainer.load_checkpoint(tag="avg")
            _release_cache(torch)                        # return the blocks left by averaging and loading (computation unchanged; not a fix for out-of-memory, see the allocator option)
            mse = trainer.evaluate_mse(X_es, y_es32, device=device, batch_size=EVAL_BATCH_SIZE)
            _release_cache(torch)
            if mse < best_mse:
                best_mse, best_epoch = float(mse), epoch
                trainer.save_checkpoint(tag="best_mse")
            trainer.load_checkpoint()                    # last
            trainer.remove_old_temp_checkpoints()
            _release_cache(torch)                        # also before the next epoch (computation unchanged)
            if epoch - best_epoch > protocol.TRANSFORMER_PATIENCE:   # patience + 1 consecutive epochs without improvement
                break
        if best_epoch == 0:   # the validation loss was never finite (NaN, inf) -> no best checkpoint; a failed configuration instead of a missing-file error
            raise ValueError(f"NODE: no finite validation loss in {epoch} epochs (config {dict(cfg)})")
        trainer.load_checkpoint(tag="best_mse")
        trainer.writer.close()
        info = dict(best_epoch=best_epoch, n_epochs=epoch, best_val_mse_std=best_mse, train_sec=time.time() - t0,
                    layer_dim=cfg["total_trees"] // cfg["num_layers"], device=device, batch_size=batch_size)
        return trainer.model, info
    finally:
        shutil.rmtree(work, ignore_errors=True)


def predict(model, X, device=None):
    import torch
    lib = _lib()
    device = device or next(model.parameters()).device
    model.train(False)
    with torch.no_grad():
        out = lib.process_in_chunks(model, torch.as_tensor(X, device=device), batch_size=EVAL_BATCH_SIZE)
    return lib.check_numpy(out).astype(np.float64)


def _y_stats(y, fit_idx):
    y_fit = y.loc[fit_idx]
    mean, std = float(y_fit.mean()), float(y_fit.std(ddof=0))
    if not (np.isfinite(std) and std > 0):
        raise ValueError(f"cannot standardize y: standard deviation {std} over {len(fit_idx)} training rows")
    return y_fit, mean, std


def fit_predict(df, y, cat_cols, num_cols, fit_idx, es_idx, pred_sets, cfg, seed, max_epochs=None, save_path=None, batch_size=None):
    """NODE trained on fit_idx with early stopping on es_idx; predictions for pred_sets ({name: rows}) on the original scale. With
    save_path the restored best weights (state_dict, including the data-aware initialization flag) and the architecture
    configuration are saved. Returns (prediction dict, record dict)."""
    import torch
    t0 = time.time()
    y_fit, mean, std = _y_stats(y, fit_idx)
    prep = Prep(df, cat_cols, num_cols, fit_idx, y_fit, seed)
    X_tr, X_es = prep.transform(df.loc[fit_idx]), prep.transform(df.loc[es_idx])
    model, info = train(X_tr, (y_fit - mean) / std, X_es, (y.loc[es_idx] - mean) / std, cfg, seed, max_epochs=max_epochs,
                        batch_size=batch_size)
    fit_sec = time.time() - t0                         # fitting the input transformation + training, as fit_sec of the other models
    preds, predict_sec = {}, {}
    for k, idx in pred_sets.items():                   # prediction time (including the input transformation) per evaluation set
        t1 = time.time()
        preds[k] = predict(model, prep.transform(df.loc[idx])) * std + mean
        predict_sec[k] = time.time() - t1
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), config=dict(cfg), in_features=int(X_tr.shape[1]), node_commit=NODE_COMMIT,
                        normalize=prep.normalize), save_path)
    info.update(config=dict(cfg), y_standardization=dict(mean=mean, std=std, ddof=0, n_rows=len(fit_idx)),
                unobserved_num_cols=prep.unobserved, n_features_in=int(X_tr.shape[1]), node_commit=NODE_COMMIT,
                preprocess=PREPROCESS if prep.normalize else PREPROCESS_A, normalize=prep.normalize,
                n_const_standardized=getattr(prep, "n_const", None), fit_sec=fit_sec, predict_sec=predict_sec, init_rows=f"{INIT_ROWS} shuffled training rows (seed)",
                optimizer="QHAdam(nus=(0.7, 1.0), betas=(0.95, 0.998)), lr 1e-3 (default)",
                checkpoint_averaging=f"average of the last {N_LAST_CHECKPOINTS} epoch checkpoints",
                early_stopping=f"val MSE of the average of the last {N_LAST_CHECKPOINTS} epoch checkpoints each epoch, stop after "
                               f"{protocol.TRANSFORMER_PATIENCE + 1} non-improving epochs, max {max_epochs or protocol.TRANSFORMER_MAX_EPOCHS}")
    del model
    return preds, info


def predict_saved(df, y, cat_cols, num_cols, fit_idx, seed, save_path, pred_sets, device=None):
    """Restore the saved weights without training and predict pred_sets on the original scale (save-and-reload check). The input
    transformation is fitted again from fit_idx and seed (deterministic)."""
    import torch
    lib = _lib()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    y_fit, mean, std = _y_stats(y, fit_idx)
    ck = torch.load(save_path, map_location=device, weights_only=True)
    prep = Prep(df, cat_cols, num_cols, fit_idx, y_fit, seed, normalize=ck.get("normalize", False))   # the preprocessing used when saving (no standardization if not recorded)
    model = _build(lib, ck["in_features"], ck["config"], device)
    model.load_state_dict(ck["state_dict"])     # also restores the initialization flag (_is_initialized_tensor = 1) -> no second data-aware initialization
    return {k: predict(model, prep.transform(df.loc[idx])) * std + mean for k, idx in pred_sets.items()}


def is_oom(e):
    """Whether the error is GPU out-of-memory: torch OutOfMemoryError, and cuBLAS or CUDA allocation failures raised as
    RuntimeError (decided by the message)."""
    import torch
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e)
    return isinstance(e, RuntimeError) and ("out of memory" in msg.lower() or "CUBLAS_STATUS_ALLOC_FAILED" in msg)


DISK_ERRNOS = (errno.ENOSPC, errno.EDQUOT, errno.EIO)


def is_fatal(e):
    """Errors that stop the grid search instead of being recorded as failed trials: out-of-memory (is_oom) and disk errors. An
    OSError counts only when its errno is no space, quota exceeded or I/O error (DISK_ERRNOS); write failures that torch.save
    raises as RuntimeError without errno are recognized by the message. Other OSErrors (missing file, permission, ...) stay failed
    trials (the original exception is recorded). A wrapped cause is checked through __cause__."""
    for x in (e, getattr(e, "__cause__", None)):
        if x is None:
            continue
        if is_oom(x) or (isinstance(x, OSError) and x.errno in DISK_ERRNOS):
            return True
        msg = str(x)
        if isinstance(x, RuntimeError) and any(s in msg for s in ("No space left on device", "PytorchStreamWriter failed", "file write failed",
                                                                 "unexpected pos")):   # secondary errors of a full disk
            return True
    return False


def tmp_dir():
    """Temporary folder for checkpoints: DEFECTSHIFT_NODE_TMP, otherwise under the repository cache (config.CACHE_DIR/tmp_node). The
    system /tmp is not used by default because it is often small."""
    global _TMP_CHECKED
    from defectshift import config
    d = os.environ.get("DEFECTSHIFT_NODE_TMP") or os.path.join(config.CACHE_DIR, "tmp_node")
    os.makedirs(d, exist_ok=True)
    if not _TMP_CHECKED:                              # notice of training folders left by a forced stop (several GiB each for 8-layer configurations), once per process
        _TMP_CHECKED = True
        old = [x for x in os.listdir(d) if x.startswith("node_")]
        if old:
            warnings.warn(f"{len(old)} earlier training folders remain in the NODE temporary folder {d} (forced stop); they can be removed if no other run is active")
    return d


def _progress_header(train_idx, val_idx, seed, max_epochs, configs):
    """Values that decide whether a grid progress record belongs to the same search (hashes of the search rows, seed, epoch cap,
    grid, batch list, versions of this file and of the official code)."""
    import hashlib
    from defectshift import hpo
    code = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    h = dict(train_idx_sha256=hpo._idx_sha(train_idx), val_idx_sha256=hpo._idx_sha(val_idx), seed=int(seed), max_epochs=max_epochs,
             configs=[dict(c) for c in configs], reduced_batch=sorted(list(k) for k in REDUCED_BATCH), normalize=NORMALIZE, code_sha256=code,
             node_commit=NODE_COMMIT)
    return json.loads(json.dumps(h))


def tune_grid(df, y, cat_cols, num_cols, train_idx, val_idx, seed, max_epochs=None, configs=None, progress_path=None):
    """Train every grid configuration (the 25 CONFIGS) on train_idx with early stopping on val_idx and choose the one with the
    best validation R^2 (original scale). configs: partial grid for smoke runs only. progress_path: progress record with one line
    per configuration (JSON lines). For the same search (matching header line) finished configurations (successful trials only)
    are not trained again and their results are reused, so that finished configurations are not lost after an interruption or
    an out-of-memory stop. Failed trials are trained again when resuming. A resumed configuration is trained with the same seed
    and rows as in a single run (if GPU repetitions vary, the values are not guaranteed to be identical). A different header
    line stops the search. Returns (chosen configuration, record dict)."""
    from defectshift import metrics
    configs = CONFIGS if configs is None else configs
    trials, best, resumed = [], None, {}
    if progress_path:
        header = _progress_header(train_idx, val_idx, seed, max_epochs, configs)
        if os.path.exists(progress_path):
            with open(progress_path) as fh:
                raw = [s for s in fh.read().split("\n") if s.strip()]
            lines = []
            for j, s in enumerate(raw):
                try:
                    lines.append(json.loads(s))
                except json.JSONDecodeError:
                    if j != len(raw) - 1:
                        raise RuntimeError(f"line {j + 1} of the NODE grid progress record {progress_path} is broken (not the last line); check and remove it")
                    warnings.warn(f"the last line of the NODE grid progress record {progress_path} was truncated and is dropped (forced stop)")
                    with open(progress_path, "w") as fh:     # rewrite the intact lines only, so that the next line is not appended to the truncated one
                        fh.write("".join(json.dumps(x) + "\n" for x in lines))
                        fh.flush()
                        os.fsync(fh.fileno())
            if not lines or lines[0].get("header") != header:
                raise RuntimeError(f"the NODE grid progress record {progress_path} belongs to a different search (rows, seed, epoch cap, grid, code version); check and remove it")
            resumed = {t["number"]: t for t in lines[1:] if t["state"] == "COMPLETE"}   # failed trials are trained again
        else:
            os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
            with open(progress_path, "w") as fh:
                fh.write(json.dumps(dict(header=header)) + "\n")
    for n, cfg in enumerate(configs):
        if n in resumed:
            trial = resumed[n]
            assert trial["params"] == json.loads(json.dumps(dict(cfg))), f"configuration {n} of the progress record differs from the grid"
        else:
            t0 = time.time()
            try:
                preds, info = fit_predict(df, y, cat_cols, num_cols, train_idx, val_idx, {"val": val_idx}, cfg, seed,
                                          max_epochs=max_epochs)
                r2 = float(metrics.regression(y.loc[val_idx], preds["val"])["r2"])
                trial = dict(number=n, state="COMPLETE", value=r2, seconds=time.time() - t0, params=dict(cfg),
                             user_attrs=dict(n_epochs=info["n_epochs"], best_epoch=info["best_epoch"], batch_size=info["batch_size"]))
            except Exception as e:
                if is_fatal(e):
                    raise        # out-of-memory (fixed batch list) and disk errors are unrelated to the configuration -> stop instead of a failed trial
                # any other failure of one configuration is recorded and the search continues (as in the searches of the other models)
                trial = dict(number=n, state="FAIL", value=None, seconds=time.time() - t0, params=dict(cfg),
                             error=f"{type(e).__name__}: {e}"[:300])
            if progress_path:
                with open(progress_path, "a") as fh:
                    fh.write(json.dumps(trial) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            free_cuda()
        trials.append(trial)
        if trial["state"] == "COMPLETE" and (best is None or trial["value"] > best[1]):
            best = (dict(cfg), trial["value"])
    if best is None:
        raise RuntimeError("NODE grid: every configuration failed")
    return best[0], dict(best_score=best[1], trials=trials, sampler="full grid (original paper A.2.3 grid + official default)",
                         grid=[dict(c) for c in configs], n_trials_resumed=len(resumed),
                         trials_sec_total=float(sum(t["seconds"] for t in trials)))   # hpo_sec of a resumed search covers the last call only


def free_cuda():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
