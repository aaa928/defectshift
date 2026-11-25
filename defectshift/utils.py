"""Shared helpers for logging, seeding and freeing memory, and the preprocessing used by several models (median
imputation and standardization fitted on the training rows, separation of categorical and numerical inputs).
Features are selected by their `num_` and `cat_` prefixes, not by hard-coded column counts.
"""
import gc
import os
import random
import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from defectshift import config

# Some consoles use an encoding that cannot represent every Unicode character in the log messages; switch to UTF-8 where
# the stream supports it and ignore streams that do not.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def log(msg: str, level: str = "INFO"):
    """Log line with a timestamp and a level tag (INFO by default)."""
    level = level.upper()
    if level not in LOG_LEVELS:
        level = "INFO"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}", flush=True)


def log_exception(context_msg: str):
    """Call inside an except block: log the message and the traceback at ERROR level, so that a long run can record
    the cause of a failed item and continue with the next one."""
    log(f"{context_msg}\n{traceback.format_exc()}", level="ERROR")


def set_seed(seed: int):
    """Fix the global random state of Python's random module, NumPy, and PyTorch on CPU and CUDA.

    Called immediately before each fit, so that a result does not depend on how much randomness earlier fits in the
    same process consumed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def free_memory():
    """Call at the end of a loop iteration (domain or model): Python garbage collection and, if PyTorch is installed,
    release of the CUDA cache, to limit memory fragmentation over long runs. The import is deferred and failures are
    ignored, so the function is safe in scripts without PyTorch."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def fit_feature_transform(df, feature_cols, needs_impute_scale, train_idx):
    """Fit median imputation and StandardScaler on the training rows (train_idx) and return a reusable transform
    (DataFrame -> DataFrame), kept separate from the model so that later rows (A_heldout, B_te) use the same transform.
    Tree models (needs_impute_scale=False) handle missing values natively, so the transform is the identity; with
    needs_impute_scale="impute" (models that reject missing values but need no scaling, such as scikit-learn's
    GradientBoostingRegressor) only the median imputation is applied.
    """
    X_all = df[feature_cols]
    if not needs_impute_scale:
        return lambda X: X[feature_cols] if hasattr(X, "columns") else X
    med = X_all.loc[train_idx].median()
    if needs_impute_scale == "impute":
        # imputation only; the medians come from the training rows
        return lambda X: (X[feature_cols] if hasattr(X, "columns") else pd.DataFrame(X, columns=feature_cols)).fillna(med)
    scaler = StandardScaler().fit(X_all.loc[train_idx].fillna(med))
    apply_scale_floor(scaler, X_all.loc[train_idx].fillna(med))  # same denominator floor as the other standardized inputs

    def _transform(X):
        Xi = X[feature_cols] if hasattr(X, "columns") else pd.DataFrame(X, columns=feature_cols)
        return pd.DataFrame(scaler.transform(Xi.fillna(med)), index=Xi.index, columns=feature_cols)

    return _transform


def cat_cardinalities(df, cat_cols):
    """Cardinality of each categorical column over the whole data frame. The cat_* columns are integer codes
    0..n-1, so the cardinalities can be used as embedding table sizes independently of the domain split.

    This is the fixed schema of the dataset (number of levels), not a value or label. Embeddings of levels that occur
    only in the target domain keep their initial values when a model is trained on the source domain."""
    return [int(df[c].max()) + 1 for c in cat_cols]


# Floor of the standardization denominator.
#
# StandardScaler sets the denominator to 1 only when the variance is exactly zero. A small single-domain training set
# can contain columns whose standard deviation is positive but tiny, and dividing by it inflates the evaluation values
# by orders of magnitude, which breaks models without normalization layers (such as the MLP).
#
# The floor is a fixed constant that uses no statistics outside the training rows. The numerical features are min-max
# scaled to [0, 1] in the released data, so 1e-3 (0.1% of the feature range) is used; it lies below the smallest
# training-set standard deviation on the source side, so ordinary columns are unaffected. Columns with exactly zero
# variance keep the scikit-learn denominator of 1.
SCALE_FLOOR_ABS = 1e-3


def apply_scale_floor(scaler, cont_train):
    """Raise the standardization denominator to SCALE_FLOOR_ABS for columns whose training standard deviation is
    positive but below SCALE_FLOOR_ABS. Uses only the training rows (cont_train); modifies the scaler in place and
    returns it."""
    raw_sd = np.asarray(cont_train.std(ddof=0))
    bump = (raw_sd > 0) & (raw_sd < SCALE_FLOOR_ABS)
    if bump.any():
        scaler.scale_ = np.where(bump, SCALE_FLOOR_ABS, scaler.scale_)
    return scaler


def prep_features_cat_cont(df, cat_cols, num_cols, train_idx, *idxs):
    """Preprocessing for models that take categorical and numerical inputs separately (such as FT-Transformer). The
    categorical columns are already integer label codes and stay as they are; the numerical columns get median
    imputation and StandardScaler fitted on the training rows.

    The standardization denominator has a fixed floor (see SCALE_FLOOR_ABS). All statistics come from the train_idx rows.
    """
    _split = fit_cat_cont_transform(df, cat_cols, num_cols, train_idx)
    return [_split(idx) for idx in (train_idx, *idxs)]


def fit_cat_cont_transform(df, cat_cols, num_cols, train_idx):
    """Fit the transform of prep_features_cat_cont once on the training rows and return it as a function
    (idx, frame=None -> (x_cat, x_cont)), so that prediction reuses the transform fitted at training time. Statistics and
    the floor rule are those of prep_features_cat_cont. With frame, the idx rows of that DataFrame are transformed;
    otherwise those of df."""
    Xcat_all = df[cat_cols]
    Xcont_all = df[num_cols]

    med = Xcont_all.loc[train_idx].median()
    scaler = StandardScaler()
    Xcont_train = Xcont_all.loc[train_idx].fillna(med)
    scaler.fit(Xcont_train)
    apply_scale_floor(scaler, Xcont_train)  # floor from the training rows only

    def _split(idx, frame=None):
        xc, xn = (Xcat_all, Xcont_all) if frame is None else (frame[cat_cols], frame[num_cols])
        x_cat = xc.loc[idx].to_numpy(dtype="int64")
        x_cont = scaler.transform(xn.loc[idx].fillna(med)).astype("float32")
        return x_cat, x_cont

    return _split


def new_run_id() -> str:
    """Default run name from the start time, so that repeated runs do not mix their saved models."""
    return time.strftime("%Y%m%d_%H%M%S")


def read_per_fit(path):
    """Read report/per_fit.csv. A relative `pred_path` is taken relative to the cache folder (config.CACHE_DIR); an absolute one (as written
    by a local run) is kept."""
    df = pd.read_csv(path)
    if "pred_path" in df.columns:
        df["pred_path"] = [q if not isinstance(q, str) or os.path.isabs(q) else os.path.join(config.CACHE_DIR, q) for q in df["pred_path"]]
    return df
