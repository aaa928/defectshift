"""Foundation models of the benchmark: TabPFN-3.5, TabICL-2.2, TabFM 1.0.1 (GitHub fbb6655), TabDPT-1.3, Kumo
Tabular-Large and EXAONE-Tabular. All hyperparameters are the package defaults (in-context models, no tuning).
Installed code check: version() compares Kumo (sdm install commit + hash of the installed source), EXAONE (hash of the
installed source) and TabFM (clean git commit + hash of the installed source) with fixed values and stops on any difference
(also when the source was modified or extended at the same commit).

Common input convention: (X_cat, X_cont) -> combined float32 array + positions of the categorical columns (`combine`).
X_cat = integer category ids fixed by the dataset. X_cont follows regimes.Ctx.fm_input: "raw_native" in the benchmark (raw
numerical values with missing entries, no external imputation or standardization; missing values are handled by each
model's official interface, MISSING_HANDLING), "impute_standardize" in screening and the selection audit (training-row
median imputation and standardization). The combined array is passed to each model's official interface:
- TabPFN, TabICL, TabDPT, EXAONE: the array as is (only TabPFN takes the categorical column positions; the others have no
  such argument)
- TabFM, Kumo: a DataFrame with the categorical columns marked as categorical (TabFM: category dtype, Kumo: sdm categorical stype)
- Kumo regression: 16 ensemble members, float16 autocast of the network (settings of NVIDIA's official benchmark adapter).
  The point prediction is the mean of the 999 quantiles produced by the default sdm recipe (sort the quantiles, then a
  20% two-sided trimmed mean over members), as in the NVIDIA adapter and the TabArena integration
  (tabarena/models/kumo_tabular/model.py: `out.numerical.float().mean(dim=-1)`)
- EXAONE regression: the readout fixed by the package

One instance (weights) per model is loaded per process, and only the seed changes between fits; creating instances
hundreds of times can fragment GPU memory until the process dies. Training = building the context from the training rows
(no parameter training).
"""
import importlib.metadata as _md
import os
from functools import lru_cache

import numpy as np
import torch

from defectshift.models import weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOUNDATION_MODELS = ("TabPFN", "TabICL", "TabFM", "TabDPT", "Kumo", "EXAONE")
# Models judged by the qualification check: the three additional models only; a failing model would be excluded from the
# benchmark without replacement. The check results of the other three are reported only.
QUALIFY_JUDGED = ("Kumo", "TabDPT", "EXAONE")
# What each model does with numerical missing values passed as is (Ctx.fm_input="raw_native"; from the papers and the installed sources)
MISSING_HANDLING = {
    "TabPFN": "native: missing/infinite indicator recorded, then training-column mean imputation; indicator read by the "
              "encoder (TabPFN-3.5 report, Fig. 8)",
    "Kumo": "native: training-row mean imputation plus a learned missing-mask embedding (sdm cell_embedding.nan_lin)",
    "EXAONE": "native: NaN mapped to a non-finite code channel read by the encoder (exaonetabular model/encoders.py)",
    "TabICL": "internal mean imputation (TabICLv2 paper: 'Missing values are currently imputed by the mean')",
    "TabDPT": "internal: zero after normalization, i.e. mean imputation (TabDPT paper)",
    "TabFM": "internal mean imputation for numerical columns (sklearn SimpleImputer in classifier_and_regressor)",
}
KUMO_SIZE = "large"
KUMO_SDM_COMMIT = "98f61289c7a4e1bce3b33771223ac2e123b63f19"   # installed commit (direct_url.json), the same as in the TabArena evaluation
KUMO_WEIGHTS_REVISION = "v1.0.1"                                 # Hub tag pinned by sdm/models/kumo/tabular/model.py
KUMO_WEIGHTS_SNAPSHOT = "3c3e10bbdb590ace29e7026847f92db3c603096d"   # commit that this tag pointed to for the paper (a Hub tag can be moved); checked below
# Kumo ensemble size and network precision: NVIDIA's official benchmark adapter at the installed commit,
# benchmark/tabular/model.py (SDMKumoTabularModel: default_num_estimators = 16, autocast_dtype = torch.float16, used as is
# for Large), the TabArena integration (default_num_estimators 16, float16 autocast) and the model card example (float16
# autocast) agree. sdm's fit defaults to num_estimators=None and runs a single member for 2-D input, so the value must be
# set explicitly. sdm runs the preprocessing recipe with autocast disabled (base.py).
KUMO_NUM_ESTIMATORS = 16
KUMO_AUTOCAST_DTYPE = torch.float16
EXAONE_CODE_COMMIT = "8638e07d09fad154249bd75ba4786181491f7025"  # official repository at this commit, without patches
# sha256 of the installed source (src/exaonetabular/**/*.py, 22 files, relative path + content), computed when it was identical to the official commit
EXAONE_SRC_SHA256 = "19b7588de62bafa1510a28b12689768193d53edf8b9203622e75d92eb44e5fe2"
TABFM_COMMIT = "fbb665569425fd2f490c6576b3af967876fe11ff"         # official main (including changes after 1.0.1)
# sha256 of the installed sources (.py, relative path + content), computed in a clean state (tabfm: identical to the
# official fbb6655, no untracked files; sdm: as installed by pip from 98f61289c)
TABFM_SRC_SHA256 = "5b4afbe9bad927f9fceea7dabe80a85c309a813bb2e6aceb18b36a6a0772c457"
SDM_SRC_SHA256 = "69b56a41e6849c146263f40b81566477d99b90b4ec771b2a7ec9d03629533b69"


def combine(X):
    """X=(X_cat, X_cont) -> (combined array, positions of the categorical columns)."""
    x_cat, x_cont = X
    return np.concatenate([x_cat.astype("float32"), x_cont], axis=1), list(range(x_cat.shape[1]))


def _frame(X, cat_idx, cat_as_category):
    """Combined array -> DataFrame. Categorical columns as integer ids (category dtype if cat_as_category)."""
    import pandas as pd
    cols = [f"cat_{i}" if i in cat_idx else f"num_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    for i in cat_idx:
        c = cols[i]
        df[c] = df[c].round().astype("int64")
        if cat_as_category:
            df[c] = df[c].astype("category")
    return df


_instances = {}   # model name -> object with loaded weights (one per process)


class _TabFM:
    """TabFM: wraps a TabFMRegressor around the shared weights (backend.load with an explicit device) for every fit. Input:
    DataFrame with category dtype."""

    def __init__(self, cat_idx, seed):
        self.cat_idx, self.seed = cat_idx, seed

    def fit(self, X, y):
        from tabfm.src.classifier_and_regressor import TabFMRegressor
        if "TabFM" not in _instances:
            from tabfm import tabfm_v1_0_0_pytorch as backend
            _instances["TabFM"] = backend.load(model_type="regression", checkpoint_path=weights.snapshot_dir("TabFM"), device=DEVICE)
        self.model = TabFMRegressor(model=_instances["TabFM"], random_state=self.seed)
        self.model.fit(_frame(X, self.cat_idx, True), y)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(_frame(X, self.cat_idx, True)), float).reshape(-1)

    def get_params(self):
        return dict(random_state=self.seed, backend="tabfm_v1_0_0_pytorch", device=DEVICE)


class _TabDPT:
    """TabDPT: the API takes the seed at prediction time, so the seed is kept and passed to predict(seed=). Everything else
    uses the package defaults."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        if "TabDPT" not in _instances:
            from tabdpt import TabDPTRegressor
            _instances["TabDPT"] = TabDPTRegressor(device=DEVICE, verbose=False, model_weight_path=weights.main_path("TabDPT"))
        self.model = _instances["TabDPT"]
        self.model.fit(np.asarray(X, np.float32), np.asarray(y, np.float32))
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(np.asarray(X, np.float32), seed=self.seed), float).reshape(-1)

    def get_params(self):
        return dict(seed=self.seed, predict_defaults="n_ensembles=8, context_size=None (package defaults)", device=DEVICE)


class _Kumo:
    """Kumo Tabular-Large (sdm). Categorical columns get the sdm categorical stype, numerical columns the numerical stype.
    Training = fitting the context and the preprocessing recipe, with the seed passed as a torch.Generator. Prediction = mean
    of the output of the default sdm recipe (999 quantiles)."""

    def __init__(self, cat_idx, seed):
        self.cat_idx, self.seed = cat_idx, seed

    def _table(self, X):
        import sdm
        df = _frame(X, self.cat_idx, False)
        stypes = {c: ("categorical" if c.startswith("cat_") else "numerical") for c in df.columns}
        return sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=DEVICE)

    def fit(self, X, y):
        import pandas as pd
        import sdm
        if "Kumo" not in _instances:
            model = sdm.models.KumoTabular(task="regression", size=KUMO_SIZE, device=DEVICE)
            _check_kumo_weights(strict=True)       # cached only after the check has passed: a failed check fails again on the next call
            _instances["Kumo"] = model
        self.model = _instances["Kumo"]
        y_t = sdm.TableTensor.from_pandas(df=pd.DataFrame({"target": np.asarray(y, float)}), stypes={"target": "numerical"},
                                          device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(int(self.seed))
        with torch.amp.autocast(DEVICE, dtype=KUMO_AUTOCAST_DTYPE, enabled=DEVICE == "cuda"):
            self.model.fit(x=self._table(X), y=y_t, num_estimators=KUMO_NUM_ESTIMATORS, generator=gen)
        return self

    def predict(self, X):
        with torch.amp.autocast(DEVICE, dtype=KUMO_AUTOCAST_DTYPE, enabled=DEVICE == "cuda"):
            out = self.model.predict(self._table(X))
        return out.numerical.float().mean(dim=-1).cpu().numpy().astype(float).reshape(-1)

    def get_params(self):
        return dict(seed=self.seed, size=KUMO_SIZE, weights_revision=KUMO_WEIGHTS_REVISION, sdm_commit=KUMO_SDM_COMMIT,
                    num_estimators=KUMO_NUM_ESTIMATORS, autocast_dtype=str(KUMO_AUTOCAST_DTYPE) if DEVICE == "cuda" else None,
                    readout="mean of 999 quantiles after sdm default recipe (sort, 20% trimmed mean over estimators)")


class _EXAONE:
    """EXAONE-Tabular regression. The weights are the public checkpoint of the pinned revision. The package pins no sha256 for it (it logs a warning when it loads the file),
    so the files are verified here against environment/weights.json before the load; the
    seed changes for every fit through manifest.with_overrides(seed=) (the weights stay the same)."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        if "EXAONE" not in _instances:
            from exaonetabular import EXAONETabularRegressor
            if os.environ.get("EXAONETABULAR_REGRESSOR_WEIGHTS"):          # the package would use that file instead of the pinned revision, which the verification below would no longer describe
                raise RuntimeError("EXAONETABULAR_REGRESSOR_WEIGHTS is set: unset it so that the pinned EXAONE-Tabular weights are used")
            weights.snapshot_dir("EXAONE")          # the pinned files must be in the local cache and equal to the pinned sha256 (the package pins none)
            _instances["EXAONE"] = EXAONETabularRegressor.from_pretrained(device="cuda:0" if DEVICE == "cuda" else "cpu",
                                                                         seed=int(self.seed), revision=weights.commit("EXAONE"))
        self.model = _instances["EXAONE"]
        self.model.manifest = self.model.manifest.with_overrides(seed=int(self.seed))
        self.model.fit(np.asarray(X, np.float32), np.asarray(y, np.float32))
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(np.asarray(X, np.float32)), float).reshape(-1)

    def get_params(self):
        rt = self.model.manifest.runtime if hasattr(self, "model") else None
        return dict(seed=self.seed, runtime=(None if rt is None else repr(rt)), code_commit=EXAONE_CODE_COMMIT)


def build(name, cat_idx, seed):
    """Regression model object (fit(X, y) / predict(X) / get_params()). Hyperparameters are the package defaults."""
    if name == "TabPFN":
        if name not in _instances:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
            _instances[name] = TabPFNRegressor.create_default_for_version(
                ModelVersion.V3_5, device=DEVICE, random_state=seed, categorical_features_indices=cat_idx,
                model_path=weights.main_path("TabPFN"))
        return _instances[name].set_params(random_state=seed, categorical_features_indices=cat_idx)
    if name == "TabICL":
        if name not in _instances:
            from tabicl import TabICLRegressor
            _instances[name] = TabICLRegressor(device=DEVICE, random_state=seed, model_path=weights.main_path("TabICL"), allow_auto_download=False)
        return _instances[name].set_params(random_state=seed)
    if name == "TabFM":
        return _TabFM(cat_idx, seed)
    if name == "TabDPT":
        return _TabDPT(seed)
    if name == "Kumo":
        return _Kumo(cat_idx, seed)
    if name == "EXAONE":
        return _EXAONE(seed)
    raise ValueError(name)


def release(name=None):
    """Unload loaded weights (all if name is None): when the benchmark moves to another model, and when the qualification
    check compares with a fresh instance."""
    for k in ([name] if name else list(_instances)):
        _instances.pop(k, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def limits(name, model):
    """Row and feature limits and reduction rules reported by the model (for the record; None if unknown). model = the
    object returned by build(), after fitting."""
    import re
    try:
        if name == "TabPFN":
            ic = model.inference_config_
            return dict(max_samples=ic.MAX_NUMBER_OF_SAMPLES, max_features=ic.MAX_NUMBER_OF_FEATURES)
        if name == "TabDPT":
            m = model.model
            return dict(max_features=m.max_features, feature_reduction=m.feature_reduction, context_reduction=m.context_reduction)
        if name == "EXAONE":
            txt = repr(model.model.manifest)
            return {k: int(v) for k, v in re.findall(r"(support_row_limit|feature_limit)=(\d+)", txt)}
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}"[:200])
    return None


def effective_shape(name, model, n_rows, n_features):
    """Context rows and features the model actually uses, relative to the input (from the installed sources, for the
    record). Called right after a foundation-model fit.
    - TabPFN, TabICL: no reduction (our sizes are within the limits) / TabFM: all rows if max_num_rows is None, features <=
      max_num_features (500)
    - TabDPT: predict context_size defaults to None -> all training rows; features > max_features (128) are reduced to 128 by PCA
    - Kumo: no row reduction in the default recipe; columns up to SelectColumns(500) (constant columns can be dropped by
      DropConstantColumns)
    - EXAONE regression: rows > support_row_limit (100,000) are subsampled at random, features > feature_limit (1,024)
      selected by f_regression
    The pretraining width (number of columns) is recorded separately: TabICLv2 and Kumo were pretrained on 2-100 columns
    (official documentation); our 221 columns lie outside."""
    rows, feats, note = n_rows, n_features, "none"
    lim = limits(name, model) or {}
    if name == "TabDPT" and isinstance(lim.get("max_features"), int) and n_features > lim["max_features"]:
        feats, note = lim["max_features"], f"features PCA {n_features}->{lim['max_features']}"
    elif name == "Kumo" and n_features > 500:
        feats, note = 500, "SelectColumns(500, first)"
    elif name == "EXAONE":
        if isinstance(lim.get("support_row_limit"), int) and n_rows > lim["support_row_limit"]:
            rows, note = lim["support_row_limit"], "rows random subsample"
        if isinstance(lim.get("feature_limit"), int) and n_features > lim["feature_limit"]:
            feats, note = lim["feature_limit"], "features f_regression selection"
    elif name == "TabFM" and n_features > 500:
        feats, note = 500, "max_num_features=500"
    pretrain_width = {"TabICL": "2-100 columns", "Kumo": "up to 100 columns"}.get(name)
    return dict(context_input_rows=n_rows, context_effective_rows=rows, feature_input_cols=n_features,
                feature_effective_cols=feats, reduction=note, pretraining_width=pretrain_width, limits=lim)


def _hf_snapshot(repo_dir, ref="main"):
    """Snapshot revision that ref points to in the active Hugging Face cache (respecting HF_HOME); None if absent."""
    from huggingface_hub.constants import HF_HUB_CACHE
    p = os.path.join(HF_HUB_CACHE, repo_dir, "refs", ref)
    return open(p).read().strip() if os.path.exists(p) else None


def _weights_snapshot(repo_dir, name):
    """Weights revision for the run record: the commit pinned in environment/weights.json, which is what the code loads. The Hugging Face cache is consulted
    only for a model that is not pinned there (a download by commit creates no refs/main)."""
    rec = weights.record(name)
    return rec["commit"] if rec else _hf_snapshot(repo_dir)


def _git_state(pkg):
    """Git commit of the source of an editable installed package and the tracked files modified since (for the record;
    None if it is not a git repository). find_spec does not execute the package's __init__."""
    import importlib.util
    import subprocess
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.origin:
        return None
    root = os.path.dirname(os.path.dirname(spec.origin))
    try:
        top = subprocess.run(["git", "-C", root, "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             check=True).stdout.strip()
        if os.path.realpath(top) != os.path.realpath(root):   # a copy inside another repository: do not record that repository's commit
            return None
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, check=True).stdout.split("\n")
    except (OSError, subprocess.CalledProcessError):
        return None
    files = [ln[3:] for ln in dirty if ln.strip()]
    return f"git {head}" + (f" (modified: {', '.join(files)})" if files else "")


def _kumo_installed_commit():
    """Installed commit from direct_url.json written by pip (structured-data-models)."""
    import json
    info = json.loads(_md.distribution("structured-data-models").read_text("direct_url.json") or "{}")
    return (info.get("vcs_info") or {}).get("commit_id")


@lru_cache(maxsize=None)
def _src_sha256(pkg):
    """sha256 of the relative paths and contents of an installed package's source (.py), without __pycache__."""
    import hashlib
    import importlib.util
    root = os.path.dirname(importlib.util.find_spec(pkg).origin)
    h = hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(root)):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            if f.endswith(".py"):
                path = os.path.join(dp, f)
                h.update(os.path.relpath(path, root).encode()); h.update(b"\0")
                h.update(open(path, "rb").read()); h.update(b"\0")
    return h.hexdigest()


def _check_kumo_weights(strict):
    """The Hub tag of the Kumo-Tabular weights can be moved, so the snapshot that it points to in the local cache must be the one of the paper. Before the first load
    the weights may not be downloaded yet (strict=False allows that); after the load they must be present and equal (strict=True)."""
    snap = _hf_snapshot("models--nvidia--Kumo-Tabular", KUMO_WEIGHTS_REVISION)
    if snap is None:
        if strict:
            raise RuntimeError(f"Kumo-Tabular weights of tag {KUMO_WEIGHTS_REVISION} not found in the Hugging Face cache after the load")
        return
    if snap != KUMO_WEIGHTS_SNAPSHOT:
        raise RuntimeError(f"Kumo-Tabular weights: tag {KUMO_WEIGHTS_REVISION} points to {snap} in the local cache, the paper used {KUMO_WEIGHTS_SNAPSHOT}. To restore it, "
                           f"download that revision with snapshot_download and write it into refs/{KUMO_WEIGHTS_REVISION} of models--nvidia--Kumo-Tabular "
                           f"(docs/reproducing.md, paragraph 'Foundation-model weights')")


def check_install(name):
    """Check that the installed code equals the fixed values; RuntimeError otherwise (so that no other code runs silently)."""
    if name == "Kumo":
        if _kumo_installed_commit() != KUMO_SDM_COMMIT:
            raise RuntimeError(f"structured-data-models installed commit {_kumo_installed_commit()} != fixed {KUMO_SDM_COMMIT}")
        if _src_sha256("sdm") != SDM_SRC_SHA256:
            raise RuntimeError("the installed sdm source differs from the fixed hash (installed files changed at the same commit)")
        _check_kumo_weights(strict=False)
    if name == "EXAONE" and _src_sha256("exaonetabular") != EXAONE_SRC_SHA256:
        raise RuntimeError("the installed exaonetabular source differs from the fixed hash (changed from the official 8638e07)")
    if name == "TabFM":
        state = _git_state("tabfm") or ""
        if state != f"git {TABFM_COMMIT}":        # a state with modified files ("(modified: ...)") is refused as well
            raise RuntimeError(f"the installed tabfm source is not a clean {TABFM_COMMIT}: {state}")
        if _src_sha256("tabfm") != TABFM_SRC_SHA256:   # changes git status (tracked files only) misses, such as an added untracked .py
            raise RuntimeError("the installed tabfm source differs from the fixed hash")


def weights_record(name):
    """Repository, commit and sha256 of the pinned weight files of `name` (None for Kumo, which is pinned by its Hub tag), for the run manifest."""
    return weights.record(name)


def version(name):
    """Version string for the record: package version and, where available, code commit and weight revision. The installed
    code is checked first with check_install."""
    check_install(name)
    if name == "TabPFN":
        return f"tabpfn {_md.version('tabpfn')}, ModelVersion.V3_5"
    if name == "TabICL":
        return (f"tabicl {_md.version('tabicl')} (TabICLv2 regressor), "
                f"weights HF snapshot {_weights_snapshot('models--jingang--TabICL', 'TabICL')}")
    if name == "TabFM":
        return (f"tabfm {_md.version('tabfm')} {_git_state('tabfm')}, backend tabfm_v1_0_0_pytorch, "
                f"weights HF snapshot {_weights_snapshot('models--google--tabfm-1.0.0-pytorch', 'TabFM')}")
    if name == "TabDPT":
        return f"tabdpt {_md.version('tabdpt')}, weights HF snapshot {_weights_snapshot('models--Layer6--TabDPT', 'TabDPT')}"
    if name == "Kumo":
        return (f"structured-data-models {_md.version('structured-data-models')} commit {KUMO_SDM_COMMIT}, "
                f"KumoTabular size={KUMO_SIZE}, weights Hub tag {KUMO_WEIGHTS_REVISION} "
                f"(snapshot {_hf_snapshot('models--nvidia--Kumo-Tabular', KUMO_WEIGHTS_REVISION)}), "
                f"num_estimators={KUMO_NUM_ESTIMATORS}, autocast {KUMO_AUTOCAST_DTYPE}")
    if name == "EXAONE":
        return (f"exaonetabular {_md.version('exaonetabular')} code commit {EXAONE_CODE_COMMIT}, "
                f"weights HF snapshot {_weights_snapshot('models--LG-AI-Research--EXAONE-Tabular', 'EXAONE')}")
    return "unknown"
