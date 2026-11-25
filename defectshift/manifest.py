"""Run manifest: one file per run that records the code, data, settings and environment behind its results.

Each run folder gets a manifest.json. Resuming a run in the same folder is allowed only if every key that determines
reproducibility is unchanged; a change of code, data, candidate list, protocol constants or package versions makes it a
different run, and resuming is refused.
"""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time

from defectshift import config

PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "torch", "xgboost", "lightgbm", "catboost", "optuna",
            "pytabkit", "rtdl", "tabpfn", "tabicl", "tabdpt", "tabfm", "exaonetabular", "structured-data-models")
# Keys that decide whether a run may be resumed (time stamps, for example, are not compared). The runtime (Python, CUDA,
# GPU, driver) is included because GPU models can give different numbers on a different device or driver even with the
# same code and packages; after such a change, start a new run.
KEYS_MUST_MATCH = ("data_sha256", "candidates_sha256", "code_sha256", "protocol", "packages", "args",
                   "python", "torch_cuda", "gpu", "nvidia_driver")


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def code_files():
    """Code files inside the repository that this process actually loaded (including the running script), plus the
    candidate list CSV, sorted by path."""
    root = os.path.realpath(config.REPO_ROOT)
    files = set()
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    for f in (sys.argv[0] if sys.argv else None, main_file):   # the running script (existing files only)
        if f and f.endswith(".py") and os.path.isfile(f) and os.path.realpath(f).startswith(root + os.sep):
            files.add(os.path.realpath(f))
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        # some external modules have a relative __file__, which would resolve inside the repository from the current
        # folder; only absolute paths of existing files are used
        if f and os.path.isabs(f) and f.endswith(".py") and os.path.isfile(f) and os.path.realpath(f).startswith(root + os.sep):
            files.add(os.path.realpath(f))
    files.add(os.path.realpath(config.CANDIDATES_CSV))
    return sorted(files)


def code_sha256(files=None):
    """Hash of the paths and contents of code_files(), concatenated."""
    h = hashlib.sha256()
    for p in files or code_files():
        h.update(os.path.relpath(p, os.path.realpath(config.REPO_ROOT)).encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()


def git_state():
    try:
        head = subprocess.run(["git", "-C", config.REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", config.REPO_ROOT, "status", "--porcelain"], capture_output=True, text=True)
        # without any commit, rev-parse fails and prints "HEAD", so the return code decides
        return dict(head=head.stdout.strip() if head.returncode == 0 else None, dirty=bool(dirty.stdout.strip()))
    except Exception as e:
        return dict(error=repr(e))


def _packages():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def protocol_constants():
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(config).items()
            if k.isupper() and not k.endswith(("_DIR", "_PATH", "_CSV", "_ROOT"))}


def _driver():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def preprocessing_policy():
    from defectshift import utils
    return dict(
        scale_floor_abs=utils.SCALE_FLOOR_ABS,
        statistics="median, standardization and its floor are all computed on the training rows of each model",
        categorical_encoding="global integer codes fixed by the dataset (cat_* 0..K-1); embedding table size = maximum over all data + 1 (schema)",
        features="cat_* + num_*, without the production-lot column (config.LOT_COL), the column that defines a process-condition split, the timestamp columns and the target",
        split_grouping="every split (A_heldout, tuning, repetitions, the four subsets of B, internal early stopping, A/B of the random-split controls) "
                       "is grouped by production lot and stratified by y > 0 (StratifiedGroupKFold); record shares are approximate",
    )


def build(args):
    """Call after all imports (after the training code is loaded), so that code_files contains all code that runs."""
    import torch
    files = code_files()
    return dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        data_path=config.DATA_PATH, data_sha256=file_sha256(config.DATA_PATH),
        candidates_sha256=file_sha256(config.CANDIDATES_CSV),
        code_sha256=code_sha256(files), code_files=[os.path.relpath(p, os.path.realpath(config.REPO_ROOT)) for p in files],
        # per-file hashes, for compatibility checks between runs (shared training code must match even when the running
        # scripts differ; runio.check_compat)
        code_file_sha256={os.path.relpath(p, os.path.realpath(config.REPO_ROOT)): file_sha256(p) for p in files},
        git=git_state(),
        protocol=dict(constants=protocol_constants(), preprocessing=preprocessing_policy()), packages=_packages(), args=args,
        python=sys.version.split()[0], platform=platform.platform(),
        torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        nvidia_driver=_driver(),
    )


def write_or_check(run_dir, new):
    """Write run_dir/manifest.json if it does not exist; otherwise check that the reproducibility keys are unchanged
    (raise otherwise)."""
    path = os.path.join(run_dir, "manifest.json")
    os.makedirs(run_dir, exist_ok=True)
    new = json.loads(json.dumps(new, ensure_ascii=False))   # same form as the saved file (integer keys -> strings, tuples -> lists)
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(new, f, ensure_ascii=False, indent=1)
        return "new"
    old = json.load(open(path))
    diff = [k for k in KEYS_MUST_MATCH if old.get(k) != new.get(k)]
    if diff:
        raise RuntimeError(f"{run_dir}: cannot resume, the conditions differ from the existing run ({diff}); use a new --run-id")
    return "resume"
