"""Reading and writing run results, and the checks that decide whether one run may use the outputs of another."""
import json
import os
import sys

import pandas as pd

STOP_EXIT_CODE = 75   # "stopped on request" (not a failure). The run scripts that chain several steps stop on this code instead of moving on


def check_stop(stop_file):
    """If the stop-request file exists, leave the records as they are and exit (code STOP_EXIT_CODE). Running the same command again
    continues with the remaining units. The file is checked just before a unit starts (tuning: (domain, model, side); final benchmark:
    (domain, model, repetition, regime)). A unit that has started runs to the end, and a record is written only after its unit
    has finished, so no half-written record is left at the stop. To stop at once, the process can be killed (only the unit in
    progress is repeated; the NODE search keeps a per-configuration progress record, so finished configurations are not repeated)."""
    if stop_file and os.path.exists(stop_file):
        from defectshift.utils import log
        log(f"[stop request] found {stop_file}: the records are kept and the run stops. Running the same command again continues", level="WARNING")
        sys.exit(STOP_EXIT_CODE)


def append_row(path, row):
    """Append one result row, with columns in the order of the existing header (missing values left empty). When a row
    brings a column that the header lacks (for example a failed row), the whole file is rewritten with the union of the
    columns (temporary file, then replace), because appending rows with a different column set would shift the cells."""
    new = pd.DataFrame([row])
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new.to_csv(path, index=False)
        return
    cols = list(pd.read_csv(path, nrows=0).columns)
    extra = [c for c in new.columns if c not in cols]
    if not extra:
        new.reindex(columns=cols).to_csv(path, mode="a", header=False, index=False)
        return
    tmp = path + ".tmp"
    pd.concat([pd.read_csv(path), new], ignore_index=True)[cols + extra].to_csv(tmp, index=False)
    os.replace(tmp, path)


def control_names(controls, only):
    """Seeds of --controls -> names of the random-split controls (placebo@<seed>). Only the seeds of the controls of the screening stage
    (config.PLACEBO_B_TARGET) are accepted. Combining with --only is refused so that controls and selected domains are not mixed."""
    from defectshift import config
    if only:
        raise SystemExit("--controls and --only are not used together")
    seeds = [int(s) for s in controls.split(",")]
    bad = [s for s in seeds if s not in config.PLACEBO_B_TARGET]
    if bad:
        raise SystemExit(f"not a control seed: {bad} (available: {sorted(config.PLACEBO_B_TARGET)})")
    return [f"placebo@{s}" for s in seeds]


def check_same_inputs(man, what):
    """Stop unless the data and candidate-list hashes in a run manifest equal those of the current files. A domain of the
    same name gives other splits and results on changed data, so results of an earlier run are not reused."""
    from defectshift import config, manifest
    for key, path in (("data_sha256", config.DATA_PATH), ("candidates_sha256", config.CANDIDATES_CSV)):
        if man.get(key) is None:
            raise SystemExit(f"{what}: the manifest has no {key}")
        if man[key] != manifest.file_sha256(path):
            raise SystemExit(f"{what}: {key} differs from the current file ({path})")


# Shared training code and packages that must match when one run uses the tuning records or qualification results of
# another (the running scripts themselves may differ)
COMPAT_FILES = ("defectshift/config.py", "defectshift/regimes.py", "defectshift/hpo.py", "defectshift/utils.py",
                "defectshift/splits.py", "defectshift/candidates.py", "defectshift/models/protocol.py",
                "defectshift/models/ml.py", "defectshift/models/transformer.py", "defectshift/models/tabular_dl.py")
COMPAT_PKGS = ("numpy", "pandas", "scikit-learn", "scipy", "torch", "xgboost", "lightgbm", "catboost", "optuna", "pytabkit", "rtdl")
# When the benchmark uses the foundation-model qualification results: input construction, the foundation-model wrappers,
# the qualification code and the foundation-model packages
FMQ_COMPAT_FILES = ("defectshift/config.py", "defectshift/regimes.py", "defectshift/utils.py", "defectshift/splits.py",
                    "defectshift/candidates.py", "defectshift/models/foundation.py", "scripts/05_qualify_foundation.py")
FMQ_COMPAT_PKGS = ("numpy", "pandas", "scikit-learn", "scipy", "torch", "tabpfn", "tabicl", "tabdpt", "tabfm", "exaonetabular",
                   "structured-data-models")


def check_compat(man, what, files=COMPAT_FILES, pkgs=COMPAT_PKGS):
    """Whether the current run may use the outputs of another run (man): the contents of the shared training code
    (per-file hashes), the versions of the relevant packages and the protocol constants must match. Manifests without
    per-file hashes cannot be checked and are refused. Raises SystemExit on a mismatch."""
    from defectshift import config, manifest
    bad = []
    fh = man.get("code_file_sha256")
    if fh is None:
        bad.append("the manifest has no per-file code hashes (run of an earlier protocol version)")
    else:
        root = os.path.realpath(config.REPO_ROOT)
        for f in files:
            if fh.get(f) != manifest.file_sha256(os.path.join(root, f)):
                bad.append(f"code {f}")
    cur_pkgs = manifest._packages()
    bad += [f"package {p} {(man.get('packages') or {}).get(p)} != {cur_pkgs.get(p)}" for p in pkgs
            if (man.get("packages") or {}).get(p) != cur_pkgs.get(p)]
    cur_const = json.loads(json.dumps(manifest.protocol_constants()))
    if (man.get("protocol") or {}).get("constants") != cur_const:
        bad.append("protocol constants (config)")
    if bad:
        raise SystemExit(f"{what}: not compatible with the current code and environment: " + "; ".join(bad[:8]))


def refuse_orphan_cache(run_dir, cache_dir, what):
    """Stop when a new run (no manifest in run_dir yet) finds a cache of the same run name (tuning records, for example).
    If only the results folder was deleted, the remaining records cannot be traced to their code through a manifest and
    must not mix into the new run."""
    if os.path.exists(os.path.join(run_dir, "manifest.json")) or not os.path.isdir(cache_dir):
        return
    if any(files for _, _, files in os.walk(cache_dir)):
        raise SystemExit(f"{what}: a cache ({cache_dir}) exists without its results folder ({run_dir}); use a new --run-id or check and clean the cache")


def check_selection_current(screening_run_id):
    """For manuscript tables and merging: the screening run behind the selection must be of the current protocol version
    (config.PROTOCOL_VERSION). Screening runs kept only for comparison are refused; paths that read them for comparison
    do not call this function."""
    from defectshift import config
    man = json.load(open(os.path.join(config.RESULTS_DIR, "screening", "runs", screening_run_id, "manifest.json")))
    v = ((man.get("protocol") or {}).get("constants") or {}).get("PROTOCOL_VERSION")
    if v != config.PROTOCOL_VERSION:
        raise SystemExit(f"screening run {screening_run_id} has protocol version {v} != {config.PROTOCOL_VERSION}; manuscript tables and merging "
                         "use only screening runs of the current version")


def load_final_domains(screening_run_id):
    """Selected domains (two temporal + process condition). Checks that the rows and manifest of the screening run still
    match the hashes recorded at selection time and that the data and candidate-list hashes of the screening run match the
    current files. Returns (domain list, hash of the selection file)."""
    import json
    from defectshift import config, manifest
    run_dir = os.path.join(config.RESULTS_DIR, "screening", "runs", screening_run_id)
    path = os.path.join(run_dir, "selection", "final_domains.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist; the benchmark runs only after the domains are selected")
    final = json.load(open(path))
    if final["rows_sha256"] != manifest.file_sha256(os.path.join(run_dir, "rows.csv")) or \
            final["manifest_sha256"] != manifest.file_sha256(os.path.join(run_dir, "manifest.json")):
        raise SystemExit("rows.csv or manifest.json of the screening run differ from selection time (hash mismatch)")
    check_same_inputs(json.load(open(os.path.join(run_dir, "manifest.json"))), f"screening run {screening_run_id}")
    return final["temporal"] + final["categorical"], manifest.file_sha256(path)


def artifacts_ok(prefix, route, need_pred=False):
    """Whether the saved artifacts are complete: row indices (.npz), settings (.json), the model file unless the model
    is a foundation model, and the predictions (.pred.npz) if need_pred."""
    import json
    need = [prefix + ".npz", prefix + ".json"] + ([] if route == "foundation" else
                                                  [prefix + (".pt" if route == "transformer" else ".joblib")])
    if need_pred:
        need.append(prefix + ".pred.npz")
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need):
        return False
    try:
        json.load(open(prefix + ".json"))
    except Exception:
        return False
    return True


def done_keys(rows_path, prefix_of=None, route_of=None, key_cols=("slot", "repeat", "model", "regime"), need_pred=False):
    """Set of keys (string tuples) with status ok. With prefix_of(row) and route_of(model), the saved artifacts must
    also be complete for a key to count as done."""
    if not os.path.exists(rows_path):
        return set()
    d = pd.read_csv(rows_path)
    d = d[d.status == "ok"]
    out = set()
    for r in d.to_dict("records"):
        if prefix_of is not None and not artifacts_ok(prefix_of(r), route_of(r["model"]), need_pred):
            continue
        out.add(tuple(str(r[k]) for k in key_cols))
    return out
