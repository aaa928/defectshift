"""Shared rules of the domain screening (scripts/01_screen_domains.py), also used by the selection audit
(scripts/03a_audit_selection.py, which must follow the same artifact paths): artifact paths, artifact checks,
finished keys and appending result rows.
"""
import os

import pandas as pd

from defectshift import registry
from defectshift.candidates import slot_dir


def artifact_prefix(model_dir, slot, model, regime, i):
    return os.path.join(model_dir, slot_dir(slot), f"{model}__{regime}__r{i}")


def artifacts_ok(prefix, route):
    """Whether the saved artifacts are intact: row indices (.npz), configuration (.json) and, except for foundation models, the
    model file. All must exist and be non-empty, and the json must be readable."""
    import json
    need = [prefix + ".npz", prefix + ".json"] + ([] if route == "foundation" else
                                                  [prefix + (".pt" if route == "transformer" else ".joblib")])
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in need):
        return False
    try:
        json.load(open(prefix + ".json"))
    except Exception:
        return False
    return True


def done_keys(path, model_dir=None):
    """Finished (slot, repeat, model, regime). The status must be ok and, if model_dir is given, the saved artifacts must be intact
    (missing or broken artifacts are trained again)."""
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    d = d[d.status == "ok"]
    out = set()
    for r in d.itertuples():
        if model_dir is not None and not artifacts_ok(artifact_prefix(model_dir, r.slot, r.model, r.regime, r.repeat),
                                                      registry.get(r.model).route):
            continue
        out.add((str(r.slot), str(r.repeat), str(r.model), str(r.regime)))
    return out


def append_row(path, row):
    """Append one result row. Columns follow the header order of the existing file (missing values left empty). If a column
    not in the header appears for the first time (a failed row, for example), the whole file is rewritten with the union of the
    columns (temporary file, then replace), because appending rows with different columns would shift the cells."""
    new = pd.DataFrame([row])
    if not os.path.exists(path):
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
