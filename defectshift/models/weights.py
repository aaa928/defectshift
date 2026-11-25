"""Pinned weights of the foundation models that download them from the Hugging Face Hub (TabPFN-3.5, TabICL, TabFM, TabDPT, EXAONE-Tabular).

environment/weights.json records the repository, the commit and the sha256 of every file that the paper used (docs/reproducing.md, "Foundation-model
weights"). The libraries download the newest file of `main` when no path is given, so this module hands them a pinned path instead:
path(name, file) returns the local path of a file of the pinned commit after checking its size and sha256; it reads the local Hugging Face cache only
and raises RuntimeError if the file is missing or differs, never downloading and never falling back to `main`. main_path(name) and snapshot_dir(name)
give what is passed to the library, fetch(name) downloads exactly the pinned files by commit (scripts/00_fetch_weights.py), and record(name) returns
repository, commit and sha256 for the run manifest. Kumo-Tabular is pinned by the Hub tag KUMO_WEIGHTS_REVISION in foundation.py.
"""
import hashlib
import json
import os

from defectshift import config

WEIGHTS_JSON = os.path.join(config.REPO_ROOT, "environment", "weights.json")
CACHE_DIR = None            # Hugging Face cache folder; None = the default of huggingface_hub (tests set a temporary folder)
HINT = ("run `python scripts/00_fetch_weights.py` once (network access; TabPFN-3.5 also needs a Hugging Face token and the accepted license)")
_verified = set()


def spec(name):
    with open(WEIGHTS_JSON) as f:
        d = json.load(f)
    if name not in d or name.startswith("_"):
        raise KeyError(f"no pinned weights for {name} in {WEIGHTS_JSON}")
    return d[name]


def commit(name):
    return spec(name)["commit"]


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _local(name, rel):
    """Path of a pinned file in the Hugging Face cache of the pinned commit, without network access; None if it is not there."""
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:                                            # older huggingface_hub
        from huggingface_hub.utils import LocalEntryNotFoundError
    s = spec(name)
    try:
        return hf_hub_download(repo_id=s["repo_id"], filename=rel, revision=s["commit"], cache_dir=CACHE_DIR, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def _verify(name, path, entry):
    key = (os.path.realpath(path), entry["sha256"])
    if key in _verified:
        return
    if os.path.getsize(path) != entry["bytes"]:
        raise RuntimeError(f"{name}: {entry['path']} has {os.path.getsize(path)} bytes, the pinned file has {entry['bytes']}")
    got = sha256_file(path)
    if got != entry["sha256"]:
        raise RuntimeError(f"{name}: {entry['path']} has sha256 {got}, the pinned file has {entry['sha256']}")
    _verified.add(key)


def path(name, rel):
    """Verified local path of the pinned file `rel` of `name`. RuntimeError if it is not in the local cache or differs from the pinned file."""
    s = spec(name)
    entry = next((e for e in s["files"] if e["path"] == rel), None)
    if entry is None:
        raise KeyError(f"{rel} is not a pinned file of {name}")
    p = _local(name, rel)
    if p is None:
        raise RuntimeError(f"{name}: the pinned file {rel} (commit {s['commit'][:8]}) is not in the Hugging Face cache; {HINT}")
    _verify(name, p, entry)
    return p


def main_path(name):
    """The weights file that is passed to the library."""
    return path(name, spec(name)["main"])


def snapshot_dir(name):
    """Folder of the pinned commit; every pinned file of `name` has been found and verified."""
    root = None
    for e in spec(name)["files"]:
        p = path(name, e["path"])
        r = p[:-(len(e["path"]) + 1)]
        if root not in (None, r):
            raise RuntimeError(f"{name}: the pinned files are not in one snapshot folder")
        root = r
    return root


def fetch(name):
    """Download the pinned files of `name` by commit (only those that are not in the cache yet) and verify them. Returns the snapshot folder."""
    from huggingface_hub import hf_hub_download
    s = spec(name)
    for e in s["files"]:
        p = _local(name, e["path"])
        if p is None:
            p = hf_hub_download(repo_id=s["repo_id"], filename=e["path"], revision=s["commit"], cache_dir=CACHE_DIR)
        _verify(name, p, e)
    return snapshot_dir(name)


def record(name):
    """Repository, commit and sha256 of the pinned files (no local paths), for the manifest of a run. None for models that are not pinned here."""
    try:
        s = spec(name)
    except KeyError:
        return None
    return dict(repo_id=s["repo_id"], commit=s["commit"], files={e["path"]: e["sha256"] for e in s["files"]})
