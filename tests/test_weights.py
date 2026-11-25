"""Checks of the pinned foundation-model weights (defectshift/models/weights.py, environment/weights.json). Run: python -m pytest tests/test_weights.py -q
A fake Hugging Face cache with tiny files stands in for the real weights; nothing is downloaded and no model is loaded or fitted.
"""
import hashlib
import json
import os
import re

import pytest

from defectshift import config
from defectshift.models import weights

COMMIT = "a" * 40
OTHER = "b" * 40


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _put(cache, repo, commit, rel, data):
    p = os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots", commit, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return p


@pytest.fixture
def fake(tmp_path, monkeypatch):
    cache = str(tmp_path / "hub")
    files = {"model.bin": b"weights", "sub/config.json": b"{}"}
    spec = {"_note": "x", "Fake": dict(repo_id="org/fake", commit=COMMIT, main="model.bin",
                                       files=[dict(path=k, sha256=_sha(v), bytes=len(v)) for k, v in files.items()])}
    sp = tmp_path / "weights.json"
    sp.write_text(json.dumps(spec))
    monkeypatch.setattr(weights, "WEIGHTS_JSON", str(sp))
    monkeypatch.setattr(weights, "CACHE_DIR", cache)
    monkeypatch.setattr(weights, "_verified", set())
    return cache, files


def test_path_returns_verified_file(fake):
    cache, files = fake
    for k, v in files.items():
        _put(cache, "org/fake", COMMIT, k, v)
    p = weights.main_path("Fake")
    with open(p, "rb") as f:
        assert f.read() == b"weights"
    root = weights.snapshot_dir("Fake")
    assert os.path.isfile(os.path.join(root, "sub", "config.json")) and root.endswith(COMMIT)


def test_missing_file_stops_with_a_hint(fake):
    """Only the local cache is read (local_files_only), so a missing file cannot trigger a download; the error names the fetch script."""
    with pytest.raises(RuntimeError, match="00_fetch_weights.py"):
        weights.main_path("Fake")


def test_different_content_is_refused(fake):
    cache, files = fake
    _put(cache, "org/fake", COMMIT, "model.bin", b"changed-longer")    # another size: refused by the size check
    with pytest.raises(RuntimeError, match="bytes"):
        weights.main_path("Fake")
    _put(cache, "org/fake", COMMIT, "model.bin", b"weighTs")           # same size, other content
    with pytest.raises(RuntimeError, match="sha256"):
        weights.main_path("Fake")


def test_no_fallback_to_another_commit(fake):
    cache, files = fake
    for k, v in files.items():
        _put(cache, "org/fake", OTHER, k, v)                            # right content, but another commit (e.g. a newer main)
    with pytest.raises(RuntimeError, match="not in the Hugging Face cache"):
        weights.main_path("Fake")


def test_fetch_downloads_by_commit_and_verifies(fake, monkeypatch):
    import huggingface_hub
    cache, files = fake
    calls = []

    def fake_download(repo_id, filename, revision=None, cache_dir=None, local_files_only=False, **k):
        calls.append((repo_id, filename, revision, local_files_only))
        if local_files_only:
            from huggingface_hub.errors import LocalEntryNotFoundError
            p = os.path.join(cache, "models--org--fake", "snapshots", revision, filename)
            if not os.path.exists(p):
                raise LocalEntryNotFoundError("absent")
            return p
        return _put(cache, repo_id, revision, filename, files[filename])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    root = weights.fetch("Fake")
    assert root.endswith(COMMIT)
    assert all(rev == COMMIT for _, _, rev, _ in calls), "files must be requested by the pinned commit, never by a branch"
    assert [c for c in calls if not c[3]], "the missing files were downloaded"
    n = len(calls)
    weights.fetch("Fake")                                                # second run: everything is in the cache, nothing is downloaded
    assert not [c for c in calls[n:] if not c[3]]


def test_record_has_no_local_paths(fake):
    r = weights.record("Fake")
    assert r["commit"] == COMMIT and set(r["files"]) == {"model.bin", "sub/config.json"}
    assert "/tmp" not in json.dumps(r) and weights.record("Kumo") is None


def test_weights_json_of_the_repository():
    with open(os.path.join(config.REPO_ROOT, "environment", "weights.json")) as f:
        d = json.load(f)
    names = [k for k in d if not k.startswith("_")]
    assert names == ["TabPFN", "TabICL", "TabFM", "TabDPT", "EXAONE"]
    for n in names:
        s = d[n]
        assert re.fullmatch(r"[0-9a-f]{40}", s["commit"]) and "/" in s["repo_id"]
        assert s["main"] in {e["path"] for e in s["files"]}
        for e in s["files"]:
            assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) and isinstance(e["bytes"], int) and e["bytes"] > 0


def test_foundation_hands_the_pinned_files_to_the_libraries():
    """The library defaults download the newest `main` file; every call must hand over the pinned file instead (source check, nothing is imported)."""
    with open(os.path.join(config.REPO_ROOT, "defectshift", "models", "foundation.py")) as f:
        src = f.read()
    for needle in ('model_path=weights.main_path("TabPFN")', 'model_path=weights.main_path("TabICL"), allow_auto_download=False',
                   'model_weight_path=weights.main_path("TabDPT")', 'checkpoint_path=weights.snapshot_dir("TabFM")',
                   'revision=weights.commit("EXAONE")', "EXAONETABULAR_REGRESSOR_WEIGHTS", "_check_kumo_weights(strict=True)"):
        assert needle in src, needle
    assert "TabICLRegressor(device=DEVICE, random_state=seed)" not in src and "TabDPTRegressor(device=DEVICE, verbose=False)" not in src
    with open(os.path.join(config.REPO_ROOT, "scripts", "06_benchmark.py")) as f:
        bench = f.read()
    assert "foundation_weights=" in bench


def test_kumo_weights_snapshot_guard(monkeypatch):
    """The Hub tag of the Kumo weights can be moved: the snapshot of the tag in the local cache must equal the one of the paper once the weights are loaded."""
    from defectshift.models import foundation as fm
    good = fm.KUMO_WEIGHTS_SNAPSHOT
    monkeypatch.setattr(fm, "_hf_snapshot", lambda repo_dir, ref="main": good)
    fm._check_kumo_weights(strict=True)
    monkeypatch.setattr(fm, "_hf_snapshot", lambda repo_dir, ref="main": "f" * 40)
    with pytest.raises(RuntimeError, match="the paper used"):
        fm._check_kumo_weights(strict=False)
    monkeypatch.setattr(fm, "_hf_snapshot", lambda repo_dir, ref="main": None)
    fm._check_kumo_weights(strict=False)                         # not downloaded yet: allowed before the first load
    with pytest.raises(RuntimeError, match="not found"):
        fm._check_kumo_weights(strict=True)


def test_qualification_must_come_from_the_pinned_weights():
    """benchmark.check_fmq_weights: the manifest of a qualification run must hold the record of the weights that environment/weights.json pins now."""
    from defectshift import benchmark
    from defectshift.models import foundation as fm
    models = ["TabPFN", "TabICL", "TabFM", "Kumo", "TabDPT", "EXAONE"]
    rec = {m: json.loads(json.dumps(fm.weights_record(m))) for m in models}          # as written to and read from manifest.json
    assert rec["Kumo"] is None and rec["TabPFN"]["commit"]                          # Kumo is pinned by its Hub tag, not here
    benchmark.check_fmq_weights(dict(models=models, foundation_weights=rec), "q")
    with pytest.raises(SystemExit, match="no foundation_weights"):
        benchmark.check_fmq_weights(dict(models=models), "q")
    other = json.loads(json.dumps(rec))
    other["TabDPT"]["commit"] = "0" * 40
    with pytest.raises(SystemExit, match=r"weights of \['TabDPT'\] differ from the pinned weights"):
        benchmark.check_fmq_weights(dict(models=models, foundation_weights=other), "q")
    first_file = next(iter(other["TabICL"]["files"]))
    other = json.loads(json.dumps(rec))
    other["TabICL"]["files"][first_file] = "0" * 64
    with pytest.raises(SystemExit, match=r"weights of \['TabICL'\] differ from the pinned weights"):
        benchmark.check_fmq_weights(dict(models=models, foundation_weights=other), "q")
    partial = {m: v for m, v in rec.items() if m != "EXAONE"}                        # a model of the run without a record
    with pytest.raises(SystemExit, match=r"weights of \['EXAONE'\] differ from the pinned weights"):
        benchmark.check_fmq_weights(dict(models=models, foundation_weights=partial), "q")


def test_kumo_instance_is_cached_only_after_the_weights_check(monkeypatch):
    """A failed snapshot check must not leave a cached instance behind: every later fit in the same process must fail the check again."""
    import sys
    import types

    import numpy as np

    from defectshift.models import foundation as fm
    monkeypatch.setitem(sys.modules, "sdm", types.SimpleNamespace(models=types.SimpleNamespace(KumoTabular=lambda **kw: object())))
    monkeypatch.setattr(fm, "_instances", {})

    def bad(strict):
        raise RuntimeError("the snapshot of the tag differs")
    monkeypatch.setattr(fm, "_check_kumo_weights", bad)
    wrapper = fm._Kumo(cat_idx=[], seed=0)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="snapshot of the tag differs"):
            wrapper.fit(np.zeros((2, 1)), np.zeros(2))
        assert "Kumo" not in fm._instances
    monkeypatch.setattr(fm, "_check_kumo_weights", lambda strict: None)
    with pytest.raises(AttributeError):                         # the fake library has no TableTensor: the fit stops after the instance is cached
        wrapper.fit(np.zeros((2, 1)), np.zeros(2))
    assert "Kumo" in fm._instances


def test_kumo_error_message_tells_how_to_restore(monkeypatch):
    from defectshift.models import foundation as fm
    monkeypatch.setattr(fm, "_hf_snapshot", lambda repo_dir, ref="main": "f" * 40)
    with pytest.raises(RuntimeError) as e:
        fm._check_kumo_weights(strict=True)
    assert fm.KUMO_WEIGHTS_SNAPSHOT in str(e.value) and f"refs/{fm.KUMO_WEIGHTS_REVISION}" in str(e.value) and "models--nvidia--Kumo-Tabular" in str(e.value)
