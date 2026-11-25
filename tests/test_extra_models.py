"""Checks of the additional models (NODE, TabNet) and the uniform preprocessing code. Only the checks that run without data or a GPU run by
default; the NODE training checks run only when qhoptim and tensorboardX are installed (skipped otherwise).
Run: python -m pytest tests/test_extra_models.py -q
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from defectshift import config, regimes, uniform_prep  # noqa: E402
from defectshift.models import extra_models  # noqa: E402
from defectshift.runio import COMPAT_FILES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extra_hpo_manifest_holds_all_compat_files():
    """The code hashes in the run record of the additional-model tuning stage must include every compatibility file to pass the strict
    check of the benchmark (check_compat). Checked in a new process so that modules loaded by other tests do not mix in."""
    code = ("import importlib.util, os, sys; root = sys.argv[1]; sys.path.insert(0, root); "
            "spec = importlib.util.spec_from_file_location('benchmark_node_tabnet', os.path.join(root, 'scripts', '06a_benchmark_node_tabnet.py')); "
            "m = importlib.util.module_from_spec(spec); sys.argv = [os.path.join(root, 'scripts', '06a_benchmark_node_tabnet.py')]; "
            "spec.loader.exec_module(m); from defectshift import manifest; "
            "print('\\n'.join(os.path.relpath(p, os.path.realpath(root)) for p in manifest.code_files()))")
    out = subprocess.run([sys.executable, "-c", code, ROOT], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert out.returncode == 0, out.stderr[-2000:]
    files = set(out.stdout.split())
    assert [f for f in COMPAT_FILES if f not in files] == []


def test_registry_of_extra_models():
    assert extra_models.spec_of("NODE").route == "node" and extra_models.spec_of("TabNet").route == "tabnet"
    assert extra_models.spec_of("XGBoost").route == "ml"
    assert set(extra_models.tuned_all()) >= {"NODE", "TabNet", "XGBoost", "TabM"}
    assert extra_models.n_trials_final("NODE") == 25 and extra_models.n_trials_final("TabNet") == config.HPO_N_TRIALS
    # NODE uses main setting C (it standardizes by itself), so the uniform preprocessing does not change its input -> not affected; TabNet is
    assert not uniform_prep.affected(extra_models.spec_of("NODE")) and uniform_prep.affected(extra_models.spec_of("TabNet"))


def test_uniform_ctx_fits_on_training_rows_only():
    """The uniform preprocessing transform is fitted on the training rows only: changing values outside the training rows leaves the
    transformed training rows unchanged, and the training rows have mean 0."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"cat_001": rng.randint(0, 3, 40), "num_001": rng.randn(40), "num_002": rng.randn(40)})
    df.loc[[3, 7], "num_001"] = np.nan
    y = pd.Series(rng.rand(40))
    fit_idx = df.index[:30]
    ctx = regimes.Ctx(df, y, ["cat_001"], ["num_001", "num_002"], cat_mode="benchmark", fm_input="raw_native")
    a, _ = uniform_prep.uniform_ctx(ctx, fit_idx)
    df2 = df.copy()
    df2.loc[35:, "num_001"] = 1e6                                   # only rows outside the training rows change
    b, _ = uniform_prep.uniform_ctx(regimes.Ctx(df2, y, ["cat_001"], ["num_001", "num_002"], "benchmark", "raw_native"), fit_idx)
    assert np.allclose(a.df.loc[fit_idx, ["num_001", "num_002"]], b.df.loc[fit_idx, ["num_001", "num_002"]])
    assert np.abs(a.df.loc[fit_idx, ["num_001", "num_002"]].mean()).max() < 1e-9
    assert not a.df[["num_001", "num_002"]].isna().any().any()
    assert (a.df["cat_001"] == df["cat_001"]).all()


def _merge_args(tmp_path, monkeypatch, run_trials, hpo_args):
    merge = _load_script("06b_merge_runs")
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    hdir = tmp_path / "benchmark" / "hpo" / "x_hpo"
    hdir.mkdir(parents=True)
    (hdir / "manifest.json").write_text(json.dumps({"args": hpo_args}))
    a = dict(stage="benchmark_extra", hpo_run_id="x_hpo", max_epochs=None, screening_final_domains_sha256="s", hpo_trials_per_model=run_trials)
    consts = dict(PROTOCOL_VERSION=config.PROTOCOL_VERSION)
    return merge.lineage_problems(a, consts, ["NODE", "TabNet"], "s")


def test_merge_lineage_reads_real_extra_budgets(tmp_path, monkeypatch):
    good_hpo = dict(stage="benchmark_tuning_extra", full_run=True, trials_per_model={"NODE": 25, "TabNet": 30}, max_epochs=None)
    assert _merge_args(tmp_path, monkeypatch, {"NODE": 25, "TabNet": 30}, good_hpo) == []


@pytest.mark.parametrize("run_trials,hpo_args", [
    ({"NODE": 30, "TabNet": 30}, dict(stage="benchmark_tuning_extra", full_run=True, trials_per_model={"NODE": 25, "TabNet": 30}, max_epochs=None)),
    ({"NODE": None, "TabNet": None}, dict(stage="benchmark_tuning_extra", full_run=True, trials_per_model={"NODE": 25, "TabNet": 30}, max_epochs=None)),
    ({"NODE": 25, "TabNet": 30}, dict(stage="benchmark_tuning_extra", full_run=False, trials_per_model={"NODE": 2, "TabNet": 3}, max_epochs=10)),
    ({"NODE": 25, "TabNet": 30}, dict(stage="benchmark_tuning", full_run=True, trials=30, max_epochs=None)),
])
def test_merge_lineage_rejects_nonfinal_extra_runs(tmp_path, monkeypatch, run_trials, hpo_args):
    assert _merge_args(tmp_path, monkeypatch, run_trials, hpo_args) != []


def test_node_fit_save_restore():
    """Small NODE training: the official checkpoint averaging, saving and restored predictions agree, and the initialization sample is drawn
    from the shuffled training rows (deterministic with a fixed seed)."""
    pytest.importorskip("qhoptim")
    pytest.importorskip("tensorboardX")
    from defectshift.models import node
    if not os.path.isdir(node.NODE_DIR):
        pytest.skip("official NODE copy not found")
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({"cat_001": rng.randint(0, 3, n), "num_001": rng.randn(n), "num_002": rng.randn(n)})
    y = pd.Series(df.num_001 * 0.5 + rng.randn(n) * 0.1)
    fit_idx, es_idx, ev = df.index[:200], df.index[200:250], df.index[250:]
    cfg = dict(num_layers=1, total_trees=16, depth=3, tree_dim=2)   # a small configuration outside the grid -> the batch size is given directly
    path = os.path.join(os.environ.get("PYTEST_TMP", "/tmp"), f"node_test_{os.getpid()}.pt")
    import torch
    # training runs on the GPU when one is visible while restoring runs on the CPU, so the two predictions can differ in the last digits
    # (about 1e-8 observed); on the CPU both sides use the same arithmetic and must agree exactly
    tol = 1e-6 if torch.cuda.is_available() else 0.0
    try:
        p, info = node.fit_predict(df, y, ["cat_001"], ["num_001", "num_002"], fit_idx, es_idx, {"ev": ev}, cfg, 42,
                                   max_epochs=2, save_path=path, batch_size=64)
        q = node.predict_saved(df, y, ["cat_001"], ["num_001", "num_002"], fit_idx, 42, path, {"ev": ev}, device="cpu")
        assert np.isfinite(p["ev"]).all() and np.max(np.abs(p["ev"] - q["ev"])) <= tol
        # restoring uses the preprocessing choice made at saving time (not overwritten even if the current default changes)
        saved = node.NORMALIZE
        try:
            node.NORMALIZE = not saved
            r = node.predict_saved(df, y, ["cat_001"], ["num_001", "num_002"], fit_idx, 42, path, {"ev": ev}, device="cpu")
        finally:
            node.NORMALIZE = saved
        assert np.max(np.abs(p["ev"] - r["ev"])) <= tol and info["normalize"] == saved
        assert info["fit_sec"] > 0 and set(info["predict_sec"]) == {"ev"} and info["batch_size"] == 64
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_node_batch_size_list():
    """Fixed list of batch sizes: only 4 grid configurations use 256, the rest of the grid 512, and configurations outside the grid stop (not
    silently set to 512)."""
    from defectshift.models import node as N
    sizes = {tuple(c[k] for k in N.CFG_KEYS): N.batch_size_for(c) for c in N.CONFIGS}
    assert len(sizes) == 25 and sorted(sizes.values()).count(256) == 4 and sorted(sizes.values()).count(512) == 21
    assert {k for k, v in sizes.items() if v == 256} == set(N.REDUCED_BATCH)
    assert all(k[1] == 2048 and k[2] == 8 and k[0] in (2, 4) for k in N.REDUCED_BATCH)
    with pytest.raises(ValueError):
        N.batch_size_for(dict(num_layers=1, total_trees=16, depth=3, tree_dim=2))


def test_node_is_oom():
    """Out-of-memory detection: not only torch's OutOfMemoryError but also cuBLAS and CUDA allocation failures raised as RuntimeError."""
    from defectshift.models import node as N
    assert N.is_oom(RuntimeError("CUDA error: out of memory"))
    assert N.is_oom(RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"))
    assert not N.is_oom(RuntimeError("shape mismatch"))
    assert not N.is_oom(ValueError("out of memory"))
    # stop = out of memory + disk or file write errors (failed checkpoint saves); anything else is a failed trial
    assert N.is_fatal(OSError(28, "No space left on device")) and N.is_fatal(RuntimeError("PytorchStreamWriter failed writing file"))
    assert N.is_fatal(RuntimeError("CUDA error: out of memory")) and not N.is_fatal(ValueError("nan in loss"))


def test_node_grid_progress_resume(tmp_path, monkeypatch):
    """Grid progress record: after a stop on out-of-memory, the finished configurations remain in the record, and calling the same search
    again trains only the remaining configurations. A different header line (rows, seed, ...) stops. Training is replaced by a fake
    function (no GPU or data)."""
    from defectshift.models import node as N
    df = pd.DataFrame({"num_001": np.arange(20.0)})
    y = pd.Series(np.arange(20.0))
    tr, va = df.index[:15], df.index[15:]
    cfgs = N.CONFIGS[:4]
    calls = []

    def fake(df_, y_, cc, nc, fit_idx, es_idx, sets, cfg, seed, max_epochs=None, **kw):
        calls.append(cfg["total_trees"] + 10 * cfg["num_layers"] + cfg["depth"] + cfg["tree_dim"])
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError("CUDA error: out of memory")
        return {"val": y_.loc[sets["val"]].to_numpy() + 0.1 * len(calls)}, dict(n_epochs=3, best_epoch=2, batch_size=N.batch_size_for(cfg))

    monkeypatch.setattr(N, "fit_predict", fake)
    monkeypatch.setattr(N, "free_cuda", lambda: None)
    path = str(tmp_path / "x.grid_progress.jsonl")
    fail_at = 3
    with pytest.raises(RuntimeError):
        N.tune_grid(df, y, [], ["num_001"], tr, va, 7, configs=cfgs, progress_path=path)
    assert len(open(path).read().strip().split("\n")) == 1 + 2          # header line + 2 finished configurations
    calls.clear()
    fail_at = None
    best, det = N.tune_grid(df, y, [], ["num_001"], tr, va, 7, configs=cfgs, progress_path=path)
    assert len(calls) == 2 and det["n_trials_resumed"] == 2 and [t["number"] for t in det["trials"]] == [0, 1, 2, 3]
    assert all(t["state"] == "COMPLETE" for t in det["trials"]) and best == dict(cfgs[0])   # the first configuration has the smallest error
    with pytest.raises(RuntimeError):                                    # another seed = another search -> stops without mixing records
        N.tune_grid(df, y, [], ["num_001"], tr, va, 8, configs=cfgs, progress_path=path)


def test_node_grid_progress_reruns_failed_and_truncated(tmp_path, monkeypatch):
    """Failed trials in the progress record are retrained on resume, and a truncated last line is dropped (with a warning). A broken middle
    line stops."""
    from defectshift.models import node as N
    df = pd.DataFrame({"num_001": np.arange(20.0)})
    y = pd.Series(np.arange(20.0))
    tr, va = df.index[:15], df.index[15:]
    cfgs = N.CONFIGS[:3]
    calls = []

    def fake(df_, y_, cc, nc, fit_idx, es_idx, sets, cfg, seed, max_epochs=None, **kw):
        calls.append(cfg)
        if fail_cfg is not None and cfg == fail_cfg:
            raise ValueError("fake configuration failure")             # a failure that does not stop -> failed trial
        return {"val": y_.loc[sets["val"]].to_numpy()}, dict(n_epochs=1, best_epoch=1, batch_size=N.batch_size_for(cfg))

    monkeypatch.setattr(N, "fit_predict", fake)
    monkeypatch.setattr(N, "free_cuda", lambda: None)
    path = str(tmp_path / "y.grid_progress.jsonl")
    fail_cfg = cfgs[1]
    _, det = N.tune_grid(df, y, [], ["num_001"], tr, va, 7, configs=cfgs, progress_path=path)
    assert [t["state"] for t in det["trials"]] == ["COMPLETE", "FAIL", "COMPLETE"]
    with open(path, "a") as fh:                        # imitate a last line truncated by a forced stop
        fh.write('{"number": 9, "state": "COMP')
    calls.clear()
    fail_cfg = None
    with pytest.warns(UserWarning):
        _, det = N.tune_grid(df, y, [], ["num_001"], tr, va, 7, configs=cfgs, progress_path=path)
    assert calls == [cfgs[1]] and [t["state"] for t in det["trials"]] == ["COMPLETE"] * 3 and det["n_trials_resumed"] == 2
    lines = open(path).read().split("\n")
    assert all(json.loads(s) for s in lines if s.strip())   # the truncated line is removed and the new trial is appended as a whole line
    lines.insert(1, "{broken")                         # a broken middle line stops
    open(path, "w").write("\n".join(lines))
    with pytest.raises(RuntimeError):
        N.tune_grid(df, y, [], ["num_001"], tr, va, 7, configs=cfgs, progress_path=path)


def test_node_prep_standardize_c():
    """NODE preprocessing C: standardization statistics from the training rows only, columns with standard deviation 0 are divided by 1
    (finite values), missing values are imputed before standardization (no NaN), and values stay finite when a constant column takes other
    values in the prediction rows. A (no standardization) follows the official example. Both settings preserve the order."""
    pytest.importorskip("category_encoders")
    from defectshift.models import node as N
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({"cat_001": rng.randint(0, 3, n), "num_001": rng.rand(n) * 0.001, "num_002": rng.rand(n), "num_003": 0.5})
    df.loc[[3, 9], "num_002"] = np.nan
    df.loc[150:, "num_003"] = 0.9                     # constant in the training rows (0-99), other values in the prediction rows
    y = pd.Series(rng.rand(n))
    fit_idx, ev = df.index[:100], df.index[100:]
    c = N.Prep(df, ["cat_001"], ["num_001", "num_002", "num_003"], fit_idx, y.loc[fit_idx], 7, normalize=True)
    a = N.Prep(df, ["cat_001"], ["num_001", "num_002", "num_003"], fit_idx, y.loc[fit_idx], 7, normalize=False)
    assert c.normalize and not a.normalize and c.n_const == 1
    Xc, Xa = c.transform(df.loc[ev]), a.transform(df.loc[ev])
    assert np.isfinite(Xc).all() and np.isfinite(Xa).all()
    df2 = df.copy()
    df2.loc[ev, "num_002"] = 1e6                      # only rows outside the training rows change -> statistics unchanged
    c2 = N.Prep(df2, ["cat_001"], ["num_001", "num_002", "num_003"], fit_idx, y.loc[fit_idx], 7, normalize=True)
    assert np.allclose(c.mu, c2.mu) and np.allclose(c.sd, c2.sd)
    tr = c.transform(df.loc[fit_idx])
    order = np.argsort(df.loc[fit_idx, "num_001"].to_numpy())
    assert np.all(np.diff(tr[order, 0]) >= -1e-6)      # monotone (order preserved)
    assert N.NORMALIZE is True and "standardize" in N.PREPROCESS and "no standardization" in N.PREPROCESS_A


def test_node_fatal_error_classes():
    """Stop decision: only disk errnos (space, quota, input/output) and out-of-memory stop. Missing files, permissions and other exceptions
    are failed trials."""
    import errno
    from defectshift.models import node as N
    assert N.is_fatal(OSError(errno.ENOSPC, "No space left on device"))
    assert N.is_fatal(OSError(errno.EDQUOT, "Disk quota exceeded")) and N.is_fatal(OSError(errno.EIO, "Input/output error"))
    assert not N.is_fatal(FileNotFoundError(errno.ENOENT, "No such file", "checkpoint_best_mse.pth"))
    assert not N.is_fatal(PermissionError(errno.EACCES, "Permission denied"))
    assert N.is_fatal(RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed"))
    wrapped = RuntimeError("save failed")
    wrapped.__cause__ = OSError(errno.ENOSPC, "No space left on device")
    assert N.is_fatal(wrapped) and not N.is_fatal(ValueError("NODE: no finite validation loss"))


def test_node_nonfinite_validation_is_trial_failure(tmp_path, monkeypatch):
    """If the validation loss is never finite, a 'no finite validation loss' ValueError (failed trial) is raised, not a missing-file error.
    Normal data trains through the same path. The real official Trainer path (CPU, small configuration)."""
    pytest.importorskip("qhoptim")
    pytest.importorskip("tensorboardX")
    from defectshift.models import node as N
    if not os.path.isdir(N.NODE_DIR):
        pytest.skip("official NODE copy not found")
    monkeypatch.setenv("DEFECTSHIFT_NODE_TMP", str(tmp_path))
    rng = np.random.RandomState(0)
    X_tr, X_es = rng.randn(120, 4).astype(np.float32), rng.randn(40, 4).astype(np.float32)
    y_tr = pd.Series(rng.randn(120))
    cfg = dict(num_layers=1, total_trees=8, depth=2, tree_dim=2)
    with pytest.raises(ValueError, match="no finite validation loss") as ex:
        N.train(X_tr, y_tr, X_es, pd.Series(np.full(40, np.nan)), cfg, 1, max_epochs=2, device="cpu", batch_size=32)
    assert not N.is_fatal(ex.value)
    model, info = N.train(X_tr, y_tr, X_es, pd.Series(rng.randn(40)), cfg, 1, max_epochs=2, device="cpu", batch_size=32)
    assert info["best_epoch"] >= 1


def test_node_tmp_dir_default_not_root_tmp(monkeypatch):
    """The default temporary folder is under the repository cache (not /tmp on the root disk), or DEFECTSHIFT_NODE_TMP if set."""
    from defectshift import config
    from defectshift.models import node as N
    monkeypatch.delenv("DEFECTSHIFT_NODE_TMP", raising=False)
    monkeypatch.setattr(config, "CACHE_DIR", os.path.join(ROOT, "cache"))   # the default of a real run (not the temporary cache set at the top of the tests)
    d = N.tmp_dir()
    assert d == os.path.join(config.CACHE_DIR, "tmp_node") and not d.startswith("/tmp")


def test_sensitivity_alloc_policy():
    """Allocator policy of the sensitivity runner: NODE and TabNet run separately with expandable_segments on, other models with the default
    (off), and mixing them stops."""
    sensitivity = _load_script("07a_sensitivity")
    on = "expandable_segments:True"
    assert sensitivity.alloc_policy(["NODE", "TabNet"], on) is None and sensitivity.alloc_policy(["NODE"], on) is None
    assert sensitivity.alloc_policy(["XGBoost", "MLP"], None) is None
    assert "separately" in sensitivity.alloc_policy(["NODE", "MLP"], on) and "separately" in sensitivity.alloc_policy(["TabNet", "MLP"], None)
    assert sensitivity.alloc_policy(["NODE"], None) and sensitivity.alloc_policy(["NODE"], "expandable_segments:False")
    assert sensitivity.alloc_policy(["MLP"], on)


def test_extra_runner_sets_allocator_only_as_script():
    """Loading the additional-model runner as a module (load_extra_hpo of the sensitivity runner) does not change the allocator environment
    variable."""
    code = ("import importlib.util, os, sys; root = sys.argv[1]; sys.path.insert(0, root); "
            "spec = importlib.util.spec_from_file_location('06a_benchmark_node_tabnet', os.path.join(root, 'scripts', '06a_benchmark_node_tabnet.py')); "
            "m = importlib.util.module_from_spec(spec); sys.argv = ['x']; spec.loader.exec_module(m); "
            "print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))")
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    out = subprocess.run([sys.executable, "-c", code, ROOT], capture_output=True, text=True, env=dict(env, CUDA_VISIBLE_DEVICES=""))
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip().splitlines()[-1] == "None"


def test_sensitivity_node_prep_a_arm(tmp_path, monkeypatch):
    """NODE preprocessing A arm: NODE only, standardization off only during the arm (restored even on an exception), and the report gives
    the A vs C differences in predictions and metrics on the same evaluation rows."""
    pytest.importorskip("category_encoders")
    from defectshift.models import node as N
    sensitivity = _load_script("07a_sensitivity")
    assert sensitivity.arm_models(sensitivity.NODE_PREP_ARM, ["NODE", "TabNet", "XGBoost"]) == ["NODE"]
    assert sensitivity.NODE_PREP_ARM in sensitivity.SRC_PARAM_ARMS                 # the source search values of C as they are (no retuning)
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"cat_001": rng.randint(0, 3, 60), "num_001": rng.rand(60)})
    y = pd.Series(rng.rand(60))
    idx = df.index[:40]
    with sensitivity.node_preprocess_a():
        assert N.NORMALIZE is False and N.Prep(df, ["cat_001"], ["num_001"], idx, y.loc[idx], 0).normalize is False
    assert N.NORMALIZE is True and N.Prep(df, ["cat_001"], ["num_001"], idx, y.loc[idx], 0).normalize is True
    with pytest.raises(RuntimeError):
        with sensitivity.node_preprocess_a():
            raise RuntimeError("training failure")
    assert N.NORMALIZE is True
    # report: a fake sensitivity run (A predictions) + a fake final-benchmark per_fit (C predictions)
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "res"))
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    rep = _load_script("07c_sensitivity_report")
    yy = pd.Series(np.linspace(0, 1, 30))
    id_idx, ood_idx = np.arange(0, 10), np.arange(10, 30)
    rows, benchmark = [], []
    for i in range(2):
        for reg in ("ID_match", "OOD_match"):
            pc_id, pc_ood = yy[id_idx].to_numpy() + 0.1, yy[ood_idx].to_numpy() + 0.1
            pa_id, pa_ood = pc_id + 0.05 * (i + 1), pc_ood - 0.02
            pre = sensitivity.pred_prefix("sensitivity_x", "cat:cat_050", "NODE", reg, sensitivity.NODE_PREP_ARM, i)
            os.makedirs(os.path.dirname(pre), exist_ok=True)
            np.savez(pre + ".pred.npz", id_idx=id_idx, id_pred=pa_id, ood_idx=ood_idx, ood_pred=pa_ood)
            cp = str(tmp_path / f"c_{i}_{reg}.pred.npz")
            np.savez(cp, id_idx=id_idx, id_pred=pc_id, ood_idx=ood_idx, ood_pred=pc_ood)
            rows.append(dict(arm=sensitivity.NODE_PREP_ARM, slot="cat:cat_050", candidate="cat_050", repeat=i, model="NODE", regime=reg, status="ok"))
            benchmark.append(dict(candidate="cat_050", repeat=i, model="NODE", regime=reg, pred_path=cp))
    os.makedirs(tmp_path / "res" / "sensitivity" / "runs" / "sensitivity_x")
    pd.DataFrame(rows).to_csv(tmp_path / "res" / "sensitivity" / "runs" / "sensitivity_x" / "rows.csv", index=False)
    nd = rep.node_prep_diff(["sensitivity_x"], pd.DataFrame(benchmark), yy)
    assert len(nd) == 4
    r = nd[(nd.repeat == 1) & (nd.regime == "ID_match")].iloc[0]
    assert np.isclose(r.id_mean_abs_pred_diff, 0.10) and np.isclose(r.ood_max_abs_pred_diff, 0.02) and np.isclose(r.ood_pred_corr, 1.0)
    assert np.isclose(r.ood_rmse_A_minus_C, 0.08 - 0.10)        # C is off by +0.1, A by +0.08
    assert np.isclose(r.ood_p95_abs_pred_diff, 0.02) and np.isclose(r.id_p95_abs_pred_diff, 0.10)


def test_sensitivity_node_c_check_arm(tmp_path, monkeypatch):
    """Same-process C check: one NODE cell (first domain x repetition 0 x ID_match), and the report gives the maximum absolute difference
    from the final-benchmark NODE prediction."""
    sensitivity = _load_script("07a_sensitivity")
    a = sensitivity.NODE_C_CHECK_ARM
    assert sensitivity.arm_models(a, ["NODE", "TabNet"]) == ["NODE"] and a in sensitivity.SRC_PARAM_ARMS
    assert sensitivity.arm_repeats(a, 3) == [0] and sensitivity.arm_regimes(a) == ["ID_match"]
    assert sensitivity.arm_repeats("default", 3) == [0, 1, 2] and sensitivity.arm_regimes("default") == ["ID_match", "OOD_match"]
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "res"))
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    rep = _load_script("07c_sensitivity_report")
    id_idx, ood_idx = np.arange(5), np.arange(5, 12)
    p_id, p_ood = np.linspace(0, 1, 5), np.linspace(1, 2, 7)
    pre = sensitivity.pred_prefix("sensitivity_c", "cat:cat_050", "NODE", "ID_match", a, 0)
    os.makedirs(os.path.dirname(pre), exist_ok=True)
    np.savez(pre + ".pred.npz", id_idx=id_idx, id_pred=p_id, ood_idx=ood_idx, ood_pred=p_ood + 1e-7)
    cp = str(tmp_path / "c.pred.npz")
    np.savez(cp, id_idx=id_idx, id_pred=p_id, ood_idx=ood_idx, ood_pred=p_ood)
    os.makedirs(tmp_path / "res" / "sensitivity" / "runs" / "sensitivity_c")
    pd.DataFrame([dict(arm=a, slot="cat:cat_050", candidate="cat_050", repeat=0, model="NODE", regime="ID_match", status="ok",
                       proc_pid=123, proc_start="2000-01-01 00:00:00", proc_prior_fits=57)]).to_csv(
        tmp_path / "res" / "sensitivity" / "runs" / "sensitivity_c" / "rows.csv", index=False)
    benchmark = pd.DataFrame([dict(candidate="cat_050", repeat=0, model="NODE", regime="ID_match", pred_path=cp)])
    out = rep.node_c_check(["sensitivity_c"], benchmark)
    assert len(out) == 1 and out[0]["id_max_abs_pred_diff"] == 0.0 and np.isclose(out[0]["ood_max_abs_pred_diff"], 1e-7)
    assert out[0]["proc_prior_fits"] == 57 and out[0]["proc_pid"] == 123      # the same-process history appears in the check report as it is
