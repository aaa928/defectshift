"""Data-flow and comparison checks of the sensitivity analysis. Run: python -m pytest tests/test_sensitivity.py -q

Training, prediction and search are replaced by recording fakes, and the real scripts (06_benchmark -> 06c_aggregate -> 07a_sensitivity ->
07c_sensitivity_report) are run.
- per arm, the rows of each training call (sets recomputed with the protocol split functions), the seed (the same formula as the
  benchmark), the categorical mode and the source of the hyperparameters
- the search split of the hpo_seed arm (A_dev 80/20 with another seed, different from the benchmark tuning split, no evaluation rows)
- resuming, and fail-closed behavior when a search fails
- comparison: models that were not rerun keep their benchmark values, rerun models get their sensitivity values, and an arm whose
  conditions equal the benchmark agrees exactly in ranks and signs
Fake prediction = mean y of the training rows + model position x 0.001 (different per model, so no ties) + (0.0003 for defaults, 0.0002
for values tuned on the new split). It does not depend on the categorical mode, so the integer arm equals the benchmark exactly.
"""
import importlib.util
import json
import os
import sys

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from defectshift import config, hpo, manifest, regimes, registry  # noqa: E402
from defectshift.candidates import build_one, load_data  # noqa: E402
from defectshift.runio import load_final_domains  # noqa: E402
from defectshift.splits import _strat_split, es_split, get_A, hpo_split, matched, repeat_split, split_B4  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_MEDIAN = "x_time<=2099-05-04 21:18:39"
DOMAINS = [X_MEDIAN, "cat_050"]
# untuned (ml, one-hot), tuned + one-hot, tuned + integer (categorical mode irrelevant), tuned + native categorical + early stopping,
# transformer, dl2, foundation
MODELS = ["LinearRegression", "Ridge", "DecisionTree", "XGBoost", "MLP", "RealMLP", "TabPFN"]
TUNED_HERE = [m for m in MODELS if m in hpo.TUNED]
INTEGER_HERE = ["LinearRegression", "Ridge", "XGBoost"]
SEED = 1999


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def data():
    return load_data()


def _inputs_sha():
    """Current values for the fake run manifests: data and candidate-list hashes (runio.check_same_inputs), per-file code hashes, packages
    and protocol constants (runio.check_compat), GPU name (benchmark.load_fmq). As if the fake runs were made with the current code and
    environment."""
    import torch
    from defectshift.runio import COMPAT_FILES, FMQ_COMPAT_FILES
    root = os.path.realpath(config.REPO_ROOT)
    return dict(data_sha256=manifest.file_sha256(config.DATA_PATH), candidates_sha256=manifest.file_sha256(config.CANDIDATES_CSV),
                code_file_sha256={f: manifest.file_sha256(os.path.join(root, f)) for f in COMPAT_FILES + FMQ_COMPAT_FILES},
                packages=manifest._packages(), protocol=dict(constants=json.loads(json.dumps(manifest.protocol_constants()))),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def _search_rows(df, y, name):
    """Search rows of domain name, with the same functions that benchmark.check_hpo_record recomputes and compares."""
    from defectshift.splits import B_view
    lot = df[config.LOT_COL]
    slot, mA, mB = build_one(df, name)
    a = get_A(slot, df.index[mA], y, lot)
    bv = B_view(slot, df.index[mB], y, lot, "benchmark")
    return slot, {"source": hpo_split(a["A_dev"], y, lot), "target": (bv["B_tr"], bv["B_val"])}


def _hpo_record(model, side, slot, rows, **extra):
    """A search record made as if it came from the final budget and the current search rows."""
    tr, va = rows
    return dict(model=model, side=side, slot=slot, n_trials=config.HPO_N_TRIALS, train_idx_sha256=hpo._idx_sha(tr),
                val_idx_sha256=hpo._idx_sha(va), **extra)


@pytest.fixture(scope="module")
def inputs(data):
    """A fake screening selection (1 temporal + 1 categorical) and a finished benchmark tuning run (params mark their source)."""
    df, _ = data
    d = os.path.join(config.RESULTS_DIR, "screening", "runs", "screening_sensitivity")
    os.makedirs(os.path.join(d, "selection"), exist_ok=True)
    open(os.path.join(d, "rows.csv"), "w").write("x\n1\n")
    json.dump(_inputs_sha(), open(os.path.join(d, "manifest.json"), "w"))
    json.dump(dict(temporal=[X_MEDIAN], categorical=["cat_050"], rows_sha256=manifest.file_sha256(os.path.join(d, "rows.csv")),
                   manifest_sha256=manifest.file_sha256(os.path.join(d, "manifest.json"))),
              open(os.path.join(d, "selection", "final_domains.json"), "w"))
    names, sha = load_final_domains("screening_sensitivity")
    assert names == DOMAINS
    h = os.path.join(config.RESULTS_DIR, "benchmark", "hpo", "hpo_sensitivity")
    os.makedirs(h, exist_ok=True)
    json.dump(dict(args=dict(cat_mode="benchmark", full_run=False, screening_final_domains_sha256=sha), **_inputs_sha()),
              open(os.path.join(h, "manifest.json"), "w"))
    y = load_data()[1]
    for n in DOMAINS:
        slot, rows = _search_rows(df, y, n)
        for m in TUNED_HERE:
            for side in hpo.SIDES:
                hpo.save(_hpo_record(m, side, slot, rows[side], params={"marker": f"{m}:{side}", "alpha": 1.0}, best_score=0.5),
                         hpo.record_prefix("hpo_sensitivity", slot, m, side))
    return df


class Recorder:
    def __init__(self, y):
        self.y, self.fits = y, []

    def fit(self, spec, ctx, fit_idx, es_idx, seed, params=None, max_epochs=None):
        self.fits.append(dict(model=spec.name, fit=set(fit_idx), es=None if es_idx is None else set(es_idx), seed=seed,
                              params=params, cat_mode=ctx.cat_mode, fm_input=ctx.fm_input, max_epochs=max_epochs))
        return regimes.Fitted(spec, None, fit_idx, es_idx, seed, params, 0.0, {})

    def predict(self, f, ctx, idx):
        off = 0.001 * (1 + list(registry.SPECS).index(f.spec.name))
        if f.params is None:
            off += 0.0003
        elif "split" in str(f.params.get("marker", "")):
            off += 0.0002
        return np.full(len(idx), float(self.y.loc[f.fit_idx].mean()) + off)

    def save(self, f, ctx, prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        np.savez(prefix + ".npz", fit_idx=np.asarray(f.fit_idx), es_idx=np.asarray([] if f.es_idx is None else f.es_idx, dtype=np.int64))
        json.dump(dict(seed=f.seed), open(prefix + ".json", "w"))
        if f.spec.route != "foundation":
            open(prefix + (".pt" if f.spec.route == "transformer" else ".joblib"), "w").write("model")

    def install(self, monkeypatch):
        for n in ("fit", "predict", "save"):
            monkeypatch.setattr(regimes, n, getattr(self, n))


def _sets(df, y, name, i):
    lot = df[config.LOT_COL]
    slot, mA, mB = build_one(df, name)
    a = get_A(slot, df.index[mA], y, lot)
    b4 = split_B4(df.index[mB], y, lot)
    a_train, _ = repeat_split(a["A_dev"], y, lot, i)
    n_star, a_match, b_match = matched(a_train, b4["B_tr"], i)
    return dict(slot=slot, A_heldout=a["A_heldout"], A_dev=a["A_dev"], A_match=a_match, B_match=b_match, **b4)


def _fake_tune(calls, fail=()):
    def tune(spec, ctx, train_idx, val_idx, n_trials=config.HPO_N_TRIALS, seed=config.HPO_SEED, max_epochs=None):
        calls.append(dict(model=spec.name, train=set(train_idx), val=set(val_idx), n_trials=n_trials, seed=seed, cat_mode=ctx.cat_mode))
        if spec.name in fail:
            raise RuntimeError("fake search failure")
        return dict(params={"marker": f"{spec.name}:split{SEED}", "alpha": 2.0}, hpo_sec=0.0, best_score=0.4, trials=[],
                    unobserved_num_cols=[], model=spec.name, n_trials=n_trials, max_epochs=max_epochs,   # the record items of the real hpo.tune
                    train_idx_sha256=hpo._idx_sha(train_idx), val_idx_sha256=hpo._idx_sha(val_idx))
    return tune


def _sensitivity_argv(run_id, *extra):
    return ["07a_sensitivity.py", "--screening-run-id", "screening_sensitivity", "--hpo-run-id", "hpo_sensitivity", "--run-id", run_id,
            "--models", ",".join(MODELS), "--hpo-seed-domains", "cat_050", "--trials", "3", *extra]


def test_sensitivity_dataflow_and_report(inputs, data, monkeypatch):
    df, y = data
    lot = df[config.LOT_COL]
    # ---- benchmark reference (the same fake training): 3 repetitions, without the foundation qualification (TabPFN is not judged)
    rec1 = Recorder(y)
    rec1.install(monkeypatch)
    benchmark = _load_script("06_benchmark")
    monkeypatch.setattr(sys, "argv", ["06_benchmark.py", "--screening-run-id", "screening_sensitivity", "--hpo-run-id", "hpo_sensitivity",
                                      "--run-id", "benchmark_sensitivity_base", "--models", ",".join(MODELS), "--repeats", "3"])
    benchmark.main()
    agg = _load_script("06c_aggregate")
    monkeypatch.setattr(sys, "argv", ["06c_aggregate.py", "--run-id", "benchmark_sensitivity_base"])
    agg.main()

    # ---- sensitivity
    rec = Recorder(y)
    rec.install(monkeypatch)
    calls = []
    monkeypatch.setattr(hpo, "tune", _fake_tune(calls))
    mod = _load_script("07a_sensitivity")
    monkeypatch.setattr(sys, "argv", _sensitivity_argv("sensitivity_flow"))
    mod.main()
    n_default = len(TUNED_HERE) * 2 * 3 * 2
    n_integer = len(INTEGER_HERE) * 2 * 3 * 2
    n_hs = len(TUNED_HERE) * 1 * 3 * 2
    assert len(rec.fits) == n_default + n_integer + n_hs
    # search: once per tuned model on cat_050, split = A_dev 80/20 of seed 1999 (different from the benchmark tuning split), the same
    # search seed as the benchmark
    s = _sets(df, y, "cat_050", 0)
    tr, va = _strat_split(s["A_dev"], y, lot, config.HPO_VAL_FRAC, SEED)
    benchmark_tr, benchmark_va = hpo_split(s["A_dev"], y, lot)
    never = set(s["A_heldout"]) | set(s["B_te"]) | set(s["B_screen"]) | set(s["B_tr"]) | set(s["B_val"])
    assert [c["model"] for c in calls] == TUNED_HERE
    for c in calls:
        assert c["train"] == set(tr) and c["val"] == set(va) and c["val"] != set(benchmark_va) and c["seed"] == config.HPO_SEED
        assert not (c["train"] | c["val"]) & never and c["n_trials"] == 3 and c["cat_mode"] == "benchmark"
    # training calls: identify the arm -> check rows and seeds
    seen = {"default": set(), "integer": set(), "hs": set()}
    for f in rec.fits:
        p = f["params"]
        arm = "integer" if f["cat_mode"] == "integer" else ("default" if p is None else "hs")
        if arm == "default":
            assert f["model"] in TUNED_HERE
        elif arm == "integer":
            assert f["model"] in INTEGER_HERE
            assert p == (None if f["model"] == "LinearRegression" else {"marker": f"{f['model']}:source", "alpha": 1.0})
        else:
            assert p["marker"] == f"{f['model']}:split{SEED}" and f["cat_mode"] == "benchmark"
        assert f["fm_input"] == "raw_native"
        regime = i = None
        for reg in ("ID_match", "OOD_match"):
            k = f["seed"] - config.RANDOM_STATE - config.SEED_OFFSET[reg]
            if 0 <= k < 3:
                regime, i = reg, k
        assert regime is not None, f["seed"]
        hits = []
        for name in DOMAINS:
            ss = _sets(df, y, name, i)
            pool = ss["A_match"] if regime == "ID_match" else ss["B_match"]
            if registry.get(f["model"]).uses_es:
                fit, es = es_split(pool, y, lot, i)
                ok = f["fit"] == set(fit) and f["es"] == set(es)
            else:
                ok = f["fit"] == set(pool) and f["es"] is None
            if ok:
                hits.append(name)
        assert len(hits) == 1
        seen[arm].add((hits[0], i, f["model"], regime))
    assert {d for d, *_ in seen["hs"]} == {"cat_050"} and len(seen["hs"]) == n_hs
    assert len(seen["default"]) == n_default and len(seen["integer"]) == n_integer
    # process history: per row, the number of fits attempted earlier in this process accumulates as 0, 1, 2, ... and the PID is the same
    r4 = pd.read_csv(os.path.join(config.RESULTS_DIR, "sensitivity", "runs", "sensitivity_flow", "rows.csv"))
    assert list(r4.proc_prior_fits) == list(range(len(r4))) and r4.proc_pid.nunique() == 1
    # resume: no new training or search
    mod.main()
    assert len(rec.fits) == n_default + n_integer + n_hs and len(calls) == len(TUNED_HERE)

    # ---- comparison
    rep = _load_script("07c_sensitivity_report")
    monkeypatch.setattr(sys, "argv", ["07c_sensitivity_report.py", "--sensitivity-run-id", "sensitivity_flow", "--benchmark-run-id",
                                      "benchmark_sensitivity_base"])
    with pytest.raises(SystemExit, match="not main runs"):          # a test run (3 searches, chosen models) is not a main run -> refused without --smoke
        rep.main()
    monkeypatch.setattr(sys, "argv", ["07c_sensitivity_report.py", "--sensitivity-run-id", "sensitivity_flow", "--benchmark-run-id",
                                      "benchmark_sensitivity_base", "--smoke"])
    rep.main()
    out = os.path.join(config.RESULTS_DIR, "sensitivity", "runs", "sensitivity_flow", "report")
    summ = json.load(open(os.path.join(out, "arm_summary.json")))
    assert summ["metric_recompute_max_abs_diff"] <= 1e-12
    vals = pd.read_csv(os.path.join(out, "arm_values.csv"))
    # integer arm: the fake prediction does not depend on the categorical mode and the hyperparameters equal the benchmark's, so it equals
    # the benchmark exactly
    iv = vals[vals.arm == "integer"]
    assert np.allclose(iv.benchmark, iv.arm_value, rtol=0, atol=1e-12) and set(iv[iv.rerun].model) == set(INTEGER_HERE)
    for q, r in summ["arms"]["integer"].items():
        assert r["n_domains"] == 2 and np.isclose(r["spearman_median"], 1.0)
        assert np.isclose(r["spearman_rerun_median"], 1.0)   # the same ranks among the 3 rerun models too
        assert r["roster_rank_corr_uninformative"] is False and r["rerun_rank_change"]["n_changed"] == 0
        if q.startswith("target_reference_gap"):
            assert r["sign_agree_rerun_median"] == 1.0 and r["consensus_sign_same"] == 2
            assert r["n_sign_flips"] == 0 and r["n_rerun_cells"] == 2 * len(INTEGER_HERE)
    assert len(pd.read_csv(os.path.join(out, "sign_flips.csv"))) == int((vals.rerun & vals.quantity.str.startswith("target_reference_gap")
                                                                        & (np.sign(vals.benchmark) != np.sign(vals.arm_value))).sum())
    # default arm: rerun models get their sensitivity value (mean over repetitions), the others (LinearRegression, TabPFN) keep the
    # benchmark value
    dv = vals[(vals.arm == "default") & (vals.quantity == "ID_match:rmse_ood")].set_index(["candidate", "model"])
    sensitivity_pf = pd.read_csv(os.path.join(out, "sensitivity_per_fit.csv"))
    for (c, m), r in dv.iterrows():
        if m in TUNED_HERE:
            exp = sensitivity_pf[(sensitivity_pf.arm == "default") & (sensitivity_pf.candidate == c) & (sensitivity_pf.model == m)
                                 & (sensitivity_pf.regime == "ID_match")].rmse_ood.mean()
            assert r.rerun and abs(r.arm_value - exp) < 1e-12 and abs(r.arm_value - r.benchmark) > 1e-6
        else:
            assert not r.rerun and r.arm_value == r.benchmark
    # hpo_seed arm: one domain, comparison of the chosen values (marker and alpha both differ, |log10(2/1)|)
    dd = pd.read_csv(os.path.join(out, "arm_domain.csv"))
    assert set(dd[dd.arm == f"hpo_seed{SEED}"].candidate) == {"cat_050"}
    hp = pd.read_csv(os.path.join(out, "hpo_seed_params.csv"))
    assert len(hp) == len(TUNED_HERE) and (hp.share_same == 0).all() and np.allclose(hp.max_abs_log10_ratio, np.log10(2))
    assert np.allclose(hp.best_score_benchmark - hp.best_score_new, 0.1)

    # ---- merging several runs (e.g. NODE and TabNet run separately): the merged report of two runs that split the models must equal the
    # report of one run, and runs that overlap in an (arm, model) are not merged
    g1, g2 = MODELS[:4], MODELS[4:]
    for rid, ms in (("sensitivity_g1", g1), ("sensitivity_g2", g2)):
        argv = _sensitivity_argv(rid)
        argv[argv.index("--models") + 1] = ",".join(ms)
        monkeypatch.setattr(sys, "argv", argv)
        mod.main()
    monkeypatch.setattr(sys, "argv", ["07c_sensitivity_report.py", "--sensitivity-run-id", "sensitivity_g1,sensitivity_g2",
                                      "--benchmark-run-id", "benchmark_sensitivity_base", "--smoke"])
    rep.main()
    out2 = os.path.join(config.RESULTS_DIR, "sensitivity", "runs", "sensitivity_g1", "report_with_sensitivity_g2")
    key = ["arm", "candidate", "model", "quantity"]
    v_one = vals.sort_values(key).reset_index(drop=True)
    v_two = pd.read_csv(os.path.join(out2, "arm_values.csv")).sort_values(key).reset_index(drop=True)
    assert len(v_one) == len(v_two) and (v_one[key] == v_two[key]).all().all()
    assert np.allclose(v_one.arm_value, v_two.arm_value, rtol=0, atol=1e-12) and (v_one.rerun == v_two.rerun).all()
    d_two = pd.read_csv(os.path.join(out2, "arm_domain.csv")).sort_values(["arm", "candidate", "quantity"]).reset_index(drop=True)
    d_one = dd.sort_values(["arm", "candidate", "quantity"]).reset_index(drop=True)
    assert np.allclose(d_one.spearman, d_two.spearman, equal_nan=True)
    assert len(pd.read_csv(os.path.join(out2, "hpo_seed_params.csv"))) == len(TUNED_HERE)
    monkeypatch.setattr(sys, "argv", ["07c_sensitivity_report.py", "--sensitivity-run-id", "sensitivity_flow,sensitivity_g1",
                                      "--benchmark-run-id", "benchmark_sensitivity_base", "--smoke"])
    with pytest.raises(SystemExit, match="is in both sensitivity runs"):
        rep.main()
    # if the domains of the same arm differ between runs, they are not merged into one roster
    argv = _sensitivity_argv("sensitivity_g3", "--only", "cat_050", "--arms", "default")
    argv[argv.index("--models") + 1] = ",".join(g2)
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    monkeypatch.setattr(sys, "argv", ["07c_sensitivity_report.py", "--sensitivity-run-id", "sensitivity_g1,sensitivity_g3",
                                      "--benchmark-run-id", "benchmark_sensitivity_base", "--smoke"])
    with pytest.raises(SystemExit, match="differ between runs"):
        rep.main()
    # completeness: if a run lacks every successful row of one (domain, model) (e.g. skipped after a failed search), it is refused
    # without --partial
    rows_p = os.path.join(config.RESULTS_DIR, "sensitivity", "runs", "sensitivity_g3", "rows.csv")
    rws = pd.read_csv(rows_p)
    gone = g2[0]
    rws[rws.model != gone].to_csv(rows_p, index=False)
    monkeypatch.setattr(sys, "argv", ["07c_sensitivity_report.py", "--sensitivity-run-id", "sensitivity_g3", "--benchmark-run-id",
                                      "benchmark_sensitivity_base", "--smoke"])
    with pytest.raises(SystemExit, match="expected keys"):
        rep.main()
    monkeypatch.setattr(sys, "argv", ["07c_sensitivity_report.py", "--sensitivity-run-id", "sensitivity_g3", "--benchmark-run-id",
                                      "benchmark_sensitivity_base", "--smoke", "--partial"])
    rep.main()
    av3 = pd.read_csv(os.path.join(config.RESULTS_DIR, "sensitivity", "runs", "sensitivity_g3", "report", "arm_values.csv"))
    assert gone not in set(av3[av3.rerun].model)


def test_sensitivity_fail_closed_and_guards(inputs, data, monkeypatch):
    df, y = data
    rec = Recorder(y)
    rec.install(monkeypatch)
    calls = []
    monkeypatch.setattr(hpo, "tune", _fake_tune(calls, fail=("MLP",)))
    mod = _load_script("07a_sensitivity")
    monkeypatch.setattr(sys, "argv", ["07a_sensitivity.py", "--screening-run-id", "screening_sensitivity", "--hpo-run-id", "hpo_sensitivity",
                                      "--run-id", "sensitivity_fail", "--models", "Ridge,MLP", "--hpo-seed-domains", "cat_050", "--trials", "3",
                                      "--arms", f"hpo_seed{SEED}"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 1                                             # MLP search fails -> incomplete exit
    assert {f["model"] for f in rec.fits} == {"Ridge"} and len(rec.fits) == 3 * 2
    rows = pd.read_csv(os.path.join(config.RESULTS_DIR, "sensitivity", "runs", "sensitivity_fail", "hpo_rows.csv"))
    assert dict(zip(rows.model, rows.status)) == {"Ridge": "ok", "MLP": "failed"}
    for bad, msg in ((["--arms", f"hpo_seed{config.HPO_SEED}"], "the same as the benchmark's tuning split seed"), (["--arms", "foo"], "unknown arm"),
                     (["--only", "cat_999"], "not selected domains"),
                     (["--arms", "default,default"], "duplicate")):
        monkeypatch.setattr(sys, "argv", ["07a_sensitivity.py", "--screening-run-id", "screening_sensitivity", "--hpo-run-id",
                                          "hpo_sensitivity", "--run-id", "sensitivity_bad", *bad])
        with pytest.raises(SystemExit, match=msg):
            mod.main()


def test_sensitivity_report_rerun_rank_and_sign_flips():
    """Rank correlation among the rerun models only, and the list of cells whose sign changed: if unchanged models are mixed in, the
    overall correlation can be high while the correlation among the rerun models is low."""
    rep = _load_script("07c_sensitivity_report")
    models = ["m1", "m2", "m3", "m4", "m5", "m6"]
    idx = pd.MultiIndex.from_product([["c1"], models, ["target_reference_gap:gap_rmse_pct"]], names=["candidate", "model", "quantity"])
    v1 = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], index=idx)
    v2 = pd.Series([3.0, 2.0, -1.0, 10.0, 20.0, 30.0], index=idx)      # m1-m3 rerun: ranks reversed, m3 changes sign
    rerun = {("c1", m) for m in models[:3]}
    r = rep.compare(v1, v2, rerun, "integer")[0]
    assert r["n_rerun"] == 3 and r["spearman"] > 0.7 and np.isclose(r["spearman_rerun"], -1.0)
    assert np.isclose(r["sign_agree_rerun"], 2 / 3)
    av = pd.DataFrame(dict(benchmark=v1, arm_value=v2)).reset_index().assign(arm="integer", rerun=lambda d: d.model.isin(models[:3]))
    f = rep.sign_flips(av)
    assert list(f.model) == ["m3"] and f.benchmark.iloc[0] == 3.0 and f.arm_value.iloc[0] == -1.0
    assert rep.sign_flips(pd.DataFrame()).empty
    av2 = av.copy()
    av2.loc[av2.model == "m1", "arm_value"] = np.nan                     # missing values are not counted as sign changes
    assert list(rep.sign_flips(av2).model) == ["m3"]
    r2 = rep.compare(v1, v2, {("c1", "m1"), ("c1", "m2")}, "integer")[0]   # not computed with 2 rerun models
    assert "spearman_rerun" not in r2
    rk = pd.DataFrame(rep.rerun_ranks(v1, v2, rerun, "integer")).set_index("model")
    assert list(rk.index) == ["m1", "m2", "m3"] and (rk.n_models == 6).all()
    assert rk.loc["m1"].rank_benchmark == 1 and rk.loc["m1"].rank_arm == 3 and rk.loc["m3"].rank_change == -2   # m3: rank 3 -> rank 1
    one = pd.DataFrame(rep.rerun_ranks(v1, v2, {("c1", "m3")}, "node_prep_a"))            # an arm where only one model was rerun
    assert len(one) == 1 and one.rank_change.iloc[0] == -2
