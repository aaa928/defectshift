"""Data-flow checks of the benchmark. Run: python -m pytest tests/test_benchmark.py -q

The training, prediction and saving functions are replaced by recording fakes, the real benchmark scripts
(scripts/06_benchmark.py, 04_tune_hyperparameters.py, 05_qualify_foundation.py) are run as they are, and the rows passed to
each call are compared with the sets recomputed with the protocol split functions. No model training, no GPU.
Selection results, search records and qualification verdicts are faked in a temporary results folder (only the check of the real
final_domains.json reads the repository results).
"""
import importlib.util
import json
import os
import sys

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from defectshift import config, hpo, manifest, metrics, regimes, registry, selection  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_candidates, load_data  # noqa: E402
from defectshift.runio import load_final_domains  # noqa: E402
from defectshift.splits import es_split, get_A, hpo_split, matched, repeat_split, split_B4  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_MEDIAN = "x_time<=2099-05-04 21:18:39"
DOMAINS = [X_MEDIAN, "cat_050"]                    # 1 temporal + 1 categorical (load_final_domains order = temporal, then categorical)
# one of each combination of route, tuning and early stopping: untuned (ml), tuned (ml), tuned + early stopping (ml), transformer, dl2,
# foundation
MODELS = ["LinearRegression", "Ridge", "XGBoost", "MLP", "RealMLP", "TabPFN"]
FINAL_CATEGORICAL = ["cat_019", "cat_039", "cat_045", "cat_005", "cat_050"]   # the final selection of the screening


@pytest.fixture(scope="module")
def data():
    return load_data()


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_screening(run_id, categorical, temporal):
    """Imitation of a screening run whose selection is finished: rows.csv, manifest.json and selection/final_domains.json with their hashes."""
    d = os.path.join(config.RESULTS_DIR, "screening", "runs", run_id)
    os.makedirs(os.path.join(d, "selection"), exist_ok=True)
    with open(os.path.join(d, "rows.csv"), "w") as f:
        f.write("x\n1\n")
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(_inputs_sha(), f)
    final = dict(temporal=temporal, categorical=categorical, rows_sha256=manifest.file_sha256(os.path.join(d, "rows.csv")),
                 manifest_sha256=manifest.file_sha256(os.path.join(d, "manifest.json")))
    with open(os.path.join(d, "selection", "final_domains.json"), "w") as f:
        json.dump(final, f)


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


def _final_sha():
    return load_final_domains("screening_fake")[1]


def _fake_hpo(run_id, df, names, models, cat_mode="benchmark", final_sha=None):
    """Imitation of a finished benchmark tuning run. params carry a (model, side) marker so that the record used can be recognized."""
    d = os.path.join(config.RESULTS_DIR, "benchmark", "hpo", run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(dict(args=dict(cat_mode=cat_mode, full_run=False, screening_final_domains_sha256=final_sha or _final_sha()),
                       **_inputs_sha()), f)
    y = load_data()[1]
    for n in names:
        slot, rows = _search_rows(df, y, n)
        for m in models:
            if m in hpo.TUNED:
                for side in hpo.SIDES:
                    hpo.save(_hpo_record(m, side, slot, rows[side], params={"marker": f"{m}:{side}"}),
                             hpo.record_prefix(run_id, slot, m, side))


def _fake_fmq(run_id, passed, models=("TabPFN", "TabICL", "TabFM", "Kumo", "TabDPT", "EXAONE"), final_sha=None):
    d = os.path.join(config.RESULTS_DIR, "benchmark", "fm_qualify", run_id)
    os.makedirs(d, exist_ok=True)
    res = {m: dict(passed=m in passed) for m in models}
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(dict(results=res), f)
    from defectshift.models import foundation as fm
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(dict(args=dict(screening_final_domains_sha256=final_sha or _final_sha(), models=list(models),
                                 foundation_weights={m: fm.weights_record(m) for m in models}), **_inputs_sha()), f)


class Recorder:
    """Replacement of regimes.fit / predict / save / load. Records the rows passed to every training and prediction call."""

    def __init__(self, y):
        self.y, self.fits, self.preds = y, [], []

    def fit(self, spec, ctx, fit_idx, es_idx, seed, params=None, max_epochs=None):
        self.fits.append(dict(model=spec.name, fit=set(fit_idx), es=None if es_idx is None else set(es_idx),
                              seed=seed, params=params, features=list(ctx.feature_cols), cat_mode=ctx.cat_mode,
                              fm_input=ctx.fm_input, max_epochs=max_epochs))
        return regimes.Fitted(spec, None, fit_idx, es_idx, seed, params or {}, 0.0, {})

    def predict(self, f, ctx, idx):
        self.preds.append(dict(model=f.spec.name, seed=f.seed, rows=set(idx)))
        return np.full(len(idx), float(self.y.loc[f.fit_idx].mean()))

    def save(self, f, ctx, prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        np.savez(prefix + ".npz", fit_idx=np.asarray(f.fit_idx),                     # the same keys as regimes.save
                 es_idx=np.asarray([] if f.es_idx is None else f.es_idx, dtype=np.int64))
        with open(prefix + ".json", "w") as fh:
            json.dump(dict(seed=f.seed), fh)
        if f.spec.route != "foundation":
            with open(prefix + (".pt" if f.spec.route == "transformer" else ".joblib"), "w") as fh:
                fh.write("model")

    def load(self, spec, ctx, prefix):
        idx = pd.Index(np.load(prefix + ".npz")["fit_idx"])
        return regimes.Fitted(spec, None, idx, None, json.load(open(prefix + ".json"))["seed"], {}, 0.0, {})

    def install(self, monkeypatch):
        for n in ("fit", "predict", "save", "load"):
            monkeypatch.setattr(regimes, n, getattr(self, n))


def _protocol_sets(df, y, name, i):
    """Row sets set by the protocol in repetition i of domain name (recomputed directly by the test)."""
    lot = df[config.LOT_COL]
    slot, mA, mB = build_one(df, name)
    a = get_A(slot, df.index[mA], y, lot)
    b4 = split_B4(df.index[mB], y, lot)
    a_train, a_val = repeat_split(a["A_dev"], y, lot, i)
    n_star, a_match, b_match = matched(a_train, b4["B_tr"], i)
    return dict(slot=slot, A_heldout=a["A_heldout"], A_dev=a["A_dev"], A_train=a_train, A_val=a_val, n_star=n_star,
                A_match=a_match, B_match=b_match, **b4)


def _decode(c, repeats):
    """Training call record c -> (regime, i). Model seed = RANDOM_STATE + SEED_OFFSET[regime] + i. OOD_match_tuned has the same seed as
    OOD_match, so it is told apart by the source of the hyperparameters (the target marker)."""
    seed, params = c["seed"], c.get("params") or {}
    for regime, off in config.SEED_OFFSET.items():
        if regime == "OOD_match_tuned":
            continue
        i = seed - config.RANDOM_STATE - off
        if 0 <= i < repeats:
            if regime == "OOD_match" and str(params.get("marker", "")).endswith(":target"):
                return "OOD_match_tuned", i
            return regime, i
    raise AssertionError(f"seed that cannot be decoded {seed}")


def _argv(run_id, *extra):
    return ["06_benchmark.py", "--screening-run-id", "screening_fake", "--hpo-run-id", "hpo_fake", "--fmq-run-id", "fmq_fake",
            "--run-id", run_id, *extra]


@pytest.fixture(scope="module")
def benchmark_inputs(data):
    df, _ = data
    _fake_screening("screening_fake", categorical=["cat_050"], temporal=[X_MEDIAN])
    assert load_final_domains("screening_fake")[0] == DOMAINS
    _fake_hpo("hpo_fake", df, DOMAINS, MODELS)
    _fake_fmq("fmq_fake", passed={"TabPFN", "TabICL", "TabDPT"})     # TabFM is not judged, so it is included even if it fails
    return df


# ------------------------------------------------------------------ real selection result
def test_real_final_domains_load(monkeypatch):
    """The selection result of the main screening in the repository results: 7 domains (2 prespecified temporal + 5 categorical), the hash
    check passes, equal to the final list."""
    real = os.path.join(ROOT, "results")
    if not os.path.exists(os.path.join(real, "screening", "runs", "screening_main", "selection", "final_domains.json")):
        pytest.skip("selection result of screening_main not found")
    monkeypatch.setattr(config, "RESULTS_DIR", real)
    domains, sha = load_final_domains("screening_main")
    final = json.load(open(os.path.join(real, "screening", "runs", "screening_main", "selection", "final_domains.json")))
    assert len(domains) == 7 and len(set(domains)) == 7 and len(sha) == 64
    assert final["temporal"] == selection.temporal_primary(load_candidates()) == domains[:2]
    assert final["categorical"] == FINAL_CATEGORICAL == domains[2:]
    assert set(domains) <= set(load_candidates())


# ------------------------------------------------------------------ benchmark main path
def test_benchmark_dataflow(benchmark_inputs, data, monkeypatch):
    """Row sets, early-stopping rows, source of the hyperparameters, seeds, prediction rows, features, result rows and artifacts, checked
    exactly per regime."""
    df, y = data
    mod = _load_script("06_benchmark")
    rec = Recorder(y)
    rec.install(monkeypatch)
    repeats = config.N_REPEATS_BENCHMARK
    monkeypatch.setattr(sys, "argv", _argv("benchmark_flow", "--models", ",".join(MODELS)))
    mod.main()

    n_reg = {m: 4 if m in hpo.TUNED else 3 for m in MODELS}
    assert len(rec.fits) == len(DOMAINS) * repeats * sum(n_reg.values())
    assert len(rec.preds) == 2 * len(rec.fits)
    lot = df[config.LOT_COL]
    rows = pd.read_csv(os.path.join(config.RESULTS_DIR, "benchmark", "runs", "benchmark_flow", "rows.csv"))
    assert len(rows) == len(rec.fits) and (rows.status == "ok").all()
    assert not rows.duplicated(list(mod.KEY)).any()
    model_dir = os.path.join(config.CACHE_DIR, "benchmark", "runs", "benchmark_flow", "models")

    # training calls happen in domain order: per domain one block of (models x repetitions x regimes)
    per_domain = len(rec.fits) // len(DOMAINS)
    for d_i, name in enumerate(DOMAINS):
        cat_cols, num_cols = feature_sets(df, name)
        fits = rec.fits[d_i * per_domain:(d_i + 1) * per_domain]
        preds = rec.preds[2 * d_i * per_domain:2 * (d_i + 1) * per_domain]
        sets = {i: _protocol_sets(df, y, name, i) for i in range(repeats)}
        s0 = sets[0]
        never = set(s0["A_heldout"]) | set(s0["B_te"]) | set(s0["B_screen"])
        seen_keys, by_key = set(), {}
        for k, c in enumerate(fits):
            regime, i = _decode(c, repeats)
            s, spec = sets[i], registry.get(c["model"])
            by_key[(c["model"], regime, i)] = c
            seen_keys.add((c["model"], regime, i))
            used = c["fit"] | (c["es"] or set())
            # evaluation rows and B_screen are in no training or early stopping
            assert not used & never, f"{name} {c['model']} {regime} r{i}: evaluation rows or B_screen used in training"
            # exact rows per regime
            if regime == "ID_full":
                assert c["fit"] == set(s["A_train"])
                assert c["es"] == (set(s["A_val"]) if spec.uses_es else None)
            else:   # ID_match = A_match, OOD_match and OOD_match_tuned = B_match (same rows, same internal 80/20)
                if regime == "OOD_match_tuned":
                    assert c["model"] in hpo.TUNED
                pool = s["A_match"] if regime == "ID_match" else s["B_match"]
                assert len(pool) == s["n_star"]
                if spec.uses_es:
                    fit_, es_ = es_split(pool, y, lot, i)
                    assert c["fit"] == set(fit_) and c["es"] == set(es_)
                    assert not set(lot.loc[fit_]) & set(lot.loc[es_])          # internal early stopping is also by lot
                else:
                    assert c["fit"] == set(pool) and c["es"] is None
            # A_val only in ID_full early stopping, B_val in no final training (target search validation only)
            if regime != "ID_full":
                assert not used & set(s["A_val"])
            assert not used & set(s["B_val"])
            assert c["cat_mode"] == "benchmark" and c["fm_input"] == "raw_native"
            # source of the hyperparameters: tuned models use source (ID_full, ID_match, OOD_match) and target (OOD_match_tuned), the
            # others the defaults
            if c["model"] in hpo.TUNED:
                side = "target" if regime == "OOD_match_tuned" else "source"
                assert c["params"] == {"marker": f"{c['model']}:{side}"}
            else:
                assert c["params"] is None
            # features: no lot column or split-defining columns, categorical representation benchmark
            assert c["features"] == cat_cols + num_cols and config.LOT_COL not in c["features"]
            if name.startswith("cat_"):
                assert name not in c["features"]
            assert not {"x_time", "y_time", "y"} & set(c["features"]) and c["cat_mode"] == "benchmark"
            # prediction: per training A_heldout (ID) then B_te (OOD), no other rows are predicted
            p_id, p_ood = preds[2 * k], preds[2 * k + 1]
            assert p_id["seed"] == p_ood["seed"] == c["seed"] and p_id["model"] == p_ood["model"] == c["model"]
            assert p_id["rows"] == set(s["A_heldout"]) and p_ood["rows"] == set(s["B_te"])
        # H_tune: OOD_match_tuned has the same training rows, early-stopping rows and seed as OOD_match of the same repetition; only the
        # hyperparameters differ
        for (m, r, i), c in by_key.items():
            if r == "OOD_match_tuned":
                o = by_key[(m, "OOD_match", i)]
                assert (c["fit"], c["es"], c["seed"]) == (o["fit"], o["es"], o["seed"]) and c["params"] != o["params"]
        expected = {(m, r, i) for m in MODELS for r in mod.REGIMES for i in range(repeats)
                    if r != "OOD_match_tuned" or m in hpo.TUNED}
        assert seen_keys == expected
        # result rows: n*, row counts, lot exposure, saved artifacts
        rd = rows[rows.candidate == name]
        assert len(rd) == len(expected) and set(zip(rd.model, rd.regime, rd.repeat)) == expected
        for i in range(repeats):
            ri = rd[rd.repeat == i]
            assert (ri.n_star == sets[i]["n_star"]).all()
            nm = ri[ri.regime.isin(["ID_match", "OOD_match"]) & ~ri.model.map(lambda m: registry.get(m).uses_es)]
            assert (nm.n_fit == sets[i]["n_star"]).all()                    # both regimes train on the same number of rows (n*)
        assert (rd.n_id == len(s0["A_heldout"])).all() and (rd.n_ood == len(s0["B_te"])).all()
        is_id = rd.regime.isin(["ID_full", "ID_match"])
        assert (rd[is_id].lot_exposure_id == 0).all()                       # A_heldout lots do not overlap A_dev
        assert (rd[~is_id].lot_exposure_ood == 0).all()                     # the four subsets of B are also by lot (B_te lots not in B_tr or B_val)
        assert set(rd.seed) == {config.RANDOM_STATE + config.SEED_OFFSET[r] + i for r in mod.REGIMES for i in range(repeats)}
        for r in rd.to_dict("records"):
            prefix = mod.artifact_prefix(model_dir, r["slot"], r["model"], r["regime"], r["repeat"])
            pr = np.load(prefix + ".pred.npz")
            assert set(pr["id_idx"]) == set(s0["A_heldout"]) and set(pr["ood_idx"]) == set(s0["B_te"])
            assert len(pr["id_pred"]) == len(pr["id_idx"]) and len(pr["ood_pred"]) == len(pr["ood_idx"])
    man = json.load(open(os.path.join(config.RESULTS_DIR, "benchmark", "runs", "benchmark_flow", "manifest.json")))
    assert man["args"]["foundation_excluded"] == ["Kumo", "EXAONE"]          # only the failed ones among the 3 judged models are excluded
    assert {"TabPFN", "TabICL", "TabFM", "TabDPT"} <= set(man["args"]["roster"])
    assert man["args"]["full_run"] is False


def test_benchmark_repeats_share_screening_draws(data):
    """A_train, A_match and B_match of repetitions 0-2 are the same draws as in the screening (same functions and seeds; B_tr is the same
    four-way split regardless of the stage)."""
    df, y = data
    from defectshift.splits import B_view
    lot = df[config.LOT_COL]
    slot, mA, mB = build_one(df, "cat_050")
    b_benchmark = B_view(slot, df.index[mB], y, lot, "benchmark")
    b_screening = B_view(slot, df.index[mB], y, lot, "screening")
    assert b_benchmark["B_tr"].equals(b_screening["B_tr"]) and set(b_benchmark) == {"B_tr", "B_val", "B_te"}
    draws = [_protocol_sets(df, y, "cat_050", i) for i in range(config.N_REPEATS_BENCHMARK)]
    for i, d in enumerate(draws):
        assert len(d["A_match"]) == len(d["B_match"]) == d["n_star"] == min(len(d["A_train"]), len(d["B_tr"]))
        for j in range(i):
            assert set(d["A_train"]) != set(draws[j]["A_train"])          # a different split per repetition


def test_benchmark_resume_and_guards(benchmark_inputs, data, monkeypatch):
    df, y = data
    mod = _load_script("06_benchmark")
    rec = Recorder(y)
    rec.install(monkeypatch)
    small = ("--only", "cat_050", "--models", "Ridge,XGBoost,TabPFN", "--repeats", "2")
    monkeypatch.setattr(sys, "argv", _argv("benchmark_resume", *small))
    mod.main()
    n_first = len(rec.fits)
    assert n_first == 2 * (4 + 4 + 3)
    rows_path = os.path.join(config.RESULTS_DIR, "benchmark", "runs", "benchmark_resume", "rows.csv")
    # again with the same conditions: no new training
    mod.main()
    assert len(rec.fits) == n_first and len(pd.read_csv(rows_path)) == n_first
    # if one artifact (predictions) breaks, only that key is retrained
    slot = build_one(df, "cat_050")[0]
    model_dir = os.path.join(config.CACHE_DIR, "benchmark", "runs", "benchmark_resume", "models")
    os.remove(mod.artifact_prefix(model_dir, slot, "XGBoost", "OOD_match", 1) + ".pred.npz")
    mod.main()
    assert len(rec.fits) == n_first + 1
    regime, i = _decode(rec.fits[-1], 2)
    assert (rec.fits[-1]["model"], regime, i) == ("XGBoost", "OOD_match", 1)
    # with changed conditions the same run-id cannot be resumed
    monkeypatch.setattr(sys, "argv", _argv("benchmark_resume", "--only", "cat_050", "--models", "Ridge", "--repeats", "2"))
    with pytest.raises(RuntimeError, match="cannot resume"):
        mod.main()
    # wrong inputs are refused before training
    for extra, msg in [(("--only", "cat_001"), "not selected domains"),
                       (("--models", "Kumo"), "outside the roster"),
                       (("--repeats", "6"), "--repeats"),
                       (("--regimes", "ID_full,B_te"), "--regimes")]:
        monkeypatch.setattr(sys, "argv", _argv("benchmark_bad", *extra))
        with pytest.raises(SystemExit, match=msg):
            mod.main()
    n = len(rec.fits)
    monkeypatch.setattr(sys, "argv", ["06_benchmark.py", "--screening-run-id", "screening_fake", "--hpo-run-id", "hpo_missing",
                                      "--fmq-run-id", "fmq_fake", "--run-id", "benchmark_bad2"])
    _fake_hpo("hpo_missing", df, [], [])
    with pytest.raises(SystemExit, match="search records missing"):
        mod.main()
    _fake_hpo("hpo_integer", df, DOMAINS, MODELS, cat_mode="integer")
    monkeypatch.setattr(sys, "argv", ["06_benchmark.py", "--screening-run-id", "screening_fake", "--hpo-run-id", "hpo_integer",
                                      "--fmq-run-id", "fmq_fake", "--run-id", "benchmark_bad3", "--models", ",".join(MODELS)])
    with pytest.raises(SystemExit, match="categorical representation"):
        mod.main()
    # whether the input runs come from the same screening selection, and whether the qualification has all 3 judged models
    _fake_hpo("hpo_other_screening", df, DOMAINS, MODELS, final_sha="0" * 64)
    _fake_fmq("fmq_other_screening", passed={"TabDPT"}, final_sha="0" * 64)
    _fake_fmq("fmq_partial", passed={"TabDPT"}, models=("TabDPT", "Kumo"))      # no EXAONE
    _fake_fmq("fmq_oldcode", passed={"TabDPT"})                                   # a qualification with other foundation wrapper code
    p = os.path.join(config.RESULTS_DIR, "benchmark", "fm_qualify", "fmq_oldcode", "manifest.json")
    m_ = json.load(open(p))
    m_["code_file_sha256"]["defectshift/models/foundation.py"] = "0" * 64
    json.dump(m_, open(p, "w"))
    _fake_fmq("fmq_otherweights", passed={"TabDPT"})                              # a qualification with other foundation-model weights
    p = os.path.join(config.RESULTS_DIR, "benchmark", "fm_qualify", "fmq_otherweights", "manifest.json")
    m_ = json.load(open(p))
    m_["args"]["foundation_weights"]["TabPFN"]["commit"] = "0" * 40
    json.dump(m_, open(p, "w"))
    _fake_fmq("fmq_noweights", passed={"TabDPT"})                                 # a qualification that did not record the weights
    p = os.path.join(config.RESULTS_DIR, "benchmark", "fm_qualify", "fmq_noweights", "manifest.json")
    m_ = json.load(open(p))
    del m_["args"]["foundation_weights"]
    json.dump(m_, open(p, "w"))
    for hpo_id, fmq_id, msg in [("hpo_other_screening", "fmq_fake", "the tuning run comes from a different domain selection"),
                                ("hpo_fake", "fmq_other_screening", "the qualification run comes from a different domain selection"),
                                ("hpo_fake", "fmq_partial", "judged models"),
                                ("hpo_fake", "fmq_oldcode", "not compatible"),
                                ("hpo_fake", "fmq_otherweights", "differ from the pinned weights"),
                                ("hpo_fake", "fmq_noweights", "no foundation_weights")]:
        monkeypatch.setattr(sys, "argv", ["06_benchmark.py", "--screening-run-id", "screening_fake", "--hpo-run-id", hpo_id,
                                          "--fmq-run-id", fmq_id, "--run-id", "benchmark_bad4", "--models", "Ridge"])
        with pytest.raises(SystemExit, match=msg):
            mod.main()
    # search records made with the smoke trial count, and records from other search rows, are refused before training (smoke runs only
    # with --allow-nonfinal-hpo)
    _fake_hpo("hpo_smoke", df, DOMAINS, MODELS)
    _fake_hpo("hpo_rows", df, DOMAINS, MODELS)
    for n_ in DOMAINS:
        slot_ = build_one(df, n_)[0]
        p_ = hpo.record_prefix("hpo_smoke", slot_, "Ridge", "source")
        hpo.save(dict(hpo.load(p_), n_trials=3), p_)
        q_ = hpo.record_prefix("hpo_rows", slot_, "Ridge", "source")
        hpo.save(dict(hpo.load(q_), train_idx_sha256="0" * 64), q_)
    for k, (hpo_id, extra, msg) in enumerate([("hpo_smoke", [], "final budget"), ("hpo_rows", [], "search training and validation rows"),
                                              ("hpo_rows", ["--allow-nonfinal-hpo"], "search training and validation rows")]):
        monkeypatch.setattr(sys, "argv", ["06_benchmark.py", "--screening-run-id", "screening_fake", "--hpo-run-id", hpo_id,
                                          "--fmq-run-id", "fmq_fake", "--run-id", f"benchmark_bad5_{k}", "--models", "Ridge", *extra])
        with pytest.raises(SystemExit, match=msg):
            mod.main()
    assert len(rec.fits) == n


# ------------------------------------------------------------------ benchmark tuning path
def test_tuning_dataflow(benchmark_inputs, data, monkeypatch):
    """source search = the tuning split of A_dev (by lot), target search = (B_tr, B_val). No A_heldout, B_screen or B_te."""
    df, y = data
    mod = _load_script("04_tune_hyperparameters")
    calls = []

    def fake_tune(spec, ctx, train_idx, val_idx, n_trials=config.HPO_N_TRIALS, seed=config.HPO_SEED, max_epochs=None):
        calls.append(dict(model=spec.name, train=set(train_idx), val=set(val_idx), cat_mode=ctx.cat_mode,
                          features=list(ctx.feature_cols), n_trials=n_trials, max_epochs=max_epochs))
        return dict(params={}, hpo_sec=0.0, best_score=0.0, trials=[], unobserved_num_cols=[])
    monkeypatch.setattr(mod.hpo, "tune", fake_tune)
    models = ["Ridge", "XGBoost", "MLP", "RealMLP"]
    monkeypatch.setattr(sys, "argv", ["04_tune_hyperparameters.py", "--screening-run-id", "screening_fake", "--run-id", "hpo_flow",
                                      "--models", ",".join(models), "--trials", "3"])
    mod.main()
    assert len(calls) == len(DOMAINS) * len(models) * 2
    lot = df[config.LOT_COL]
    per = len(models) * 2
    for d_i, name in enumerate(DOMAINS):
        s = _protocol_sets(df, y, name, 0)
        src_tr, src_va = hpo_split(s["A_dev"], y, lot)
        never = set(s["A_heldout"]) | set(s["B_te"]) | set(s["B_screen"])
        for k, c in enumerate(calls[d_i * per:(d_i + 1) * per]):
            side = ("source", "target")[k % 2]
            assert c["model"] == models[k // 2] and c["cat_mode"] == "benchmark" and c["n_trials"] == 3
            assert not (c["train"] | c["val"]) & never and not c["train"] & c["val"]
            if side == "source":
                assert c["train"] == set(src_tr) and c["val"] == set(src_va)
                assert not set(lot.loc[src_tr]) & set(lot.loc[src_va])
            else:
                assert c["train"] == set(s["B_tr"]) and c["val"] == set(s["B_val"])
            assert config.LOT_COL not in c["features"]
    mod.main()                                                              # resume: no new search
    assert len(calls) == len(DOMAINS) * len(models) * 2
    monkeypatch.setattr(sys, "argv", ["04_tune_hyperparameters.py", "--screening-run-id", "screening_fake", "--run-id", "hpo_bad",
                                      "--models", "TabPFN"])
    with pytest.raises(SystemExit, match="models not tuned"):
        mod.main()


# ------------------------------------------------------------------ foundation-model qualification path
def test_fm_qualify_uses_A_dev_only(benchmark_inputs, data, monkeypatch):
    df, y = data
    mod = _load_script("05_qualify_foundation")
    rec = Recorder(y)
    rec.install(monkeypatch)
    monkeypatch.setattr(mod.fm, "release", lambda name=None: None)
    monkeypatch.setattr(sys, "argv", ["05_qualify_foundation.py", "--screening-run-id", "screening_fake", "--run-id", "fmq_flow",
                                      "--models", "TabPFN"])
    mod.main()
    from defectshift.candidates import subsample
    A_dev = _protocol_sets(df, y, mod.DOMAIN, 0)["A_dev"]
    # per context size: 5 trainings (default, reproduction, other context, isolation check, new instance) and 6 predictions (+ save and
    # restore), all inside A_dev
    assert len(rec.fits) == 5 * len(mod.CONTEXT_SIZES) and len(rec.preds) == 6 * len(mod.CONTEXT_SIZES)
    for g, n in enumerate(mod.CONTEXT_SIZES):
        ctx_idx = subsample(A_dev, min(n, len(A_dev)), config.RANDOM_STATE)
        rest = A_dev.difference(ctx_idx)
        q = set(subsample(rest, min(mod.QUERY_MAX, len(rest)), config.RANDOM_STATE + 1))
        assert q and q <= set(A_dev)
        fits = rec.fits[5 * g:5 * g + 5]
        for k, c in enumerate(fits):
            assert c["es"] is None and c["fit"] <= set(A_dev) and not c["fit"] & q
            if k == 2:                                                       # other context: the same size from A_dev outside the query
                assert len(c["fit"]) == len(ctx_idx) and c["seed"] == mod.SEED + 1
            else:
                assert c["fit"] == set(ctx_idx) and c["seed"] == mod.SEED
        assert all(p["rows"] == q for p in rec.preds[6 * g:6 * g + 6])
    summary = json.load(open(os.path.join(config.RESULTS_DIR, "benchmark", "fm_qualify", "fmq_flow", "summary.json")))
    assert summary["results"]["TabPFN"]["passed"] is True


# ------------------------------------------------------------------ foundation-model input modes
def test_foundation_input_modes(data, monkeypatch):
    """Screening and selection-audit default (impute_standardize) = the imputed and standardized input, benchmark (raw_native) = raw numeric
    values (missing values included) as they are. Prediction uses the mode recorded at training. Columns without observed values in the
    training rows are constant 0 in both modes (exception rule)."""
    from defectshift.models import foundation as fm
    from defectshift.utils import prep_features_cat_cont
    df, y = data
    lot = df[config.LOT_COL]
    slot, mA, _ = build_one(df, "cat_050")
    A_dev = get_A(slot, df.index[mA], y, lot)["A_dev"]
    fit_idx, ev_idx = A_dev[:400], A_dev[400:600]
    cat_cols, num_cols = feature_sets(df, "cat_050")
    seen = []

    class Capture:
        def __init__(self, *a):
            pass

        def fit(self, X, y_):
            seen.append(("fit", X.copy()))
            return self

        def predict(self, X):
            seen.append(("predict", X.copy()))
            return np.zeros(len(X))

        def get_params(self):
            return {}
    monkeypatch.setattr(fm, "build", lambda name, cat_idx, seed: Capture())
    monkeypatch.setattr(fm, "effective_shape", lambda *a: {})
    monkeypatch.setattr(fm, "version", lambda name: "test")
    spec = registry.get("TabPFN")
    nc = len(cat_cols)
    for mode in ("impute_standardize", "raw_native"):
        seen.clear()
        ctx = regimes.Ctx(df, y, cat_cols, num_cols, cat_mode="benchmark", fm_input=mode)
        f = regimes.fit(spec, ctx, fit_idx, None, 1)
        regimes.predict(f, ctx, ev_idx)
        (_, Xf), (_, Xp) = seen
        unobs = f.config["unobserved_num_cols"]
        keep = [j for j, c in enumerate(num_cols) if c not in unobs]
        if mode == "impute_standardize":
            ref_f, ref_p = prep_features_cat_cont(regimes._neutralized(ctx, unobs).df, cat_cols, num_cols, fit_idx, ev_idx)
            assert np.array_equal(Xf, fm.combine(ref_f)[0]) and np.array_equal(Xp, fm.combine(ref_p)[0])
            assert not np.isnan(Xf).any() and f.config["external_imputation"] is True
        else:
            raw_f = df.loc[fit_idx, num_cols].to_numpy("float32")[:, keep]
            raw_p = df.loc[ev_idx, num_cols].to_numpy("float32")[:, keep]
            assert np.array_equal(Xf[:, nc:][:, keep], raw_f, equal_nan=True)
            assert np.array_equal(Xp[:, nc:][:, keep], raw_p, equal_nan=True)
            assert np.isnan(Xf).any()                                            # missing values are passed as they are
            assert f.config["external_imputation"] is False and f.config["external_standardization"] is False
            assert f.config["missing_value_handling"] == fm.MISSING_HANDLING["TabPFN"]
        assert f.config["fm_input"] == mode
        drop = [j for j, c in enumerate(num_cols) if c in unobs]
        if drop:
            assert (Xf[:, nc:][:, drop] == 0).all() and (Xp[:, nc:][:, drop] == 0).all()
        assert np.array_equal(Xf[:, :nc], df.loc[fit_idx, cat_cols].to_numpy("float32"))


# ------------------------------------------------------------------ benchmark aggregation (no retraining)
def test_benchmark_aggregate(benchmark_inputs, data, monkeypatch):
    """Aggregation of fake benchmark artifacts: completeness, recomputation from the saved predictions = rows metrics, numbers of target-reference gap,
    H_tune and ID-OOD gap pairs, the RQ3 identity, tail bins."""
    df, y = data
    mod = _load_script("06_benchmark")
    rec = Recorder(y)
    rec.install(monkeypatch)
    monkeypatch.setattr(sys, "argv", _argv("benchmark_agg", "--only", "cat_050", "--models", "Ridge,XGBoost,TabPFN", "--repeats", "2"))
    mod.main()
    slot = build_one(df, "cat_050")[0]
    os.remove(mod.artifact_prefix(os.path.join(config.CACHE_DIR, "benchmark", "runs", "benchmark_agg", "models"), slot, "Ridge", "ID_match", 0)
              + ".pred.npz")
    mod.main()                                                            # 2 successful rows for the same key (the last one counts)
    agg = _load_script("06c_aggregate")
    monkeypatch.setattr(sys, "argv", ["06c_aggregate.py", "--run-id", "benchmark_agg"])
    agg.main()
    out = os.path.join(config.RESULTS_DIR, "benchmark", "runs", "benchmark_agg", "report")
    rep = json.load(open(os.path.join(out, "report.json")))
    c = rep["completeness"]
    assert c["expected"] == c["ok"] == 2 * (4 + 4 + 3) and c["missing"] == 0 and c["n_duplicate_ok_rows"] == 1
    assert rep["metric_recompute_max_abs_diff"] <= 1e-12 and rep["rq3_identity_max_abs_err"] < 1e-9
    target_reference_gap = pd.read_csv(os.path.join(out, "target_reference_gap.csv"))
    h_tune = pd.read_csv(os.path.join(out, "h_tune.csv"))
    id_ood_gap = pd.read_csv(os.path.join(out, "id_ood_gap.csv"))
    assert len(target_reference_gap) == 2 * 3 and len(h_tune) == 2 * 2 and len(id_ood_gap) == 2 * 3 * 2
    pf = pd.read_csv(os.path.join(out, "per_fit.csv"))
    for tag in ("id", "ood"):                                              # unseen-category exposure diagnostics (share, column count)
        assert pf[f"unseen_cat_rows_{tag}"].between(0, 1).all() and (pf[f"unseen_cat_cols_{tag}"] >= 0).all()
    one = pf[(pf.model == "Ridge") & (pf.repeat == 1)].set_index("regime")
    exp = metrics.target_reference_gap(dict(r2=one.loc["ID_match", "r2_ood"], rmse=one.loc["ID_match", "rmse_ood"], mae=one.loc["ID_match", "mae_ood"]),
                         dict(r2=one.loc["OOD_match", "r2_ood"], rmse=one.loc["OOD_match", "rmse_ood"], mae=one.loc["OOD_match", "mae_ood"]))
    got = target_reference_gap[(target_reference_gap.model == "Ridge") & (target_reference_gap.repeat == 1)].iloc[0]
    assert all(abs(got[k] - exp[k]) < 1e-12 for k in exp)
    tail = pd.read_csv(os.path.join(out, "tail.csv"))
    n_te = len(np.load(pf.pred_path.iloc[0])["ood_idx"])
    per_fit_bins = tail.groupby(["repeat", "model", "regime"]).n.sum()
    assert set(tail.bin) == {"0-50", "50-80", "80-90", "90-95", "95-100"} and (per_fit_bins == n_te).all()   # the bins cover B_te completely
    monkeypatch.setattr(sys, "argv", ["06c_aggregate.py", "--run-id", "benchmark_resume_missing_none"])
    with pytest.raises(FileNotFoundError):
        agg.main()


# ------------------------------------------------------------------ robustness (fail-closed, resume, row indices)
def test_benchmark_fail_closed_and_resume_checks(benchmark_inputs, data, monkeypatch):
    """A broken prediction file is not counted as finished and is retrained, a remaining failure gives exit code 1, and different stored
    evaluation rows stop the run."""
    df, y = data
    mod = _load_script("06_benchmark")
    rec = Recorder(y)
    rec.install(monkeypatch)
    small = ("--only", "cat_050", "--models", "Ridge,TabPFN", "--repeats", "1")
    monkeypatch.setattr(sys, "argv", _argv("benchmark_hard", *small))
    mod.main()
    n0 = len(rec.fits)
    slot = build_one(df, "cat_050")[0]
    mdir = os.path.join(config.CACHE_DIR, "benchmark", "runs", "benchmark_hard", "models")
    with open(mod.artifact_prefix(mdir, slot, "Ridge", "ID_full", 0) + ".pred.npz", "wb") as f:
        f.write(b"not an npz")                                              # nonzero size but cannot be opened
    mod.main()
    assert len(rec.fits) == n0 + 1
    # if one training fails, its row stays failed and the run exits with code 1
    orig = rec.fit

    def flaky(spec, ctx, fit_idx, es_idx, seed, params=None):
        if spec.name == "Ridge" and seed == config.RANDOM_STATE + config.SEED_OFFSET["ID_match"]:
            raise RuntimeError("deliberate failure")
        return orig(spec, ctx, fit_idx, es_idx, seed, params)
    monkeypatch.setattr(regimes, "fit", flaky)
    monkeypatch.setattr(sys, "argv", _argv("benchmark_hard2", *small))
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 1
    # stop if the stored evaluation rows differ from the current split
    rec.install(monkeypatch)
    p = mod.artifact_prefix(mdir, slot, "TabPFN", "OOD_match", 0) + ".pred.npz"
    z = dict(np.load(p))
    z["ood_idx"] = z["ood_idx"][::-1]
    np.savez(p, **z)
    monkeypatch.setattr(sys, "argv", _argv("benchmark_hard", *small))
    with pytest.raises(SystemExit, match="stored evaluation rows differ from the current split"):
        mod.main()


def test_benchmark_aggregate_formulas():
    """The aggregation formulas compared with values computed by hand (deterministic fake metrics)."""
    agg = _load_script("06c_aggregate")
    base = dict(candidate="d", repeat=0, model="m")
    pf = pd.DataFrame([
        dict(base, regime="ID_match", r2_id=0.6, rmse_id=1.0, mae_id=0.8, r2_ood=0.3, rmse_ood=2.0, mae_ood=1.5),
        dict(base, regime="OOD_match", r2_id=0.2, rmse_id=1.8, mae_id=1.4, r2_ood=0.5, rmse_ood=1.6, mae_ood=1.2),
        dict(base, regime="OOD_match_tuned", r2_id=0.2, rmse_id=1.8, mae_id=1.4, r2_ood=0.55, rmse_ood=1.5, mae_ood=1.1),
    ])
    ds = agg._pair(pf, "ID_match", "OOD_match", lambda a, b: agg._d(a, b, "ood")).iloc[0]
    assert np.isclose(ds.r2_difference, 0.5 - 0.3) and np.isclose(ds.gap_rmse_pct, (2.0 - 1.6) / 1.6 * 100) and np.isclose(ds.gap_mae_pct, (1.5 - 1.2) / 1.2 * 100)
    ht = agg._pair(pf, "OOD_match", "OOD_match_tuned", lambda a, b: agg._d(a, b, "ood")).iloc[0]
    assert np.isclose(ht.r2_difference, 0.55 - 0.5) and np.isclose(ht.gap_rmse_pct, (1.6 - 1.5) / 1.5 * 100)   # positive = the target search values are better
    # ID-OOD gap: A_heldout vs B_te of the same source model (positive = worse on B_te)
    a = pf[pf.regime == "ID_match"].iloc[0]
    assert np.isclose(a.r2_id - a.r2_ood, 0.3) and np.isclose((a.rmse_ood - a.rmse_id) / a.rmse_id * 100, 100.0)
    # RQ3 identity: (1 - R²_OOD)/(1 - R²_ID) = (MSE_OOD/MSE_ID)(Var_ID/Var_OOD)
    rng = np.random.default_rng(0)
    yi, yo = rng.normal(0, 2, 500), rng.normal(1, 1, 400)
    pi, po = yi + rng.normal(0, 1, 500), yo + rng.normal(0, 0.8, 400)
    mi, mo = agg._metrics(yi, pi), agg._metrics(yo, po)
    assert np.isclose((1 - mo["r2"]) / (1 - mi["r2"]), (mo["mse"] / mi["mse"]) * (mi["var_y"] / mo["var_y"]))


def test_benchmark_aggregate_rejects_wrong_eval_rows(benchmark_inputs, data, monkeypatch):
    """If the rows of a prediction file differ from B_te of the split code, the aggregation stops (even if the metrics equal rows)."""
    df, y = data
    mod = _load_script("06_benchmark")
    rec = Recorder(y)
    rec.install(monkeypatch)
    monkeypatch.setattr(sys, "argv", _argv("benchmark_aggbad", "--only", "cat_050", "--models", "Ridge", "--repeats", "1"))
    mod.main()
    slot = build_one(df, "cat_050")[0]
    p = mod.artifact_prefix(os.path.join(config.CACHE_DIR, "benchmark", "runs", "benchmark_aggbad", "models"), slot, "Ridge", "ID_full", 0) + ".pred.npz"
    z = dict(np.load(p))
    z["ood_idx"] = z["ood_idx"][::-1]                                       # constant predictions: the metrics stay, only the rows change
    np.savez(p, **z)
    agg = _load_script("06c_aggregate")
    monkeypatch.setattr(sys, "argv", ["06c_aggregate.py", "--run-id", "benchmark_aggbad"])
    with pytest.raises(SystemExit, match="differ from A_heldout and B_te"):
        agg.main()
