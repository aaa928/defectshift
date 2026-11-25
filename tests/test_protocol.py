"""Checks of the protocol rules. Run: python -m pytest tests -q

The split rules are checked on the real data.parquet. Caches and results are written to a temporary folder (the repository cache/ and
results/ are not touched).
"""
import json
import math
import os
import sys

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from defectshift import config, manifest, metrics, regimes, registry, selection  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_candidates, load_data, parse_candidate  # noqa: E402
from defectshift.splits import (B_view, b_te_eligible, check_A, es_split, get_A, hpo_split,  # noqa: E402
                                matched, repeat_split, split_B4)
from defectshift.utils import SCALE_FLOOR_ABS, fit_feature_transform, prep_features_cat_cont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_MEDIAN = "x_time<=2099-05-04 21:18:39"
CANDS = ["cat_050", X_MEDIAN, "placebo@9001"]


@pytest.fixture(scope="module")
def data():
    return load_data()


def _lots(df, idx):
    return set(df.loc[idx, config.LOT_COL])


def _setup(df, name):
    slot, mA, mB = build_one(df, name)
    return slot, df.index[mA], df.index[mB]


# ------------------------------------------------------------------ candidate construction
def test_candidate_schema_and_temporal_medians(data):
    df, _ = data
    names = load_candidates()
    kinds = pd.Series([parse_candidate(n)[0] for n in names]).value_counts().to_dict()
    assert kinds == {"cat": 48, "x_time": 15, "y_time": 15}
    tx, ty = selection.temporal_primary(names)
    assert tx == X_MEDIAN
    for name in (tx, ty):   # A share 0.50 on both axes
        _, A_idx, _ = _setup(df, name)
        assert len(A_idx) / len(df) == pytest.approx(0.50, abs=0.005)
    slots = [build_one(df, n)[0] for n in load_candidates(include_placebo=True)]
    assert len(slots) == len(set(slots)) == 78 + len(config.PLACEBO_B_TARGET) == 88


# ------------------------------------------------------------------ A/B splits
@pytest.mark.parametrize("name", CANDS)
def test_A_heldout_never_used(data, name):
    df, y = data
    slot, A_idx, B_idx = _setup(df, name)
    lot = df[config.LOT_COL]
    a = get_A(slot, A_idx, y, lot)
    assert not set(a["A_heldout"]) & set(a["A_dev"])
    assert not _lots(df, a["A_heldout"]) & _lots(df, a["A_dev"])             # no lot on both sides
    assert len(a["A_heldout"]) == pytest.approx(0.2 * len(A_idx), abs=0.02 * len(A_idx))   # approximate because of lot grouping
    h_tr, h_va = hpo_split(a["A_dev"], y, lot)
    assert not _lots(df, h_tr) & _lots(df, h_va)
    used = {"hpo_train": h_tr, "hpo_val": h_va}
    for i in range(3):
        A_train, A_val = repeat_split(a["A_dev"], y, lot, i)
        assert not _lots(df, A_train) & _lots(df, A_val)
        assert len(A_train) == pytest.approx(0.6 * len(A_idx), abs=0.02 * len(A_idx))
        used.update({f"A_train{i}": A_train, f"A_val{i}": A_val})
    check_A(a, used)
    assert get_A(slot, A_idx, y, lot)["A_heldout"].equals(a["A_heldout"])   # cache = recomputation


@pytest.mark.parametrize("name", CANDS)
def test_B_view_stages(data, name):
    df, y = data
    slot, A_idx, B_idx = _setup(df, name)
    lot = df[config.LOT_COL]
    assert set(B_view(slot, B_idx, y, lot, "screening")) == {"B_tr", "B_screen"}      # screening: no B_val or B_te
    assert set(B_view(slot, B_idx, y, lot, "shift_statistics")) == {"B_tr", "B_screen"}     # shift statistics: no B_te
    assert set(B_view(slot, B_idx, y, lot, "benchmark")) == {"B_tr", "B_val", "B_te"}
    b4 = split_B4(B_idx, y, lot)
    for k, f in config.B_FRACS.items():
        assert len(b4[k]) == pytest.approx(f * len(B_idx), abs=0.02 * len(B_idx))
    parts = list(b4.values())
    for i_ in range(4):
        for j_ in range(i_ + 1, 4):
            assert not _lots(df, parts[i_]) & _lots(df, parts[j_])                # the four subsets are disjoint by lot


def test_b_te_eligible_exact():
    for n in range(1900, 2100):
        assert b_te_eligible(n) == (math.floor(0.25 * n) >= 500)
    assert b_te_eligible(2000) and not b_te_eligible(1999)


@pytest.mark.parametrize("name", CANDS)
def test_matched_and_internal_es(data, name):
    df, y = data
    slot, A_idx, B_idx = _setup(df, name)
    lot = df[config.LOT_COL]
    a = get_A(slot, A_idx, y, lot)
    bv = B_view(slot, B_idx, y, lot, "screening")
    for i in range(2):
        A_train, _ = repeat_split(a["A_dev"], y, lot, i)
        n_star, A_m, B_m = matched(A_train, bv["B_tr"], i)
        assert n_star == min(len(A_train), len(bv["B_tr"])) == len(A_m) == len(B_m)
        assert set(A_m) <= set(A_train) and set(B_m) <= set(bv["B_tr"])
        for pool in (A_m, B_m):
            fit, es = es_split(pool, y, lot, i)
            assert not set(fit) & set(es) and len(fit) + len(es) == n_star
            assert not _lots(df, fit) & _lots(df, es)
            assert len(es) == pytest.approx(config.ES_VAL_FRAC * n_star, abs=0.03 * n_star)
            assert abs((y.loc[fit] > 0).mean() - (y.loc[es] > 0).mean()) < 0.03


def test_feature_sets(data):
    df, _ = data
    n_cat = sum(c.startswith("cat_") for c in df.columns)
    cat, num = feature_sets(df, "cat_050")
    assert "cat_050" not in cat and len(cat) == n_cat - 2                      # lot column + split column
    for n in CANDS[1:]:
        cat2, num2 = feature_sets(df, n)
        assert "cat_050" in cat2 and num2 == num and len(cat2) == n_cat - 1    # lot column only
    for n in load_candidates(include_placebo=True):
        assert config.LOT_COL not in feature_sets(df, n)[0]                    # the lot is a feature in no candidate
    assert not {"y", "x_time", "y_time"} & set(cat + num)


def test_placebo_lot_level_size_matched(data):
    df, y = data
    N = len(df)
    # |B| targets = the q_B quantiles (5, 15, ..., 95%) of the eligible categorical candidates x N; the config values must equal a
    # recomputation from the raw data
    qB = []
    for n in load_candidates():
        if parse_candidate(n)[0] == "cat":
            nB = int(build_one(df, n)[2].sum())
            if b_te_eligible(nB):
                qB.append(nB / N)
    assert len(qB) == 46
    want = [int(round(q * N)) for q in pd.Series(qB).quantile([(j - .5) / 10 for j in range(1, 11)])]
    assert list(config.PLACEBO_B_TARGET.values()) == want
    for seed, target in config.PLACEBO_B_TARGET.items():
        _, A_idx, B_idx = _setup(df, f"placebo@{seed}")
        assert len(A_idx) + len(B_idx) == N
        assert len(B_idx) == pytest.approx(target, abs=0.01 * N)                  # approximate by lot (grid of 0.01 steps)
        assert not _lots(df, A_idx) & _lots(df, B_idx)                            # no lot split between A and B
        assert abs((y.loc[A_idx] > 0).mean() - (y.loc[B_idx] > 0).mean()) < 0.01


# ------------------------------------------------------------------ preprocessing
def _toy():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cat_a": rng.integers(0, 3, 40), "num_x": rng.random(40), "num_c": 0.5})
    df.loc[:19, "num_c"] = 0.5 + np.r_[1e-6, -1e-6] .repeat(10)   # nearly constant in the training rows (standard deviation 1e-6)
    df.loc[3, "num_x"] = np.nan
    return df, pd.Index(range(20)), pd.Index(range(20, 40))


def test_preprocessing_uses_train_rows_only():
    df, tr, te = _toy()
    base = prep_features_cat_cont(df, ["cat_a"], ["num_x", "num_c"], tr, te[:1])
    base_ml = fit_feature_transform(df, ["num_x", "num_c"], True, tr)(df.loc[te[:1]])
    df2 = df.copy()
    df2.loc[te[1:], ["num_x", "num_c"]] = 1e3      # extreme values in rows other than the training rows
    new = prep_features_cat_cont(df2, ["cat_a"], ["num_x", "num_c"], tr, te[:1])
    new_ml = fit_feature_transform(df2, ["num_x", "num_c"], True, tr)(df2.loc[te[:1]])
    for (c0, x0), (c1, x1) in zip(base, new):
        assert (c0 == c1).all() and np.allclose(x0, x1)
    assert np.allclose(base_ml.values, new_ml.values)


def test_scale_floor_applied():
    df, tr, te = _toy()
    (_, x_tr), (_, x_te) = prep_features_cat_cont(df, ["cat_a"], ["num_x", "num_c"], tr, te)
    # num_c: training standard deviation 1e-6 -> denominator SCALE_FLOOR_ABS. The evaluation value 0.5 gives (0.5 - 0.5)/1e-3, near 0
    assert np.abs(x_te[:, 1]).max() < 1e-2 / SCALE_FLOOR_ABS


def test_impute_only_uses_train_median():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 100.0], "b": [np.nan, 2.0, 2.0, 50.0]})
    out = fit_feature_transform(df, ["a", "b"], "impute", pd.Index([0, 1, 2]))(df)
    assert out.loc[1, "a"] == 2.0 and out.loc[0, "b"] == 2.0 and out.loc[3, "a"] == 100.0


# ------------------------------------------------------------------ quantities and selection
def test_target_reference_gap_sign():
    worse, better = dict(r2=0.2, rmse=0.06, mae=0.04), dict(r2=0.5, rmse=0.05, mae=0.03)
    d = metrics.target_reference_gap(id_match=worse, ood_match=better)
    assert d["r2_difference"] == pytest.approx(0.3) and d["gap_rmse_pct"] == pytest.approx(20.0) and d["gap_mae_pct"] > 0
    assert all(v < 0 for v in metrics.target_reference_gap(id_match=better, ood_match=worse).values())


def _rows(spec):
    """spec: {candidate: {model: [gap_rmse_pct per repeat]}} -> rows.csv form (ID_match rmse set to match D)."""
    out = []
    for cand, per_model in spec.items():
        kind = parse_candidate(cand)[0]
        for m, ds in per_model.items():
            for i, d in enumerate(ds):
                if d is None:   # failure
                    out.append(dict(slot=cand, candidate=cand, kind=kind, model=m, repeat=i, regime="ID_match",
                                    status="failed", n_A=5000, n_B=4000, n_star=1000, b_te_eligible=True))
                    continue
                for regime, rmse in (("OOD_match", 1.0), ("ID_match", 1.0 + d / 100)):
                    out.append(dict(slot=cand, candidate=cand, kind=kind, model=m, repeat=i, regime=regime, status="ok",
                                    r2_screen=0.5, rmse_screen=rmse, mae_screen=rmse, n_A=5000, n_B=4000, n_star=1000,
                                    b_te_eligible=(cand != "cat_inelig")))
    return pd.DataFrame(out)


MODELS = list(config.SCREENING_MODELS)


def test_selection_rules():
    names = load_candidates()
    tx, ty = selection.temporal_primary(names)
    spec = {
        "cat_a": {m: [10, 12, 8] for m in MODELS},                        # 4/4 consistently positive, consensus 10
        "cat_b": {**{m: [5, 5, 5] for m in MODELS[:3]}, MODELS[3]: [-5, -5, -5]},   # 3/4 -> passes, consensus 5
        "cat_c": {**{m: [9, -1, -2] for m in MODELS[:2]}, **{m: [9, 9, 9] for m in MODELS[2:]}},  # mean > 0 but 1/3 positive -> 2/4
        "cat_d": {**{m: [20, 20, 20] for m in MODELS[:3]}, MODELS[3]: [20, 20, None]},             # incomplete
        "cat_inelig": {m: [30, 30, 30] for m in MODELS},                  # B_te eligibility not met
        "cat_tie": {m: [5, 5, 5] for m in MODELS},                        # consensus 5, tie with cat_b -> by name
        tx: {m: [-3, -3, -3] for m in MODELS},                            # prespecified temporal: included regardless of the score
        ty: {m: [1, 1, 1] for m in MODELS},
        "x_time<=2099-06-04 10:07:30.500000": {m: [50, 50, 50] for m in MODELS},   # temporal, not prespecified: not selected
        "placebo@9001": {m: [0, 0, 0] for m in MODELS},
    }
    rows = _rows(spec)
    summ = selection.candidate_summary(rows, MODELS, 3)
    assert len(summ) == len(spec)                                         # every candidate stays in the table (shift association uses all)
    s = summ.set_index("candidate")
    assert s.loc["cat_a", "n_models_consistent"] == 4 and s.loc["cat_a", "consensus_gap_rmse_pct"] == pytest.approx(10.0)
    assert s.loc["cat_b", "n_models_consistent"] == 3 and s.loc["cat_c", "n_models_consistent"] == 2
    assert not s.loc["cat_d", "complete"]
    table, final = selection.select(summ, names, n_cat=3)
    t = table.set_index("candidate")
    assert final["categorical"] == ["cat_a", "cat_b", "cat_tie"]          # consensus descending, ties by name
    assert final["temporal"] == [tx, ty]
    assert len(final["temporal"]) + len(final["categorical"]) == 2 + 3
    assert t.loc["cat_c", "reason"].startswith("consistently positive models") and t.loc["cat_d", "reason"].startswith("incomplete")
    assert t.loc["cat_inelig", "reason"] == "B_te eligibility not met"
    assert not t.loc["x_time<=2099-06-04 10:07:30.500000", "selected"] and t.loc[tx, "selected"]
    assert not t.loc["placebo@9001", "selected"]
    with pytest.raises(AssertionError):
        selection.select(summ, names, n_cat=None)


def test_selection_fewer_passing_than_n_cat():
    names = load_candidates()
    tx, ty = selection.temporal_primary(names)
    rows = _rows({"cat_a": {m: [10, 12, 8] for m in MODELS}, "cat_b": {m: [-5, -5, -5] for m in MODELS},
                  tx: {m: [1, 1, 1] for m in MODELS}, ty: {m: [1, 1, 1] for m in MODELS}})
    _, final = selection.select(selection.candidate_summary(rows, MODELS, 3), names, n_cat=5)
    assert final["categorical"] == ["cat_a"] and final["n_cat_passed"] == 1      # only 1 if only 1 passes


def test_final_domains_immutable(tmp_path):
    p = tmp_path / "final_domains.json"
    selection.write_final_domains(str(p), dict(categorical=["cat_a"]))
    with pytest.raises(FileExistsError):
        selection.write_final_domains(str(p), dict(categorical=["cat_b"]))
    assert json.load(open(p))["categorical"] == ["cat_a"]


# ------------------------------------------------------------------ run records
def test_manifest_code_files_exist():
    import types
    fake = types.ModuleType("fake_rel")
    fake.__file__ = "_classes.py"                      # imitation of an external module with a relative __file__
    sys.modules["fake_rel"] = fake
    try:
        files = manifest.code_files()
    finally:
        del sys.modules["fake_rel"]
    assert files and all(os.path.isfile(f) for f in files)
    rel = [os.path.relpath(f, os.path.realpath(ROOT)) for f in files]
    assert "defectshift/config.py" in rel and "splits/candidates.csv" in rel


def test_manifest_resume_guard(tmp_path):
    m = dict(data_sha256="d", candidates_sha256="c", code_sha256="x", protocol={"a": 1}, packages={"p": "1"}, args={"k": 1})
    assert manifest.write_or_check(str(tmp_path), m) == "new"
    assert manifest.write_or_check(str(tmp_path), dict(m, created="other")) == "resume"
    for k in ("code_sha256", "data_sha256", "args"):
        with pytest.raises(RuntimeError):
            manifest.write_or_check(str(tmp_path), dict(m, **{k: "changed"}))


# ------------------------------------------------------------------ screening script structure, save and restore
def _load_screening_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("01_screen_domains", os.path.join(ROOT, "scripts", "01_screen_domains.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_screening_dataflow_never_touches_heldout_or_bte(data, monkeypatch):
    """Records the rows passed to the training and prediction functions on the real screening path: A_heldout, B_val and B_te must appear
    nowhere, and predictions are made on B_screen only."""
    df, y = data
    mod = _load_screening_module()
    assert mod.EVAL_SETS == ("B_screen",)
    seen = dict(fit=[], es=[], pred=[])

    def fake_fit(spec, ctx, fit_idx, es_idx, seed, params=None, max_epochs=None):
        assert max_epochs is None                                            # no epoch cap without the smoke argument
        seen["fit"].append(set(fit_idx)); seen["es"].append(set() if es_idx is None else set(es_idx))
        return regimes.Fitted(spec, None, fit_idx, es_idx, seed, {}, 0.0, {})

    def fake_predict(f, ctx, idx):
        seen["pred"].append(set(idx)); return np.zeros(len(idx))
    monkeypatch.setattr(mod.regimes, "fit", fake_fit)
    monkeypatch.setattr(mod.regimes, "predict", fake_predict)
    monkeypatch.setattr(sys, "argv", ["01_screen_domains.py", "--run-id", "dataflow_test", "--only", "cat_050", "--repeats", "2",
                                      "--models", "XGBoost,TabPFN", "--no-save"])
    mod.main()
    slot, A_idx, B_idx = _setup(df, "cat_050")
    lot = df[config.LOT_COL]
    a, b4 = get_A(slot, A_idx, y, lot), split_B4(B_idx, y, lot)
    forbidden = set(a["A_heldout"]) | set(b4["B_val"]) | set(b4["B_te"])
    assert len(seen["fit"]) == 2 * 2 * 2                                   # 2 repetitions x 2 models x 2 regimes
    for s_ in seen["fit"] + seen["es"]:
        assert not s_ & forbidden
    assert seen["pred"] and all(p == set(b4["B_screen"]) for p in seen["pred"])


def test_screening_tuned_dataflow_source_only(data, monkeypatch):
    """Tuned screening (--tune): only tunable models are searched, once per candidate; the search rows are the tuning split of A_dev (no
    A_heldout or B); the chosen values are passed as they are to the training of both regimes and all repetitions; TabPFN trains without
    search or params. The smoke epoch cap (--max-epochs 10) is passed to the search and to every training."""
    df, y = data
    mod = _load_screening_module()
    tuned, fits = [], []

    def fake_tune(spec, ctx, tr, va, n_trials=None, seed=None, max_epochs=None):
        tuned.append((spec.name, set(tr), set(va), n_trials, max_epochs))
        return dict(params={"tag": spec.name}, hpo_sec=0.0, best_score=0.0, trials=[], unobserved_num_cols=[])

    def fake_fit(spec, ctx, fit_idx, es_idx, seed, params=None, max_epochs=None):
        fits.append((spec.name, params, max_epochs))
        return regimes.Fitted(spec, None, fit_idx, es_idx, seed, {}, 0.0, {})
    monkeypatch.setattr(mod.hpo, "tune", fake_tune)          # saving the record uses the real function (temporary cache): the completeness check at the end counts the records
    monkeypatch.setattr(mod.regimes, "fit", fake_fit)
    monkeypatch.setattr(mod.regimes, "predict", lambda f, ctx, idx: np.zeros(len(idx)))
    monkeypatch.setattr(sys, "argv", ["01_screen_domains.py", "--run-id", "tuned_dataflow_test", "--only", "cat_050", "--repeats", "2",
                                      "--models", "XGBoost,TabPFN", "--no-save", "--tune", "--trials", "3", "--max-epochs", "10"])
    mod.main()
    slot, A_idx, B_idx = _setup(df, "cat_050")
    lot = df[config.LOT_COL]
    a = get_A(slot, A_idx, y, lot)
    tr, va = hpo_split(a["A_dev"], y, lot)
    assert [t[0] for t in tuned] == ["XGBoost"] and tuned[0][1:] == (set(tr), set(va), 3, 10)
    assert not (tuned[0][1] | tuned[0][2]) & (set(a["A_heldout"]) | set(B_idx))
    assert len(fits) == 2 * 2 * 2
    assert all(p == {"tag": "XGBoost"} for n, p, _ in fits if n == "XGBoost")
    assert all(p is None for n, p, _ in fits if n == "TabPFN")
    assert all(me == 10 for *_, me in fits)


def test_screening_exits_nonzero_when_incomplete(monkeypatch):
    """Completeness check at the end of the run: if any training fails, the count is reported and the exit code is 1 (so that a run with
    continuing failures does not look finished)."""
    mod = _load_screening_module()

    def fake_fit(spec, ctx, fit_idx, es_idx, seed, params=None, max_epochs=None):
        if spec.name == "TabPFN":
            raise RuntimeError("fake device error")
        return regimes.Fitted(spec, None, fit_idx, es_idx, seed, {}, 0.0, {})
    monkeypatch.setattr(mod.regimes, "fit", fake_fit)
    monkeypatch.setattr(mod.regimes, "predict", lambda f, ctx, idx: np.zeros(len(idx)))
    monkeypatch.setattr(sys, "argv", ["01_screen_domains.py", "--run-id", "incomplete_test", "--only", "cat_050", "--repeats", "1",
                                      "--models", "XGBoost,TabPFN", "--no-save"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 1


def test_full_run_requires_n_cat(monkeypatch):
    mod = _load_screening_module()
    monkeypatch.setattr(mod.selection, "N_CAT", None)
    # --tune and the final budget are part of the full-run decision, so --tune is given (without it the run is not a full run, this guard
    # is not tested and real training could start). The training functions are blocked so that nothing runs by mistake
    monkeypatch.setattr(mod.regimes, "fit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("training must not start")))
    monkeypatch.setattr(mod.hpo, "tune", lambda *a, **k: (_ for _ in ()).throw(AssertionError("search must not start")))
    monkeypatch.setattr(sys, "argv", ["01_screen_domains.py", "--run-id", "full_guard_test", "--placebo", "--tune"])
    with pytest.raises(SystemExit):
        mod.main()


def test_resume_requires_artifacts(tmp_path):
    mod = _load_screening_module()
    rows = tmp_path / "rows.csv"
    pd.DataFrame([dict(slot="cat:cat_050", repeat=0, model="XGBoost", regime="ID_match", status="ok"),
                  dict(slot="cat:cat_050", repeat=0, model="XGBoost", regime="OOD_match", status="ok"),
                  dict(slot="cat:cat_050", repeat=0, model="TabPFN", regime="ID_match", status="ok")]).to_csv(rows, index=False)
    md = tmp_path / "models"
    for model, regime, files in (("XGBoost", "ID_match", (".npz", ".json", ".joblib")), ("XGBoost", "OOD_match", (".npz", ".json")),
                                 ("TabPFN", "ID_match", (".npz", ".json"))):
        prefix = mod.artifact_prefix(str(md), "cat:cat_050", model, regime, 0)
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        for ext in files:
            open(prefix + ext, "w").write("{}" if ext == ".json" else "x")
    done = mod.done_keys(str(rows), str(md))
    assert ("cat:cat_050", "0", "XGBoost", "ID_match") in done
    assert ("cat:cat_050", "0", "XGBoost", "OOD_match") not in done     # no model file -> retrained
    assert ("cat:cat_050", "0", "TabPFN", "ID_match") in done           # foundation models need only the context rows


def test_stratification_failure_raises():
    from defectshift.splits import _strat_split
    y = pd.Series([0.0] * 9 + [1.0])                                    # 1 positive -> one side has no y > 0 stratum
    lots = pd.Series(range(10))
    with pytest.raises(RuntimeError, match="B_screen split"):              # the message names the split that failed
        _strat_split(pd.Index(range(10)), y, lots, 0.5, 0, "B_screen split")
    with pytest.raises(RuntimeError, match="3 lots"):                  # number of lots < k
        _strat_split(pd.Index(range(9)), pd.Series([0.0, 1.0] * 4 + [0.0]), pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     0.2, 0, "A_heldout split")


def test_preflight_names_candidate(monkeypatch):
    mod = _load_screening_module()
    df = pd.DataFrame({"cat_050": ["a"] * 400 + ["b"] * 400, config.LOT_COL: np.arange(800) // 10})
    y = pd.Series([0.0] * 799 + [1.0])                                  # 1 positive on the B side -> B splits cannot be stratified
    monkeypatch.setattr(mod, "build_one", lambda d, n: (n, (d.cat_050 == "a").values, (d.cat_050 == "b").values))
    with pytest.raises(RuntimeError, match="candidate cat_050: B_te split.*invalid under the fixed protocol"):
        mod.preflight(df, y, df[config.LOT_COL], ["cat_050"], repeats=1)


def test_validate_run_fail_closed():
    all_names = load_candidates(include_placebo=True)
    names = load_candidates()
    placebo = [n for n in all_names if n not in names]
    assert len(names) == 78 and len(placebo) == 10
    need = len(MODELS) * 2 * 3
    good = pd.DataFrame([dict(candidate=c, model=m, repeat=i, regime=r, status="ok")
                         for c in all_names for m in MODELS for i in range(3) for r in ("ID_match", "OOD_match")])
    args = dict(models=MODELS, repeats=3, save_models=True, full_run=True, candidates=all_names,
                tune=True, trials=config.HPO_N_TRIALS)
    assert selection.validate_run(good, args, names, placebo) == dict(n_duplicate_ok_rows=0)   # passes when complete
    assert len(good) == 88 * need
    bad_cases = {
        "one candidate missing": (good[good.candidate != names[0]], args),
        "random-split control result missing": (good[good.candidate != placebo[0]], args),
        "model saving off": (good, dict(args, save_models=False)),
        "not full_run": (good, dict(args, full_run=False)),
        "partial run": (good, dict(args, candidates=names[:10])),
        "no controls among the argument candidates": (good, dict(args, candidates=names)),
        "duplicate argument candidates": (good, dict(args, candidates=all_names + [names[0]])),
        "unknown candidate in the arguments": (good, dict(args, candidates=all_names + ["cat_999"])),
        "unknown candidate in the results": (pd.concat([good, good.iloc[:1].assign(candidate="cat_999")]), args),
        "1 failed row": (good.assign(status=np.where(good.index == 5, "failed", "ok")), args),
        # same number of rows but one wrong key (repetition 2 -> 3): a row-count check would not catch it
        "wrong key": (good.assign(repeat=np.where(good.index == 5, 3, good.repeat)), args),
        # tuned-screening conditions: fixed settings (no tuning) and smoke trial counts cannot be used for the selection
        "no tuning": (good, dict(args, tune=False)),
        "no tune record (format of a first screening)": (good, {k: v for k, v in args.items() if k not in ("tune", "trials")}),
        "smoke trial count": (good, dict(args, trials=3)),
        "smoke epoch cap": (good, dict(args, max_epochs=10)),
    }
    for label, (rows, a) in bad_cases.items():
        with pytest.raises(ValueError):
            selection.validate_run(rows, a, names, placebo)
            pytest.fail(label)
    dup = pd.concat([good, good.iloc[:3]])                              # retraining after broken artifacts is allowed, the count is recorded
    assert selection.validate_run(dup, args, names, placebo) == dict(n_duplicate_ok_rows=3)
    # a first screening (fixed settings) is allowed explicitly, only when read again for comparison
    old = {k: v for k, v in args.items() if k not in ("tune", "trials")}
    assert selection.validate_run(good, old, names, placebo, require_tuned=False) == dict(n_duplicate_ok_rows=0)


def test_selection_needs_temporal_results_and_hashes():
    names = load_candidates()
    tx, ty = selection.temporal_primary(names)
    rows = _rows({"cat_a": {m: [10, 12, 8] for m in MODELS}, tx: {m: [1, 1, 1] for m in MODELS}})   # no y_time results
    summ = selection.candidate_summary(rows, MODELS, 3)
    with pytest.raises(AssertionError):
        selection.select(summ, names, n_cat=1)
    h = selection.code_hashes()
    assert set(h) == {"selection_py_sha256", "select_domains_py_sha256", "metrics_py_sha256"} and all(len(v) == 64 for v in h.values())


def test_save_load_roundtrip(data, tmp_path):
    df, y = data
    cat, num = feature_sets(df, "cat_050")
    ctx = regimes.Ctx(df, y, cat, num)
    idx = df.index[:600]
    for name in ("DecisionTree", "Ridge"):   # two fast CPU routes (no imputation / imputation + standardization)
        spec = registry.get(name)
        f = regimes.fit(spec, ctx, idx[:500], None, seed=1)
        p0 = regimes.predict(f, ctx, idx[500:])
        prefix = str(tmp_path / name)
        regimes.save(f, ctx, prefix)
        g = regimes.load(spec, ctx, prefix)
        assert np.array_equal(p0, regimes.predict(g, ctx, idx[500:]))
        assert json.load(open(prefix + ".json"))["config"]["n_fit"] == 500


def test_unobserved_train_column_is_neutralized(data, tmp_path):
    """Numeric columns that are entirely missing in the training rows (there is no imputation median to fit): constant
    in training and evaluation and recorded, with the same predictions after saving and restoring."""
    df, y = data
    cat, num = feature_sets(df, "cat_050")
    idx = df.index[:700]
    fit_idx, es_idx, ev = idx[:450], idx[450:550], idx[550:]
    df2 = df.copy()
    df2.loc[fit_idx, num[:2]] = np.nan                       # entirely missing only in the training rows (the evaluation rows have values)
    ctx = regimes.Ctx(df2, y, cat, num)
    for name in ("Ridge", "MLP", "TabM", "XGBoost"):
        spec = registry.get(name)
        f = regimes.fit(spec, ctx, fit_idx, es_idx if spec.uses_es else None, seed=3)
        assert f.config["unobserved_num_cols"] == num[:2]
        p0 = regimes.predict(f, ctx, ev)
        assert np.isfinite(p0).all()
        prefix = str(tmp_path / name)
        regimes.save(f, ctx, prefix)
        assert np.allclose(p0, regimes.predict(regimes.load(spec, ctx, prefix), ctx, ev))
    f = regimes.fit(registry.get("Ridge"), regimes.Ctx(df, y, cat, num), fit_idx, None, seed=3)
    assert f.config["unobserved_num_cols"] == []          # nothing changes without missing columns


def test_append_row_keeps_columns_aligned(tmp_path):
    mod = _load_screening_module()
    path = str(tmp_path / "rows.csv")
    base = dict(slot="s", repeat=0, model="MLP", regime="ID_match", n_fit=10)
    mod.append_row(path, dict(base, fit_sec=1.0, r2_screen=0.5, rmse_screen=0.1, status="ok"))
    mod.append_row(path, dict(base, fit_sec=1.0, status="failed", error="ValueError: x"))   # no metrics + error column
    mod.append_row(path, dict(base, repeat=1, fit_sec=2.0, r2_screen=0.4, rmse_screen=0.2, status="ok"))
    r = pd.read_csv(path)
    assert list(r.status) == ["ok", "failed", "ok"] and list(r.repeat) == [0, 0, 1]
    assert r.r2_screen.dtype.kind == "f" and np.isnan(r.r2_screen[1]) and r.error[1] == "ValueError: x"
    assert r.columns[-1] == "error"


# ------------------------------------------------------------------ categorical representation modes (benchmark; integer = screening, selection audit, sensitivity)
def test_integer_mode_is_unchanged(data):
    from defectshift.models import protocol
    df, _ = data
    cat, num = feature_sets(df, "cat_050")
    feats, idx = cat + num, df.index[:500]
    X = df.loc[df.index[500:600], feats]
    for impl in ("XGBoost", "LinearModel", "GBR", "kNN"):
        old = fit_feature_transform(df, feats, protocol.ml_needs_scale(impl), idx)(X)
        assert protocol.ml_cat_representation(impl, "integer") == "integer"
        pd.testing.assert_frame_equal(protocol.ml_transform(df, feats, impl, idx, "integer")(X), old)
    for impl in ("DecisionTree", "RandomForest", "GBR"):                       # the 3 tree models use integer codes in the benchmark too
        assert protocol.ml_cat_representation(impl, "benchmark") == "integer"


def test_onehot_representation(data):
    from defectshift.models import protocol
    df, _ = data
    cat, num = feature_sets(df, "cat_050")
    feats, tr = cat + num, df.index[:400]
    tf = protocol.ml_transform(df, feats, "LinearModel", tr, "benchmark")
    X = df.loc[df.index[400:600], feats].copy()
    X.loc[X.index[0], cat[0]] = 999                                           # a category not seen in training
    out = tf(X)
    dummies = [c for c in out.columns if c not in num]
    assert list(out.columns[:len(num)]) == num and len(dummies) == sum(df.loc[tr, c].nunique() for c in cat)
    assert set(np.unique(out[dummies].to_numpy())) <= {0.0, 1.0}              # dummies are not standardized
    assert out.loc[X.index[0], [d for d in dummies if d.startswith(cat[0] + "_")]].sum() == 0
    df2 = df.copy()
    df2.loc[df.index[600:], feats] = df2.loc[df.index[600:], feats].sample(frac=1, random_state=0).to_numpy()
    pd.testing.assert_frame_equal(protocol.ml_transform(df2, feats, "LinearModel", tr, "benchmark")(X), out)   # rows outside training have no effect


def test_native_categorical_gbdt_roundtrip(data, tmp_path):
    df, y = data
    cat, num = feature_sets(df, "cat_050")
    ctx = regimes.Ctx(df, y, cat, num, cat_mode="benchmark")
    idx = df.index[:900]
    fit_idx, es_idx, ev = idx[:600], idx[600:750], idx[750:]
    for name in ("XGBoost", "LightGBM", "CatBoost"):
        spec = registry.get(name)
        f = regimes.fit(spec, ctx, fit_idx, es_idx, seed=5)
        assert f.config["cat_mode"] == "benchmark" and f.config["cat_representation"] == "native"
        gp = f.config["get_params"]
        assert (name != "XGBoost" or gp["enable_categorical"]) and (name != "CatBoost" or gp["cat_features"] == repr(cat) or gp["cat_features"] == cat)
        p0 = regimes.predict(f, ctx, ev)
        assert np.isfinite(p0).all()
        prefix = str(tmp_path / name)
        regimes.save(f, ctx, prefix)
        assert np.allclose(p0, regimes.predict(regimes.load(spec, ctx, prefix), ctx, ev))


def test_tune_ml_returns_searched_params_only(data):
    from defectshift.models import protocol
    df, y = data
    cat, num = feature_sets(df, "cat_050")
    best = protocol.tune_ml("reg", "XGBoost", df, cat + num, y, df.index[:500], df.index[500:650], n_trials=2, cat_mode="benchmark")
    assert best and "enable_categorical" not in best


# ------------------------------------------------------------------ benchmark tuning
def test_hpo_ml_record_and_reuse(data, tmp_path):
    from defectshift import hpo
    df, y = data
    cat, num = feature_sets(df, "cat_050")
    ctx = regimes.Ctx(df, y, cat, num, cat_mode="benchmark")
    tr, va = df.index[:400], df.index[400:500]
    rec = hpo.tune(registry.get("Ridge"), ctx, tr, va, n_trials=3)
    assert set(rec["params"]) == {"alpha"} and len(rec["trials"]) == 3 and rec["hpo_sec"] > 0
    assert np.isfinite(rec["best_score"]) and rec["cat_mode"] == "benchmark" and rec["n_train"] == 400
    prefix = str(tmp_path / "Ridge__source")
    hpo.save(rec, prefix)
    assert hpo.done(prefix) and hpo.load(prefix)["params"] == rec["params"]
    assert json.load(open(prefix + ".json"))["params"]["alpha"] == rec["params"]["alpha"]
    f = regimes.fit(registry.get("Ridge"), ctx, tr, None, seed=1, params=hpo.load(prefix)["params"])
    assert f.config["get_params"]["alpha"] == rec["params"]["alpha"] and f.config["cat_representation"] == "onehot"
    with pytest.raises(AssertionError, match="not a tuned model"):
        hpo.tune(registry.get("LinearRegression"), ctx, tr, va)


@pytest.mark.parametrize("name", ["RealMLP", "TabM"])
def test_pytabkit_hpo_fixed_refit_reproduces(data, tmp_path, name):
    """[GPU] The configuration chosen by pytabkit HPO is retrained with a fixed-configuration estimator of the same class: with the same rows
    and seed the predictions agree, and the applied configuration (fit_params_) equals the one chosen by HPO (not replaced by the TD
    defaults)."""
    from defectshift.models import tabular_dl
    df, y = data
    cat, num = feature_sets(df, "cat_050")
    feats = cat + num
    tr, va = df.index[:500], df.index[500:650]
    r = tabular_dl.hpo_pytabkit(name, df.loc[tr, feats], y.loc[tr], df.loc[va, feats], y.loc[va], cat, 3, 7)
    est = tabular_dl.build_fixed_pytabkit(name, r["fit_params"], 3, 7)
    med = df.loc[tr, feats].median(numeric_only=True)
    est.fit(df.loc[tr, feats].fillna(med), y.loc[tr].to_numpy("float32"), X_val=df.loc[va, feats].fillna(med),
            y_val=y.loc[va].to_numpy("float32"), cat_indicator=[c in cat for c in feats])
    after = tabular_dl._strip_fit_params(est.fit_params_)
    assert after == tabular_dl._strip_fit_params(r["fit_params"])                     # the whole configuration is the same
    p = np.asarray(est.predict(df.loc[va, feats].fillna(med))).reshape(-1)
    assert np.allclose(p, r["val_pred"], atol=1e-5)                            # same rows and seed give the same predictions
    import joblib
    joblib.dump(est, tmp_path / "m.joblib")
    assert np.allclose(np.asarray(joblib.load(tmp_path / "m.joblib").predict(df.loc[va, feats].fillna(med))).reshape(-1), p)


def test_hpo_transformer_mlp(data):
    """[GPU] The MLP search finishes with its trial records, and regimes.fit works with the chosen values."""
    from defectshift import hpo
    df, y = data
    cat, num = feature_sets(df, "cat_050")
    ctx = regimes.Ctx(df, y, cat, num, cat_mode="benchmark")
    tr, va = df.index[:500], df.index[500:650]
    rec = hpo.tune(registry.get("MLP"), ctx, tr, va, n_trials=2)
    assert rec["params"] and len(rec["trials"]) == 2
    f = regimes.fit(registry.get("MLP"), ctx, tr[:400], tr[400:], seed=3, params=rec["params"])
    assert np.isfinite(regimes.predict(f, ctx, va)).all()


# ------------------------------------------------------------------ selection audit (B_te re-evaluation without retraining)
def _load_selection_audit_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("03a_audit_selection", os.path.join(ROOT, "scripts", "03a_audit_selection.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_selection_audit_refuses_without_final_domains(monkeypatch):
    mod = _load_selection_audit_module()
    monkeypatch.setattr(sys, "argv", ["03a_audit_selection.py", "--screening-run-id", "no_such_run"])
    with pytest.raises(SystemExit, match="final_domains"):
        mod.main()


def test_selection_audit_guard_blocks_training():
    from defectshift.models import protocol
    mod = _load_selection_audit_module()
    before = protocol.fit_ml
    with mod.no_training():
        with pytest.raises(RuntimeError, match="training call"):
            regimes.fit(registry.get("XGBoost"), None, None, None, 0)
        for n in ("fit_ml", "fit_transformer", "fit_dl2", "tune_ml"):
            with pytest.raises(RuntimeError, match="training route"):
                getattr(protocol, n)()
    assert regimes.fit is mod._ORIG_FIT and protocol.fit_ml is before      # the original functions are restored after the run


def test_selection_audit_dataflow_reload_only(data, monkeypatch):
    """The selection audit only loads saved models (no training) and predicts only on B_te and, for the restore check, on B_screen."""
    df, y = data
    mod = _load_selection_audit_module()
    lot = df[config.LOT_COL]
    slot, A_idx, B_idx = _setup(df, "cat_050")
    a, b4 = get_A(slot, A_idx, y, lot), split_B4(B_idx, y, lot)
    models, offset = ["XGBoost", "TabPFN"], {"ID_match": 0.01, "OOD_match": 0.005}
    screening_rows = pd.DataFrame([dict(slot=slot, candidate="cat_050", kind="cat", repeat=i, model=m, regime=r, status="ok",
                                        n_A=len(A_idx), n_B=len(B_idx), n_star=100, b_te_eligible=True,
                                        r2_screen=0.5, rmse_screen=offset[r], mae_screen=offset[r])
                                   for i in range(3) for m in models for r in ("ID_match", "OOD_match")])   # the screening has 3 repetitions
    final = dict(categorical=["cat_050"], temporal=[], rows_sha256="x", manifest_sha256="y")
    monkeypatch.setattr(mod, "check_screening_run", lambda p, require_tuned=True: (screening_rows, dict(args=dict(models=models, repeats=3)), final))
    monkeypatch.setattr(mod.screening, "artifacts_ok", lambda prefix, route: True)
    monkeypatch.setattr(mod.manifest, "file_sha256", lambda path: "h")
    seen = dict(pred=[])
    A_train, _ = repeat_split(a["A_dev"], y, lot, 0)
    _, A_m, B_m = matched(A_train, b4["B_tr"], 0)

    def fake_load(spec, ctx, prefix):                          # imitation of a saved model with the same training rows as the protocol
        regime = "OOD_match" if "OOD_match" in prefix else "ID_match"
        pool = B_m if regime == "OOD_match" else A_m
        fit_, es_ = es_split(pool, y, lot, 0) if spec.uses_es else (pool, None)
        if spec.route == "foundation":                          # context rebuild: one call through the blocked training route
            return regimes.fit(spec, ctx, fit_, es_, 0)
        return regimes.Fitted(spec, None, fit_, es_, 0, {}, 0.0, {"regime": regime})
    monkeypatch.setattr(mod, "_ORIG_FIT", lambda spec, ctx, fit_, es_, seed: regimes.Fitted(
        spec, None, fit_, es_, seed, {}, 0.0, {"regime": "OOD_match" if set(fit_) <= set(B_m) else "ID_match"}))

    def fake_predict(f, ctx, idx):
        seen["pred"].append(set(idx))
        return y.loc[idx].to_numpy() + offset[f.config["regime"]]
    monkeypatch.setattr(mod.regimes, "load", fake_load)
    monkeypatch.setattr(mod.regimes, "predict", fake_predict)
    monkeypatch.setattr(sys, "argv", ["03a_audit_selection.py", "--screening-run-id", "fake", "--run-id", "selection_audit_dataflow_test",
                                      "--only", "cat_050", "--repeats", "1"])
    mod.main()
    te, scr = set(b4["B_te"]), set(b4["B_screen"])
    assert len(seen["pred"]) == 2 * 2 * 2 and all(p_ in (te, scr) for p_ in seen["pred"])      # 2 models x 2 regimes x (B_te, B_screen)
    assert sum(p_ == te for p_ in seen["pred"]) == 4
    run_dir = os.path.join(config.RESULTS_DIR, "selection_audit", "runs", "selection_audit_dataflow_test")
    r = pd.read_csv(os.path.join(run_dir, "rows.csv"))
    assert len(r) == 4 and (r.status == "ok").all() and (r.screen_rmse_diff <= 1e-12).all()
    assert (r[r.regime == "OOD_match"].te_lot_in_train == 0).all() and (r.n_B_te == len(te)).all()
    assert r[r.model == "TabPFN"].context_rebuilt.all() and not r[r.model == "XGBoost"].context_rebuilt.any()
    assert r.rows_match_protocol.all()
    rep = json.load(open(os.path.join(run_dir, "report", "report.json")))
    assert [d_["candidate"] for d_ in rep["selected_categorical"]] == ["cat_050"]
    assert np.isfinite(rep["selected_categorical"][0]["screen"])   # the B_screen consensus score from the evaluated repetition (1) only


def test_registry():
    assert len(registry.SPECS) == 20
    fam = pd.Series([s.family for s in registry.SPECS.values()]).value_counts().to_dict()
    assert fam == {"classical": 4, "tree": 6, "deep": 4, "foundation": 6}
    assert {s.name for s in registry.SPECS.values() if s.uses_es} == {"XGBoost", "LightGBM", "CatBoost", "MLP",
                                                                      "FT-Transformer", "RealMLP", "TabM"}


# ------------------------------------------------------------------ run lineage and input assumptions
def test_check_compat_requires_same_code_and_packages():
    """Using the search records of another run requires the shared training code (per-file hashes), packages and protocol constants to equal
    the current ones; an old manifest without per-file hashes is refused."""
    import json as _json
    from defectshift import manifest as _m
    from defectshift.runio import COMPAT_FILES, check_compat
    root = os.path.realpath(config.REPO_ROOT)
    good = dict(code_file_sha256={f: _m.file_sha256(os.path.join(root, f)) for f in COMPAT_FILES}, packages=_m._packages(),
                protocol=dict(constants=_json.loads(_json.dumps(_m.protocol_constants()))))
    check_compat(good, "same run")
    for label, man in [("old manifest", {k: v for k, v in good.items() if k != "code_file_sha256"}),
                       ("different code", dict(good, code_file_sha256=dict(good["code_file_sha256"], **{COMPAT_FILES[3]: "0" * 64}))),
                       ("different packages", dict(good, packages=dict(good["packages"], xgboost="0.0"))),
                       ("different constants", dict(good, protocol=dict(constants=dict(good["protocol"]["constants"], HPO_SEED=-1))))]:
        with pytest.raises(SystemExit):
            check_compat(man, label)


def test_categorical_assumption_checked():
    """Missing, negative or non-integer values in a categorical column stop the run before training (the real data passes)."""
    from defectshift.candidates import check_categorical
    df, _ = load_data()
    check_categorical(df)
    for bad in (np.nan, -1, 0.5):
        d2 = df[["cat_050"]].astype(float).copy()
        d2.iloc[3, 0] = bad
        with pytest.raises(ValueError, match="categorical columns violate the assumptions"):
            check_categorical(d2)


def test_safe_ratio_undefined_is_nan():
    assert np.isnan(metrics.safe_ratio(1.0, 0.0)) and np.isnan(metrics.pct_change(1.0, 0.0))
    assert metrics.pct_change(1.1, 1.0) == pytest.approx(10.0)
    assert np.isnan(metrics.target_reference_gap(dict(r2=0.1, rmse=1.0, mae=1.0), dict(r2=0.2, rmse=0.0, mae=1.0))["gap_rmse_pct"])


def test_fm_qualify_unmeasured_memory_fails():
    """A qualification run without a GPU (memory not measured) fails even if every other condition passes."""
    mod = _load_script_generic("05_qualify_foundation")
    row = dict(status="ok", shape_ok=True, finite=True, nonconstant=True, peak_gib=None, repro_max_abs_diff=0.0,
               isolation_max_abs_diff=0.0, fresh_max_abs_diff=0.0, reload_max_abs_diff=0.0, fit_sec=1.0, predict_sec=1.0)
    rows = [dict(row, n_context_target=n) for n in mod.CONTEXT_SIZES]
    j = mod.judge(rows)
    assert not j["passed"] and any("not measured" in r for r in j["reasons"])
    j2 = mod.judge([dict(r, peak_gib=1.0) for r in rows])
    assert j2["passed"]


def _load_script_generic(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hpo_check_record():
    """Search-record check (shared by the final benchmark and the screening): a different model, side, domain, required trial count, epoch
    cap or search-row hash is refused. With a required trial count of None (final-benchmark smoke runs), the trial count and epoch cap are
    not checked."""
    from defectshift import hpo
    spec = registry.get("Ridge")
    tr, va = np.arange(10), np.arange(10, 15)
    rec = dict(model="Ridge", side="source", slot="s", n_trials=3, max_epochs=10,
               train_idx_sha256=hpo._idx_sha(tr), val_idx_sha256=hpo._idx_sha(va))
    hpo.check_record(rec, spec, "source", "s", (tr, va), 3, 10)                  # screening smoke run (3 trials, 10 epochs)
    hpo.check_record(rec, spec, "source", "s", (tr, va), None)                   # final-benchmark smoke run (--allow-nonfinal-hpo)
    for args in [(spec, "target", "s", (tr, va), 3, 10), (spec, "source", "t", (tr, va), 3, 10),
                 (registry.get("kNN"), "source", "s", (tr, va), 3, 10),
                 (spec, "source", "s", (tr, va), config.HPO_N_TRIALS, None),        # full-run budget but a smoke record
                 (spec, "source", "s", (tr, va), 3, None),                          # different epoch cap
                 (spec, "source", "s", (va, tr), 3, 10)]:                           # different search rows
        with pytest.raises(SystemExit, match="search record cannot be used"):
            hpo.check_record(rec, *args)


def test_fit_max_epochs_caps_mlp(data):
    """The smoke-only epoch cap is applied to MLP training and recorded in the configuration (None = protocol value)."""
    df, y = data
    cat, num = feature_sets(df, "cat_050")
    ctx = regimes.Ctx(df, y, cat, num, cat_mode="benchmark")
    idx = df.index[:400]
    f = regimes.fit(registry.get("MLP"), ctx, idx[:300], idx[300:], seed=1, max_epochs=2)
    assert f.config["n_epochs"] <= 2 and f.config["max_epochs"] == 2 and f.config["smoke_max_epochs"] == 2
    f0 = regimes.fit(registry.get("Ridge"), ctx, idx[:300], None, seed=1)
    assert f0.config["smoke_max_epochs"] is None


def test_selection_current_and_orphan_cache(tmp_path, monkeypatch):
    """Version check of screening runs (for manuscript tables and merging), and refusal of run names whose cache remains without a results
    folder."""
    from defectshift import runio
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    for run, consts in (("screening_new", manifest.protocol_constants()), ("screening_old", {})):
        d = tmp_path / "screening" / "runs" / run
        d.mkdir(parents=True)
        json.dump(dict(protocol=dict(constants=json.loads(json.dumps(consts)))), open(d / "manifest.json", "w"))
    runio.check_selection_current("screening_new")
    with pytest.raises(SystemExit, match="protocol version"):
        runio.check_selection_current("screening_old")
    run_dir, cache = tmp_path / "res" / "r1", tmp_path / "cache" / "r1"
    runio.refuse_orphan_cache(str(run_dir), str(cache), "t")                 # neither exists: passes
    (cache / "slot").mkdir(parents=True)
    runio.refuse_orphan_cache(str(run_dir), str(cache), "t")                 # empty cache folder: passes
    (cache / "slot" / "Ridge__source.json").write_text("{}")
    with pytest.raises(SystemExit, match="a cache"):
        runio.refuse_orphan_cache(str(run_dir), str(cache), "t")
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")
    runio.refuse_orphan_cache(str(run_dir), str(cache), "t")                 # resume (manifest exists): passes
