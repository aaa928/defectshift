"""Checks of the target (y) standardization rule. Run: python -m pytest tests/test_target_scaling.py -q

- The 12 models of the ml and transformer routes train on y standardized with training-row statistics and map predictions back to the
  original scale.
- tabular_dl (RealMLP, TabM) and the foundation models are not standardized (official implementations, official interfaces).
Checks that use the GPU (the three boosting models, MLP, FT-Transformer) run only with DEFECTSHIFT_RUN_GPU_TESTS=1 (so that they do not
compete with other GPU work).
"""
import json
import os

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from defectshift import regimes, registry  # noqa: E402
from defectshift.candidates import feature_sets, load_data  # noqa: E402

gpu = pytest.mark.skipif(os.environ.get("DEFECTSHIFT_RUN_GPU_TESTS") != "1", reason="GPU check: only with DEFECTSHIFT_RUN_GPU_TESTS=1")
CPU_MODELS = ("LinearRegression", "Ridge", "kNN", "SVR", "DecisionTree", "RandomForest", "GBR")
GPU_MODELS = ("XGBoost", "LightGBM", "CatBoost", "MLP", "FT-Transformer")


@pytest.fixture(scope="module")
def data():
    return load_data()


def _ctx(df, y):
    cat, num = feature_sets(df, "cat_050")
    return regimes.Ctx(df, y, cat, num, cat_mode="benchmark")


def _rows(df):
    idx = df.index[:900]
    return idx[:600], idx[600:720], idx[720:]          # training, early stopping, evaluation


def test_routes():
    """Standardized = the 12 ml and transformer models; excluded = the 2 tabular_dl and 6 foundation models."""
    on = sorted(n for n, s in registry.SPECS.items() if s.route in regimes.Y_STANDARDIZED_ROUTES)
    assert on == sorted(CPU_MODELS + GPU_MODELS)
    import pandas as pd
    y = pd.Series(np.arange(10, dtype=float))
    for n, s in registry.SPECS.items():
        out, rec = regimes.standardized_target(s, y, y.index[:6])
        if n in on:
            assert rec["mean"] == pytest.approx(2.5) and rec["std"] == pytest.approx(np.std(np.arange(6.0)))
            assert np.allclose(out, (y - 2.5) / rec["std"])
        else:
            assert rec is None and out is y


def test_constant_target_refused():
    import pandas as pd
    with pytest.raises(ValueError, match="cannot standardize y"):
        regimes.y_stats(pd.Series([0.05] * 8), range(8))


@pytest.mark.parametrize("name", CPU_MODELS)
def test_cpu_models_roundtrip(data, tmp_path, name):
    """Recorded statistics = mean and standard deviation of the training rows, predictions on the original scale (finite), the same
    predictions after save and reload."""
    df, y = data
    ctx = _ctx(df, y)
    fit_idx, _, ev = _rows(df)
    spec = registry.get(name)
    f = regimes.fit(spec, ctx, fit_idx, None, seed=1)
    ys = f.config["y_standardization"]
    assert ys["mean"] == pytest.approx(float(y.loc[fit_idx].mean()))
    assert ys["std"] == pytest.approx(float(y.loc[fit_idx].std(ddof=0))) and ys["n_rows"] == len(fit_idx)
    p0 = regimes.predict(f, ctx, ev)
    assert np.isfinite(p0).all()
    assert abs(np.mean(p0) - y.loc[ev].mean()) < 5 * y.std()          # original scale (on the standardized scale the mean would be near 0)
    prefix = str(tmp_path / name)
    regimes.save(f, ctx, prefix)
    assert json.load(open(prefix + ".json"))["config"]["y_standardization"]["std"] == pytest.approx(ys["std"])
    assert np.allclose(p0, regimes.predict(regimes.load(spec, ctx, prefix), ctx, ev), rtol=0, atol=1e-12)


def test_stats_use_fit_rows_only(data):
    """Changing y outside the training rows leaves the statistics and predictions unchanged (y of evaluation or other rows does not leak
    into training)."""
    df, y = data
    fit_idx, _, ev = _rows(df)
    y2 = y.copy()
    y2.loc[ev] = 1e6
    spec = registry.get("Ridge")
    f1 = regimes.fit(spec, _ctx(df, y), fit_idx, None, seed=1)
    f2 = regimes.fit(spec, _ctx(df, y2), fit_idx, None, seed=1)
    assert f1.config["y_standardization"] == f2.config["y_standardization"]
    assert np.array_equal(regimes.predict(f1, _ctx(df, y), ev), regimes.predict(f2, _ctx(df, y2), ev))


@pytest.mark.parametrize("name", ["LinearRegression", "Ridge", "kNN"])
def test_invariant_models_match_raw_scale(data, monkeypatch, name):
    """Models invariant to affine transformations of y: original-scale predictions of standardized training = predictions of
    original-scale training (within numerical error). Decision trees are invariant in theory but are not included: y takes fractional
    values (0.03125 and so on), so ties in split improvement are frequent, and floating-point rounding after standardization picks other
    splits (a maximum per-row difference of 0.125 was measured for fully grown trees with default settings)."""
    df, y = data
    ctx = _ctx(df, y)
    fit_idx, _, ev = _rows(df)
    spec = registry.get(name)
    p_std = regimes.predict(regimes.fit(spec, ctx, fit_idx, None, seed=1), ctx, ev)
    monkeypatch.setattr(regimes, "Y_STANDARDIZED_ROUTES", ())
    f_raw = regimes.fit(spec, ctx, fit_idx, None, seed=1)
    assert f_raw.config["y_standardization"] is None
    assert np.max(np.abs(p_std - regimes.predict(f_raw, ctx, ev))) < 1e-9 * max(1.0, float(y.abs().max()))


def test_old_artifact_without_record_is_raw_scale(data):
    """Old artifacts without a y_standardization record do not map predictions back (compatibility for re-inferring old results)."""
    df, y = data
    ctx = _ctx(df, y)
    fit_idx, _, ev = _rows(df)
    f = regimes.fit(registry.get("Ridge"), ctx, fit_idx, None, seed=1)
    model_scale = regimes._predict_model_scale(f, ctx, ev)
    f.config.pop("y_standardization")
    assert np.array_equal(regimes.predict(f, ctx, ev), model_scale)


def test_hpo_standardizes_with_train_rows(data):
    """The search also standardizes with the statistics of the search training rows and records them. The validation R^2 (best score)
    equals the R^2 measured after retraining on the original scale."""
    from sklearn.metrics import r2_score
    from defectshift import hpo
    df, y = data
    ctx = _ctx(df, y)
    tr, va = df.index[:500], df.index[500:650]
    rec = hpo.tune(registry.get("Ridge"), ctx, tr, va, n_trials=3)
    assert rec["y_standardization"]["mean"] == pytest.approx(float(y.loc[tr].mean()))
    f = regimes.fit(registry.get("Ridge"), ctx, tr, None, seed=1, params=rec["params"])
    assert r2_score(y.loc[va], regimes.predict(f, ctx, va)) == pytest.approx(rec["best_score"], abs=1e-9)


@gpu
@pytest.mark.parametrize("name", GPU_MODELS)
def test_gpu_models_roundtrip(data, tmp_path, name):
    df, y = data
    ctx = _ctx(df, y)
    fit_idx, es_idx, ev = _rows(df)
    spec = registry.get(name)
    f = regimes.fit(spec, ctx, fit_idx, es_idx, seed=1)
    assert f.config["y_standardization"]["mean"] == pytest.approx(float(y.loc[fit_idx].mean()))
    p0 = regimes.predict(f, ctx, ev)
    assert np.isfinite(p0).all() and abs(np.mean(p0) - y.loc[ev].mean()) < 5 * y.std()
    prefix = str(tmp_path / name)
    regimes.save(f, ctx, prefix)
    assert np.allclose(p0, regimes.predict(regimes.load(spec, ctx, prefix), ctx, ev), rtol=0, atol=1e-6)


@gpu
def test_catboost_overfitting_detector_applied(data):
    """With the fixed values of the official Gorishniy configuration (od_pval 0.001 + early stopping 50) given together, training works and
    both remain in the settings actually applied."""
    df, y = data
    ctx = _ctx(df, y)
    fit_idx, es_idx, _ = _rows(df)
    f = regimes.fit(registry.get("CatBoost"), ctx, fit_idx, es_idx, seed=1)
    allp = f.model.get_all_params()
    got = {k: allp.get(k) for k in ("od_type", "od_pval", "od_wait")}
    assert got["od_type"] == "IncToDec" and got["od_pval"] == pytest.approx(0.001) and int(got["od_wait"]) == 50, got


# Comparing the early-stopping rounds of the three boosting models on the original and the standardized scale is a measurement, not a
# pass/fail test (complete numerical invariance is not guaranteed) -> recorded by scripts/checks/check_gpu_models.py.


def _toy_ctx(seed=0, n=600, scale=0.06):
    """Synthetic regression data on the small scale of defect rates (mean 0.05, standard deviation about scale). 3 numerical columns + 1 categorical."""
    import pandas as pd
    g = np.random.RandomState(seed)
    df = pd.DataFrame({"num_a": g.rand(n), "num_b": g.rand(n), "num_c": g.rand(n), "cat_x": g.randint(0, 3, n)})
    y = pd.Series(0.05 + scale * (2 * df.num_a - 1) + 0.2 * scale * g.randn(n), name="y")
    return regimes.Ctx(df, y, ["cat_x"], ["num_a", "num_b", "num_c"], cat_mode="benchmark")


def test_gbr_not_constant_on_standardized_target(monkeypatch):
    """min_impurity_decrease 0.01 of Grinsztajn Table 7 is a normal split criterion on standardized y (the predictions vary). On
    original-scale y (variance about 0.001) it blocks every split and yields a constant predictor — one of the reasons y standardization
    covers every model, checked through the actual training path."""
    ctx = _toy_ctx()
    idx = ctx.df.index
    params = dict(n_estimators=50, learning_rate=0.1, max_depth=3, min_impurity_decrease=0.01)
    spec = registry.get("GBR")
    p_std = regimes.predict(regimes.fit(spec, ctx, idx[:400], None, seed=1, params=params), ctx, idx[400:])
    assert np.std(p_std) > 0.5 * float(ctx.y.loc[idx[400:]].std())
    monkeypatch.setattr(regimes, "Y_STANDARDIZED_ROUTES", ())
    p_raw = regimes.predict(regimes.fit(spec, ctx, idx[:400], None, seed=1, params=params), ctx, idx[400:])
    assert np.ptp(p_raw) == 0.0


@pytest.mark.parametrize("name", CPU_MODELS)
def test_cpu_models_same_seed_refit(data, name):
    """Retraining from scratch on the same rows with the same seed gives the same predictions (a property separate from save and reload).
    Only the random forest differs at machine precision (measured 2.8e-17) because of the summation order of parallel prediction
    (n_jobs=-1), so up to 1e-12 is allowed; the others must be exactly equal."""
    df, y = data
    ctx = _ctx(df, y)
    fit_idx, _, ev = _rows(df)
    spec = registry.get(name)
    p1 = regimes.predict(regimes.fit(spec, ctx, fit_idx, None, seed=7), ctx, ev)
    p2 = regimes.predict(regimes.fit(spec, ctx, fit_idx, None, seed=7), ctx, ev)
    assert np.max(np.abs(p1 - p2)) <= (1e-12 if name == "RandomForest" else 0.0)


@pytest.mark.parametrize("name", ["Ridge", "kNN", "DecisionTree"])
def test_cached_eval_transform_matches_refit(data, tmp_path, name):
    """Prediction reusing the evaluation transform made at training time = prediction of the reloaded model that builds the transform anew
    (exactly the same values)."""
    df, y = data
    ctx = _ctx(df, y)
    fit_idx, _, ev = _rows(df)
    spec = registry.get(name)
    f = regimes.fit(spec, ctx, fit_idx, None, seed=1)
    assert getattr(f, "_eval_tf", None) is not None
    p_cached = regimes.predict(f, ctx, ev)
    prefix = str(tmp_path / name)
    regimes.save(f, ctx, prefix)
    g = regimes.load(spec, ctx, prefix)
    assert getattr(g, "_eval_tf", None) is None
    assert np.array_equal(p_cached, regimes.predict(g, ctx, ev))
