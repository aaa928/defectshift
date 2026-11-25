"""Checks of the search spaces and training rules. Run: python -m pytest tests/test_search_spaces.py -q

- For every model, the values drawn from the search space reach the actual estimator arguments as they are (names, ranges).
- Early stopping of MLP and FT-Transformer = stop after patience + 1 consecutive epochs without improvement, improvement threshold 0
  (Gorishniy et al. 2021).
- FT-Transformer weight decay excludes the tokenizer, normalization and biases; MLP decays all parameters.
XGBoost and CatBoost estimators are only built, not trained. MLP and FT-T train very small models for 1-2 epochs (on the GPU if visible).
To hide the GPU, set CUDA_VISIBLE_DEVICES="" when running (changing the environment variable in this file would hide the GPU from the
other tests of the same pytest run too, and the CatBoost GPU test would fail).
"""
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from defectshift.models import ml, protocol, transformer as tr  # noqa: E402

ML = ("LinearModel", "kNN", "SVM_rbf", "DecisionTree", "RandomForest", "GBR", "XGBoost", "LightGBM", "CatBoost")
N_DRAW = 25


def _draws(suggest, name):
    """N_DRAW draws from a random sampler as (parameter dict, params recorded by Optuna) pairs."""
    out = []
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    def obj(trial):
        out.append((suggest(trial), dict(trial.params)))
        return 0.0
    study.optimize(obj, n_trials=N_DRAW)
    return out


@pytest.mark.parametrize("name", ML)
def test_ml_params_reach_estimator(name):
    """The drawn values are in the estimator's get_params as they are (not silently ignored or passed under another name)."""
    for p, _ in _draws(lambda t: ml._suggest_params("reg", name, t), name):
        est = ml.build_one_model("reg", name, p)
        got = est.get_params()
        for k, v in p.items():
            assert k in got, (name, k)
            assert got[k] == v, (name, k, got[k], v)


def test_ml_ranges():
    """The same ranges (bounds included) and choices as the source tables."""
    def span(name, key):
        vals = [p[key] for p, _ in _draws(lambda t: ml._suggest_params("reg", name, t), name) if p[key] is not None]
        return min(vals), max(vals)
    lo, hi = span("LinearModel", "alpha")
    assert 1e-4 <= lo and hi <= 1e5
    lo, hi = span("kNN", "n_neighbors")
    assert 1 <= lo and hi <= 50
    assert {p["p"] for p, _ in _draws(lambda t: ml._suggest_params("reg", "kNN", t), "kNN")} <= {1, 2, 3, 4, 5}
    for p, _ in _draws(lambda t: ml._suggest_params("reg", "GBR", t), "GBR"):
        assert p["max_depth"] in (None, 2, 3, 4, 5) and p["min_impurity_decrease"] in (0.0, 0.01, 0.02, 0.05)
        assert 10 <= p["n_estimators"] <= 1000 and 2 <= p["min_samples_leaf"] <= 50 and 1e-4 <= p["learning_rate"] <= 1
    for p, raw in _draws(lambda t: ml._suggest_params("reg", "XGBoost", t), "XGBoost"):
        for k, lab in (("gamma", "gamma"), ("reg_lambda", "lambda"), ("reg_alpha", "alpha")):
            assert (p[k] == 0.0) == (not raw[f"optional_{lab}"])            # '?loguniform': 0 when False
            assert p[k] == 0.0 or 1e-8 <= p[k] <= 1e2
    for p, _ in _draws(lambda t: ml._suggest_params("reg", "RandomForest", t), "RandomForest"):
        assert p["max_features"] in ("sqrt", None) and (p["max_depth"] is None or 3 <= p["max_depth"] <= 50)


def test_fixed_boosting_settings():
    """Fixed values of the sources: LightGBM bagging_freq = 1 (TabArena), CatBoost od_pval = 0.001 (official Gorishniy configuration),
    boosting at most 2000 rounds with early stopping after 50."""
    assert ml.build_one_model("reg", "LightGBM", {}).get_params()["subsample_freq"] == 1
    cb = ml.build_one_model("reg", "CatBoost", {}).get_params()
    # both values must be in the constructor for CatBoost to apply IncToDec (p 0.001, wait 50) (early stopping passed to fit switches to Iter and ignores od_pval)
    assert cb["od_pval"] == 0.001 and cb["early_stopping_rounds"] == 50 and cb["iterations"] == ml.BOOSTING_MAX_ITER == 2000
    assert cb["allow_writing_files"] is False
    assert ml.EARLY_STOPPING_ROUNDS == 50 and ml.LIGHTGBM_DEVICE == "cpu" and ml.device_of("LightGBM") == "cpu"


def test_transformer_spaces_build():
    """MLP: first, middle and last layer sizes (the middle ones shared), FT-T: d_token a multiple of 8, ffn = int(d_token x factor). Models
    are built from the drawn values."""
    for p, raw in _draws(lambda t: tr._suggest_params("MLP", t), "MLP"):
        d = p["d_layers"]
        assert len(d) == raw["n_layers"] and all(1 <= x <= 512 for x in d)
        if len(d) > 2:
            assert len(set(d[1:-1])) == 1
        assert (p["dropout"] == 0.0) == (not raw["optional_dropout"])
        tr.build_one_model("MLP", p, [3, 4], 5)
    for p, _ in _draws(lambda t: tr._suggest_params("FT_Transformer", t), "FT"):
        assert p["d_token"] % 8 == 0 and 64 <= p["d_token"] <= 512 and 1 <= p["n_blocks"] <= 4
        assert 2 / 3 <= p["d_ffn_factor"] <= 8 / 3 and 1e-5 <= p["lr"] <= 1e-3
        m = tr.build_one_model("FT_Transformer", p, [3, 4], 5)
        ffn = m.transformer.blocks[0]["ffn"].linear_first.out_features
        assert ffn == 2 * int(p["d_token"] * p["d_ffn_factor"])                 # ReGLU doubles the output


def test_legacy_default_params_unchanged():
    """The FT-T architecture of the fixed settings (DEFAULT_PARAMS) keeps the earlier fixed values (attention 0.2, ffn 0.1, residual 0, ffn = 2 x d_token)."""
    m = tr.build_one_model("FT_Transformer", tr.DEFAULT_PARAMS["FT_Transformer"], [3], 4)
    assert m.transformer.blocks[0]["ffn"].linear_first.out_features == 2 * 2 * 64


def _toy_xy(n=64, seed=0):
    g = np.random.RandomState(seed)
    return (g.randint(0, 3, (n, 1)), g.randn(n, 4).astype("float32")), g.randn(n).astype("float32")


def test_early_stopping_counts_patience_plus_one():
    """With learning rate 0 only the first epoch improves (infinity -> value); afterwards the validation MSE stays the same, so training
    stops at 1 + (patience + 1) epochs."""
    X, y = _toy_xy()
    m = tr.build_one_model("MLP", dict(d_layers=[8], dropout=0.0, cat_emb_dim=8, lr=0.0), [3], 4)
    tr.fit_with_val("reg", "MLP", m, X, y, X, y, lr=0.0, weight_decay=0.0, max_epochs=100, patience=16, batch_size=32)
    assert m._best_epoch == 1 and m._n_epochs == 1 + 17
    assert protocol.TRANSFORMER_PATIENCE == 16


def test_weight_decay_groups(monkeypatch):
    """FT-T: a decayed group + a group with decay 0 (tokenizer, normalization, biases); MLP: all parameters in one group."""
    seen = {}

    class _Spy(torch.optim.AdamW):
        def __init__(self, params, **kw):
            params = list(params)
            seen["groups"], seen["wd"] = params, kw["weight_decay"]
            super().__init__(params, **kw)
    monkeypatch.setattr(torch.optim, "AdamW", _Spy)
    X, y = _toy_xy()
    p = dict(d_token=16, n_blocks=1, lr=1e-3, weight_decay=3e-4)
    m = tr.build_one_model("FT_Transformer", p, [3], 4)
    tr.fit_with_val("reg", "FT_Transformer", m, X, y, X, y, lr=1e-3, weight_decay=3e-4, max_epochs=1, batch_size=32)
    g = seen["groups"]
    assert seen["wd"] == 3e-4 and len(g) == 2 and g[1]["weight_decay"] == 0.0
    names_no_wd = {n for n, v in m.named_parameters() if any(v is x for x in g[1]["params"])}
    assert names_no_wd and all(("feature_tokenizer" in n) or ("normalization" in n) or n.endswith(".bias")
                               for n in names_no_wd)
    m2 = tr.build_one_model("MLP", dict(d_layers=[8], dropout=0.0, cat_emb_dim=8, lr=1e-3), [3], 4)
    tr.fit_with_val("reg", "MLP", m2, X, y, X, y, lr=1e-3, weight_decay=1e-4, max_epochs=1, batch_size=32)
    assert isinstance(seen["groups"][0], torch.nn.Parameter) and len(seen["groups"]) == len(list(m2.parameters()))


def test_tail_gain():
    """Decrease of the lowest validation MSE over the last k epochs: positive if it keeps improving to the end, 0 if it stopped earlier,
    None with k epochs or fewer."""
    assert tr.tail_gain(list(range(40, 0, -1)), k=30) == pytest.approx(30.0)
    assert tr.tail_gain([3.0, 1.0] + [2.0] * 35, k=30) == 0.0
    assert tr.tail_gain([1.0] * 30, k=30) is None
