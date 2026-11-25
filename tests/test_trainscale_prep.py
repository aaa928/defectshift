"""Tests of the input transformation of the trainscale arm (synthetic data; no data file or GPU needed)."""
import numpy as np
import pandas as pd

from defectshift import regimes, trainscale_prep


def _ctx():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "a": rng.uniform(0.2, 0.9, n), "b": rng.uniform(0.0, 1.0, n), "const": np.full(n, 0.5),
        "unobs": np.full(n, np.nan), "c1": rng.integers(0, 3, n),
    })
    df.loc[[3, 7, 11], "a"] = np.nan
    y = pd.Series(rng.uniform(size=n))
    return regimes.Ctx(df, y, ["c1"], ["a", "b", "const", "unobs"], cat_mode="benchmark", fm_input="raw_native")


def test_fit_rows_min_max_and_missing_kept():
    ctx = _ctx()
    fit = list(range(0, 40))
    out, info = trainscale_prep.trainscale_ctx(ctx, fit)
    assert out.df["a"].isna().equals(ctx.df["a"].isna())              # missing positions unchanged
    for c in ("a", "b"):
        v = out.df.loc[fit, c]
        assert abs(v.min()) < 1e-12 and abs(v.max() - 1.0) < 1e-12     # fills [0, 1] on the fit rows
    assert info["trainscale_constant_num_cols"] == ["const"]
    assert (out.df.loc[fit, "const"] == 0.0).all()
    assert info["trainscale_unobserved_num_cols"] == ["unobs"] and out.df["unobs"].isna().all()   # untouched
    assert out.df["c1"].equals(ctx.df["c1"])                          # categorical column unchanged
    assert ctx.df["a"].equals(_ctx().df["a"])                         # the input context is not modified


def test_affine_and_eval_rows_outside_range():
    ctx = _ctx()
    out, _ = trainscale_prep.trainscale_ctx(ctx, list(range(0, 20)))
    ok = ctx.df["b"].notna()
    assert abs(np.corrcoef(ctx.df.loc[ok, "b"], out.df.loc[ok, "b"])[0, 1] - 1.0) < 1e-12   # affine transformation
    assert out.df.loc[20:, "b"].max() > 1.0 or out.df.loc[20:, "b"].min() < 0.0           # evaluation rows can lie outside the range


def test_affected_models():
    class S:
        def __init__(self, route): self.route = route
    assert [r for r in ("ml", "transformer", "tabular_dl", "foundation", "tabnet", "node") if trainscale_prep.affected(S(r))] == ["tabular_dl", "foundation", "tabnet"]
