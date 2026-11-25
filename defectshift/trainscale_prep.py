"""Input transformation of the training-row scale sensitivity analysis (arm trainscale).

The released numerical features are values placed in [0, 1] with the extremes over all records, the target-side records
included. A model whose internal preprocessing is not invariant to this affine map can therefore depend on the released scale.
This arm gives such models the released values re-min-max scaled with the extremes of the fit rows (fit_idx) of each training.

- A released value x' is an affine transformation of the original value x, so x'' = (x' - lo) / (hi - lo), with lo and hi the
  minimum and maximum of the observed x' over the fit rows, equals the original value min-max scaled with the fit-row extremes.
  Neither the original values nor their physical meaning is needed.
- Missing values are kept (not imputed). The uniform-preprocessing arm (uniform_prep) changes imputation and standardization
  together; this arm changes only the scale.
- A column with no observed value in the fit rows is left untouched. A column with a single value in the fit rows
  (hi == lo) is shifted to x' - lo.
- Early-stopping and evaluation rows only receive the same transformation (values outside the fit-row range lie outside [0, 1]).
- Categorical columns are unchanged.

Affected models are those whose internal preprocessing was not confirmed to be invariant to an affine map: RealMLP and TabM
(pytabkit), TabNet and the six foundation models. Tree models are invariant to monotone maps; linear, Ridge, kNN, SVR, MLP,
FT-Transformer and NODE are invariant to training-row standardization, so they are not affected.
The compatibility files (runio.COMPAT_FILES) are not changed; the transformation is defined in this file only. Called by the
trainscale arm of scripts/07a_sensitivity.py.
"""
import numpy as np

from defectshift import regimes

DESCRIPTION = ("num: released value re-min-max scaled with the min and max over the fit rows of each training (missing values kept, "
               "columns without observed fit-row values untouched, constant fit-row columns shifted only); early-stopping and "
               "evaluation rows transformed with the same constants; cat: unchanged")


def affected(spec):
    """Models whose input this arm changes: those whose internal preprocessing is not affine-invariant (tabular_dl = RealMLP and TabM, foundation, TabNet)."""
    return spec.route in ("tabular_dl", "foundation", "tabnet")


def change(spec):
    """What actually changes in this model's input (for the interpretation table). None if the model is not affected."""
    if not affected(spec):
        return None
    if spec.route == "tabular_dl":
        return "released scale replaced by fit-row min-max before the official preprocessing (robust scaling is affine-invariant)"
    if spec.route == "tabnet":
        return "fit-row min-max before the input batch normalization (its epsilon is absolute)"
    return "released scale replaced by fit-row min-max; the foundation model applies its own preprocessing"


def trainscale_ctx(ctx, fit_idx):
    """New context with the column-wise min-max fitted on fit_idx applied to all rows, and the information to record."""
    df = ctx.df.copy()
    cols = list(ctx.num_cols)
    fit = df.loc[fit_idx, cols].astype("float64")
    lo, hi = fit.min(), fit.max()                       # NaN if nothing is observed
    observed = [c for c in cols if not np.isnan(lo[c])]
    unobserved = [c for c in cols if np.isnan(lo[c])]
    span = (hi - lo)[observed]
    constant = [c for c in observed if span[c] <= 0.0]
    span = span.where(span > 0.0, 1.0)
    df[observed] = (df[observed].astype("float64") - lo[observed]) / span
    out = regimes.Ctx(df, ctx.y, ctx.cat_cols, ctx.num_cols, cat_mode=ctx.cat_mode, fm_input=ctx.fm_input)
    return out, dict(trainscale_unobserved_num_cols=unobserved, trainscale_constant_num_cols=constant, trainscale_fit_rows=len(fit_idx))
