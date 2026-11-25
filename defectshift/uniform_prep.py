"""Input transformation of the uniform-preprocessing sensitivity analysis.

In the main benchmark each model keeps its own preprocessing (models/protocol.py, regimes.py): LinearRegression, Ridge, kNN,
SVR, MLP and FT-Transformer use training-row median imputation and standardization, GBR uses imputation only, DecisionTree,
RandomForest, XGBoost, LightGBM and CatBoost receive missing values as they are (model-specific handling), RealMLP and TabM
use the official pytabkit preprocessing, and the foundation models receive raw numerical values (fm_input="raw_native").
This sensitivity analysis first applies training-row numerical median imputation and standardization, only to the models
whose input actually changes, and retrains them with the same hyperparameters.

- Transformation: utils.fit_feature_transform(..., True, fit_idx) as is (medians, means and standard deviations from the
  training rows fit_idx only, including the 1e-3 floor on the standard deviation). Early-stopping and evaluation rows only
  receive the same transformation.
- Numerical columns with no observed value in the training rows are 0 in every row (the same effect as the unobserved-column
  rule of regimes).
- Categorical columns: the cat_* columns of the released data have no missing values, so a fixed missing-category token
  would change nothing. If a missing value appears, the transformation stops instead of passing it silently.
- Preprocessing inside a model (the internal transformations of pytabkit and the foundation models, the categorical handling
  of boosting) is part of the model and is left as is. What changes is the numerical input given to the model.

The compatibility files (runio.COMPAT_FILES) are not changed; the transformation is defined in this file only. Called by the
uniform-preprocessing arm of scripts/07a_sensitivity.py.
"""
from defectshift import regimes
from defectshift.models import protocol
from defectshift.utils import fit_feature_transform

DESCRIPTION = ("num: train-median impute + standardize (scale floor) fitted on fit_idx, applied before the model's own "
               "preprocessing; num columns unobserved in fit_idx -> 0; cat: unchanged (no missing values in cat_*)")



def change(spec):
    """What the uniform preprocessing actually changes in this model's input (for the result interpretation table). None if
    the model is not affected."""
    if not affected(spec):
        return None
    if spec.route == "ml":       # DT, RF, XGBoost, LightGBM, CatBoost: trees that received missing values; standardization does not change splits
        return "missing values imputed (standardization does not change tree splits)"
    if spec.route == "tabular_dl":      # RealMLP, TabM: already training-row median imputation, followed by the official scaling
        return "standardization added before the official preprocessing (little change expected)"
    if spec.route == "foundation":
        return "raw values with missing entries replaced by imputed and standardized values"
    if spec.route == "node":     # the quantile transform is invariant to monotone transformations; only the relative scale of its noise changes
        from defectshift.models import node
        if node.NORMALIZE:   # NODE standardizes its input itself, so the input is unchanged (not affected)
            return None
        return "standardization before the quantile transform (only the relative scale of its noise changes)"
    if spec.route == "tabnet":
        return "standardization before the input batch normalization (little change expected)"
    raise ValueError(spec.route)


def affected(spec):
    """Whether the uniform preprocessing actually changes the model input. The transformer route (MLP, FT-T) and the models
    with imputation and standardization (LinearRegression, Ridge, kNN, SVR) already use the same numerical preprocessing; GBR
    imputes only and is a tree, so standardization does not change its splits."""
    if spec.route in ("tabular_dl", "foundation"):
        return True
    if spec.route == "node":    # NODE standardizes on the training rows before its quantile transform (node.NORMALIZE) -> input unchanged
        from defectshift.models import node
        return not node.NORMALIZE
    if spec.route == "tabnet":   # TabNet takes raw values (before its input batch normalization)
        return True
    if spec.route == "ml":
        return protocol.ml_needs_scale(spec.impl) is False
    return False


def uniform_ctx(ctx, fit_idx):
    """A new context with the numerical imputation and standardization fitted on fit_idx applied to every row, and the
    information to record."""
    if ctx.df[ctx.cat_cols].isna().any().any():
        bad = [c for c in ctx.cat_cols if ctx.df[c].isna().any()]
        raise ValueError(f"missing values in categorical columns (no missing-category token rule): {bad[:5]}")
    df = ctx.df.copy()
    unobserved = [c for c in ctx.num_cols if df.loc[fit_idx, c].isna().all()]
    observed = [c for c in ctx.num_cols if c not in unobserved]
    tf = fit_feature_transform(df, observed, True, fit_idx)
    df[observed] = tf(df[observed])
    if unobserved:
        df[unobserved] = 0.0
    out = regimes.Ctx(df, ctx.y, ctx.cat_cols, ctx.num_cols, cat_mode=ctx.cat_mode, fm_input=ctx.fm_input)
    return out, dict(uniform_preprocess=DESCRIPTION, uniform_unobserved_num_cols=unobserved, uniform_fit_rows=len(fit_idx))
