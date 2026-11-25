"""Evaluation metrics and the target-reference gap. For every metric, a positive gap means that the source-trained model
(A_match) is worse."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    # R^2 is undefined when the evaluation target is constant (zero denominator; sklearn would return 0 or 1): NaN instead
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 and np.ptp(y_true) > 0 else float("nan")
    return dict(r2=r2, rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                mae=mean_absolute_error(y_true, y_pred))


def target_reference_gap(id_match, ood_match):
    """id_match / ood_match: regression() results on the same evaluation set.
    R^2: difference (OOD_match - ID_match); RMSE and MAE: percent increase ((ID_match - OOD_match) / OOD_match x 100)."""
    return dict(
        r2_difference=ood_match["r2"] - id_match["r2"],
        gap_rmse_pct=pct_change(id_match["rmse"], ood_match["rmse"]),
        gap_mae_pct=pct_change(id_match["mae"], ood_match["mae"]),
    )


def safe_ratio(a, b):
    """a / b; NaN where the denominator is zero or a value is not finite, so that infinities and division errors do
    not reach tables and figures."""
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    np.divide(a, b, out=out, where=ok)
    return out if out.ndim else float(out)


def pct_change(a, b):
    """(a - b) / b x 100, NaN under the same conditions as safe_ratio."""
    return safe_ratio(a - b, b) * 100
