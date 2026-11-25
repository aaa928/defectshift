"""Gaps of the paper computed from predictions (no model code, no data access).

Give the predictions of the source-trained model f_S on the source test set A_heldout and on the target rows, and the predictions of the
target-trained reference f_T on the same target rows; gaps_from_predictions returns the ID-OOD gap, the target-reference gap (RMSE and MAE, in percent;
positive = a loss of the source-trained model) and the outcome-scale ratios s. k_among_controls places a gap among the gaps of the randomized lot-split
controls (not a significance test). docs/usage.md, "Evaluating a model", defines the quantities.
"""
import numpy as np


def rmse_mae(y_true, y_pred):
    err = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def gaps_from_predictions(y_heldout, pred_s_heldout, y_target, pred_s_target, pred_t_target):
    """Returns a dict with s_rmse, s_mae, id_ood_gap_rmse, target_reference_gap_rmse, id_ood_gap_mae and target_reference_gap_mae.

    y_heldout, pred_s_heldout: outcomes of A_heldout and the predictions of f_S on it; y_target, pred_s_target, pred_t_target: outcomes of the evaluated
    target rows and the predictions of f_S and f_T on them. s is the outcome-scale ratio of the two test sets: the standard deviation of the target
    outcomes over that of A_heldout for RMSE, and the mean absolute deviation around the median over the same for MAE."""
    err_id = rmse_mae(y_heldout, pred_s_heldout)
    err_s = rmse_mae(y_target, pred_s_target)
    err_t = rmse_mae(y_target, pred_t_target)
    yt, yh = np.asarray(y_target, float), np.asarray(y_heldout, float)
    out = dict(s_rmse=yt.std() / yh.std(), s_mae=np.abs(yt - np.median(yt)).mean() / np.abs(yh - np.median(yh)).mean())
    for k, measure in enumerate(("rmse", "mae")):
        out[f"id_ood_gap_{measure}"] = 100 * (err_s[k] / err_id[k] - 1)
        out[f"target_reference_gap_{measure}"] = 100 * (err_s[k] / err_t[k] - 1)
    return out


def k_among_controls(gap, control_gaps):
    """Number of controls whose target-reference gap is smaller than `gap`."""
    return int((np.asarray(control_gaps, float) < gap).sum())
