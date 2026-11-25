"""Gap formulas of defectshift/evaluation.py on a toy example (numpy only, no data and no model)."""
import numpy as np
import pytest

from defectshift.evaluation import gaps_from_predictions, k_among_controls, rmse_mae


def test_rmse_mae():
    assert rmse_mae([0, 2], [1, 1]) == (1.0, 1.0)


def test_gaps_from_predictions_toy():
    g = gaps_from_predictions(y_heldout=[0, 2], pred_s_heldout=[1, 1], y_target=[0, 4], pred_s_target=[0, 2], pred_t_target=[1, 3])
    assert g["s_rmse"] == pytest.approx(2.0) and g["s_mae"] == pytest.approx(2.0)
    assert g["id_ood_gap_rmse"] == pytest.approx(100 * (np.sqrt(2) - 1)) and g["target_reference_gap_rmse"] == pytest.approx(100 * (np.sqrt(2) - 1))
    assert g["id_ood_gap_mae"] == pytest.approx(0.0) and g["target_reference_gap_mae"] == pytest.approx(0.0)
    assert list(g) == ["s_rmse", "s_mae", "id_ood_gap_rmse", "target_reference_gap_rmse", "id_ood_gap_mae", "target_reference_gap_mae"]


def test_k_among_controls():
    assert k_among_controls(2.5, [1.0, 2.0, 3.0, 4.0]) == 2
    assert k_among_controls(0.0, [1.0, 2.0]) == 0
