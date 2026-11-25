"""Direction helper and the constant-target guard of the R^2 metric. No model is fitted.
Run: python -m pytest tests/test_directions.py -q"""
import numpy as np
import pandas as pd

from defectshift.directions import same_direction
from defectshift.metrics import regression


def test_same_direction_scalars_and_pandas():
    assert same_direction(1.0, 2.0) and same_direction(-1, -3)
    assert not same_direction(1.0, -2.0)
    assert not same_direction(0.0, 0.0) and not same_direction(0.0, 1.0)    # zero is no direction
    assert not same_direction(np.nan, np.nan)
    got = same_direction(pd.Series([1.0, -1.0, 0.0, 2.0]), pd.Series([3.0, -2.0, 0.0, -1.0]))
    assert got.tolist() == [True, True, False, False]


def test_constant_target_has_no_r2():
    const = regression([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
    assert np.isnan(const["r2"]) and const["rmse"] == 0.0 and const["mae"] == 0.0
    assert regression([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])["r2"] == 1.0
