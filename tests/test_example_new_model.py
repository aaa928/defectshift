"""The minimal example (examples/evaluate_new_model.py) follows the protocol: matched sample sizes and the target rows used. Run: python -m pytest tests/test_example_new_model.py -q

Needs data/data.parquet (skipped otherwise); the split caches go to the temporary folder set by conftest.py.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from defectshift import config
from defectshift.candidates import load_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(config.DATA_PATH), reason="data/data.parquet is not present")


def _example():
    spec = importlib.util.spec_from_file_location("evaluate_new_model", os.path.join(ROOT, "examples", "evaluate_new_model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fast_model():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy="median"), Ridge())


def test_sizes_and_finite_gaps():
    ex = _example()
    df, y = load_data()
    overlap = pd.read_csv(ex.OVERLAP)
    res = ex.evaluate_slot(df, y, df[config.LOT_COL], "cat_050", overlap, model_factory=_fast_model)
    assert (res["n_star"], res["n_target_rows"]) == (1611, 850)                  # paper Table 2: n* and B_te of the cat_050 domain
    ctl = ex.evaluate_slot(df, y, df[config.LOT_COL], "placebo@9001", overlap, model_factory=_fast_model)
    assert (ctl["n_star"], ctl["n_target_rows"]) == (1315, 685)                  # smallest control
    for r in (res, ctl):
        assert all(np.isfinite(v) for k, v in r.items() if k.endswith("rmse") or k.endswith("mae"))


def test_all_rows_option_uses_every_target_record():
    ex = _example()
    df, y = load_data()
    overlap = pd.read_csv(ex.OVERLAP)
    # cat_005 has 197 of 1,736 target records whose lot also occurs in the source development set
    ld = ex.evaluate_slot(df, y, df[config.LOT_COL], "cat_005", overlap, model_factory=_fast_model)
    al = ex.evaluate_slot(df, y, df[config.LOT_COL], "cat_005", overlap, model_factory=_fast_model, all_rows=True)
    assert (ld["n_target_rows"], al["n_target_rows"]) == (1539, 1736)
