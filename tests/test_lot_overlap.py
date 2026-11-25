"""The lot-overlap annotation (splits/target_test_lot_overlap.csv and splits/source_heldout_lot_overlap.csv): structure, paper values, and agreement with a fresh computation.
Run: python -m pytest tests/test_lot_overlap.py -q

The first three tests read only the released tables. The last recomputes the tables from data/data.parquet with the released split code
(it is skipped when the data file is absent; the split caches go to the temporary folder set by conftest.py).
"""
import importlib.util
import os

import pandas as pd
import pytest

from defectshift import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "splits", "target_test_lot_overlap.csv")
SOURCE_TABLE = os.path.join(ROOT, "splits", "source_heldout_lot_overlap.csv")

# target test records and records whose lot also occurs in the source development set (paper Table III, "Shared")
CHALLENGE = {
    "cat_005": (1736, 197), "cat_019": (1959, 214), "cat_039": (2080, 84), "cat_045": (1528, 20), "cat_050": (850, 0),
    "x_time<=2099-05-04 21:18:39": (1784, 0), "y_time<=2099-05-10 02:20:07.500000": (1789, 30),
}


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(TABLE)


def test_columns_and_unique_rows(table):
    assert list(table.columns) == ["slot", "row", "source_train_lot_overlap"]
    assert table["source_train_lot_overlap"].dtype == bool
    assert not table.duplicated(["slot", "row"]).any()
    assert table["slot"].nunique() == 17


def test_paper_values(table):
    for slot, (n, marked) in CHALLENGE.items():
        t = table[table.slot == slot]
        assert (len(t), int(t["source_train_lot_overlap"].sum())) == (n, marked), slot
    controls = table[table.slot.str.startswith("placebo@")]
    assert sorted(controls.slot.unique()) == sorted(f"placebo@{s}" for s in config.PLACEBO_B_TARGET)
    assert not controls["source_train_lot_overlap"].any()                       # whole lots are assigned to one side
    sizes = controls.groupby("slot").size()
    assert (sizes.min(), sizes.max()) == (685, 2247)


def test_source_heldout_table():
    s = pd.read_csv(SOURCE_TABLE)
    assert list(s.columns) == ["slot", "row", "target_test_lot_overlap"]
    assert s["target_test_lot_overlap"].dtype == bool
    assert not s.duplicated(["slot", "row"]).any()
    assert s["slot"].nunique() == 17
    controls = s[s.slot.str.startswith("placebo@")]
    assert not controls["target_test_lot_overlap"].any()                        # whole lots are assigned to one side


@pytest.mark.skipif(not os.path.exists(config.DATA_PATH), reason="data/data.parquet is not present")
def test_matches_fresh_computation(table):
    spec = importlib.util.spec_from_file_location("make_lot_overlap", os.path.join(ROOT, "splits", "make_lot_overlap.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from defectshift.candidates import load_data
    df, y = load_data()
    fresh, fresh_source = mod.build(df, y, df[config.LOT_COL])
    assert fresh.equals(table)
    assert fresh_source.equals(pd.read_csv(SOURCE_TABLE))
