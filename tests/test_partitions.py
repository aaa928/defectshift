"""The partition table (splits/partitions.csv): structure, agreement with the lot-overlap tables, and agreement with a fresh computation.
Run: python -m pytest tests/test_partitions.py -q

The first two tests read only the released tables. The last recomputes the table from data/data.parquet with the released split code
(it is skipped when the data file is absent; the split caches go to the temporary folder set by conftest.py). No model is fitted.
"""
import importlib.util
import os

import pandas as pd
import pytest

from defectshift import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "splits", "partitions.csv")
PARTITIONS = {"A_heldout", "A_dev", "B_tr", "B_val", "B_screen", "B_te"}


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(TABLE)


def test_structure(table):
    assert list(table.columns) == ["slot", "row", "partition"]
    assert table["slot"].nunique() == 17
    assert set(table["partition"]) == PARTITIONS
    assert not table.duplicated(["slot", "row"]).any()                          # every record in exactly one subset of a split
    sizes = table.groupby("slot").size()
    assert sizes.nunique() == 1                                                  # all records of the data in every split


def test_agrees_with_lot_overlap_tables(table):
    target = pd.read_csv(os.path.join(ROOT, "splits", "target_test_lot_overlap.csv"))
    source = pd.read_csv(os.path.join(ROOT, "splits", "source_heldout_lot_overlap.csv"))
    for slot, g in table.groupby("slot"):
        assert set(g.loc[g.partition == "B_te", "row"]) == set(target.loc[target.slot == slot, "row"]), slot
        assert set(g.loc[g.partition == "A_heldout", "row"]) == set(source.loc[source.slot == slot, "row"]), slot


@pytest.mark.skipif(not os.path.exists(config.DATA_PATH), reason="data/data.parquet is not present")
def test_matches_fresh_computation(table):
    spec = importlib.util.spec_from_file_location("make_partitions", os.path.join(ROOT, "splits", "make_partitions.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from defectshift.candidates import load_data
    df, y = load_data()
    assert mod.build(df, y, df[config.LOT_COL]).equals(table)
