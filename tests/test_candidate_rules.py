"""The candidate generation rule (defectshift.candidate_rules) reproduces the candidate list CSV. Run: python -m pytest tests -q

Only reads data.parquet (writes no cache or results).
"""
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from defectshift import config  # noqa: E402
from defectshift.candidate_rules import (candidate_names, candidate_table, categorical_candidates,  # noqa: E402
                                         time_candidates)
from defectshift.candidates import load_candidates, load_data  # noqa: E402
from defectshift.selection import temporal_primary  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_data()[0]


@pytest.fixture(scope="module")
def ranking():
    return pd.read_csv(config.CANDIDATES_CSV)


def test_rule_reproduces_candidate_csv(data, ranking):
    names = candidate_names(data)
    assert len(names) == len(set(names)) == 78
    assert set(names) == set(ranking["candidate"])
    t = candidate_table(data).set_index("candidate").loc[ranking["candidate"]]
    assert (t["n_A"].to_numpy() == ranking["n_A"].to_numpy()).all()
    assert (t["n_B"].to_numpy() == ranking["n_B"].to_numpy()).all()


def test_rule_parts(data):
    cats = categorical_candidates(data)
    all_cats = {c for c in data.columns if c.startswith("cat_")}
    assert len(cats) == 48 and all_cats - set(cats) == {config.LOT_COL, "cat_003"}
    times = time_candidates(data)
    assert [sum(n.startswith(ax + "<=") for n in times) for ax in ("x_time", "y_time")] == [15, 15]
    assert temporal_primary(times) == [times[7], times[22]]   # the 8th quantile (0.50) = the pre-specified median


def test_table_with_placebo(data):
    t = candidate_table(data, include_placebo=True)
    assert list(t["candidate"]) == candidate_names(data) + [f"placebo@{s}" for s in config.PLACEBO_B_TARGET]
    assert set(t["candidate"]) == set(load_candidates(include_placebo=True))
    assert t["kind"].value_counts().to_dict() == {"cat": 48, "x_time": 15, "y_time": 15, "placebo": 10}
    assert ((t["n_A"] + t["n_B"]) == len(data)).all() and (t[["n_A", "n_B"]] >= config.MIN_DOMAIN).all().all()
