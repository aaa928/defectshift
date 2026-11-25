"""Checks of paper/analysis/size_matched_verdict.py (the verdict against the size-matched controls only) with small synthetic tables and values known by hand.
No data file and no fit are needed. Run: python -m pytest tests/test_size_matched_verdict.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ["cat_005", "cat_019", "cat_039", "cat_045", "cat_050", "x_time", "y_time"]


def LOTS_CTRL(i):
    return 100 + 3 * i                                               # target test lots of control i


def LOTS_DOM(j):
    return 90 + j                                                    # lot-disjoint target test lots of domain j


@pytest.fixture
def sm(monkeypatch):
    monkeypatch.syspath_prepend(os.path.join(ROOT, "paper", "analysis"))
    import extra_controls_verdict
    import size_matched_verdict
    return size_matched_verdict, extra_controls_verdict


def _inputs(tmp_path, ev):
    """50 controls (10 original, 40 added) with simple values; control i is size-matched for domain j when i % 5 == j % 5."""
    names = [f"c{i:02d}" for i in range(50)]
    vals = pd.DataFrame(dict(candidate=names, tr_rmse=[i - 25.0 for i in range(50)], tr_mae=[(i % 7) - 3.0 for i in range(50)],
                             io_rmse=0.0, io_mae=0.0, src=["original10"] * 10 + ["added40"] * 40))
    ch8 = pd.DataFrame(dict(dom=DOMAINS, tr_rmse=[float(k * 4) for k in range(7)], tr_mae=[0.5] * 7, io_rmse=0.0, io_mae=0.0))
    rows = []
    for j, d in enumerate(DOMAINS):
        for i, c in enumerate(names):
            ok = i % 5 == j % 5
            rows.append(dict(domain=d, candidate=c, rel_n_star=0.05 if ok else (0.2 if i % 2 else 0.05), rel_n_B_te=0.05 if ok else (0.05 if i % 2 else 0.2), matched=ok,
                             n_star_ctrl=1000.0 + i, n_star_dom=1000.0 + j, n_B_te_ctrl=500 + i, n_B_te_dom=500 + j, lots_ctrl=LOTS_CTRL(i), lots_dom=0))
    agg, sizes = tmp_path / "agg", tmp_path / "sizes"
    agg.mkdir(exist_ok=True), sizes.mkdir(exist_ok=True)
    vals.to_csv(agg / "ctrl40_values.csv", index=False)
    ev.verdict(ch8, vals, "T2", "8", names).to_csv(agg / "ctrl40_verdict.csv", index=False)
    pd.DataFrame(rows).to_csv(sizes / "control_matching.csv", index=False)
    near = tmp_path / "near"
    near.mkdir(exist_ok=True)
    nrows = [dict(dom=d, rank=r + 1, control=names[(5 * j + r) % 50]) for j, d in enumerate(DOMAINS) for r in range(10)]     # ten consecutive controls per domain
    pd.DataFrame(nrows).to_csv(near / "lot_match_selection.csv", index=False)
    feat = [dict(group="control", candidate=c, dom=c, test_lots=LOTS_CTRL(i)) for i, c in enumerate(names)] + [dict(group="challenge", candidate=f"{d}<=x", dom=d, test_lots=LOTS_DOM(j)) for j, d in enumerate(DOMAINS)]
    pd.DataFrame(feat).to_csv(near / "lot_match_features.csv", index=False)
    return vals, ch8


def _run(sm_mod, monkeypatch, tmp_path, out="out"):
    monkeypatch.setattr(sys, "argv", ["size_matched_verdict.py", "--run-id", "t", "--agg-run-id", "agg", "--sizes-run-id", "sizes", "--nearest-run-id", "near",
                                      "--analysis-dir", str(tmp_path), "--out-dir", str(tmp_path / out)])
    sm_mod.main()
    return pd.read_csv(tmp_path / out / "size_matched_verdict.csv")


def test_k_and_n_against_the_size_matched_controls(sm, tmp_path, monkeypatch):
    mod, ev = sm
    vals, ch8 = _inputs(tmp_path, ev)
    v = _run(mod, monkeypatch, tmp_path).set_index("dom")
    for j, d in enumerate(DOMAINS):
        idx = [i for i in range(50) if i % 5 == j % 5]
        m = vals.iloc[idx]
        g = ch8.set_index("dom").loc[d]
        assert v.loc[d, "n_ctrl"] == len(idx) == 10
        assert v.loc[d, "n_original10"] == sum(i < 10 for i in idx) and v.loc[d, "n_added40"] == sum(i >= 10 for i in idx)
        assert v.loc[d, "k_rmse"] == int((m.tr_rmse < g.tr_rmse).sum()) and v.loc[d, "k_mae"] == int((m.tr_mae < g.tr_mae).sum())
        assert v.loc[d, "ctrl_max_rmse"] == m.tr_rmse.max() and v.loc[d, "margin_rmse"] == pytest.approx(g.tr_rmse - m.tr_rmse.max())
        assert bool(v.loc[d, "exceed_rmse_only"]) == (v.loc[d, "k_rmse"] == v.loc[d, "n_ctrl"])


def test_ties_are_not_counted(sm, tmp_path, monkeypatch):
    """k counts strictly smaller gaps: a domain value equal to a control value does not exceed that control."""
    mod, ev = sm
    vals, ch8 = _inputs(tmp_path, ev)
    ch8.loc[0, "tr_rmse"] = vals.tr_rmse.iloc[5]                     # cat_005: control 5 is size-matched and has exactly this value
    ev.verdict(ch8, vals, "T2", "8", vals.candidate.tolist()).to_csv(tmp_path / "agg" / "ctrl40_verdict.csv", index=False)
    v = _run(mod, monkeypatch, tmp_path).set_index("dom")
    idx = [i for i in range(50) if i % 5 == 0]
    assert v.loc["cat_005", "k_rmse"] == int((vals.iloc[idx].tr_rmse < vals.tr_rmse.iloc[5]).sum()) == 1


def test_refusals(sm, tmp_path, monkeypatch):
    mod, ev = sm
    _inputs(tmp_path, ev)
    _run(mod, monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="exists"):                  # existing outputs are not overwritten
        _run(mod, monkeypatch, tmp_path)
    t = pd.read_csv(tmp_path / "sizes" / "control_matching.csv")
    t.loc[3, "matched"] = not t.loc[3, "matched"]                    # a flag that differs from the relative differences
    t.to_csv(tmp_path / "sizes" / "control_matching.csv", index=False)
    with pytest.raises(SystemExit, match="recomputed"):
        _run(mod, monkeypatch, tmp_path, out="out2")
    _inputs(tmp_path, ev)
    vd = pd.read_csv(tmp_path / "agg" / "ctrl40_verdict.csv")
    vd.loc[vd.dom == "cat_050", "k_rmse"] += 1                       # a verdict file that is not the one of the 50 controls
    vd.to_csv(tmp_path / "agg" / "ctrl40_verdict.csv", index=False)
    with pytest.raises(SystemExit, match="track T2"):
        _run(mod, monkeypatch, tmp_path, out="out3")


def test_the_tolerance_of_the_size_table_must_be_the_one_asked_for(sm, tmp_path, monkeypatch):
    """A size table made with another tolerance cannot be read as 10 percent (control_sizes.py records tol)."""
    mod, ev = sm
    _inputs(tmp_path, ev)
    t = pd.read_csv(tmp_path / "sizes" / "control_matching.csv")
    t["tol"] = 0.2
    t.to_csv(tmp_path / "sizes" / "control_matching.csv", index=False)
    with pytest.raises(SystemExit, match="tol"):
        _run(mod, monkeypatch, tmp_path)
    t["tol"] = 0.1
    t.to_csv(tmp_path / "sizes" / "control_matching.csv", index=False)
    assert len(_run(mod, monkeypatch, tmp_path, out="out_ok")) == 7


def test_nearest_controls_that_are_size_matched(sm, tmp_path, monkeypatch):
    """Of the ten nearest controls of a domain, the number that also meet the size rule (here: control i is matched for domain j when i % 5 == j % 5)."""
    mod, ev = sm
    _inputs(tmp_path, ev)
    v = _run(mod, monkeypatch, tmp_path).set_index("dom")
    for j, d in enumerate(DOMAINS):
        near = [(5 * j + r) % 50 for r in range(10)]
        assert v.loc[d, "n_nearest10"] == 10 and v.loc[d, "n_nearest10_matched"] == sum(i % 5 == j % 5 for i in near) == 2
    t = pd.read_csv(tmp_path / "near" / "lot_match_selection.csv")
    t = t[~((t.dom == "cat_005") & (t["rank"] == 10))]                        # a domain with nine nearest controls
    t.to_csv(tmp_path / "near" / "lot_match_selection.csv", index=False)
    with pytest.raises(SystemExit, match="ten distinct controls"):
        _run(mod, monkeypatch, tmp_path, out="out2")


def test_a_missing_tolerance_and_an_empty_domain_are_refused(sm, tmp_path, monkeypatch):
    """A tol column with missing values cannot be read as 10 percent, and a domain without any size-matched control has no verdict ("exceeds all" would hold for an empty set)."""
    mod, ev = sm
    _inputs(tmp_path, ev)
    t = pd.read_csv(tmp_path / "sizes" / "control_matching.csv")
    t["tol"] = 0.1
    t.loc[5, "tol"] = np.nan
    t.to_csv(tmp_path / "sizes" / "control_matching.csv", index=False)
    with pytest.raises(SystemExit, match="tol"):
        _run(mod, monkeypatch, tmp_path)
    t["tol"] = 0.1
    t.loc[t.domain == "x_time", ["rel_n_star", "matched"]] = [0.5, False]                # no size-matched control for x_time
    t.to_csv(tmp_path / "sizes" / "control_matching.csv", index=False)
    with pytest.raises(SystemExit, match="empty set"):
        _run(mod, monkeypatch, tmp_path, out="out2")


def test_lots_of_the_size_matched_controls_and_of_the_domain(sm, tmp_path, monkeypatch):
    """Minimum, median and maximum of the target test lots over the size-matched controls, the lot-disjoint value of the domain, and one row per domain and size-matched control."""
    mod, ev = sm
    _inputs(tmp_path, ev)
    v = _run(mod, monkeypatch, tmp_path).set_index("dom")
    for j, d in enumerate(DOMAINS):
        lots = [LOTS_CTRL(i) for i in range(50) if i % 5 == j % 5]
        assert v.loc[d, "lots_ctrl_min"] == min(lots) and v.loc[d, "lots_ctrl_max"] == max(lots) and v.loc[d, "lots_ctrl_median"] == np.median(lots)
        assert v.loc[d, "lots_domain_disjoint"] == LOTS_DOM(j)
    mc = pd.read_csv(tmp_path / "out" / "size_matched_controls.csv")
    assert len(mc) == 70 and list(mc.domain.unique()) == DOMAINS
    assert "n_B_te_dom_all" in mc.columns and "n_B_te_dom" not in mc.columns and "lots_domain_disjoint" in mc.columns
    assert (mc.groupby("domain").size() == 10).all() and set(mc.src) == {"original10", "added40"}
    assert (mc.lots_ctrl == 100 + 3 * mc.candidate.str[1:].astype(int)).all()


def test_lots_that_differ_between_the_tables_are_refused(sm, tmp_path, monkeypatch):
    mod, ev = sm
    _inputs(tmp_path, ev)
    f = pd.read_csv(tmp_path / "near" / "lot_match_features.csv")
    f.loc[3, "test_lots"] += 1                                       # a control whose lots differ from the size table
    f.to_csv(tmp_path / "near" / "lot_match_features.csv", index=False)
    with pytest.raises(SystemExit, match="lots of the controls"):
        _run(mod, monkeypatch, tmp_path)
    f.loc[3, "test_lots"] -= 1
    f = f[f.dom != "x_time"]                                         # a domain without lots
    f.to_csv(tmp_path / "near" / "lot_match_features.csv", index=False)
    with pytest.raises(SystemExit, match="features table"):
        _run(mod, monkeypatch, tmp_path, out="out2")


def test_the_second_output_file_is_not_overwritten_either(sm, tmp_path, monkeypatch):
    mod, ev = sm
    _inputs(tmp_path, ev)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "size_matched_controls.csv").write_text("keep")
    with pytest.raises(SystemExit, match="size_matched_controls.csv exists"):
        _run(mod, monkeypatch, tmp_path)
    assert (tmp_path / "out" / "size_matched_controls.csv").read_text() == "keep"
