"""Checks of the two analyses of the appendix on the direction comparison (paper/analysis/direction_checks.py and the table generator
paper/tools/tables_direction.py) with small synthetic inputs and values known by hand: the outcome-scale term D, the resampling weights, the
rule that assigns agree / disagree / unresolved, and the two generated tables. No data file and no fit are needed.
Run: python -m pytest tests/test_direction_checks.py -q
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["cat_005", "cat_019", "cat_039", "cat_045", "cat_050", "x_time", "y_time"]


@pytest.fixture
def analysis(monkeypatch):
    """The modules of paper/analysis (they import each other by their file names)."""
    monkeypatch.syspath_prepend(os.path.join(ROOT, "paper", "analysis"))
    import direction_checks
    import lot_block_bootstrap
    import scale_normalized_bootstrap
    return direction_checks, scale_normalized_bootstrap, lot_block_bootstrap


def test_outcome_scale_term_by_hand(analysis):
    """D = |log s| - |log(1 + g_IO) - log s| for models whose errors are constant in magnitude: the outcome-scale ratio is s = 2 (target
    spread 1, source spread 0.5), and the error ratio target/source is 2 (D = log 2), 1 (D = 0), 8 (D = -log 2) and 1.2 (D = log 1.2:
    the term is the larger one in magnitude although the ID-OOD gap is only 20 percent)."""
    dc, _, _ = analysis
    yT, yS = np.tile([0.0, 2.0], 10), np.tile([0.0, 1.0], 10)
    sign = np.tile([1.0, -1.0], 10)
    c_target, c_source = {"two": 1.0, "one": 0.5, "eight": 4.0, "small": 0.6}, 0.5          # error magnitudes (rmse = mae = magnitude)
    preds = {m: [(yT + c * sign,)] for m, c in c_target.items()}
    spreds = {m: [yS + c_source * sign] for m in c_target}
    d = pd.DataFrame(dc.fit_level_dominance("dom", yT, yS, preds, spreds, list(c_target), 1))
    assert len(d) == 2 * len(c_target) and set(d.metric) == {"rmse", "mae"}
    assert np.allclose(d.s, 2.0)
    want = {"two": (100.0, np.log(2)), "one": (0.0, 0.0), "eight": (700.0, -np.log(2)), "small": (20.0, np.log(1.2))}
    for m, (g, dd) in want.items():
        for met in ("rmse", "mae"):
            r = d[(d.model == m) & (d.metric == met)].iloc[0]
            assert r.g_io_pct == pytest.approx(g) and r.D == pytest.approx(dd), (m, met)


def test_classification_rule(analysis):
    """Unresolved if either 95 percent interval contains 0 (an end at 0 counts as containing it); otherwise agree or disagree by the sign of the point estimates."""
    _, sn, _ = analysis
    lo_a, hi_a = np.array([1.0, -1.0, 1.0, 1.0, 0.0]), np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    lo_b, hi_b = np.array([1.0, 1.0, -1.0, -2.0, 1.0]), np.array([2.0, 2.0, 2.0, -1.0, 2.0])
    pa, pb = np.array([1.5, 0.5, 1.5, 1.5, 1.0]), np.array([1.5, 1.5, 0.5, -1.5, 1.5])
    got = list(sn.classify(lo_a, hi_a, lo_b, hi_b, pa, pb))
    assert got == ["agree", "unresolved", "unresolved", "disagree", "unresolved"]


def test_block_weights(analysis):
    """Lot weights of the block bootstrap: non-negative integers that sum to the number of lots; blocks as long as the whole sequence give every lot once;
    the same generator seed gives the same weights."""
    _, _, lb = analysis
    L = 7
    order = np.random.default_rng(0).permutation(L)
    for circular in (True, False):
        for b in (1, 2, 5, L):
            W = lb.block_weights(order, b, 50, np.random.default_rng(1), circular=circular)
            assert W.shape == (50, L) and (W >= 0).all() and (W == np.round(W)).all() and (W.sum(1) == L).all()
        assert (lb.block_weights(order, L, 20, np.random.default_rng(2), circular=circular) == 1).all()
        a = lb.block_weights(order, 2, 30, np.random.default_rng(3), circular=circular)
        assert (a == lb.block_weights(order, 2, 30, np.random.default_rng(3), circular=circular)).all()


def test_scheme_weights(analysis):
    """One independent scheme and the blocks 2, 5, 10 in a circular and a moving version, with fixed seeds per domain."""
    dc, _, _ = analysis
    LT = 12
    order = np.random.default_rng(0).permutation(LT)
    out = dc.scheme_weights(order, LT, "cat_050", 30)
    assert [(s, b) for s, b, _ in out] == [("iid", 1), ("block", 2), ("block", 5), ("block", 10), ("moving", 2), ("moving", 5), ("moving", 10)]
    for _, _, W in out:
        assert W.shape == (30, LT) and (W.sum(1) == LT).all()
    again = dc.scheme_weights(order, LT, "cat_050", 30)
    assert all((a[2] == b[2]).all() for a, b in zip(out, again))
    other = dc.scheme_weights(order, LT, "cat_019", 30)
    assert not (out[0][2] == other[0][2]).all()                                              # another domain, another seed


def _tables_module(tmp_path):
    spec = importlib.util.spec_from_file_location("tables_direction", os.path.join(ROOT, "paper", "tools", "tables_direction.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "tables").mkdir()
    mod.R, mod.HERE = tmp_path, tmp_path
    return mod


def _block_csv(n_models=22):
    rows = []
    for d in ORDER:
        for m in ("rmse", "mae"):
            base = dict(dom=d, metric=m, n_models=n_models, class_io="agree", class_sn="disagree", io_changed_vs_iid=False, sn_changed_vs_iid=False)
            rows.append(dict(base, scheme="iid", block=1, tr_lo=1.7, tr_hi=9.0))
            for scheme in ("block", "moving"):
                for b in (2, 5, 10):
                    rows.append(dict(base, scheme=scheme, block=b, tr_lo=1.0 + 0.1 * b, tr_hi=9.0 + 0.1 * b,
                                     io_changed_vs_iid=(d, m, scheme, b) == ("y_time", "mae", "moving", 5)))
    return pd.DataFrame(rows)


def test_block_table_generator(tmp_path):
    mod = _tables_module(tmp_path)
    _block_csv().to_csv(tmp_path / "block_directions.csv", index=False)
    changed = mod.tab_block()
    assert len(changed) == 1 and tuple(changed.iloc[0][["dom", "metric", "scheme", "block"]]) == ("y_time", "mae", "moving", 5)
    text = (tmp_path / "tables" / "tab_block.tex").read_text()
    assert r"[1.7, 9.0] & [1.2, 10.0] & A/D & 0 \\" in text                                   # interval of the independent scheme, range over the block schemes
    lines = [ln for ln in text.splitlines() if ln.startswith("y\\_time")]
    assert len(lines) == 1 and lines[0].endswith("& 0 \\\\")                                  # y_time RMSE
    assert any(ln.strip().startswith("& MAE &") and ln.endswith("& 1 \\\\") for ln in text.splitlines()[text.splitlines().index(lines[0]):])   # y_time MAE: one changed case
    _block_csv(n_models=21).to_csv(tmp_path / "block_directions.csv", index=False)
    with pytest.raises(AssertionError, match="22 models"):
        mod.tab_block()


def _scale_csv(tmp_path, n_fits=110):
    rows = [dict(dom=d, metric=m, n_fits=n_fits, share_fits_scale_term_larger=0.86, D_fit_median=0.3123) for d in ORDER for m in ("rmse", "mae")]
    pd.DataFrame(rows).to_csv(tmp_path / "scale_dominance_domain.csv", index=False)
    pd.DataFrame(dict(metric=["rmse", "mae"], n_larger_fit_median=[5, 6], n_domains=[7, 7])).to_csv(tmp_path / "scale_dominance_summary.csv", index=False)


def test_scaleterm_table_generator(tmp_path):
    mod = _tables_module(tmp_path)
    _scale_csv(tmp_path)
    mod.tab_scaleterm()
    text = (tmp_path / "tables" / "tab_scaleterm.tex").read_text()
    assert r"cat\_005 & 86 & +0.31 & 86 & +0.31 \\" in text and "110 fits (22 models times five repetitions)" in text
    assert "in 5 of 7 domains for RMSE and 6 of 7 for MAE" in text
    _scale_csv(tmp_path, n_fits=100)
    with pytest.raises(AssertionError, match="22 models times five repetitions"):
        mod.tab_scaleterm()


def test_reference_checks_are_required_only_in_the_paper_configuration(analysis, tmp_path):
    """The paper configuration (all models, 2,000 draws, all domains) must be compared with the saved references; a smoke run (--models, --only or another --n-boot) is not."""
    dc, _, lb = analysis
    assert dc.paper_configuration("", 2000, "") is True
    assert dc.paper_configuration("Ridge", 2000, "") is False
    assert dc.paper_configuration("", 200, "") is False
    assert dc.paper_configuration("", 2000, "cat_050") is False                       # --only alone makes the run a smoke run
    assert dc.load_references(str(tmp_path), "sn", "blk", required=False) == (None, None)       # nothing is read when no check is wanted
    with pytest.raises(SystemExit, match="reference files missing"):
        dc.load_references(str(tmp_path), "sn", "blk", required=True)
    (tmp_path / "sn").mkdir(), (tmp_path / "blk").mkdir()
    pd.DataFrame(dict(group=["challenge", "other"], scheme=["target", "target"], dom=["cat_050", "cat_050"], metric=["rmse", "rmse"], tr=[1.0, 2.0])).to_csv(tmp_path / "sn" / "sn_domain.csv", index=False)
    with pytest.raises(SystemExit, match="reference files missing"):                  # one of the two files is still missing
        dc.load_references(str(tmp_path), "sn", "blk", required=True)
    pd.DataFrame(dict(dom=["cat_050"], scheme=["block"], block=[2], rmse_lo=[0.0])).to_csv(tmp_path / "blk" / "block_boot.csv", index=False)
    sn_ref, block_ref = dc.load_references(str(tmp_path), "sn", "blk", required=True)
    assert list(sn_ref.index) == [("cat_050", "rmse")] and sn_ref.loc[("cat_050", "rmse"), "tr"] == 1.0      # only the challenge rows of scheme "target"
    assert ("cat_050", "block", 2) in block_ref.index


def test_every_domain_measure_and_scheme_must_be_compared(analysis):
    dc, _, lb = analysis
    n_blocks = 2 * len(lb.BLOCKS)                                                     # circular and moving, per block length
    dc.check_coverage(n_iid=2 * 7, n_block=2 * n_blocks * 7, n_block_nokey=0, n_domains=7)
    for bad in (dict(n_iid=13, n_block=2 * n_blocks * 7, n_block_nokey=0), dict(n_iid=14, n_block=2 * n_blocks * 7 - 1, n_block_nokey=0),
                dict(n_iid=14, n_block=2 * n_blocks * 7, n_block_nokey=1)):
        with pytest.raises(SystemExit, match="reference checks incomplete"):
            dc.check_coverage(n_domains=7, **bad)
