"""Checks of the analysis scripts (no retraining): robustness ranks, dominance and bootstrap, shift statistics, shift association. Run:
python -m pytest tests/test_analysis.py -q
Fake inputs (tables with preset values, synthetic data) are compared with values known by hand."""
import importlib.util
import os
import sys

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from defectshift import config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ shift statistics
def test_shift_statistics_units():
    mod = _load_script("10a_shift_statistics")
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, (400, 3)), rng.normal(0, 1, (400, 3))
    assert abs(mod.mmd_rbf(a, b, np.random.default_rng(1))) < 0.02                       # same distribution: about 0
    assert mod.mmd_rbf(a, b + 1.5, np.random.default_rng(1)) > 0.1                        # positive under a mean shift
    toy = pd.DataFrame({"num_a": [1, np.nan, 3, np.nan], "num_b": [5, np.nan, 7, np.nan], "num_c": [np.nan, 1, 1, 1],
                        "num_d": [1, 2, 3, 4]})
    g = mod.missing_groups(toy, list(toy.columns))
    assert sorted(map(sorted, g.values())) == [["num_a", "num_b"], ["num_c"]]            # columns with the same missing positions, columns without missing values excluded


def test_shift_statistics_one_candidate_rows_and_outputs(monkeypatch):
    """One candidate: rows used = A_dev + B_tr + B_screen (no A_heldout or B_te), features = the benchmark feature set, 29 missing-value
    groups."""
    from defectshift.candidates import build_one, feature_sets, load_data
    from defectshift.splits import B_view, get_A, split_B4
    mod = _load_script("10a_shift_statistics")
    monkeypatch.setattr(mod, "HGB", dict(max_iter=10, learning_rate=0.3, max_leaf_nodes=15, random_state=mod.SEED))
    monkeypatch.setattr(sys, "argv", ["10a_shift_statistics.py", "--run-id", "shift_statistics_test", "--only", "cat_050"])
    mod.main()
    out = os.path.join(config.RESULTS_DIR, "shift_statistics", "shift_statistics_test")
    st = pd.read_csv(os.path.join(out, "shift_stats.csv")).iloc[0]
    df, y = load_data()
    lot = df[config.LOT_COL]
    slot, mA, mB = build_one(df, "cat_050")
    a = get_A(slot, df.index[mA], y, lot)
    b4 = split_B4(df.index[mB], y, lot)
    assert st.n_A_dev == len(a["A_dev"]) and st.n_B == len(b4["B_tr"]) + len(b4["B_screen"])
    assert 0 <= st.domain_auc <= 1 and st.mmd_rbf > -0.01 and 0 <= st.cat_tvd_mean <= 1 and st.n_star > 0
    fs = pd.read_csv(os.path.join(out, "feature_stats.csv"))
    cat, num = feature_sets(df, "cat_050")
    assert len(fs) == len(cat) + len(num) and set(fs.feature) == set(cat + num) and config.LOT_COL not in set(fs.feature)
    assert len(pd.read_csv(os.path.join(out, "missing_groups.csv"))) == 29
    mod.main()                                                              # resume: finished candidates are skipped
    assert len(pd.read_csv(os.path.join(out, "shift_stats.csv"))) == 1 and len(pd.read_csv(os.path.join(out, "feature_stats.csv"))) == len(fs)


# ------------------------------------------------------------------ shift association
def test_shift_association_synthetic(monkeypatch):
    """48 categorical + 15 x 2 temporal fake candidates. domain_auc is monotone in target-reference gap (rho = 1, still large after controlling for n*);
    mean_shift_smd equals n*, so its association with target-reference gap runs through n* and the partial rank correlation controlling for n* is near 0."""
    rng = np.random.default_rng(1)
    rows, stats = [], []
    for kind, n in (("cat", 48), ("x_time", 15), ("y_time", 15)):
        for i in range(n):
            name = f"{kind}_{i}"
            ns = float(rng.integers(500, 4000))
            auc = rng.uniform(0.5, 1.0)
            target_reference_gap = 10 * auc + 0.0001 * ns if kind == "cat" else -auc     # a small share of n* mixed in
            rows.append(dict(candidate=name, kind=kind, consensus_gap_rmse_pct=target_reference_gap, b_te_eligible=True))
            st = dict(candidate=name, n_star=ns, domain_auc=auc, mean_shift_smd=ns + rng.normal(0, 1))
            for q in ("wasserstein_mean", "mmd_rbf", "scale_shift", "missing_shift_mean", "missing_shift_max", "cat_tvd_mean",
                      "cat_tvd_max", "dmean_y", "var_ratio_y", "dpos_y", "resid_shift", "rmse_ratio"):
                st[q] = rng.normal()
            stats.append(st)
    screening = os.path.join(config.RESULTS_DIR, "screening", "runs", "screening_fake9", "selection")
    os.makedirs(screening, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(screening, "summary.csv"), index=False)
    shift_stats = os.path.join(config.RESULTS_DIR, "shift_statistics", "shift_statistics_fake9")
    os.makedirs(shift_stats, exist_ok=True)
    pd.DataFrame(stats).to_csv(os.path.join(shift_stats, "shift_stats.csv"), index=False)
    mod = _load_script("10b_shift_association")
    monkeypatch.setattr(sys, "argv", ["10b_shift_association.py", "--screening-run-id", "screening_fake9", "--shift-statistics-run-id",
                                      "shift_statistics_fake9", "--run-id", "shift_association_t", "--boot", "300"])
    mod.main()
    out = os.path.join(config.RESULTS_DIR, "shift_association", "shift_association_t")
    c = pd.read_csv(os.path.join(out, "assoc_categorical.csv")).set_index("quantity")
    assert c.loc["domain_auc", "n"] == 48 and c.loc["domain_auc", "spearman"] > 0.95 and c.loc["domain_auc", "partial_nstar"] > 0.9
    assert c.loc["domain_auc", "ci_lo"] <= c.loc["domain_auc", "spearman"] <= c.loc["domain_auc", "ci_hi"] and c.loc["domain_auc", "perm_p"] < 0.01
    assert abs(c.loc["mean_shift_smd", "partial_nstar"]) < 0.3                       # the association through n* disappears after the control
    assert set(c.stage) == {"X-only", "Y-oracle"} and np.isnan(c.loc["n_star"].get("partial_nstar", np.nan))
    t = pd.read_csv(os.path.join(out, "assoc_temporal.csv"))
    assert set(t.axis) == {"x_time", "y_time"} and "ci_lo" not in t.columns and "perm_p" not in t.columns   # descriptive statistics only on the temporal axes
    auc_t = t[(t.quantity == "domain_auc") & (t.axis == "x_time")].spearman.iloc[0]
    assert auc_t < -0.95                                                              # fake temporal target-reference gap = -auc


# ------------------------------------------------------------------ manuscript tables
FAKE_FINAL = dict(temporal=["x_time<=2099-01-01 00:00:00", "y_time<=2099-01-02 00:00:00"],
                  categorical=["cat_900", "cat_901", "cat_902", "cat_903", "cat_904"])


# ------------------------------------------------------------------ repetitions 0-2 vs 3-4, residuals


def test_shift_association_constant_metric_and_candidate_mismatch(monkeypatch):
    """A constant quantity does not stop the run but gives an empty value with the reason, and an empty interval even if every bootstrap
    sample is constant. The run stops if the listed screening candidates and the shift-statistics candidates differ."""
    mod = _load_script("10b_shift_association")
    rng = np.random.default_rng(0)
    x, y, z = np.ones(20), rng.normal(size=20), rng.normal(size=20)
    r = mod.assoc(x, y, z, 50, rng, control=True)
    assert "spearman" not in r and "constant" in r["note"]
    x2 = np.r_[np.zeros(19), 1.0]                    # nearly constant: most bootstrap samples are constant
    r2 = mod.assoc(x2, y, z, 30, rng, control=False)
    assert "spearman" in r2 and r2["n_boot_used"] <= 30
    rows = [dict(candidate=f"cat_{i}", kind="cat", consensus_gap_rmse_pct=float(i), b_te_eligible=True) for i in range(10)]
    rows.append(dict(candidate="placebo@9001", kind="placebo", consensus_gap_rmse_pct=0.0, b_te_eligible=True))
    stats = [dict(candidate=f"cat_{i}", n_star=1000.0 + i, dmean_y=0.0) for i in range(9)]     # cat_9 missing
    screening = os.path.join(config.RESULTS_DIR, "screening", "runs", "screening_fake9b", "selection")
    os.makedirs(screening, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(screening, "summary.csv"), index=False)
    shift_stats = os.path.join(config.RESULTS_DIR, "shift_statistics", "shift_statistics_fake9b")
    os.makedirs(shift_stats, exist_ok=True)
    pd.DataFrame(stats).to_csv(os.path.join(shift_stats, "shift_stats.csv"), index=False)
    monkeypatch.setattr(sys, "argv", ["10b_shift_association.py", "--screening-run-id", "screening_fake9b", "--shift-statistics-run-id",
                                      "shift_statistics_fake9b", "--run-id", "shift_association_t2"])
    with pytest.raises(SystemExit, match="candidates differ"):
        mod.main()


def test_read_per_fit_paths(tmp_path, monkeypatch):
    """A relative pred_path is resolved under the cache folder, an absolute one (local run) and a missing value
    are kept, and a file without the column is read unchanged. No model is trained."""
    from defectshift.utils import read_per_fit
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    rel = "benchmark/runs/r/models/s/a.pred.npz"
    pd.DataFrame(dict(pred_path=[rel, "/abs/b.pred.npz", None], v=[1, 2, 3])).to_csv(tmp_path / "pf.csv", index=False)
    got = read_per_fit(str(tmp_path / "pf.csv"))
    assert got.pred_path[0] == os.path.join(str(tmp_path / "cache"), rel)
    assert got.pred_path[1] == "/abs/b.pred.npz"
    assert pd.isna(got.pred_path[2])
    pd.DataFrame(dict(v=[1, 2])).to_csv(tmp_path / "plain.csv", index=False)
    assert list(read_per_fit(str(tmp_path / "plain.csv")).columns) == ["v"]
