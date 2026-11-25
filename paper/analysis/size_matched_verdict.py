"""Verdict of the target-reference gap against the size-matched controls only (8 models). It reads only saved outputs (no retraining, CPU only).

Size-matched split (control_sizes.py): the number of training records n* and the number of target test rows are each within +-TOL (default 10 %) of the value
of the challenge domain (both conditions, the same rule for every domain). The rule compares the rows of B_te of a control with ALL rows of B_te of the domain (n_B_te_dom_all);
the evaluated rows of a domain are the lot-disjoint ones, up to about 11 % fewer (lots shared with A_dev), which the rule does not use. The controls of a domain are those of the 50 randomized lot-split controls (original 10 + additional 40)
that meet this rule for that domain; the other controls are not used for it. The number of controls differs by domain.
Track T4 (T0 to T3 are the tracks of extra_controls_verdict.py): the challenge values (8 models) and the control values are those of its track T2 (value definition and k as there:
k = number of controls with a strictly smaller gap, ties are not counted; "exceeds all" = k == N). k / N is a relative position among the controls and not a p-value.
For each domain it also counts how many of its ten nearest controls (lot_matched_controls.py, track T3 of extra_controls_verdict.py; chosen from split metadata alone, without the size rule) meet the size rule,
and describes the number of target test lots (not part of the size rule): minimum, median and maximum over the size-matched controls and the value of the domain. The lots are those of the evaluated rows
(lot_match_features.csv of lot_matched_controls.py): for a domain the lots of the lot-disjoint test rows (lots_domain_disjoint; lots shared with A_dev excluded), for a control the lots of B_te (a control has no lot shared with A_dev).
The lots of the domain in the size table (lots_dom of control_sizes.py) are those of ALL rows of B_te and are not used here; they differ from lots_domain_disjoint wherever the domain shares lots with A_dev.
Checks (the run stops on a difference): the size table has 7 domains x 50 controls and the same controls as the value table; the features table has the 7 domains and the 50 controls, and the lots of the controls equal those of the size table; every domain has ten nearest controls, all among the 50; with --tol 0.10 the flag recomputed from the relative
differences equals the stored flag; the verdict over all 50 controls reproduces the k values of track T2.
Input (analysis folders): <agg-run-id>/{ctrl40_values.csv, ctrl40_verdict.csv} (extra_controls_verdict.py), <sizes-run-id>/control_matching.csv (control_sizes.py), <nearest-run-id>/{lot_match_selection.csv, lot_match_features.csv} (lot_matched_controls.py)
Output: <out>/size_matched_verdict.csv (one row per domain: matched controls, of them original / additional, gaps, k, maximum, margin, how many of the ten nearest controls are size-matched, and the lots described above),
<out>/size_matched_controls.csv (one row per domain and size-matched control: n*, target test rows (all rows of B_te for the domain, n_B_te_dom_all) and target test lots (lots_domain_disjoint for the domain) of the control and of the domain); default <out> = results/analysis/<run-id>
Usage: python paper/analysis/size_matched_verdict.py --run-id size_matched_verdict
"""
import argparse
import os

import pandas as pd

import extra_controls_verdict as ev
from defectshift import config
from defectshift.utils import log

DOMAINS = ("cat_005", "cat_019", "cat_039", "cat_045", "cat_050", "x_time", "y_time")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--agg-run-id", default="extra_controls_verdict")
    ap.add_argument("--sizes-run-id", default="control_sizes")
    ap.add_argument("--nearest-run-id", default="lot_match")
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--analysis-dir", default=None, help="folder that holds the run folders (default: results/analysis)")
    ap.add_argument("--out-dir", default=None, help="output folder (default: <analysis-dir>/<run-id>)")
    a = ap.parse_args()
    rf = a.analysis_dir or os.path.join(config.RESULTS_DIR, "analysis")
    out = a.out_dir or os.path.join(rf, a.run_id)
    dst = os.path.join(out, "size_matched_verdict.csv")
    dst_ctrl = os.path.join(out, "size_matched_controls.csv")
    for f in (dst, dst_ctrl):
        if os.path.exists(f):
            raise SystemExit(f"{f} exists (existing outputs are not overwritten; use another --run-id or --out-dir)")
    vals = pd.read_csv(os.path.join(rf, a.agg_run_id, "ctrl40_values.csv"))
    vd = pd.read_csv(os.path.join(rf, a.agg_run_id, "ctrl40_verdict.csv"))
    sizes = pd.read_csv(os.path.join(rf, a.sizes_run_id, "control_matching.csv"))
    nearest = pd.read_csv(os.path.join(rf, a.nearest_run_id, "lot_match_selection.csv"))
    feat = pd.read_csv(os.path.join(rf, a.nearest_run_id, "lot_match_features.csv"))
    t2 = vd[vd.track == "T2"]
    if sorted(t2.dom) != sorted(DOMAINS) or len(vals) != 50 or vals.candidate.nunique() != 50:
        raise SystemExit("ctrl40 outputs: expected the 7 challenge domains (track T2) and 50 controls")
    ch8 = pd.DataFrame(dict(dom=t2.dom.values, tr_rmse=t2.gap_rmse.values, tr_mae=t2.gap_mae.values, io_rmse=t2.io_rmse.values, io_mae=t2.io_mae.values))
    # size table: 7 domains x the same 50 controls
    if len(sizes) != 7 * 50 or set(sizes.domain) != set(DOMAINS) or set(sizes.candidate) != set(vals.candidate):
        raise SystemExit("the size table must hold the 7 domains for the same 50 controls as the value table")
    if "tol" in sizes.columns and (sizes.tol.isna().any() or (sizes.tol - a.tol).abs().max() > 1e-12):
        raise SystemExit(f"the size table was made with tol {sorted(sizes.tol.dropna().unique())} (missing values: {int(sizes.tol.isna().sum())}), not {a.tol}")
    sizes = sizes.assign(ok=(sizes.rel_n_star <= a.tol) & (sizes.rel_n_B_te <= a.tol))
    if abs(a.tol - 0.10) < 1e-12 and (sizes.ok != sizes.matched.astype(bool)).any():
        raise SystemExit("the size-matched flag recomputed from rel_n_star and rel_n_B_te differs from the stored flag")
    names = {d: sizes[(sizes.domain == d) & sizes.ok].candidate.tolist() for d in DOMAINS}
    empty = [d for d in DOMAINS if not names[d]]
    if empty:                                                  # "exceeds all controls" would hold for an empty set
        raise SystemExit(f"no size-matched control for {empty} at tol {a.tol}: the verdict against an empty set is not defined")
    if set(nearest.dom) != set(DOMAINS) or (nearest.groupby("dom").control.nunique() != 10).any() or not set(nearest.control) <= set(vals.candidate):
        raise SystemExit("the nearest-control table must hold ten distinct controls per challenge domain, all among the 50")
    fc, fd = feat[feat["group"] == "control"], feat[feat["group"] == "challenge"]
    if sorted(fd.dom) != sorted(DOMAINS) or sorted(fc.candidate) != sorted(vals.candidate):
        raise SystemExit("the features table must hold the 7 challenge domains and the same 50 controls as the value table")
    lots_ctrl = fc.set_index("candidate").test_lots
    lots_dom = fd.set_index("dom").test_lots
    if (sizes.candidate.map(lots_ctrl) != sizes.lots_ctrl).any():
        raise SystemExit("the lots of the controls in the features table differ from those of the size table")
    # check: the verdict over all 50 controls is the one of track T2
    chk = ev.verdict(ch8, vals, "T2", "8", vals.candidate.tolist()).set_index("dom")
    ref = t2.set_index("dom")
    for col in ("k_rmse", "k_mae", "k_both"):
        if not (chk[col].reindex(ref.index) == ref[col]).all():
            raise SystemExit(f"{col} over all 50 controls differs from track T2 of ctrl40_verdict.csv")
    v = ev.verdict(ch8, vals, "T4", "8", names)
    src = vals.set_index("candidate").src
    v.insert(4, "n_original10", [int(src.reindex(names[d]).eq("original10").sum()) for d in v.dom])
    v.insert(5, "n_added40", [int(src.reindex(names[d]).eq("added40").sum()) for d in v.dom])
    v["n_nearest10"] = [int((nearest.dom == d).sum()) for d in v.dom]
    v["n_nearest10_matched"] = [int(nearest[nearest.dom == d].control.isin(names[d]).sum()) for d in v.dom]
    v["tol"] = a.tol
    lc = [lots_ctrl.reindex(names[d]) for d in v.dom]
    v["lots_ctrl_min"], v["lots_ctrl_median"], v["lots_ctrl_max"] = [int(x.min()) for x in lc], [float(x.median()) for x in lc], [int(x.max()) for x in lc]
    v["lots_domain_disjoint"] = [int(lots_dom[d]) for d in v.dom]
    mc = sizes[sizes.ok][["domain", "candidate", "n_star_ctrl", "n_star_dom", "n_B_te_ctrl", "n_B_te_dom"]].rename(columns=dict(n_B_te_dom="n_B_te_dom_all")).assign(
        lots_ctrl=lambda t: t.candidate.map(lots_ctrl), lots_domain_disjoint=lambda t: t.domain.map(lots_dom), src=lambda t: t.candidate.map(src))
    mc = mc.set_index("domain").loc[list(DOMAINS)].reset_index()
    os.makedirs(out, exist_ok=True)
    v.to_csv(dst, index=False)
    mc.to_csv(dst_ctrl, index=False)
    log(f"size-matched controls (n* and target test rows each within +-{a.tol:.0%}); k / N for RMSE and MAE:")
    for _, r in v.iterrows():
        log(f"  {r.dom:8s} N {r.n_ctrl:2d} (original {r.n_original10}, added {r.n_added40}; nearest ten size-matched {r.n_nearest10_matched})  RMSE gap {r.gap_rmse:+6.1f}  k {r.k_rmse:2d}  maximum {r.ctrl_max_rmse:+5.1f}"
            f"   MAE gap {r.gap_mae:+6.1f}  k {r.k_mae:2d}  maximum {r.ctrl_max_mae:+5.1f}"
            f"   test lots {r.lots_ctrl_min} / {r.lots_ctrl_median:g} / {r.lots_ctrl_max} (domain, lot-disjoint rows: {r.lots_domain_disjoint})")
    log(f"written: {dst}")


if __name__ == "__main__":
    main()
