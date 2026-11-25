"""Figures that describe why the range of the MAE target-reference gap over the 10 controls is narrower than that of RMSE. It reads only saved results (CPU only).
Input: results/analysis/scale_normalized_bootstrap/{sn_domain.csv, sn_cells.csv} (test-lot resampling version; lot-disjoint evaluation rows = all of B_te for the controls),
      results/analysis/robust/robust_cells.csv (cell values per row configuration; rest99 = rows without the 1 % farthest from the median of B_tr)
Output (for the controls and the challenge domains separately): the median of the absolute cell gap (RMSE, MAE), the median width of the 95 % bootstrap interval of the domain value, the range of the control domain values
(all rows, top 1 % excluded), the controls that set the two ends of the range and their s.
Output file: results/analysis/<run-id>/mae_band.csv
Usage: python paper/analysis/mae_gap_range.py --run-id mae_gap_range
"""
import argparse
import os


import pandas as pd  # noqa: E402

from defectshift import config  # noqa: E402
from defectshift.utils import log  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    D = pd.read_csv(os.path.join(rf, "scale_normalized_bootstrap", "sn_domain.csv"))
    D = D[D.scheme == "target"].assign(width=lambda x: x.tr_hi - x.tr_lo)
    C = pd.read_csv(os.path.join(rf, "scale_normalized_bootstrap", "sn_cells.csv"))
    C = C[C.scheme == "target"]
    rb = pd.read_csv(os.path.join(rf, "robust", "robust_cells.csv"))
    rb = rb[rb.group == "control"].groupby(["variant", "candidate"])[["rmse", "mae"]].median().reset_index()
    # the all-rows control values of robust must equal the point estimates of sn_boot (the same cell definition)
    chk = rb[rb.variant == "all"].set_index("candidate")
    for met in ("rmse", "mae"):
        sb = D[(D.group == "control") & (D.metric == met)].set_index("candidate").tr
        diff = float((chk[met].reindex(sb.index) - sb).abs().max())
        if diff > 1e-7:
            raise SystemExit(f"{met}: the control values of robust_cells and sn_boot differ (maximum difference {diff})")
    rows = []
    for grp in ("challenge", "control"):
        for met in ("rmse", "mae"):
            d, c = D[(D.group == grp) & (D.metric == met)], C[(C.group == grp) & (C.metric == met)]
            r = dict(group=grp, metric=met, n_domains=len(d), cell_abs_gap_median=float(c.tr.abs().median()),
                     boot_width_median=float(d.width.median()), domain_min=float(d.tr.min()), domain_max=float(d.tr.max()),
                     domain_range=float(d.tr.max() - d.tr.min()), min_dom=d.loc[d.tr.idxmin()].dom, min_dom_s=float(d.loc[d.tr.idxmin()].s),
                     max_dom=d.loc[d.tr.idxmax()].dom, max_dom_s=float(d.loc[d.tr.idxmax()].s))
            if grp == "control":
                x = rb[rb.variant == "rest99"][met]
                r.update(rest99_min=float(x.min()), rest99_max=float(x.max()), rest99_range=float(x.max() - x.min()))
                o = d[d.tr != d.tr.min()]
                r.update(range_without_min_dom=float(o.tr.max() - o.tr.min()))
            rows.append(r)
    R = pd.DataFrame(rows)
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    R.to_csv(os.path.join(out, "mae_band.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(2).to_string(index=False))
    log("[mae_band] done")


if __name__ == "__main__":
    main()
