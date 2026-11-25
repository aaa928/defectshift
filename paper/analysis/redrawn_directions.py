"""Aggregation of the directions (ID-OOD, target-reference) for the redrawn splits and of the comparison with the nearest controls (saved predictions only, no new training, CPU only).
Directions: the 5 redraws of the target split (run names redrawn_s<k>_a/_b) and the canonical split (benchmark_canonical) are recomputed with the same definitions.
  For every cell (model) the per-repetition values are computed and averaged over the repetitions, domain value = median over the 8 models (the same aggregation as verdict_robustness.cell).
  - target-reference gap  = 100 (error_T(f_S) / error_T(f_T) - 1), evaluation rows are the lot-disjoint target test rows
  - ID-OOD gap            = 100 (error_T(f_S) / error_S(f_S) - 1), numerator on the lot-disjoint rows, denominator on all of the source test set A_held (definition of the paper)
  - spread ratio s        = spread(y_T, lot-disjoint rows) / spread(y_S, A_held); standard deviation (np.std) for RMSE, mean absolute deviation around the median for MAE
  - normalized comparison = 100 ((1 + ID-OOD/100) / s - 1), computed per fit
  Direction: whether the signs of the domain values agree for ID-OOD against target-reference and for normalized against target-reference (point estimates; no bootstrap classification).
Nearest controls: the redrawn gaps (redrawn_verdict/resplit_verdict.csv) are compared as k / 10 with the 10 controls whose lot composition is closest per domain (lot_match/lot_match_selection.csv,
  a list fixed from the canonical split information). Control values are from extra_controls_compose/compose_values.csv (8 models, not_shared and rest99).
Checks: the target-reference gap and the ID-OOD gap of the canonical split must equal T0 (8 models) of ctrl40_agg, and the redrawn gaps must equal resplit_agg.
  The normalized comparison is compared with the 8-model median of scale_normalized_bootstrap/sn_cells.csv (scheme target) and the difference is printed (0 if the definitions agree).
Output: results/analysis/<run-id>/{rq1_resplit_cells.csv, rq1_resplit_domain.csv, resplit_nearest10.csv}
Usage: python paper/analysis/redrawn_directions.py --run-id redrawn_directions
"""
import argparse
import json
import os


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as cm  # noqa: E402
from defectshift import config  # noqa: E402
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data, slot_dir  # noqa: E402
from defectshift.splits import B_view, get_A  # noqa: E402
from defectshift.splits_redraw import B_view_resplit  # noqa: E402
from defectshift.utils import log  # noqa: E402
from defectshift.directions import same_direction  # noqa: E402

M8 = ("Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL")
SRC, REF = "ID_match", "OOD_match"
HEAD_DOMAINS = ("cat_050", "x_time<=2099-05-04 21:18:39")   # domains of redraw step 1 (_a), the same as redrawn_verdict


def npz(path):
    """Reads the prediction files directly into arrays and closes them (so that thousands of files are not kept open)."""
    with np.load(path) as z:
        return {k: z[k] for k in ("id_idx", "id_pred", "ood_idx", "ood_pred")}


def spread(v, met):
    return float(np.std(v)) if met == "rmse" else float(np.mean(np.abs(v - np.median(v))))


def err(p, v, met):
    return float(np.sqrt(np.mean((p - v) ** 2))) if met == "rmse" else float(np.mean(np.abs(p - v)))


def cells(zs, y, lot, a_dev_lots, a_held):
    """zs: {model: [(src_npz, ref_npz) x repetitions]}. Returns, per model, {met: (tr, io, sn)} repetition means, s and the number of rows."""
    out = {}
    s_val = None
    idx0 = None
    for m, reps in zs.items():
        acc = {met: [] for met in ("rmse", "mae")}
        for zs_, zr in reps:
            idx = zs_["ood_idx"]
            idx0 = idx if idx0 is None else idx0
            if not np.array_equal(idx, idx0):
                raise SystemExit(f"{m}: the B_te row order differs between models and repetitions")
            if not (np.array_equal(idx, zr["ood_idx"]) and np.array_equal(zs_["id_idx"], np.asarray(a_held))):
                raise SystemExit(f"{m}: the test rows or the source test rows differ from the expectation")
            mk = ~np.isin(lot.loc[idx].to_numpy(), a_dev_lots)
            yT, yS = y.loc[idx].to_numpy()[mk], y.loc[zs_["id_idx"]].to_numpy()
            for met in ("rmse", "mae"):
                eT = err(zs_["ood_pred"][mk], yT, met)
                eR = err(zr["ood_pred"][mk], yT, met)
                eS = err(zs_["id_pred"], yS, met)
                s = spread(yT, met) / spread(yS, met)
                io = eT / eS - 1
                acc[met].append((100 * (eT / eR - 1), 100 * io, 100 * ((1 + io) / s - 1), s))
        out[m] = {met: np.mean(acc[met], axis=0) for met in acc}
        s_val = {met: out[m][met][3] for met in out[m]}
    return out, s_val, int(mk.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--screening-run-id", default="screening_main")
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--ks", default="1,2,3,4,5")
    ap.add_argument("--only", default="", help="comma-separated domain names (for smoke runs)")
    a = ap.parse_args()
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    od = os.path.join(rf, a.run_id)
    os.makedirs(od, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    fin = json.load(open(os.path.join(config.RESULTS_DIR, "screening", "runs", a.screening_run_id, "selection", "final_domains.json")))
    domains = fin["categorical"] + fin["temporal"]
    if a.only:
        domains = [d for d in domains if d in a.only.split(",")]
    ks = [int(k) for k in a.ks.split(",")]
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    cell_rows, dom_rows = [], []
    for name in domains:
        dom = cm.short(name)
        slot, mA, mB = build_one(df, name)
        A = get_A(slot, df.index[mA], y, lot)
        a_dev_lots = lot.loc[A["A_dev"]].unique()
        B_idx = df.index[mB]
        parts = {}
        g = pf[(pf.candidate == name) & pf.regime.isin([SRC, REF]) & pf.model.isin(M8)]
        parts[0] = ({m: [tuple(npz(g[(g.model == m) & (g.repeat == i) & (g.regime == r)].pred_path.iloc[0]) for r in (SRC, REF))
                         for i in range(config.N_REPEATS_BENCHMARK)] for m in M8}, B_view(slot, B_idx, y, lot, "benchmark")["B_te"])
        for k in ks:
            run = f"redrawn_s{k}_{'a' if name in HEAD_DOMAINS else 'b'}"
            pdir = os.path.join(config.CACHE_DIR, "benchmark", "runs", run, "models", slot_dir(slot))
            parts[k] = ({m: [tuple(npz(os.path.join(pdir, f"{m}__{r}__r{i}.pred.npz")) for r in (SRC, REF))
                             for i in range(config.N_REPEATS_BENCHMARK)] for m in M8}, B_view_resplit(slot, B_idx, y, lot, "benchmark", k)["B_te"])
        for k, (zs, bte) in parts.items():
            first = zs[M8[0]][0][0]["ood_idx"]
            if not np.array_equal(np.sort(first), np.sort(np.asarray(bte))):
                raise SystemExit(f"{dom} split {k}: B_te of the predictions differs from the split")
            c, s_val, n_rows = cells(zs, y, lot, a_dev_lots, A["A_heldout"])
            for m, v in c.items():
                for met in ("rmse", "mae"):
                    cell_rows.append(dict(dom=dom, k=k, model=m, metric=met, tr=v[met][0], io=v[met][1], sn=v[met][2], s=v[met][3]))
            for met in ("rmse", "mae"):
                tr = float(np.median([c[m][met][0] for m in M8]))
                io = float(np.median([c[m][met][1] for m in M8]))
                sn = float(np.median([c[m][met][2] for m in M8]))
                dom_rows.append(dict(dom=dom, k=k, metric=met, tr=tr, io=io, sn=sn, s_pct=100 * (s_val[met] - 1), n_rows=n_rows,
                                     dir_io=("same" if same_direction(io, tr) else "opposite"),
                                     dir_sn=("same" if same_direction(sn, tr) else "opposite")))
        log(f"[resplit_rq1] {dom} done")
    C, D = pd.DataFrame(cell_rows), pd.DataFrame(dom_rows)
    # check 1: canonical tr and io equal T0 of ctrl40_agg
    t0 = pd.read_csv(os.path.join(rf, "extra_controls_verdict/ctrl40_verdict.csv"))
    t0 = t0[t0.track == "T0"].set_index("dom")
    for met in ("rmse", "mae"):
        c0 = D[(D.k == 0) & (D.metric == met)].set_index("dom")
        assert len(c0) == len(domains)
        for col, ref in (("tr", f"gap_{met}"), ("io", f"io_{met}")):
            d = float((c0[col] - t0[ref].reindex(c0.index)).abs().max())
            if d > 1e-6:
                raise SystemExit(f"canonical {col} ({met}) differs from T0 of ctrl40_agg (maximum difference {d})")
    # check 2: the redrawn tr equals resplit_agg
    rv = pd.read_csv(os.path.join(rf, "redrawn_verdict/resplit_verdict.csv"))
    rv = rv[rv.variant == "not_shared"].set_index(["dom", "k"])
    for met in ("rmse", "mae"):
        dd = D[D.metric == met].set_index(["dom", "k"])
        ref = rv[f"gap_{met}"].reindex(dd.index)
        assert ref.notna().all()
        d = float((dd.tr - ref).abs().max())
        if d > 1e-6:
            raise SystemExit(f"the redrawn tr ({met}) differs from resplit_agg (maximum difference {d})")
    # comparison: the normalized comparison against the 8-model median of sn_cells
    sc = pd.read_csv(os.path.join(rf, "scale_normalized_bootstrap/sn_cells.csv"))
    sc = sc[(sc.group == "challenge") & (sc.scheme == "target") & sc.model.isin(M8)].groupby(["dom", "metric"]).sn.median()
    c0 = D[D.k == 0].set_index(["dom", "metric"]).sn
    ref = sc.reindex(c0.index)
    assert ref.notna().all()
    diff = float((c0 - ref).abs().max())
    log(f"[resplit_rq1] checks passed (T0, resplit_agg). Maximum difference between the normalized comparison and the 8-model median of sn_cells {diff:.6f}")
    C.to_csv(os.path.join(od, "rq1_resplit_cells.csv"), index=False)
    D.to_csv(os.path.join(od, "rq1_resplit_domain.csv"), index=False)
    # the 10 nearest controls as reference
    sel = pd.read_csv(os.path.join(rf, "lot_match/lot_match_selection.csv"))
    comp = pd.read_csv(os.path.join(rf, "extra_controls_compose/compose_values.csv"))
    comp = comp[comp.group == "control"]
    rows = []
    for (dom, k, var), r in pd.read_csv(os.path.join(rf, "redrawn_verdict/resplit_verdict.csv")).set_index(["dom", "k", "variant"]).iterrows():
        names = sel[sel.dom == dom].control.tolist()
        cc = comp[(comp.variant == var) & comp.candidate.isin(names)]
        if len(names) != 10 or len(cc) != 10:
            raise SystemExit(f"{dom}: the 10 nearest controls were not found")
        row = dict(dom=dom, k=k, variant=var, gap_rmse=r.gap_rmse, gap_mae=r.gap_mae,
                   ctrl_max_rmse=float(cc.rmse.max()), ctrl_max_mae=float(cc.mae.max()),
                   k_rmse=int((cc.rmse < r.gap_rmse).sum()), k_mae=int((cc.mae < r.gap_mae).sum()),
                   k_both=int(((cc.rmse < r.gap_rmse) & (cc.mae < r.gap_mae)).sum()))
        rows.append(row)
    N = pd.DataFrame(rows)
    N.to_csv(os.path.join(od, "resplit_nearest10.csv"), index=False)
    pd.set_option("display.width", 250)
    print(D.round(1).to_string())
    print(N[N.variant == "not_shared"].round(1).to_string())
    log(f"[resplit_rq1] done -> {od}")


if __name__ == "__main__":
    main()
