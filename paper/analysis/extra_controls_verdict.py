"""Aggregation of the 40 additional randomized lot-split controls (8 models). It splits them into three tracks and gives the verdict of the target-reference gap for the 7 challenge domains.
It reads only saved results (no retraining, CPU only). The three tracks are not mixed.
  T1  22 models, original 10 controls          (challenge values of lot_disjoint and ctrl_values.csv; the main result)
  T0  8 models, original 10 controls           (linking row: an intermediate step to separate whether the difference between T1 and T2 is due to the model set or to the control set)
  T2  8 models, 50 controls                    (original 10 + additional 40)
  T3  8 models, the 10 controls with the closest lot composition per domain   (lot_match/lot_match_selection.csv, a list fixed before the gaps were seen; the nearest controls
      are not required to be size-matched in the sense of control_sizes.py, whose rule is applied by size_matched_verdict.py)
8 models = Ridge, kNN, XGBoost, CatBoost, MLP, TabM, TabPFN, TabICL (2 per family; the same as M8 of scripts/extra/run_redrawn_target_splits.py).
Value definition (the same as lot_disjoint_gaps.control_values): control domain value = median over the 8 models of the repetition-mean gap per model.
  target-reference gap = 100 (ood_source / ood_reference - 1), ID-OOD gap = 100 (ood_source / id_source - 1), metrics rmse and mae. All of B_te of the controls is lot-disjoint.
Challenge values (8 models): median over the models of sn_cells.csv (target version, lot-disjoint point estimate) of scale_normalized_bootstrap.
k = number of controls with a strictly smaller gap (ties are not counted; the same definition as control_rank.py), "exceeds all" = k == N. k / N is the relative position among the controls and not a p-value.
Checks: for T1 the 22-model values must equal criteria, and for T0 the original-10 control values must equal eight_model_verdict/m8_verdict.csv. The additional 40 have 3,200 rows, status ok and no duplicates.
Output: results/analysis/<run-id>/{ctrl40_values.csv, ctrl40_verdict.csv, ctrl40_summary.csv}
Usage: python paper/analysis/extra_controls_verdict.py --run-id extra_controls_verdict
"""
import argparse
import os


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as cm  # noqa: E402
from defectshift import config  # noqa: E402
from defectshift.utils import log  # noqa: E402

M8 = ("Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL")
KEYS = ["candidate", "model", "repeat"]
SRC, REF = cm.SRC, cm.REF


def ctrl_cells(rows):
    """Control rows.csv format -> model cell values (repetition mean): columns tr_rmse, tr_mae, io_rmse, io_mae."""
    rows = rows.drop_duplicates(["candidate", "model", "repeat", "regime"])
    s = rows[rows.regime == SRC].set_index(KEYS)
    r = rows[rows.regime == REF].set_index(KEYS).reindex(s.index)
    out = pd.DataFrame(index=s.index)
    for met in ("rmse", "mae"):
        out[f"tr_{met}"] = 100 * (s[f"{met}_ood"] / r[f"{met}_ood"] - 1)
        out[f"io_{met}"] = 100 * (s[f"{met}_ood"] / s[f"{met}_id"] - 1)
    return out.groupby(["candidate", "model"]).mean().reset_index()


def domain_values(cells, models):
    c = cells[cells.model.isin(models)]
    if c.groupby("candidate").model.nunique().min() != len(models):
        raise SystemExit("a split lacks some model in the controls")
    return c.groupby("candidate")[["tr_rmse", "tr_mae", "io_rmse", "io_mae"]].median().reset_index()


def verdict(ch_vals, ctl, track, models_label, ctl_names):
    """ch_vals: DataFrame(dom, tr_rmse, tr_mae, io_rmse, io_mae). ctl: control domain values. k is the number of controls that are strictly smaller."""
    rows = []
    for _, r in ch_vals.iterrows():
        names = ctl_names[r.dom] if isinstance(ctl_names, dict) else ctl_names
        c = ctl[ctl.candidate.isin(names)]
        n = len(c)
        row = dict(track=track, models=models_label, dom=r.dom, n_ctrl=n)
        for met in ("rmse", "mae"):
            row[f"gap_{met}"] = r[f"tr_{met}"]
            row[f"ctrl_max_{met}"] = float(c[f"tr_{met}"].max())
            row[f"ctrl_min_{met}"] = float(c[f"tr_{met}"].min())
            row[f"margin_{met}"] = r[f"tr_{met}"] - row[f"ctrl_max_{met}"]
            row[f"k_{met}"] = int((c[f"tr_{met}"] < r[f"tr_{met}"]).sum())
            row[f"io_{met}"] = r[f"io_{met}"]
            row[f"k_io_{met}"] = int((c[f"io_{met}"] < r[f"io_{met}"]).sum())
        row["k_both"] = int(((c.tr_rmse < r.tr_rmse) & (c.tr_mae < r.tr_mae)).sum())
        row["exceed_both"] = row["k_both"] == n
        row["exceed_rmse_only"] = row["k_rmse"] == n
        row["exceed_mae_only"] = row["k_mae"] == n
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--new-run-id", default="controls_supplementary")
    ap.add_argument("--ctrl-run-ids", default=cm.CTRL_RUNS)
    ap.add_argument("--eight-model-run-id", default="eight_model_verdict")
    ap.add_argument("--sn-run-id", default="scale_normalized_bootstrap")
    ap.add_argument("--match-run-id", default="lot_match")
    a = ap.parse_args()
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    # original 10 (22 models and 8 models)
    pf = cm.load_challenge("benchmark_canonical")
    models22 = sorted(pf.model.unique())
    old_rows, _ = cm.load_controls(a.ctrl_run_ids, models22)
    old_cells = ctrl_cells(old_rows)
    # additional 40
    new = pd.read_csv(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.new_run_id, "rows.csv"))
    new = new[new.regime.isin([SRC, REF])]
    if (new.status != "ok").any() or new.duplicated(["candidate", "model", "repeat", "regime"]).any():
        raise SystemExit("the additional 40 runs contain rows that are not ok, or duplicates")
    if sorted(new.model.unique()) != sorted(M8) or new.candidate.nunique() != 40:
        raise SystemExit(f"the number of models or splits of the additional 40 runs differs: {new.model.nunique()}, {new.candidate.nunique()}")
    cnt = new.groupby(["candidate", "model"]).size()
    if not (cnt == config.N_REPEATS_BENCHMARK * 2).all():
        raise SystemExit("a (split, model) of the additional 40 lacks repetitions or regimes")
    new_cells = ctrl_cells(new)
    cells = pd.concat([old_cells.assign(src="original10"), new_cells.assign(src="added40")])
    cells.to_csv(os.path.join(out, "ctrl40_cells.csv"), index=False)
    v22_old = domain_values(old_cells, models22)
    v8_old = domain_values(old_cells, M8)
    v8_new = domain_values(new_cells, M8)
    v8_all = pd.concat([v8_old, v8_new])
    v8_all.assign(src=["original10"] * len(v8_old) + ["added40"] * len(v8_new)).to_csv(os.path.join(out, "ctrl40_values.csv"), index=False)
    # check 1: 22 models, the values of the original 10 controls equal the existing output
    cv = pd.read_csv(os.path.join(rf, "lot_disjoint", "ctrl_values.csv"))
    for met in ("rmse", "mae"):
        o = cv[cv.metric == met].set_index("candidate")
        d = float((v22_old.set_index("candidate")[f"tr_{met}"] - o.tr.reindex(v22_old.candidate)).abs().max())
        if d > 1e-7:
            raise SystemExit(f"the 22-model control values differ from lot_disjoint/ctrl_values.csv ({met}, maximum difference {d})")
    # challenge values: 22 models (existing disjoint output) and 8 models (target point estimate of sn_cells)
    dd = pd.read_csv(os.path.join(rf, "lot_disjoint", "disjoint_domain.csv"))
    dd = dd[dd.variant == "disjoint"].assign(dom=lambda x: x.candidate.map(cm.short))
    ch22 = pd.DataFrame({"dom": dd[dd.metric == "rmse"].dom.values})
    for met in ("rmse", "mae"):
        d = dd[dd.metric == met].set_index("dom")
        ch22[f"tr_{met}"] = ch22.dom.map(d.tr_med)
        ch22[f"io_{met}"] = ch22.dom.map(d.idood_med)
    C = pd.read_csv(os.path.join(rf, a.sn_run_id, "sn_cells.csv"))
    C = C[(C.scheme == "target") & (C.group == "challenge") & C.model.isin(M8)]
    ch8 = C.pivot_table(index="dom", columns="metric", values=["tr", "io"], aggfunc="median")
    ch8.columns = [f"{a_}_{b}" for a_, b in ch8.columns]
    ch8 = ch8.reset_index()
    # check 2: T0 equals the 8-model verdict of the eight-model check
    m8 = pd.read_csv(os.path.join(rf, a.eight_model_run_id, "m8_verdict.csv")).set_index("dom")
    t0 = verdict(ch8, v8_old, "T0", "8", v8_old.candidate.tolist())
    for met in ("rmse", "mae"):
        d = float((t0.set_index("dom")[f"k_{met}"] - m8[f"k_{met}"]).abs().max())
        if d != 0:
            raise SystemExit(f"k_{met} of T0 differs from eight_model_verdict/m8_verdict.csv")
    t1 = verdict(ch22, v22_old, "T1", "22", v22_old.candidate.tolist())
    t2 = verdict(ch8, v8_all, "T2", "8", v8_all.candidate.tolist())
    sel = pd.read_csv(os.path.join(rf, a.match_run_id, "lot_match_selection.csv"))
    names3 = {d: g.control.tolist() for d, g in sel.groupby("dom")}
    t3 = verdict(ch8, v8_all, "T3", "8", names3)
    V = pd.concat([t1, t0, t2, t3])
    V.to_csv(os.path.join(out, "ctrl40_verdict.csv"), index=False)
    # summary: control range (per track) and the range of the additional 40 alone
    rows = []
    for name, vv in (("original10", v8_old), ("added40", v8_new), ("all50", v8_all)):
        for met in ("rmse", "mae"):
            rows.append(dict(set=name, metric=met, n=len(vv), tr_min=vv[f"tr_{met}"].min(), tr_max=vv[f"tr_{met}"].max(),
                             tr_pos_share=float((vv[f"tr_{met}"] > 0).mean()), io_min=vv[f"io_{met}"].min(), io_max=vv[f"io_{met}"].max()))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, "ctrl40_summary.csv"), index=False)
    pd.set_option("display.width", 260)
    show = ["track", "dom", "n_ctrl", "gap_rmse", "ctrl_max_rmse", "margin_rmse", "k_rmse", "gap_mae", "ctrl_max_mae", "margin_mae", "k_mae", "k_both", "exceed_both"]
    print(V[show].round(2).to_string(index=False))
    print(S.round(2).to_string(index=False))
    log("[ctrl40_agg] done")


if __name__ == "__main__":
    main()
