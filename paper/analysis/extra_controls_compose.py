"""Follow-up of the additional controls: the five evaluation-row configurations are applied in the same way to the 50 controls (original 10 + additional 40) and the 7 challenge domains, and the relative position of the target-reference gap is computed again.
It reads only saved predictions (no retraining, CPU only). The configuration rules use load_domain and cell of verdict_robustness.py (--base-disjoint) unchanged.
  all         all of B_te
  not_shared  lot-disjoint rows (the main evaluation; all rows of the controls by construction)
  rest99      lot-disjoint rows without the 1 % tail of the outcome (tail = the ceil(1 % |B_te|) rows with the largest |y - median(y of B_tr)|, the same rule for all challenge domains and controls)
  half1/half2 the two halves of B_te obtained by shuffling its lots with a fixed seed (HALF_SEED) and halving, intersected with the lot-disjoint rows (the controls in the same way)
8 models = Ridge, kNN, XGBoost, CatBoost, MLP, TabM, TabPFN, TabICL. Domain value = median over the 8 models of the repetition-mean gap per model.
Tracks: T0 original 10, T2 50 controls, T3 the 10 controls with the closest lot composition per domain (lot_match/lot_match_selection.csv).
k = number of controls with a strictly smaller gap (ties not counted), "exceeds all" = k == N. k / N is the relative position among the controls and not a p-value.
Check: the 'not_shared' values must equal those of extra_controls_verdict.py (challenge = gap of T0, controls = ctrl40_values.csv).
Output: results/analysis/<run-id>/{compose_values.csv, compose_verdict.csv}
Usage: python paper/analysis/extra_controls_compose.py --run-id extra_controls_compose
"""
import argparse
import os


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import verdict_robustness as rb  # noqa: E402
import common as cm  # noqa: E402
from defectshift import config, extra_controls  # noqa: E402
from defectshift.candidates import load_data  # noqa: E402
from defectshift.utils import log  # noqa: E402

M8 = ("Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL")
VARIANTS = ("all", "not_shared", "rest99", "half1", "half2")
SRC, REF = rb.SRC, rb.REF


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--ctrl-run-ids", default=cm.CTRL_RUNS)
    ap.add_argument("--new-run-id", default="controls_supplementary")
    ap.add_argument("--agg-run-id", default="extra_controls_verdict")
    ap.add_argument("--match-run-id", default="lot_match")
    ap.add_argument("--only", default="", help="comma-separated split names (for smoke runs)")
    a = ap.parse_args()
    extra_controls.register()   # register the target fold counts of the 40 additional seeds at run time (the config file is not changed)
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = cm.load_challenge(a.benchmark_run_id, list(M8))
    old, old_dirs = cm.load_controls(a.ctrl_run_ids, list(M8))
    old = old.drop_duplicates(["candidate", "model", "repeat", "regime"])
    new = pd.read_csv(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.new_run_id, "rows.csv"))
    new = new[(new.status == "ok") & new.regime.isin([SRC, REF])].drop_duplicates(["candidate", "model", "repeat", "regime"])
    new_dir = os.path.join(config.CACHE_DIR, "benchmark", "runs", a.new_run_id, "models")
    if old.candidate.nunique() != 10 or new.candidate.nunique() != 40:
        raise SystemExit(f"the number of controls is not 10 + 40: {old.candidate.nunique()}, {new.candidate.nunique()}")
    jobs = [("challenge", nm, g, None) for nm, g in pf.groupby("candidate", sort=False)]
    jobs += [("control", nm, g, old_dirs) for nm, g in old.groupby("candidate", sort=False)]
    jobs += [("control", nm, g, {m: new_dir for m in M8}) for nm, g in new.groupby("candidate", sort=False)]
    if a.only:
        keep = a.only.split(",")
        jobs = [j for j in jobs if cm.short(j[1]) in keep or j[1] in keep]
    rows = []
    for group, name, g, dirs in jobs:
        yv, lots, preds, masks = rb.load_domain(group, name, g, df, y, lot, list(M8), dirs, base_disjoint=True)
        for var in VARIANTS:
            mk = masks[var]
            cells = np.array([rb.cell(preds[m], yv, mk) for m in M8])
            rows.append(dict(group=group, candidate=name, dom=cm.short(name), variant=var, n_rows=int(mk.sum()), n_lots=int(len(np.unique(lots[mk]))),
                             rmse=float(np.median(cells[:, 0])), mae=float(np.median(cells[:, 1]))))
        log(f"[compose] {group} {cm.short(name)} done")
    V = pd.DataFrame(rows)
    V.to_csv(os.path.join(out, "compose_values.csv"), index=False)
    # check: not_shared equals the value of ctrl40_agg
    if not a.only:
        cv = pd.read_csv(os.path.join(rf, a.agg_run_id, "ctrl40_values.csv")).set_index("candidate")
        ns = V[(V.group == "control") & (V.variant == "not_shared")].set_index("candidate")
        for met in ("rmse", "mae"):
            d = float((ns[met] - cv[f"tr_{met}"].reindex(ns.index)).abs().max())
            if d > 1e-7:
                raise SystemExit(f"the not_shared values of the controls differ from ctrl40_agg ({met}, maximum difference {d})")
        ver = pd.read_csv(os.path.join(rf, a.agg_run_id, "ctrl40_verdict.csv"))
        t0 = ver[ver.track == "T0"].set_index("dom")
        chn = V[(V.group == "challenge") & (V.variant == "not_shared")].set_index("dom")
        for met in ("rmse", "mae"):
            d = float((chn[met] - t0[f"gap_{met}"].reindex(chn.index)).abs().max())
            if d > 1e-7:
                raise SystemExit(f"the not_shared values of the challenge domains differ from T0 of ctrl40_agg ({met}, maximum difference {d})")
        log("[compose] check passed: the not_shared values equal those of ctrl40_agg")
    # verdict
    ch = V[V.group == "challenge"]
    ct = V[V.group == "control"]
    names_old = sorted(old.candidate.unique())
    sel = pd.read_csv(os.path.join(rf, a.match_run_id, "lot_match_selection.csv"))
    near = {d: g.control.tolist() for d, g in sel.groupby("dom")}
    vr = []
    for _, r in ch.iterrows():
        for track, names in (("T0", names_old), ("T2", None), ("T3", near.get(r.dom))):
            if names is None:
                c = ct[ct.variant == r.variant]
            else:
                c = ct[(ct.variant == r.variant) & ct.candidate.isin(names)]
            if c.empty:
                continue
            row = dict(track=track, dom=r.dom, variant=r.variant, n_ctrl=len(c), n_rows=r.n_rows)
            for met in ("rmse", "mae"):
                row[f"gap_{met}"] = r[met]
                row[f"ctrl_max_{met}"] = float(c[met].max())
                row[f"margin_{met}"] = r[met] - row[f"ctrl_max_{met}"]
                row[f"k_{met}"] = int((c[met] < r[met]).sum())
            row["k_both"] = int(((c.rmse < r.rmse) & (c.mae < r.mae)).sum())
            vr.append(row)
    VR = pd.DataFrame(vr)
    VR.to_csv(os.path.join(out, "compose_verdict.csv"), index=False)
    pd.set_option("display.width", 250)
    show = ["track", "dom", "variant", "n_ctrl", "gap_rmse", "ctrl_max_rmse", "k_rmse", "gap_mae", "ctrl_max_mae", "k_mae", "k_both"]
    for var in VARIANTS:
        print(f"\n== {var}\n", VR[(VR.variant == var) & VR.track.isin(["T0", "T2"])][show].round(2).to_string(index=False))
    log("[compose] done")


if __name__ == "__main__":
    main()
