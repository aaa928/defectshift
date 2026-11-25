"""Check of the difference in the verdict between x_time and y_time. It reads only saved predictions (no new training, CPU only).

The two temporal domains have almost the same source/target membership (98.6 % of all rows), yet only x_time lies beyond the control range. The following is given.
  (1) Membership agreement: the share of all rows with the same A/B membership in the two domains, and the row overlap (Jaccard) of B_tr and B_te of the two domains
  (2) Rescoring on the common test rows: on the intersection of the two B_te, the domain values (RMSE and MAE) are recomputed from the stored predictions of each domain,
      and also on the test rows of each domain alone. The training rows and the trained models of the two domains differ, so this check does not identify the cause of the difference in the verdict;
      it only shows how much the composition of the test rows contributes
  (3) From the same predictions the part of the intersection rows is redrawn by lot (with replacement, 2,000 draws) to describe the fluctuation of the difference between the two domain values (not a test)
Output: results/analysis/<run-id>/{xy_overlap.json, xy_rescore.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/time_axes_comparison.py --run-id time_axes_comparison
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import common as cm
from defectshift import config
from defectshift.candidates import build_one, load_data
from defectshift.utils import log

BOOT_SEED = 20261105


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = cm.load_challenge(a.benchmark_run_id)
    models = sorted(pf.model.unique())
    names = {cm.short(n): n for n in pf.candidate.unique()}
    if not {"x_time", "y_time"} <= set(names):
        raise SystemExit(f"no temporal domain: {sorted(names)}")
    D = {k: cm.load_domain("challenge", names[k], pf[pf.candidate == names[k]], df, y, lot, models) for k in ("x_time", "y_time")}
    mB = {k: build_one(df, names[k])[2] for k in D}
    ov = dict(same_side_all_rows=float(np.mean(mB["x_time"] == mB["y_time"])),
              n_rows=int(len(df)),
              jaccard_B_tr=jac(D["x_time"]["bv"]["B_tr"], D["y_time"]["bv"]["B_tr"]),
              jaccard_B_te=jac(D["x_time"]["idx"], D["y_time"]["idx"]),
              n_B_te=dict(x_time=int(len(D["x_time"]["idx"])), y_time=int(len(D["y_time"]["idx"]))))
    common = np.intersect1d(D["x_time"]["idx"], D["y_time"]["idx"])
    ov["n_B_te_common"] = int(len(common))
    log(f"[xy] membership agreement {ov['same_side_all_rows']:.1%} | B_tr Jaccard {ov['jaccard_B_tr']:.2f} | B_te Jaccard {ov['jaccard_B_te']:.2f} | common test rows {len(common)}")
    rows = []
    for k, d in D.items():
        inc = np.isin(d["idx"], common)
        for part, mk in (("all", np.ones(len(inc), bool)), ("common", inc), ("own_only", ~inc)):
            r, e = cm.domain_value(d["preds"], d["yv"], mk)
            rows.append(dict(domain=k, part=part, n_rows=int(mk.sum()), n_lots=int(len(np.unique(d["lots"][mk]))), rmse=r, mae=e))
    rs = pd.DataFrame(rows)
    # lot resampling of the common rows: the same lot weights for both domains (the lots of the common rows are the same in the two domains)
    rng = np.random.default_rng(BOOT_SEED)
    cl = np.unique(lot.loc[common].to_numpy())
    W = rng.multinomial(len(cl), np.full(len(cl), 1.0 / len(cl)), size=a.n_boot)
    diff = {"rmse": [], "mae": []}
    pos = {k: {lt: np.where((d["lots"] == lt) & np.isin(d["idx"], common))[0] for lt in cl} for k, d in D.items()}
    for w in W:
        vals = {}
        for k, d in D.items():
            rows_k = np.concatenate([np.repeat(pos[k][lt], c) for lt, c in zip(cl, w) if c > 0])
            yv = d["yv"][rows_k]
            pr = {m: [(ps[rows_k], pq[rows_k]) for ps, pq in v] for m, v in d["preds"].items()}
            vals[k] = cm.domain_value(pr, yv, np.ones(len(yv), bool))
        diff["rmse"].append(vals["x_time"][0] - vals["y_time"][0])
        diff["mae"].append(vals["x_time"][1] - vals["y_time"][1])
    for met in ("rmse", "mae"):
        v = np.array(diff[met])
        ov[f"common_boot_diff_{met}"] = dict(p2_5=float(np.percentile(v, 2.5)), p97_5=float(np.percentile(v, 97.5)), share_pos=float(np.mean(v > 0)))
    ov["n_boot"], ov["boot_seed"] = a.n_boot, BOOT_SEED
    json.dump(ov, open(os.path.join(out, "xy_overlap.json"), "w"), ensure_ascii=False, indent=1)
    rs.to_csv(os.path.join(out, "xy_rescore.csv"), index=False)
    print(json.dumps(ov, ensure_ascii=False, indent=1))
    print(rs.round(2).to_string(index=False))
    log("[xy] done")


if __name__ == "__main__":
    main()
