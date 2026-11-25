"""Aggregation of the time-ordered configuration. It reads only saved results (no new training, CPU only).
Three rows and their comparison references:
  R0  canonical, 8 models: reads the existing aggregation (T0 of extra_controls_verdict = 8 models, original 10 controls, lot-disjoint) and the canonical rest99 values of the redrawn-split aggregation
  R1  canonical split, only the number of training rows reduced to n*_FO: time_ordered_size_only_<axis>. The reference is the original 10 controls (8 models, canonical split, for information)
  R2  time-ordered split, earlier lots only: time_ordered_time_ordered_<axis>. The reference is only the 10 time-ordered controls of the same axis (time_ordered_controls_<axis>); not combined with the original controls
Value definition (the same as verdict_robustness.cell): cell = repetition mean of 100 (error_source / error_reference - 1) per repetition, domain value = median over models.
Evaluation rows: lot-disjoint (test rows whose lot occurs in A_dev excluded) and rest99, which in addition leaves out the top 1% of |y - median(B_tr)|.
k = number of reference controls with a strictly smaller gap (ties excluded), k = N means "exceeds all". k is a position among the controls and is not read as a probability.
Checks: for every run, mode, axis and canonical=False in time_ordered.json, all model x repetition x regime entries have status ok, n* equals n*_FO,
the test rows of the predictions are the same across models and repetitions and equal the B_te of that mode (the rebuilt split), and the predictions are finite.
Output: results/analysis/<run-id>/{forward_values.csv, forward_verdict.csv, forward_split.csv}
Usage: python paper/analysis/time_ordered_verdict.py --run-id time_ordered_verdict
      to test with smoke results: --suffix smk2 --models Ridge,XGBoost --repeats 1 --skip-r0
"""
import argparse
import json
import os


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from defectshift import config, splits  # noqa: E402
from defectshift.candidates import build_one, load_data, slot_dir  # noqa: E402
from defectshift.splits_time_ordered import forward_view  # noqa: E402
from defectshift.utils import log  # noqa: E402

M8 = ("Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL")
SRC, REF = "ID_match", "OOD_match"
TOP_FRAC = 0.01                     # the same as verdict_robustness.TOP_FRAC
AXES = ("x_time", "y_time")
DOM = {"x_time": "x_time<=2099-05-04 21:18:39", "y_time": "y_time<=2099-05-10 02:20:07.500000"}
CTRL = tuple(f"placebo@{s}" for s in range(9001, 9011))


def run_dir(rid):
    return os.path.join(config.RESULTS_DIR, "benchmark", "runs", rid)


def load_run(rid, mode, axis, models, repeats, df, y, lot):
    """One run: per name the cell values (dict variant -> (rmse, mae)) and the split figures. Stops if a check fails."""
    meta = json.load(open(os.path.join(run_dir(rid), "time_ordered.json")))
    if meta["mode"] != mode or meta.get("canonical") is not False or (mode == "controls" and meta["axis"] != axis):
        raise SystemExit(f"{rid}: mode, axis or canonical marker of time_ordered.json differs from the expectation")
    rows = pd.read_csv(os.path.join(run_dir(rid), "rows.csv")).drop_duplicates(["candidate", "model", "repeat", "regime"], keep="last")
    model_dir = os.path.join(config.CACHE_DIR, "benchmark", "runs", rid, "models")
    out, split = {}, []
    for cand, info in meta["domains"].items():
        g = rows[(rows.candidate == cand) & rows.model.isin(models) & (rows.repeat < repeats)]
        need = len(models) * repeats * 2
        if len(g) != need or not (g.status == "ok").all():
            raise SystemExit(f"{rid} {cand}: rows {len(g)} / {need}, not ok {int((g.status != 'ok').sum())}")
        for i in range(repeats):
            nfo = info["repeats"][str(i)]["n_star_fo"]
            if not (g[g.repeat == i].n_star == nfo).all():
                raise SystemExit(f"{rid} {cand} r{i}: n* differs from n*_FO({nfo})")
        slot, mA, mB = build_one(df, cand)
        a = splits.get_A(slot, df.index[mA], y, lot)
        if mode == "size_only":
            bv = splits.B_view(slot, df.index[mB], y, lot, "benchmark")
            bte, btr = bv["B_te"], bv["B_tr"]
        else:
            fv = forward_view(a["A_dev"], df.index[mB], lot, df[info["axis"]])
            bte, btr = fv["B_te"], fv["B_tr"]
            if fv["info"] != info["split"]:
                raise SystemExit(f"{rid} {cand}: the rebuilt split figures differ from the run record")
        idx0 = None
        pairs = {m: [] for m in models}
        for m in models:
            for i in range(repeats):
                z = {reg: np.load(os.path.join(model_dir, slot_dir(slot), f"{m}__{reg}__r{i}.pred.npz")) for reg in (SRC, REF)}
                for reg in (SRC, REF):
                    if idx0 is None:
                        idx0 = z[reg]["ood_idx"]
                    if not np.array_equal(idx0, z[reg]["ood_idx"]):
                        raise SystemExit(f"{rid} {cand} {m} r{i} {reg}: the test row order differs")
                    if not np.isfinite(z[reg]["ood_pred"]).all():
                        raise SystemExit(f"{rid} {cand} {m} r{i} {reg}: the predictions contain a non-finite value")
                pairs[m].append((z[SRC]["ood_pred"], z[REF]["ood_pred"]))
        if not np.array_equal(idx0, np.asarray(bte)):
            raise SystemExit(f"{rid} {cand}: the saved test rows differ from the rebuilt B_te")
        yv = y.loc[idx0].to_numpy()
        lots = lot.loc[idx0].to_numpy()
        ld = ~np.isin(lots, lot.loc[a["A_dev"]].unique())
        score = np.abs(yv - float(y.loc[btr].median()))
        k = max(1, int(np.ceil(TOP_FRAC * len(yv))))
        top = np.zeros(len(yv), bool)
        top[np.argsort(-score, kind="stable")[:k]] = True
        masks = dict(not_shared=ld, rest99=ld & ~top)
        vals = {}
        for var, mk in masks.items():
            cells = []
            for m in models:
                d = [(100 * (np.sqrt(np.mean((ps[mk] - yv[mk]) ** 2)) / np.sqrt(np.mean((pr[mk] - yv[mk]) ** 2)) - 1),
                      100 * (np.mean(np.abs(ps[mk] - yv[mk])) / np.mean(np.abs(pr[mk] - yv[mk])) - 1)) for ps, pr in pairs[m]]
                cells.append(np.mean(d, axis=0))
            cells = np.array(cells)
            vals[var] = (float(np.median(cells[:, 0])), float(np.median(cells[:, 1])), int(mk.sum()), int(len(np.unique(lots[mk]))))
        out[cand] = vals
        split.append(dict(run=rid, mode=mode, cand=cand, axis=info["axis"], n_star_fo=info["repeats"]["0"]["n_star_fo"],
                          n_star_fo_min=min(r["n_star_fo"] for r in info["repeats"].values()),
                          n_star_fo_max=max(r["n_star_fo"] for r in info["repeats"].values()), **info["split"]))
        log(f"[forward_agg] {rid} {cand[:14]} done")
    return out, split


def kcount(ctrl, gap):
    return int(sum(c < gap for c in ctrl))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--suffix", default="", help="tail of the run name (for smoke tests, e.g. smk2)")
    ap.add_argument("--models", default=",".join(M8))
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS_BENCHMARK)
    ap.add_argument("--controls", default=",".join(c.split("@")[1] for c in CTRL), help="seeds of the time-ordered controls (comma-separated)")
    ap.add_argument("--skip-r0", action="store_true", help="do not read R0 and the original control rows (for smoke tests)")
    a = ap.parse_args()
    models = a.models.split(",")
    ctrls = [f"placebo@{s}" for s in a.controls.split(",")]
    name = lambda mode, ax: f"time_ordered_{mode}{'_' + a.suffix if a.suffix else ''}_{ax}" if a.suffix else f"time_ordered_{mode}_{ax}"
    df, y = load_data()
    lot = df[config.LOT_COL]
    values, splits_rows, verdict = [], [], []
    R = {}
    for ax in AXES:
        for mode in ("time_ordered", "size_only", "controls"):
            vals, sp = load_run(name(mode, ax), mode, ax, models, a.repeats, df, y, lot)
            R[(mode, ax)] = vals
            splits_rows += sp
            for cand, v in vals.items():
                for var, (r_, m_, n_, l_) in v.items():
                    values.append(dict(row=mode, axis=ax, cand=cand, variant=var, gap_rmse=r_, gap_mae=m_, n_rows=n_, n_lots=l_))
        missing = [c for c in ctrls if c not in R[("controls", ax)]]
        if missing:
            raise SystemExit(f"{ax}: no time-ordered controls {missing}")
    if not a.skip_r0:
        base = os.path.join(config.RESULTS_DIR, "analysis")
        t0 = pd.read_csv(os.path.join(base, "extra_controls_verdict/ctrl40_verdict.csv"))
        t0 = t0[t0.track == "T0"].set_index("dom")
        cv = pd.read_csv(os.path.join(base, "extra_controls_verdict/ctrl40_values.csv"))
        cv10 = cv[cv.src == "original10"]
        rs = pd.read_csv(os.path.join(base, "redrawn_verdict/resplit_verdict.csv"))
        rs = rs[(rs.k == 0)].set_index(["dom", "variant"])
        comp = pd.read_csv(os.path.join(base, "extra_controls_compose/compose_values.csv"))
        c10_r99 = comp[(comp.group == "control") & (comp.variant == "rest99") & comp.candidate.isin(cv10.candidate)]
        if len(cv10) != 10 or len(c10_r99) != 10:
            raise SystemExit("the values of the original 10 controls were not found")
    for ax in AXES:
        d = DOM[ax]
        for var in ("not_shared", "rest99"):
            r2 = R[("time_ordered", ax)][d][var]
            r1 = R[("size_only", ax)][d][var]
            ct = [R[("controls", ax)][c][var] for c in ctrls]
            row = dict(axis=ax, variant=var, n_ctrl=len(ct),
                       r2_rmse=r2[0], r2_mae=r2[1], r2_rows=r2[2], r2_lots=r2[3],
                       ctrl_max_rmse=max(c[0] for c in ct), ctrl_max_mae=max(c[1] for c in ct),
                       ctrl_min_rmse=min(c[0] for c in ct), ctrl_min_mae=min(c[1] for c in ct),
                       k_rmse=kcount([c[0] for c in ct], r2[0]), k_mae=kcount([c[1] for c in ct], r2[1]),
                       k_both=int(sum((c[0] < r2[0]) and (c[1] < r2[1]) for c in ct)),
                       r1_rmse=r1[0], r1_mae=r1[1], r1_rows=r1[2])
            if not a.skip_r0:
                if var == "not_shared":
                    o_r, o_m = cv10.tr_rmse.tolist(), cv10.tr_mae.tolist()
                    row.update(r0_rmse=float(t0.loc[ax, "gap_rmse"]), r0_mae=float(t0.loc[ax, "gap_mae"]))
                else:
                    o_r, o_m = c10_r99.rmse.tolist(), c10_r99.mae.tolist()
                    row.update(r0_rmse=float(rs.loc[(ax, var), "gap_rmse"]), r0_mae=float(rs.loc[(ax, var), "gap_mae"]))
                row.update(orig_ctrl_max_rmse=max(o_r), orig_ctrl_max_mae=max(o_m),
                           r0_k_rmse=kcount(o_r, row["r0_rmse"]), r0_k_mae=kcount(o_m, row["r0_mae"]),
                           r1_k_rmse=kcount(o_r, r1[0]), r1_k_mae=kcount(o_m, r1[1]))
            verdict.append(row)
    od = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(values).to_csv(os.path.join(od, "forward_values.csv"), index=False)
    pd.DataFrame(verdict).to_csv(os.path.join(od, "forward_verdict.csv"), index=False)
    pd.DataFrame(splits_rows).to_csv(os.path.join(od, "forward_split.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(verdict).round(2).to_string())
    log(f"[forward_agg] done -> {od}")


if __name__ == "__main__":
    main()
