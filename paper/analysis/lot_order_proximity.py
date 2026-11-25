"""Proximity in lot order (diagnostic). It reads only saved predictions (no new training, CPU only).

The values of the time axes can be trusted only for their order (each axis is renumbered from an arbitrary reference date), so the rank distance of lots is used instead of the elapsed time. It is computed within an axis only and the two axes are not mixed.
Definitions
  Lot time = median of the times of all rows of that lot (axis = y_time for the y_time domain, x_time otherwise). Lot rank r = ordinal in the order (lot time, lot id) (1, 2, ... over all lots)
  For every repetition i the lot sets S_i and T_i of the rows actually used in training (source A_match, reference B_match; the same repeat_split and matched as in the benchmark)
  For every test lot l
    dT_prev, dT_next = rank distance to the nearest reference (target) training lot before / after l (missing if none), dT = the smaller of the two
    dS = rank distance to the nearest source training lot (0 if the same lot is in the source training)
    delta_r = dS - dT  (positive = closer in rank to a reference training lot than to a source training lot)
    later = share of cases in which the nearest reference training lot is after l (on the future side)
  One value per lot as the mean over the 5 repetitions. Error difference = (source - reference) of the mean squared error (MSE) and mean absolute error (MAE) of the rows in the lot, repetition mean per model -> median over models
  (the lot RMSE ratio is not used because the number of rows per lot differs)
Scope: lot-disjoint test lots, i.e. test rows of lots shared with the source training (shared) are excluded (temporal domains have no shared rows and equal all lots)
Output: Spearman correlation of delta_r and the error difference per lot (95 % interval by test-lot resampling), domain values per delta_r quartile bin (RMSE and MAE gap), share of cases in which the nearest reference training lot is later.
It is a diagnostic and not an analysis that shows a causal effect. Whether the time-axis values are actual elapsed times is not used.
Output files: results/analysis/<run-id>/{rankprox_lots.csv, rankprox_bins.csv, rankprox_summary.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/lot_order_proximity.py --run-id lot_order_proximity
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as cm
from defectshift import config
from defectshift.candidates import load_data
from defectshift.splits import matched, repeat_split
from defectshift.utils import log

N_BINS = 4
BOOT_SEED = 20261105
N_BOOT = 2000


def used_rows(d, y, lot):
    """Rows actually used in training in each repetition: source A_match, reference B_match (the same repeat_split and matched as in the benchmark). Returns: [(a_match, b_match), ...]"""
    out = []
    for i in range(config.N_REPEATS_BENCHMARK):
        a_train, _ = repeat_split(d["a"]["A_dev"], y, lot, i)
        _, a_match, b_match = matched(a_train, d["bv"]["B_tr"], i)
        out.append((np.asarray(a_match), np.asarray(b_match)))
    return out


def lot_ranks(df, lot, axis):
    """Lot id -> rank (ordinal obtained by counting all lots in the order (median time, id))."""
    t = df[axis].groupby(lot).median()
    order = pd.DataFrame(dict(lot=t.index, t=t.to_numpy())).sort_values(["t", "lot"], kind="stable")
    return pd.Series(np.arange(1, len(order) + 1), index=order.lot.to_numpy())


def nearest_prev_next(r, pool_sorted):
    """For the sorted rank array pool_sorted: the distance to the nearest value smaller than r (NaN if none) and the distance to the nearest value larger than r (NaN if none)."""
    lo = np.searchsorted(pool_sorted, r, side="left")          # position where a value >= r first occurs
    hi = np.searchsorted(pool_sorted, r, side="right")         # position where a value > r first occurs
    prev = np.where(lo > 0, r - pool_sorted[np.maximum(lo - 1, 0)], np.nan)
    nxt = np.where(hi < len(pool_sorted), pool_sorted[np.minimum(hi, len(pool_sorted) - 1)] - r, np.nan)
    return prev, nxt


def run_domain(group, name, g, df, y, lot, models, dirs, rank_by_axis, rng):
    d = cm.load_domain(group, name, g, df, y, lot, models, dirs)
    axis = "y_time" if cm.short(name) == "y_time" else "x_time"
    rk = rank_by_axis[axis]
    lots = d["lots"]
    ul, codes = np.unique(lots, return_inverse=True)
    L = len(ul)
    r_te = rk.loc[ul].to_numpy()
    shared = np.isin(ul, lot.loc[d["a"]["A_dev"]].unique())
    reps = used_rows(d, y, lot)
    dT_prev, dT_next, dS = [], [], []
    for a_match, b_match in reps:
        T = np.sort(np.unique(rk.loc[lot.loc[b_match].to_numpy()].to_numpy()))
        S = np.sort(np.unique(rk.loc[lot.loc[a_match].to_numpy()].to_numpy()))
        p, n = nearest_prev_next(r_te, T)
        sp, sn = nearest_prev_next(r_te, S)
        # on the source side the distance is 0 if the same lot is in the training rows
        same = np.isin(r_te, S)
        dS.append(np.where(same, 0.0, np.fmin(sp, sn)))
        dT_prev.append(p)
        dT_next.append(n)
    dT_prev, dT_next, dS = (np.array(x) for x in (dT_prev, dT_next, dS))            # (repetition, lot)
    dT = np.fmin(dT_prev, dT_next)
    later = np.where(np.isnan(dT_next), False, np.where(np.isnan(dT_prev), True, dT_next < dT_prev)).astype(float)
    # error difference per lot: model x repetition x lot
    cnt = np.bincount(codes, minlength=L).astype(float)
    dm, de = [], []
    for m in models:
        mse_d, mae_d = [], []
        for ps, pr in d["preds"][m]:
            es, er = ps - d["yv"], pr - d["yv"]
            mse_d.append((np.bincount(codes, es ** 2, L) - np.bincount(codes, er ** 2, L)) / cnt)
            mae_d.append((np.bincount(codes, np.abs(es), L) - np.bincount(codes, np.abs(er), L)) / cnt)
        dm.append(np.mean(mse_d, axis=0))
        de.append(np.mean(mae_d, axis=0))
    d_mse, d_mae = np.median(np.array(dm), axis=0), np.median(np.array(de), axis=0)
    lots_df = pd.DataFrame(dict(group=group, candidate=name, axis=axis, lot=ul, n_rows=cnt.astype(int), rank=r_te, shared=shared,
                                dT_prev=np.nanmean(dT_prev, axis=0),
                                dT_next=np.nanmean(dT_next, axis=0), dT=np.nanmean(dT, axis=0), dS=dS.mean(axis=0),
                                later_share=later.mean(axis=0), d_mse=d_mse, d_mae=d_mae))
    lots_df["delta_r"] = lots_df.dS - lots_df.dT
    ok = ~lots_df.shared.to_numpy() & lots_df.delta_r.notna().to_numpy()
    sub = lots_df[ok].reset_index(drop=True)
    row = dict(group=group, candidate=name, axis=axis, n_test_lots=L, n_shared_lots=int(shared.sum()), n_lots_used=len(sub),
               dT_median=float(sub.dT.median()), dS_median=float(sub.dS.median()), share_closer_to_target=float((sub.delta_r > 0).mean()),
               share_nearest_target_later=float(sub.later_share.mean()))
    for k, col in (("mse", "d_mse"), ("mae", "d_mae")):
        for xname, xcol in (("delta_r", "delta_r"), ("dT", "dT")):
            rho = spearmanr(sub[xcol], sub[col]).statistic
            bs = []
            for _ in range(N_BOOT if group == "challenge" else 0):
                ix = rng.integers(0, len(sub), len(sub))
                bs.append(spearmanr(sub[xcol].to_numpy()[ix], sub[col].to_numpy()[ix]).statistic)
            row[f"rho_{xname}_{k}"] = float(rho)
            if bs:
                row[f"rho_{xname}_{k}_lo"], row[f"rho_{xname}_{k}_hi"] = (float(v) for v in np.nanpercentile(bs, [2.5, 97.5]))
    bins = []
    q = pd.qcut(sub.delta_r.rank(method="first"), N_BINS, labels=False)
    for b in range(N_BINS):
        sel = sub.lot[q == b].to_numpy()
        mk = np.isin(lots, sel)
        r, e = cm.domain_value(d["preds"], d["yv"], mk)
        bins.append(dict(group=group, candidate=name, bin=b + 1, n_lots=len(sel), n_rows=int(mk.sum()), delta_r_lo=float(sub.delta_r[q == b].min()),
                         delta_r_hi=float(sub.delta_r[q == b].max()), rmse=r, mae=e))
    log(f"[rankprox] {group} {cm.short(name)[:12]:12s} axis {axis} | test lots {L} (shared {int(shared.sum())}) | dT median {row['dT_median']:.1f} dS median {row['dS_median']:.1f} "
        f"| lots closer to the reference side {row['share_closer_to_target']:.0%} | nearest reference lot later {row['share_nearest_target_later']:.0%} "
        f"| rho(Δr, ΔMSE) {row['rho_delta_r_mse']:.2f} rho(Δr, ΔMAE) {row['rho_delta_r_mae']:.2f}")
    return lots_df, bins, row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--ctrl-run-ids", default=cm.CTRL_RUNS)
    ap.add_argument("--only", default=None, help="first part of the challenge-domain names (comma-separated). Default = all 7")
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    rank_by_axis = {ax: lot_ranks(df, lot, ax) for ax in ("x_time", "y_time")}
    pf = cm.load_challenge(a.benchmark_run_id)
    models = sorted(pf.model.unique())
    if a.only:
        pf = pf[pf.candidate.str.startswith(tuple(a.only.split(",")))]
    if pf.empty:
        raise SystemExit(f"no challenge domain: --only({a.only})")
    jobs = [("challenge", nm, g, None) for nm, g in pf.groupby("candidate", sort=False)]
    if not a.no_controls:
        ctrl, dirs = cm.load_controls(a.ctrl_run_ids, models)
        jobs += [("control", nm, g, dirs) for nm, g in ctrl.groupby("candidate", sort=False)]
    log(f"[rankprox] {len(models)} models, {len(jobs)} domains, {len(rank_by_axis['x_time'])} lots")
    rng = np.random.default_rng(BOOT_SEED)
    lot_frames, bin_rows, summ = [], [], []
    for group, name, g, dirs in jobs:
        lf, br, sm = run_domain(group, name, g, df, y, lot, models, dirs, rank_by_axis, rng)
        lot_frames.append(lf)
        bin_rows += br
        summ.append(sm)
    pd.concat(lot_frames).to_csv(os.path.join(out, "rankprox_lots.csv"), index=False)
    bins, sm = pd.DataFrame(bin_rows), pd.DataFrame(summ)
    bins.to_csv(os.path.join(out, "rankprox_bins.csv"), index=False)
    sm.to_csv(os.path.join(out, "rankprox_summary.csv"), index=False)
    sm["dom"] = sm.candidate.map(cm.short)
    ch = sm[sm.group == "challenge"].set_index("dom")
    cols = ["n_lots_used", "dT_median", "dS_median", "share_closer_to_target", "share_nearest_target_later",
            "rho_delta_r_mse", "rho_delta_r_mse_lo", "rho_delta_r_mse_hi", "rho_delta_r_mae", "rho_delta_r_mae_lo", "rho_delta_r_mae_hi"]
    print("\nchallenge domain summary (Spearman of delta_r and the error difference per lot, 95 % interval by test-lot resampling):\n", ch[cols].round(2).to_string())
    b = bins.assign(dom=bins.candidate.map(cm.short))
    for k in ("rmse", "mae"):
        print(f"\n[{k}] gap per delta_r quartile bin (1 = closest to the source side ... 4 = closest to the reference side), challenge domains:\n",
              b[b.group == "challenge"].pivot(index="dom", columns="bin", values=k).round(2).to_string())
    if not a.no_controls:
        c = sm[sm.group == "control"]
        print("\n10 randomized splits: share of nearest reference lots on the later side %.2f-%.2f, share of lots closer to the reference side %.2f-%.2f, rho(delta_r, delta MSE) %.2f-%.2f" % (
            c.share_nearest_target_later.min(), c.share_nearest_target_later.max(), c.share_closer_to_target.min(), c.share_closer_to_target.max(),
            c.rho_delta_r_mse.min(), c.rho_delta_r_mse.max()))
    log("[rankprox] done")


if __name__ == "__main__":
    main()
