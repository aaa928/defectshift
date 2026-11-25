"""Block bootstrap that preserves the production order of the test lots. It reads only saved predictions (no retraining, CPU only).
Lots that are produced close in time can resemble each other, in which case the independent lot resampling interval could come out too narrow;
this script checks the interval with block resampling of the lots in production order.

The statistic is the same as in evaluation_criteria.py (--base-disjoint) and repetition_uncertainty.boot: with the resampled lot weights W, the
RMSE and MAE target-reference gap per model and repetition -> repetition mean per model -> median over the 22 models = domain value. Evaluation rows = lot-disjoint test rows.
Resampling:
  iid      lots are drawn independently with replacement (the existing method). With the same random-number rule it is checked that the interval of the existing criteria_boot.csv (criteria, disjoint) is reproduced.
  block b  circular block bootstrap (Politis-Romano): the test lots are joined in a circle in the order of the lot rank (lot_order_proximity.lot_ranks; axis = y_time for the y_time domain, x_time otherwise),
           the start points of runs of b consecutive lots are drawn uniformly ceil(L/b) times, and the first L lots are used. b = 2, 5, 10.
  moving b non-circular moving block bootstrap (Kuensch): from the same order, without joining the ends, the start points are drawn from 0..L-b. It is a comparison to check
           whether the circular joining (treating the last and the first lot as adjacent) affects the interval. Lots at both ends are drawn less often.
2,000 draws, 2.5-97.5 percentiles. The interval is conditional on the trained models and the saved predictions and does not include the uncertainty of training or domain selection.
Output: results/analysis/<run-id>/block_boot.csv
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/lot_block_bootstrap.py --run-id lot_block_bootstrap
"""
import argparse
import os
import zlib

import numpy as np
import pandas as pd

import common as cm
import repetition_uncertainty as unc
from lot_order_proximity import lot_ranks
from defectshift import config
from defectshift.candidates import load_data
from defectshift.utils import log

BLOCKS = (2, 5, 10)
BLOCK_SEED = 20261107


def domain_stats(yv, codes, L, preds, models, W):
    """Domain value (median over the models of the RMSE and MAE gap), the same as in repetition_uncertainty.boot, computed with the given lot weights W (n_boot, L)."""
    cnt = np.bincount(codes, minlength=L).astype(float)
    n = W @ cnt
    rm, ma = [], []
    for m in models:
        a_rm, a_ma = [], []
        for ps, pr in preds[m]:
            es, er = ps - yv, pr - yv
            s = [W @ np.bincount(codes, w, minlength=L) for w in (es ** 2, er ** 2, np.abs(es), np.abs(er))]
            a_rm.append(100 * (np.sqrt(s[0] / n) / np.sqrt(s[1] / n) - 1))
            a_ma.append(100 * ((s[2] / n) / (s[3] / n) - 1))
        rm.append(np.mean(a_rm, axis=0))
        ma.append(np.mean(a_ma, axis=0))
    return np.median(np.stack(rm), axis=0), np.median(np.stack(ma), axis=0)


def block_weights(order, b, n_boot, rng, circular=True):
    """Lot weights of the block bootstrap. order = positions of the lots laid out in lot-rank order (index in np.unique order). Returns (n_boot, L).
    circular=True: circular (start points 0..L-1, the first lot follows the last lot). False: non-circular moving block (Kuensch; start points 0..L-b, ends not joined)."""
    L = len(order)
    nb = int(np.ceil(L / b))
    starts = rng.integers(0, L if circular else L - b + 1, size=(n_boot, nb))
    pos = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)[:, :L]
    if circular:
        pos = pos % L   # rank positions joined in a circle
    W = np.zeros((n_boot, L))
    lots = order[pos]
    for r in range(n_boot):
        np.add.at(W[r], lots[r], 1.0)
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--criteria-run-id", default="criteria")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    df, y = load_data()
    lot = df[config.LOT_COL]
    rank_by_axis = {ax: lot_ranks(df, lot, ax) for ax in ("x_time", "y_time")}
    pf = cm.load_challenge(a.benchmark_run_id)
    models = sorted(pf.model.unique())
    old = pd.read_csv(os.path.join(config.RESULTS_DIR, "analysis", a.criteria_run_id, "criteria_boot.csv"))
    old = old[old.variant == "disjoint"].set_index("candidate")
    rows = []
    for name, g in pf.groupby("candidate", sort=False):
        d = cm.load_domain("challenge", name, g, df, y, lot, models)
        mk = ~np.isin(d["lots"], lot.loc[d["a"]["A_dev"]].unique())
        yv, lots = d["yv"][mk], d["lots"][mk]
        preds = {m: [(ps[mk], pr[mk]) for ps, pr in pairs] for m, pairs in d["preds"].items()}
        ul, codes = np.unique(lots, return_inverse=True)
        L = len(ul)
        # iid: reproduces the existing interval (same seed rule, same multinomial call)
        rng = np.random.default_rng([unc.BOOT_SEED, zlib.crc32(cm.short(name).encode())])
        W = rng.multinomial(L, np.full(L, 1.0 / L), size=a.n_boot).astype(float)
        r, e = domain_stats(yv, codes, L, preds, models, W)
        lo, hi = np.percentile(r, [2.5, 97.5])
        o = old.loc[name]
        if abs(lo - o.rmse_lo) > 1e-9 or abs(hi - o.rmse_hi) > 1e-9:
            raise SystemExit(f"{name}: the independent resampling interval differs from the existing criteria_boot ({lo:.4f}, {hi:.4f}) versus ({o.rmse_lo:.4f}, {o.rmse_hi:.4f})")
        axis = "y_time" if cm.short(name) == "y_time" else "x_time"
        order = np.argsort(rank_by_axis[axis].loc[ul].to_numpy(), kind="stable")
        base = dict(candidate=name, dom=cm.short(name), axis=axis, n_lots=L, n_rows=int(mk.sum()))
        rows.append(dict(base, scheme="iid", block=1, rmse_lo=lo, rmse_hi=hi, mae_lo=float(np.percentile(e, 2.5)), mae_hi=float(np.percentile(e, 97.5))))
        for b in BLOCKS:
            rng_b = np.random.default_rng([BLOCK_SEED, zlib.crc32(cm.short(name).encode()), b])
            r, e = domain_stats(yv, codes, L, preds, models, block_weights(order, b, a.n_boot, rng_b))
            rows.append(dict(base, scheme="block", block=b, rmse_lo=float(np.percentile(r, 2.5)), rmse_hi=float(np.percentile(r, 97.5)),
                             mae_lo=float(np.percentile(e, 2.5)), mae_hi=float(np.percentile(e, 97.5))))
        for b in BLOCKS:
            rng_m = np.random.default_rng([BLOCK_SEED, zlib.crc32(cm.short(name).encode()), b, 1])
            r, e = domain_stats(yv, codes, L, preds, models, block_weights(order, b, a.n_boot, rng_m, circular=False))
            rows.append(dict(base, scheme="moving", block=b, rmse_lo=float(np.percentile(r, 2.5)), rmse_hi=float(np.percentile(r, 97.5)),
                             mae_lo=float(np.percentile(e, 2.5)), mae_hi=float(np.percentile(e, 97.5))))
        log(f"[block_boot] {cm.short(name)} done (test lots {L}, independent resampling interval reproduced)")
    t = pd.DataFrame(rows)
    t["rmse_width"] = t.rmse_hi - t.rmse_lo
    t["mae_width"] = t.mae_hi - t.mae_lo
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    t.to_csv(os.path.join(out, "block_boot.csv"), index=False)
    pd.set_option("display.width", 200)
    print(t[["dom", "scheme", "block", "n_lots", "rmse_lo", "rmse_hi", "rmse_width", "mae_lo", "mae_hi", "mae_width"]].round(1).to_string(index=False))
    log("[block_boot] done")


if __name__ == "__main__":
    main()
