"""Supplementary analyses of the uncertainty and the composition of the target test set; they read only saved predictions and existing results (no retraining, CPU only).

  boot    Paired bootstrap that resamples B_te by production lot. The same resampling is applied to the source (ID_match) and the reference
          (OOD_match); it gives intervals for the cells (model x domain), the domain medians and the overall share of positive cells. Training
          is fixed (only the uncertainty of the test sample).
  shared  B_te rows are split into rows whose lot also occurs in A_dev (the source training and early-stopping pool) and the other rows, and the
          transfer loss is computed for each group. If both groups have at least MIN_SHARED_ROWS rows, a lot-wise bootstrap of the difference
          (shared - not shared; resampling inside each group) is added.
  timed   For temporal domains, the time distance (hours) from every B_te row to the nearest B_tr row is measured and the transfer loss
          is reported by distance quartile.
  screen  Reliability across repetitions of the transfer loss in the screening stage (4 models x 3 repetitions): Spearman correlation between the
          consensus of single repetitions and the Spearman-Brown reliability of the mean of 3 repetitions.
  absdiff Absolute RMSE difference per cell (in defect-rate units) and its size relative to the standard deviation of the B_te outcome.
Input: results/benchmark/runs/<benchmark>/report/per_fit.csv (with pred_path), the prediction files .pred.npz, the split caches (splits.get_A and
      B_view, benchmark stage) and results/screening/runs/<screening>/rows.csv. Output: results/analysis/<run-id>/.
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/test_set_checks.py --run-id <run> --only cat_019 --n-boot 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from defectshift import config  # noqa: E402
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data, parse_candidate  # noqa: E402
from defectshift.splits import B_view, get_A  # noqa: E402
from defectshift.utils import log  # noqa: E402

SRC, REF = "ID_match", "OOD_match"
BOOT_SEED = 20261028
MIN_SHARED_ROWS = 30   # the bootstrap of the shared minus not-shared difference is run only for domains with at least this many rows in both groups


def load_fits(benchmark, only):
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", benchmark, "report", "per_fit.csv"))
    pf = pf[pf.regime.isin([SRC, REF])]
    if only:
        pf = pf[pf.candidate.isin(only)]
    n = pf.groupby(["candidate", "model", "repeat"]).regime.nunique()
    assert (n == 2).all(), "a repetition lacks the source-reference pair"
    return pf


def domain_preds(g):
    """Collects all fitted predictions of one domain in the same B_te row order. Returns the row index and {(model, repeat, regime): prediction}."""
    idx, out = None, {}
    for r in g.itertuples():
        z = np.load(r.pred_path)
        if idx is None:
            idx = z["ood_idx"]
        assert np.array_equal(idx, z["ood_idx"]), f"{r.candidate} {r.model} the B_te row order differs between fits"
        out[(r.model, r.repeat, r.regime)] = z["ood_pred"]
    return pd.Index(idx), out


def gap_pct(src, ref):
    return 100 * (src - ref) / ref


def cell_table(yv, preds, keys):
    """Repetition-mean transfer loss per cell (RMSE and MAE in %) on a row subset (yv and preds are already cut to that subset)."""
    out = []
    for m, reps in keys.items():
        d_rmse, d_mae = [], []
        for i in reps:
            e_s, e_r = preds[(m, i, SRC)] - yv, preds[(m, i, REF)] - yv
            d_rmse.append(gap_pct(np.sqrt(np.mean(e_s ** 2)), np.sqrt(np.mean(e_r ** 2))))
            d_mae.append(gap_pct(np.mean(np.abs(e_s)), np.mean(np.abs(e_r))))
        out.append(dict(model=m, gap_rmse_pct=np.mean(d_rmse), gap_mae_pct=np.mean(d_mae), n_rows=len(yv)))
    return pd.DataFrame(out)


def boot_domain(name, idx, preds, keys, y, lot, n_boot, rng):
    """Paired bootstrap by lot. Returns the resampled matrices per cell {metric: (n_boot, n_models)} and the point estimates."""
    yv = y.loc[idx].to_numpy()
    codes, lots = pd.factorize(lot.loc[idx].to_numpy())
    L = len(lots)
    cnt = np.bincount(codes, minlength=L).astype(float)
    W = rng.multinomial(L, np.full(L, 1 / L), size=n_boot).astype(float)   # number of times each lot is resampled
    models = list(keys)
    res = {k: np.zeros((n_boot, len(models))) for k in ("gap_rmse_pct", "gap_mae_pct")}
    point = {k: np.zeros(len(models)) for k in res}
    for j, m in enumerate(models):
        acc = {k: [] for k in res}
        acc_pt = {k: [] for k in res}
        for i in keys[m]:
            sse, sae = {}, {}
            for reg in (SRC, REF):
                e = preds[(m, i, reg)] - yv
                sse[reg], sae[reg] = np.bincount(codes, e ** 2, L), np.bincount(codes, np.abs(e), L)
            n_b = W @ cnt
            acc["gap_rmse_pct"].append(gap_pct(np.sqrt(W @ sse[SRC] / n_b), np.sqrt(W @ sse[REF] / n_b)))
            acc["gap_mae_pct"].append(gap_pct(W @ sae[SRC] / n_b, W @ sae[REF] / n_b))
            acc_pt["gap_rmse_pct"].append(gap_pct(np.sqrt(sse[SRC].sum() / cnt.sum()), np.sqrt(sse[REF].sum() / cnt.sum())))
            acc_pt["gap_mae_pct"].append(gap_pct(sae[SRC].sum() / cnt.sum(), sae[REF].sum() / cnt.sum()))
        for k in res:
            res[k][:, j] = np.mean(acc[k], axis=0)
            point[k][j] = np.mean(acc_pt[k])
    return models, res, point, L


def run_boot(pf, y, lot, n_boot, out):
    rng = np.random.default_rng(BOOT_SEED)
    cells, doms, share = [], [], {k: [] for k in ("gap_rmse_pct", "gap_mae_pct")}
    pos_count = {k: [] for k in share}
    fam = pf.groupby("model").family.first()
    for name, g in pf.groupby("candidate", sort=False):
        t0 = time.time()
        idx, preds = domain_preds(g)
        keys = g.groupby("model").repeat.apply(lambda s: sorted(set(s))).to_dict()
        models, res, point, L = boot_domain(name, idx, preds, keys, y, lot, n_boot, rng)
        for k in res:
            lo, hi = np.percentile(res[k], [2.5, 97.5], axis=0)
            for j, m in enumerate(models):
                cells.append(dict(candidate=name, model=m, family=fam[m], metric=k, point=point[k][j], lo=lo[j], hi=hi[j],
                                  excl0=("pos" if lo[j] > 0 else "neg" if hi[j] < 0 else "none"), p_pos=(res[k][:, j] > 0).mean()))
            med = np.median(res[k], axis=1)
            doms.append(dict(candidate=name, metric=k, n_lots=L, n_rows=len(idx), median_point=np.median(point[k]),
                             median_lo=np.percentile(med, 2.5), median_hi=np.percentile(med, 97.5), p_median_pos=(med > 0).mean(),
                             share_pos_point=(point[k] > 0).mean(), share_pos_lo=np.percentile((res[k] > 0).mean(1), 2.5),
                             share_pos_hi=np.percentile((res[k] > 0).mean(1), 97.5)))
            pos_count[k].append((res[k] > 0).sum(1))
            share[k].append(len(models))
        log(f"[boot] {name:12s} lots {L} rows {len(idx)} models {len(models)} ({time.time() - t0:.0f}s)")
    cells, doms = pd.DataFrame(cells), pd.DataFrame(doms)
    cells.to_csv(os.path.join(out, "boot_cells.csv"), index=False)
    doms.to_csv(os.path.join(out, "boot_domains.csv"), index=False)
    summ = {}
    for k in share:   # independent resampling per domain; the draws with the same index b are combined into the distribution of the overall positive share
        tot = np.sum(pos_count[k], axis=0) / np.sum(share[k])
        c = cells[cells.metric == k]
        d = doms[doms.metric == k]
        summ[k] = dict(n_cells=len(c), share_pos_point=float((c.point > 0).mean()), share_pos_lo=float(np.percentile(tot, 2.5)),
                       share_pos_hi=float(np.percentile(tot, 97.5)), cells_ci_above0=int((c.excl0 == "pos").sum()),
                       cells_ci_below0=int((c.excl0 == "neg").sum()), domains_median_pos=int((d.median_point > 0).sum()),
                       domains_median_ci_above0=int((d.median_lo > 0).sum()), domains_median_ci_below0=int((d.median_hi < 0).sum()),
                       n_domains=len(d))
    json.dump(dict(n_boot=n_boot, seed=BOOT_SEED, unit="production lot within B_te", summary=summ),
              open(os.path.join(out, "boot_summary.json"), "w"), ensure_ascii=False, indent=1)
    return summ


def domain_split_rows(df, y, lot, name):
    slot, mA, mB = build_one(df, name)
    a = get_A(slot, df.index[mA], y, lot)
    bv = B_view(slot, df.index[mB], y, lot, "benchmark")
    return a, bv


def shared_boot(name, idx, preds, keys, y, lot, shared, n_boot, rng):
    """Lot-wise bootstrap of the shared minus not-shared difference. A lot belongs to only one of the two groups (sharing is a property of the lot),
    so lots are resampled separately inside the two groups; for every model the repetition mean of (RMSE loss on shared rows - RMSE loss on
    not-shared rows) is formed, and the interval of the model median is given."""
    yv = y.loc[idx].to_numpy()
    codes, lots = pd.factorize(lot.loc[idx].to_numpy())
    L = len(lots)
    lot_shared = np.zeros(L, bool)
    lot_shared[codes[shared]] = True
    assert not lot_shared[codes[~shared]].any(), "a lot occurs in both the shared and the not-shared group"
    W = np.zeros((n_boot, L))
    for g in (lot_shared, ~lot_shared):      # resample lots inside each group (group size fixed)
        k = int(g.sum())
        W[:, g] = rng.multinomial(k, np.full(k, 1 / k), size=n_boot)
    cnt = np.bincount(codes, minlength=L).astype(float)
    diffs, diffs_pt = [], []
    for m, reps in keys.items():
        d_b, d_pt = [], []
        for i in reps:
            sse = {reg: np.bincount(codes, (preds[(m, i, reg)] - yv) ** 2, L) for reg in (SRC, REF)}
            r_b, r_pt = {}, {}
            for part, g in (("s", lot_shared), ("n", ~lot_shared)):
                Wg, n_b = W * g, (W * g) @ cnt
                r_b[part] = gap_pct(np.sqrt(Wg @ sse[SRC] / n_b), np.sqrt(Wg @ sse[REF] / n_b))
                r_pt[part] = gap_pct(np.sqrt(sse[SRC][g].sum() / cnt[g].sum()), np.sqrt(sse[REF][g].sum() / cnt[g].sum()))
            d_b.append(r_b["s"] - r_b["n"])
            d_pt.append(r_pt["s"] - r_pt["n"])
        diffs.append(np.mean(d_b, axis=0))
        diffs_pt.append(np.mean(d_pt))
    diffs = np.array(diffs)                  # (models, n_boot)
    med = np.median(diffs, axis=0)
    lo_m, hi_m = np.percentile(diffs, [2.5, 97.5], axis=1)
    return dict(candidate=name, n_shared_lots=int(lot_shared.sum()), n_other_lots=int((~lot_shared).sum()),
                diff_median_point=float(np.median(diffs_pt)), diff_median_lo=float(np.percentile(med, 2.5)),
                diff_median_hi=float(np.percentile(med, 97.5)), p_diff_median_pos=float((med > 0).mean()),
                models_diff_pos=int((np.array(diffs_pt) > 0).sum()), models_ci_above0=int((lo_m > 0).sum()),
                models_ci_below0=int((hi_m < 0).sum()), n_models=len(keys))


def run_shared(pf, df, y, lot, out, n_boot=2000):
    rng = np.random.default_rng(BOOT_SEED + 1)
    sboot = []
    rows = []
    fam = pf.groupby("model").family.first()
    for name, g in pf.groupby("candidate", sort=False):
        idx, preds = domain_preds(g)
        a, _ = domain_split_rows(df, y, lot, name)
        shared = lot.loc[idx].isin(set(lot.loc[a["A_dev"]])).to_numpy()
        keys = g.groupby("model").repeat.apply(lambda s: sorted(set(s))).to_dict()
        yv = y.loc[idx].to_numpy()
        for part, m in (("all", np.ones(len(idx), bool)), ("shared", shared), ("not_shared", ~shared)):
            if m.sum() == 0:
                continue
            t = cell_table(yv[m], {k: v[m] for k, v in preds.items()}, keys)
            t["candidate"], t["part"], t["family"] = name, part, t.model.map(fam)
            t["share_rows"] = m.mean()
            t["n_lots"] = lot.loc[idx[m]].nunique()
            rows.append(t)
        log(f"[shared] {name:12s} share of B_te rows whose lot also occurs in A_dev {shared.mean():.1%} ({shared.sum()} rows)")
        if min(shared.sum(), (~shared).sum()) >= MIN_SHARED_ROWS:
            sboot.append(shared_boot(name, idx, preds, keys, y, lot, shared, n_boot, rng))
    t = pd.concat(rows)
    t.to_csv(os.path.join(out, "shared_cells.csv"), index=False)
    if sboot:
        sb = pd.DataFrame(sboot)
        sb.to_csv(os.path.join(out, "shared_boot.csv"), index=False)
        print(sb.round(2).to_string(index=False))
    s = t.groupby(["candidate", "part"]).agg(share_rows=("share_rows", "first"), n_rows=("n_rows", "first"), n_lots=("n_lots", "first"),
                                             median_d_rmse=("gap_rmse_pct", "median"), pos_share_rmse=("gap_rmse_pct", lambda v: (v > 0).mean()))
    f = t.groupby(["candidate", "part", "family"]).gap_rmse_pct.median().unstack("family")
    s.join(f).to_csv(os.path.join(out, "shared_summary.csv"))
    return s.join(f)


def run_timed(pf, df, y, lot, out, n_bins=4):
    rows = []
    fam = pf.groupby("model").family.first()
    for name, g in pf.groupby("candidate", sort=False):
        kind, _ = parse_candidate(name)
        if kind not in ("x_time", "y_time"):
            continue
        idx, preds = domain_preds(g)
        _, bv = domain_split_rows(df, y, lot, name)
        t = pd.to_datetime(df[kind])
        tr = np.sort(t.loc[bv["B_tr"]].to_numpy().astype("datetime64[s]").astype(np.int64))
        te = t.loc[idx].to_numpy().astype("datetime64[s]").astype(np.int64)
        pos = np.searchsorted(tr, te).clip(1, len(tr) - 1)
        dist_h = np.minimum(np.abs(te - tr[pos - 1]), np.abs(tr[pos] - te)) / 3600
        q = pd.qcut(dist_h, n_bins, labels=False, duplicates="drop")
        keys = g.groupby("model").repeat.apply(lambda s: sorted(set(s))).to_dict()
        yv = y.loc[idx].to_numpy()
        for b in sorted(set(q)):
            m = q == b
            c = cell_table(yv[m], {k: v[m] for k, v in preds.items()}, keys)
            c["candidate"], c["bin"], c["family"] = name, int(b), c.model.map(fam)
            c["dist_h_min"], c["dist_h_max"], c["dist_h_median"] = dist_h[m].min(), dist_h[m].max(), np.median(dist_h[m])
            rows.append(c)
        log(f"[timed] {name[:12]:12s} median time distance to the nearest B_tr row {np.median(dist_h):.2f} h, maximum {dist_h.max():.1f} h")
    if not rows:
        return None
    t = pd.concat(rows)
    t.to_csv(os.path.join(out, "timed_cells.csv"), index=False)
    s = t.groupby(["candidate", "bin"]).agg(dist_h_min=("dist_h_min", "first"), dist_h_max=("dist_h_max", "first"), n_rows=("n_rows", "first"),
                                            median_d_rmse=("gap_rmse_pct", "median"), pos_share_rmse=("gap_rmse_pct", lambda v: (v > 0).mean()),
                                            median_d_mae=("gap_mae_pct", "median"))
    s.to_csv(os.path.join(out, "timed_summary.csv"))
    return s


def run_screen(screening, out):
    r = pd.read_csv(os.path.join(config.RESULTS_DIR, "screening", "runs", screening, "rows.csv"))
    r = r[(r.status == "ok") & r.regime.isin([SRC, REF]) & r.candidate.str.startswith("cat_")]
    p = r.pivot_table(index=["candidate", "model", "repeat"], columns="regime", values="rmse_screen")
    p["d"] = gap_pct(p[SRC], p[REF])
    cons = p.d.groupby(["candidate", "repeat"]).median().unstack("repeat")   # consensus per repetition (median over the 4 models)
    cons = cons.dropna()
    pairs = [(i, j) for i in cons.columns for j in cons.columns if i < j]
    rho = {f"{i}-{j}": float(cons[i].corr(cons[j], method="spearman")) for i, j in pairs}
    r_bar = float(np.mean(list(rho.values())))
    k = cons.shape[1]
    res = dict(n_candidates=len(cons), repeats=k, pairwise_spearman=rho, mean_pairwise=r_bar,
               spearman_brown_mean_of_k=k * r_bar / (1 + (k - 1) * r_bar))
    per_model = {}
    for m, g in p.d.groupby("model"):
        u = g.unstack("repeat").dropna()
        per_model[m] = float(np.mean([u[i].corr(u[j], method="spearman") for i, j in pairs]))
    res["per_model_mean_pairwise"] = per_model
    json.dump(res, open(os.path.join(out, "screen_reliability.json"), "w"), ensure_ascii=False, indent=1)
    return res


def run_absdiff(pf, y, out):
    w = pf.pivot_table(index=["candidate", "model", "repeat"], columns="regime", values="rmse_ood")
    w["abs"] = w[SRC] - w[REF]
    c = w.groupby(["candidate", "model"])[["abs", SRC, REF]].mean().reset_index()
    sd = {}
    for name, g in pf.groupby("candidate"):
        z = np.load(g.pred_path.iloc[0])
        sd[name] = float(y.loc[z["ood_idx"]].std())
    c["sd_y_te"] = c.candidate.map(sd)
    c["abs_over_sd"] = c["abs"] / c.sd_y_te
    c.to_csv(os.path.join(out, "absdiff_cells.csv"), index=False)
    s = c.groupby("candidate").agg(sd_y_te=("sd_y_te", "first"), rmse_ref_median=(REF, "median"), abs_median=("abs", "median"),
                                   abs_q25=("abs", lambda v: v.quantile(.25)), abs_q75=("abs", lambda v: v.quantile(.75)),
                                   abs_over_sd_median=("abs_over_sd", "median"))
    s.to_csv(os.path.join(out, "absdiff_summary.csv"))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--screening-run-id", default="screening_main")
    ap.add_argument("--parts", default="boot,shared,timed,screen,absdiff")
    ap.add_argument("--only", default=None, help="comma-separated domains (for smoke runs)")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    parts = a.parts.split(",")
    only = a.only.split(",") if a.only else None
    json.dump(dict(vars(a), python=sys.executable, created=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = load_fits(a.benchmark_run_id, only)
    log(f"[analysis] run {a.run_id} domains {pf.candidate.nunique()} models {pf.model.nunique()} fits {len(pf)} parts {parts}")
    if "boot" in parts:
        s = run_boot(pf, y, lot, a.n_boot, out)
        for k, v in s.items():
            log(f"[boot] {k}: positive cells {v['share_pos_point']:.3f} [{v['share_pos_lo']:.3f}, {v['share_pos_hi']:.3f}], "
                f"interval above 0: {v['cells_ci_above0']} / below 0: {v['cells_ci_below0']} / {v['n_cells']} cells, "
                f"domain median positive {v['domains_median_pos']}/{v['n_domains']} (interval above 0: {v['domains_median_ci_above0']})")
    if "shared" in parts:
        print(run_shared(pf, df, y, lot, out, a.n_boot).round(2).to_string())
    if "timed" in parts:
        s = run_timed(pf, df, y, lot, out)
        if s is not None:
            print(s.round(2).to_string())
    if "screen" in parts:
        print(json.dumps(run_screen(a.screening_run_id, out), ensure_ascii=False, indent=1))
    if "absdiff" in parts:
        print(run_absdiff(pf, y, out).round(4).to_string())
    log("[analysis] done")


if __name__ == "__main__":
    main()
