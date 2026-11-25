"""Uncertainty of the challenge domains: variation across repetitions and the B_te lot-resampling interval of the target-reference gap on the same test set and of R^2.
No new training: only the saved predictions of ID_match (source trained on n*) and OOD_match (target trained on n*) are read. The controls are not used.

The unit of computation is the same as cell/boot_exceed in verdict_robustness.py: per model and repetition, the target-reference gap = 100*(source error/reference error - 1),
domain value = median over models (repetition mean per model). The model set is all 22 models in per_fit of the challenge domains (no restriction by the controls).

(a) Per repetition: in every repetition i the model median of the gap is formed, and the mean, standard deviation, minimum and maximum over the 5 repetitions are given, together with the share of
    (model, repetition) cells with gap > 0. R^2 (source, reference) is also formed per repetition, and the range of the number of models with R^2 > 0 (per repetition) is given.
(b) Lot resampling: the lots of B_te are drawn with replacement (the same resampling for source and reference, fixed seed) and the domain gap (RMSE, MAE), the domain median R^2 (source, reference)
    and the number of models with R^2 > 0 are recomputed. 2.5/97.5 percentile interval. The repetitions are not treated as independent samples and the interval reflects the lot resampling only.

Output: results/analysis/<run-id>/{unc_perrepeat.csv, unc_boot.csv, unc_summary.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/repetition_uncertainty.py --run-id unc_smoke --only cat_050 --n-boot 50
"""
import argparse
import os

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from defectshift import config  # noqa: E402
from defectshift.utils import read_per_fit
from defectshift.candidates import load_data  # noqa: E402
from defectshift.utils import log  # noqa: E402
import verdict_robustness as rb  # noqa: E402  (load_domain is reused)

BOOT_SEED = 20261102


def r2(yv, p):
    return 1.0 - np.sum((yv - p) ** 2) / np.sum((yv - yv.mean()) ** 2)


def per_repeat(yv, preds, models):
    """Rows: model x repetition. rmse/mae are the target-reference gap (%), r2_src/r2_ref are on all B_te rows."""
    rows = []
    for m in models:
        for j, (ps, pr) in enumerate(preds[m]):
            es, er = ps - yv, pr - yv
            rows.append(dict(model=m, rep=j,
                             rmse=100 * (np.sqrt(np.mean(es ** 2)) / np.sqrt(np.mean(er ** 2)) - 1),
                             mae=100 * (np.mean(np.abs(es)) / np.mean(np.abs(er)) - 1),
                             r2_src=r2(yv, ps), r2_ref=r2(yv, pr)))
    return pd.DataFrame(rows)


def boot(yv, lots, preds, models, n_boot, rng):
    """n_boot lot resamplings. Returns a dict in the form of a DataFrame (statistic x n_boot). The cell definition is the same as in per_repeat."""
    ul, codes = np.unique(lots, return_inverse=True)
    L = len(ul)
    cnt = np.bincount(codes, minlength=L).astype(float)
    W = rng.multinomial(L, np.full(L, 1.0 / L), size=n_boot).astype(float)   # (n_boot, L)
    n = W @ cnt
    sy = W @ np.bincount(codes, yv, minlength=L)
    syy = W @ np.bincount(codes, yv ** 2, minlength=L)
    sst = syy - sy ** 2 / n                                                    # SST of the resampled test set
    rm, ma, r2s, r2r = [], [], [], []
    for m in models:
        a_rm, a_ma, a_rs, a_rr = [], [], [], []
        for ps, pr in preds[m]:
            es, er = ps - yv, pr - yv
            s = [W @ np.bincount(codes, w, minlength=L) for w in (es ** 2, er ** 2, np.abs(es), np.abs(er))]
            a_rm.append(100 * (np.sqrt(s[0] / n) / np.sqrt(s[1] / n) - 1))
            a_ma.append(100 * ((s[2] / n) / (s[3] / n) - 1))
            a_rs.append(1 - s[0] / sst)
            a_rr.append(1 - s[1] / sst)
        rm.append(np.mean(a_rm, axis=0)); ma.append(np.mean(a_ma, axis=0))
        r2s.append(np.mean(a_rs, axis=0)); r2r.append(np.mean(a_rr, axis=0))
    rm, ma, r2s, r2r = (np.stack(v) for v in (rm, ma, r2s, r2r))              # (models, n_boot)
    return dict(d_rmse=np.median(rm, axis=0), d_mae=np.median(ma, axis=0),
                r2_src_med=np.median(r2s, axis=0), r2_ref_med=np.median(r2r, axis=0),
                n_src_pos=(r2s > 0).sum(axis=0), n_ref_pos=(r2r > 0).sum(axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--only", default=None, help="first part of the challenge-domain names (comma-separated)")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    pf = pf[pf.regime.isin([rb.SRC, rb.REF])]
    if a.only:
        pf = pf[pf.candidate.str.startswith(tuple(a.only.split(",")))]
    if pf.empty:
        raise SystemExit(f"no challenge domain: check --only({a.only})")
    rng = np.random.default_rng(BOOT_SEED)
    rep_all, boot_all, summ = [], [], []
    for name, g in pf.groupby("candidate", sort=False):
        models = sorted(g.model.unique())
        n = g.groupby("model").repeat.nunique()
        if not (n == config.N_REPEATS_BENCHMARK).all():
            raise SystemExit(f"{name}: the number of repetitions differs from config.N_REPEATS_BENCHMARK({config.N_REPEATS_BENCHMARK}) {n.to_dict()}")
        yv, lots, preds, _ = rb.load_domain("challenge", name, g, df, y, lot, models, None)
        pr = per_repeat(yv, preds, models)
        pr.insert(0, "candidate", name)
        rep_all.append(pr)
        bt = boot(yv, lots, preds, models, a.n_boot, rng)
        bd = pd.DataFrame(bt); bd.insert(0, "candidate", name)
        boot_all.append(bd)
        # domain value per repetition: median over the models in every repetition
        byrep = pr.groupby("rep")[["rmse", "mae"]].median()
        pos = pr.groupby("rep").apply(lambda d: pd.Series(dict(src_pos=int((d.r2_src > 0).sum()), ref_pos=int((d.r2_ref > 0).sum()))))
        # domain value with the repetition mean per model (the same definition as robust.cell)
        mm = pr.groupby("model")[["rmse", "mae", "r2_src", "r2_ref"]].mean()
        row = dict(candidate=name, n_models=len(models), n_rows=len(yv), n_lots=int(len(np.unique(lots))), n_boot=a.n_boot, boot_seed=BOOT_SEED,
                   src_regime=rb.SRC, ref_regime=rb.REF,
                   d_rmse_dom=float(mm.rmse.median()), d_mae_dom=float(mm.mae.median()),
                   d_rmse_rep_sd=float(byrep.rmse.std(ddof=1)), d_rmse_rep_min=float(byrep.rmse.min()), d_rmse_rep_max=float(byrep.rmse.max()),
                   d_mae_rep_sd=float(byrep.mae.std(ddof=1)), d_mae_rep_min=float(byrep.mae.min()), d_mae_rep_max=float(byrep.mae.max()),
                   share_cells_pos_rmse=float((pr.rmse > 0).mean()), share_cells_pos_mae=float((pr.mae > 0).mean()),
                   src_pos_rep_min=int(pos.src_pos.min()), src_pos_rep_max=int(pos.src_pos.max()),
                   ref_pos_rep_min=int(pos.ref_pos.min()), ref_pos_rep_max=int(pos.ref_pos.max()),
                   r2_src_med=float(mm.r2_src.median()), r2_ref_med=float(mm.r2_ref.median()))
        for k in ("d_rmse", "d_mae", "r2_src_med", "r2_ref_med", "n_src_pos", "n_ref_pos"):
            lo, hi = np.percentile(bt[k], [2.5, 97.5])
            row[f"{k}_lo"], row[f"{k}_hi"] = float(lo), float(hi)
        summ.append(row)
        log(f"[unc] {name[:16]:16s} rows {len(yv)} lots {row['n_lots']} | target-reference gap RMSE {row['d_rmse_dom']:+.1f} "
            f"[{row['d_rmse_lo']:+.1f}, {row['d_rmse_hi']:+.1f}] MAE {row['d_mae_dom']:+.1f} [{row['d_mae_lo']:+.1f}, {row['d_mae_hi']:+.1f}]")
    pd.concat(rep_all).to_csv(os.path.join(out, "unc_perrepeat.csv"), index=False)
    pd.concat(boot_all).to_csv(os.path.join(out, "unc_boot.csv"), index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(out, "unc_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(s.T.round(3).to_string())
    log("[unc] done")


if __name__ == "__main__":
    main()
