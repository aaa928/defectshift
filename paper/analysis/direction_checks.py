"""Two checks of the direction comparison of RQ1 that the paper reports in Results (the 7 challenge domains, all models). It reads only saved predictions (no new training, CPU only).

(1) Direction verdict under block resampling of the test lots.
    scale_normalized_bootstrap.py resamples the target test lots independently with replacement and classifies each domain value as agree (A), disagree (D) or unresolved (U)
    from the 95% intervals of the target-reference gap and of the ID-OOD gap before (io) and after (sn) outcome-scale normalization. This script repeats the classification
    with the resampling of lot_block_bootstrap.py: the lot-disjoint target test lots are laid out in the production order of their lot rank (axis = y_time for the y_time domain, x_time
    otherwise) and resampled in blocks of 2, 5 and 10 lots, circular (Politis-Romano) and non-circular moving (Kuensch). The three quantities and the classification are computed from the
    SAME resamples; the source test set A_heldout is held fixed (the same as scheme "target" of scale_normalized_bootstrap.py). The weights and seeds are those of lot_block_bootstrap.py.
    Checks (paper configuration: all models, 2,000 draws, all domains; the run stops on a difference, on a missing reference file and on an incomplete comparison; reference_checks.csv records what was checked):
      Err_S    recomputed from the source test predictions equals the stored value (always);
      iid      the point values (tr, io, sn, s), the interval ends and the classes equal scheme "target" of sn_domain.csv of --sn-run-id;
      block, moving   the g_TR interval ends equal block_boot.csv of --block-run-id (RMSE and MAE). There is no reference for the intervals of g_IO and g_SN under block resampling;
                      they come from the same sn.stats function as the iid ones.
    All prediction files of the selected models must exist (they are written by the benchmark run; the data release has no predictions). With --models the domain values differ
    from those of the paper, which uses all 22 models. A run with --models, --only or another --n-boot is a smoke run: the reference checks are skipped and reference_checks.csv says so;
    --skip-reference-checks does the same for the paper configuration.
(2) Share of the outcome-scale term in the ID-OOD gap.
    log(1 + g_IO) = log s + log q + log(1 + g_TR) for each fit, where s is the outcome-scale ratio. The outcome-scale term is larger in magnitude than the rest when
    |log s| > |log(1 + g_IO) - log s|, i.e. when D = |log s| - |log(1 + g_IO) - log s| > 0. (The paper writes this statistic as Delta; the output columns keep the name D.) D compares two magnitudes: when the two terms have opposite signs they cancel, so the term can
    be larger in magnitude although g_IO is small. Report the result as "larger", not as "determines". D is computed for every fit (model x repetition, point estimates on the lot-disjoint target rows) and summarized per domain and measure:
    the median of D over the fits, the share of fits with D > 0, and, with the aggregation of the domain tables of the paper (repetition mean per model, then median over the models), D of the
    domain value. The number of domains in which the term is the larger one is counted for each of the three definitions.
Output: results/analysis/<run-id>/{block_directions.csv, scale_dominance_fits.csv, scale_dominance_domain.csv, scale_dominance_summary.csv, reference_checks.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/direction_checks.py --run-id direction_checks
Smoke run: add --only cat_050,y_time --models Ridge,XGBoost,kNN --n-boot 200 (the reference checks are skipped).
"""
import argparse
import os
import zlib

import numpy as np
import pandas as pd

import common as cm
import repetition_uncertainty as unc
import scale_normalized_bootstrap as sn
from lot_block_bootstrap import BLOCKS, BLOCK_SEED, block_weights
from lot_order_proximity import lot_ranks
from defectshift import config
from defectshift.candidates import build_one, load_data
from defectshift.utils import log, read_per_fit

TOL = 1e-9


def paper_configuration(models_arg, n_boot, only):
    """The configuration of the paper: all models, 2,000 draws, all domains. Only a run in this configuration requires the reference checks."""
    return not models_arg and n_boot == 2000 and not only


def load_references(rf, sn_run_id, block_run_id, required):
    """The saved outputs that the new values are compared with ((None, None) when no check is wanted). When the checks are required, a missing file stops the run."""
    if not required:
        return None, None
    p_sn, p_blk = os.path.join(rf, sn_run_id, "sn_domain.csv"), os.path.join(rf, block_run_id, "block_boot.csv")
    missing = [p for p in (p_sn, p_blk) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"reference files missing: {missing}. The paper configuration compares its values with them (run scale_normalized_bootstrap.py and "
                         f"lot_block_bootstrap.py first, see docs/reproducing.md); --skip-reference-checks runs without the comparison and records that.")
    sn_ref = pd.read_csv(p_sn)
    sn_ref = sn_ref[(sn_ref.group == "challenge") & (sn_ref.scheme == "target")].set_index(["dom", "metric"])
    return sn_ref, pd.read_csv(p_blk).set_index(["dom", "scheme", "block"])


def check_coverage(n_iid, n_block, n_block_nokey, n_domains):
    """In the paper configuration every domain, measure and scheme must have been compared with its reference."""
    want_iid, want_block = len(sn.METS) * n_domains, len(sn.METS) * 2 * len(BLOCKS) * n_domains
    if n_block_nokey or n_iid != want_iid or n_block != want_block:
        raise SystemExit(f"reference checks incomplete: iid {n_iid} of {want_iid}, block g_TR {n_block} of {want_block}, block keys without reference {n_block_nokey}")


def interval(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def scheme_weights(order, LT, dom, n_boot):
    """[(scheme, block, weights (n_boot, LT))] with the seeds of the existing analyses: iid as scale_normalized_bootstrap.py (target resampling), block and moving as lot_block_bootstrap.py."""
    crc = zlib.crc32(dom.encode())
    rng = np.random.default_rng([unc.BOOT_SEED, crc])
    out = [("iid", 1, rng.multinomial(LT, np.full(LT, 1.0 / LT), size=n_boot).astype(float))]
    for b in BLOCKS:
        out.append(("block", b, block_weights(order, b, n_boot, np.random.default_rng([BLOCK_SEED, crc, b]))))
    for b in BLOCKS:
        out.append(("moving", b, block_weights(order, b, n_boot, np.random.default_rng([BLOCK_SEED, crc, b, 1]), circular=False)))
    return out


def fit_level_dominance(dom, yT, yS, preds, spreds, models, n_rep):
    """Rows (one per measure, model, repetition) with D = |log s| - |log(1 + g_IO) - log s| from the point estimates on the lot-disjoint target rows and the full source test set."""
    rows = []
    for met in sn.METS:
        s = float(sn.wspread(yT, np.ones((1, len(yT))), met)[0] / sn.wspread(yS, np.ones((1, len(yS))), met)[0])
        for m in models:
            for i in range(n_rep):
                eT_r, eS_r = preds[m][i][0] - yT, spreds[m][i] - yS
                eT = np.sqrt(np.mean(eT_r ** 2)) if met == "rmse" else np.mean(np.abs(eT_r))
                eS = np.sqrt(np.mean(eS_r ** 2)) if met == "rmse" else np.mean(np.abs(eS_r))
                d = abs(np.log(s)) - abs(np.log(eT / eS) - np.log(s))
                rows.append(dict(dom=dom, metric=met, model=m, repeat=i, s=s, g_io_pct=100 * (eT / eS - 1), D=float(d)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--sn-run-id", default="scale_normalized_bootstrap")
    ap.add_argument("--block-run-id", default="lot_block_bootstrap")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--only", default="", help="comma-separated short domain names (smoke runs)")
    ap.add_argument("--models", default="", help="comma-separated models (smoke runs; default: all)")
    ap.add_argument("--skip-reference-checks", action="store_true", help="run the paper configuration without the comparison with the saved references (recorded in reference_checks.csv)")
    a = ap.parse_args()
    rf = os.path.join(config.RESULTS_DIR, "analysis")
    df, y = load_data()
    lot = df[config.LOT_COL]
    rank_by_axis = {ax: lot_ranks(df, lot, ax) for ax in ("x_time", "y_time")}
    pf = cm.load_challenge(a.benchmark_run_id)
    models = sorted(pf.model.unique())
    paper_config = paper_configuration(a.models, a.n_boot, a.only)
    check_refs = paper_config and not a.skip_reference_checks
    if a.models:
        unknown = sorted(set(a.models.split(",")) - set(models))
        if unknown:
            raise SystemExit(f"unknown models {unknown}; the models of run {a.benchmark_run_id} are {models}")
        models = [m for m in models if m in a.models.split(",")]
    domains = [cm.short(n) for n in pf.candidate.unique()]
    if a.only:
        unknown = sorted(set(a.only.split(",")) - set(domains))
        if unknown:
            raise SystemExit(f"unknown domains {unknown}; the domains of run {a.benchmark_run_id} are {domains}")
    absent = sorted({p for p in pf[pf.model.isin(models)].pred_path if isinstance(p, str) and not os.path.exists(p)})
    if absent:
        raise SystemExit(f"{len(absent)} prediction files of the selected models are missing (first: {absent[0]}). Predictions are written by the benchmark run "
                         f"and are not part of the data release (docs/reproducing.md).")
    pfa = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    src_ch = pfa[pfa.regime == cm.SRC].set_index(["candidate", "model", "repeat"])
    sn_ref, block_ref = load_references(rf, a.sn_run_id, a.block_run_id, check_refs)
    if not paper_config:
        log("[direction_checks] --models, --only or --n-boot differs from the paper configuration: reference checks skipped")
    elif a.skip_reference_checks:
        log("[direction_checks] reference checks skipped by --skip-reference-checks")
    else:
        log("[direction_checks] references sn_domain.csv and block_boot.csv found")
    n_rep = config.N_REPEATS_BENCHMARK
    rows, fit_rows = [], []
    n_iid = n_block = n_block_nokey = 0
    for name, g in pf.groupby("candidate", sort=False):
        dom = cm.short(name)
        if a.only and dom not in a.only.split(","):
            continue
        d = cm.load_domain("challenge", name, g, df, y, lot, models)
        slot = build_one(df, name)[0]
        spreds, sidx = sn.load_src("challenge", name, g, models, None, slot, n_rep)
        if not np.array_equal(sidx, np.asarray(d["a"]["A_heldout"])):
            raise SystemExit(f"{dom}: id_idx differs from A_heldout")
        mk = ~np.isin(d["lots"], lot.loc[d["a"]["A_dev"]].unique())
        yT = d["yv"][mk]
        preds = {m: [(ps[mk], pr[mk]) for ps, pr in pairs] for m, pairs in d["preds"].items()}
        ulT, codesT = np.unique(d["lots"][mk], return_inverse=True)
        yS = y.loc[sidx].to_numpy()
        ulS, codesS = np.unique(lot.loc[sidx].to_numpy(), return_inverse=True)
        LT, LS = len(ulT), len(ulS)
        for m in models:                                      # the stored Err_S must equal the value recomputed from id_pred
            for i in range(n_rep):
                e = spreds[m][i] - yS
                for met, v in (("rmse", np.sqrt(np.mean(e ** 2))), ("mae", np.mean(np.abs(e)))):
                    rec = float(src_ch.loc[(name, m, i), f"{met}_id"])
                    if abs(v - rec) > 1e-9 * max(1.0, abs(rec)):
                        raise SystemExit(f"{dom} {m} r{i} {met}: Err_S recomputed from id_pred {v} != recorded {rec}")
        point = sn.stats(yT, codesT, LT, preds, yS, codesS, LS, spreds, models, np.ones((1, LT)), None)
        axis = "y_time" if dom == "y_time" else "x_time"
        order = np.argsort(rank_by_axis[axis].loc[ulT].to_numpy(), kind="stable")
        iid_class = {}
        for scheme, b, W in scheme_weights(order, LT, dom, a.n_boot):
            bs = sn.stats(yT, codesT, LT, preds, yS, codesS, LS, spreds, models, W, None)
            for met in sn.METS:
                tr_p, io_p, sn_p, s_p = point[met]
                tr_b, io_b, sn_b, s_b = bs[met]
                dm = {k: np.median(v, axis=0) for k, v in (("tr", tr_b), ("io", io_b), ("sn", sn_b))}
                row = dict(dom=dom, metric=met, scheme=scheme, block=b, axis=axis, n_lots_T=LT, n_rows_T=int(mk.sum()), n_models=len(models), s=float(s_p[0]))
                for k, pv in (("tr", tr_p), ("io", io_p), ("sn", sn_p)):
                    row[k] = float(np.median(pv))
                    row[f"{k}_lo"], row[f"{k}_hi"] = interval(dm[k])
                for k in ("io", "sn"):
                    row[f"class_{k}"] = str(sn.classify(row["tr_lo"], row["tr_hi"], row[f"{k}_lo"], row[f"{k}_hi"], row["tr"], row[k]))
                if scheme == "iid":
                    iid_class[met] = (row["class_io"], row["class_sn"])
                    if sn_ref is not None:
                        o = sn_ref.loc[(dom, met)]
                        for k in ("tr", "io", "sn"):
                            if abs(row[k] - o[k]) > TOL:
                                raise SystemExit(f"{dom} {met}: the point value of {k} differs from sn_domain.csv")
                            if abs(row[f"{k}_lo"] - o[f"{k}_lo"]) > TOL or abs(row[f"{k}_hi"] - o[f"{k}_hi"]) > TOL:
                                raise SystemExit(f"{dom} {met}: the iid interval of {k} differs from sn_domain.csv")
                        if abs(row["s"] - o["s"]) > TOL:
                            raise SystemExit(f"{dom} {met}: s differs from sn_domain.csv")
                        if (row["class_io"], row["class_sn"]) != (o.class_io, o.class_sn):
                            raise SystemExit(f"{dom} {met}: the iid classification differs from sn_domain.csv")
                        n_iid += 1
                elif block_ref is not None:
                    if (dom, scheme, b) not in block_ref.index:
                        n_block_nokey += 1
                    else:
                        o = block_ref.loc[(dom, scheme, b)]
                        lo, hi = interval(dm["tr"])
                        if abs(lo - o[f"{met}_lo"]) > TOL or abs(hi - o[f"{met}_hi"]) > TOL:
                            raise SystemExit(f"{dom} {met} {scheme} {b}: the g_TR interval differs from block_boot.csv")
                        n_block += 1
                row["io_changed_vs_iid"] = row["class_io"] != iid_class[met][0]
                row["sn_changed_vs_iid"] = row["class_sn"] != iid_class[met][1]
                rows.append(row)
        fit_rows += fit_level_dominance(dom, yT, yS, preds, spreds, models, n_rep)
        log(f"[direction_checks] {dom} done (target lots {LT}; Err_S checked; values equal to the references so far: iid {n_iid}, block g_TR {n_block}"
            + (f"; block keys without reference {n_block_nokey}" if n_block_nokey else "") + ")")
    if check_refs:
        check_coverage(n_iid, n_block, n_block_nokey, len(domains))
    BD = pd.DataFrame(rows)
    F = pd.DataFrame(fit_rows)
    # domain summaries of the outcome-scale term, three definitions
    dom_rows = []
    for (dom, met), x in F.groupby(["dom", "metric"], sort=False):
        cell = x.groupby("model").agg(g=("g_io_pct", "mean"), s=("s", "first"))          # repetition mean per model, then the median over the models (the domain tables of the paper)
        g_dom = float(np.median(cell.g))
        s = float(x.s.iloc[0])
        D_dom = abs(np.log(s)) - abs(np.log(1 + g_dom / 100) - np.log(s))
        dom_rows.append(dict(dom=dom, metric=met, s=s, n_fits=len(x), D_fit_median=float(x.D.median()), share_fits_scale_term_larger=float((x.D > 0).mean()),
                             g_io_domain_pct=g_dom, D_domain=float(D_dom)))
    DD = pd.DataFrame(dom_rows)
    DD["scale_term_larger_fit_median"] = DD.D_fit_median > 0
    DD["scale_term_larger_domain"] = DD.D_domain > 0
    SUM = DD.groupby("metric", sort=False).agg(n_domains=("dom", "size"), n_larger_fit_median=("scale_term_larger_fit_median", "sum"),
                                             n_larger_domain=("scale_term_larger_domain", "sum")).reset_index()
    out = os.path.join(rf, a.run_id)
    os.makedirs(out, exist_ok=True)
    BD.to_csv(os.path.join(out, "block_directions.csv"), index=False)
    F.to_csv(os.path.join(out, "scale_dominance_fits.csv"), index=False)
    DD.to_csv(os.path.join(out, "scale_dominance_domain.csv"), index=False)
    SUM.to_csv(os.path.join(out, "scale_dominance_summary.csv"), index=False)
    pd.DataFrame([dict(paper_configuration=paper_config, reference_checks=check_refs, skipped_by_request=bool(a.skip_reference_checks and paper_config), only=a.only,
                       n_models=len(models), n_boot=a.n_boot, n_iid_compared=n_iid, n_block_compared=n_block, n_block_keys_without_reference=n_block_nokey)]
                 ).to_csv(os.path.join(out, "reference_checks.csv"), index=False)
    pd.set_option("display.width", 250)
    print(BD[["dom", "metric", "scheme", "block", "tr_lo", "tr_hi", "io_lo", "io_hi", "sn_lo", "sn_hi", "class_io", "class_sn", "io_changed_vs_iid", "sn_changed_vs_iid"]].round(1).to_string(index=False))
    print(DD.round(3).to_string(index=False))
    print(SUM.to_string(index=False))
    log("[direction_checks] done")


if __name__ == "__main__":
    main()
