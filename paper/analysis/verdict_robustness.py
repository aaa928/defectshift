"""Robustness of the verdict "the domain lies beyond the range of the controls": how sensitive is it to the choice of evaluation rows and to the B_te lot sample?
Only saved predictions are read (no retraining, CPU only). By default (`--models auto`), only the
**models for which all control domains, repetitions and regimes are finished** are used, taken from the control runs (--ctrl-run-ids; default: the five runs of the ten original controls, `common.CTRL_RUNS`); unfinished models are left out and
their names are logged. `--models` also takes a comma-separated list (for example the eight models M8). The same models are used for the challenge domains.

Variants (the same rule is applied to the challenge domains and the controls, and the maximum over the controls is recomputed for every variant)
  all         all B_te rows (the current verdict)
  rest99      rows without the 1 % tail of the outcome. Tail = the 1 % largest |y - median(y of B_tr)| (the same rounding-up rule as tail_concentration.py)
  not_shared  rows whose lot is not in A_dev (the source training and early-stopping pool; the same definition of shared as in test_set_checks.py). For the controls, all rows by definition
  half1/half2 two halves of B_te, formed by shuffling the lots with a fixed seed and splitting in half (the controls are halved in the same way so that the evaluation sample size matches)
`--base-disjoint`: the evaluation population is unified to the lot-disjoint rows. rest99, half1 and half2 are intersected with the not_shared rows
(the controls have no shared rows and stay as they are), all (all rows) is kept as a complementary analysis, and the lot resampling is also done on the not_shared rows. Without the option the behavior is the same as before.
Cell = model x domain, value = repetition mean of (source error / reference error - 1) x 100 per repetition. Domain value = median over the model set.
Verdict = domain value > maximum of the domain values of the 10 control domains of the same variant. It is a comparison with the observed range, not a test.
In addition, for the variant all, the share of B_te lot resamplings (n-boot draws, the same resampling for source and reference) in which the domain value exceeds the
maximum over the controls (held fixed) is reported. The control maximum is not resampled, so this share reflects only the variation of the test sample on the challenge-domain side.
Output: results/analysis/<run-id>/{robust_cells.csv, robust_table.csv, robust_boot.csv}
Usage: CUDA_VISIBLE_DEVICES="" python paper/analysis/verdict_robustness.py --run-id robust_smoke --only cat_050 --ctrl-only placebo@9001,placebo@9010 --n-boot 50
      CUDA_VISIBLE_DEVICES="" python paper/analysis/verdict_robustness.py --run-id robust_m19 --models auto --ctrl-run-ids controls_batch_01,controls_batch_02
"""
import argparse
import json
import os

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from defectshift import config  # noqa: E402
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, load_data, slot_dir  # noqa: E402
from defectshift.splits import B_view, get_A  # noqa: E402
from defectshift.utils import log  # noqa: E402

M8 = ["Ridge", "kNN", "XGBoost", "CatBoost", "MLP", "TabM", "TabPFN", "TabICL"]
SRC, REF = "ID_match", "OOD_match"
TOP_FRAC = 0.01
HALF_SEED = 20261029
BOOT_SEED = 20261030
VARIANTS = ("all", "rest99", "not_shared", "half1", "half2")


def load_domain(group, name, g, df, y, lot, models, ctrl_dirs, base_disjoint=False):
    """One domain: the B_te index, the (source, reference) predictions per model and repetition, and the row masks per variant. ctrl_dirs: model -> prediction folder of the control run that trained that model."""
    slot, mA, mB = build_one(df, name)
    bv = B_view(slot, df.index[mB], y, lot, "benchmark")
    a = get_A(slot, df.index[mA], y, lot)
    preds, idx0 = {}, None
    for m in models:
        pairs = []
        for i in sorted(g[g.model == m].repeat.unique()):
            z = {}
            for reg in (SRC, REF):
                if group == "challenge":
                    path = g[(g.model == m) & (g.repeat == i) & (g.regime == reg)].pred_path.iloc[0]
                else:
                    path = os.path.join(ctrl_dirs[m], slot_dir(slot), f"{m}__{reg}__r{i}.pred.npz")
                z[reg] = np.load(path)
            assert np.array_equal(z[SRC]["ood_idx"], z[REF]["ood_idx"])
            idx0 = z[SRC]["ood_idx"] if idx0 is None else idx0
            assert np.array_equal(idx0, z[SRC]["ood_idx"]), "the B_te row order differs between models or repetitions"
            pairs.append((z[SRC]["ood_pred"], z[REF]["ood_pred"]))
        preds[m] = pairs
    yv = y.loc[idx0].to_numpy()
    lots = lot.loc[idx0].to_numpy()
    score = np.abs(yv - float(y.loc[bv["B_tr"]].median()))
    k = max(1, int(np.ceil(TOP_FRAC * len(yv))))
    top = np.zeros(len(yv), bool)
    top[np.argsort(-score, kind="stable")[:k]] = True
    shared = np.isin(lots, lot.loc[a["A_dev"]].unique())
    ul = np.unique(lots)
    perm = np.random.default_rng(HALF_SEED).permutation(ul)
    h1 = np.isin(lots, perm[: len(ul) // 2])
    masks = dict(all=np.ones(len(yv), bool), rest99=~top, not_shared=~shared, half1=h1, half2=~h1)
    if base_disjoint:
        masks.update(rest99=~top & ~shared, half1=h1 & ~shared, half2=~h1 & ~shared)
    return yv, lots, preds, masks


def cell(pairs, yv, mask):
    d_rmse, d_mae = [], []
    for ps, pr in pairs:
        es, er = ps[mask] - yv[mask], pr[mask] - yv[mask]
        d_rmse.append(100 * (np.sqrt(np.mean(es ** 2)) / np.sqrt(np.mean(er ** 2)) - 1))
        d_mae.append(100 * (np.mean(np.abs(es)) / np.mean(np.abs(er)) - 1))
    return float(np.mean(d_rmse)), float(np.mean(d_mae))


def boot_exceed(yv, lots, preds, thr, n_boot, rng):
    """Recomputes the domain value (median over models) from the lot resamplings and returns the share exceeding the maximum over the controls (thr, held fixed). The cell definition is the same as in cell."""
    ul, codes = np.unique(lots, return_inverse=True)
    L = len(ul)
    sums = {}   # (model, repeat) -> per lot (source sum of squares, reference sum of squares, source sum of absolute values, reference sum of absolute values), number of rows
    cnt = np.bincount(codes, minlength=L).astype(float)
    for m, pairs in preds.items():
        for j, (ps, pr) in enumerate(pairs):
            es, er = ps - yv, pr - yv
            sums[(m, j)] = np.stack([np.bincount(codes, w, minlength=L) for w in (es ** 2, er ** 2, np.abs(es), np.abs(er))])
    W = rng.multinomial(L, np.full(L, 1.0 / L), size=n_boot).astype(float)   # (n_boot, L) lot weights
    n = W @ cnt
    out = {}
    for k, (i_s, i_r, f) in dict(rmse=(0, 1, np.sqrt), mae=(2, 3, lambda v: v)).items():
        cells = []
        for m in preds:
            reps = [100 * (f((W @ sums[(m, j)][i_s]) / n) / f((W @ sums[(m, j)][i_r]) / n) - 1)
                    for j in range(len(preds[m]))]
            cells.append(np.mean(reps, axis=0))
        dom = np.median(np.stack(cells), axis=0)
        out[k] = float(np.mean(dom > thr[k]))
    return out


def main():
    import common as cm  # imported here because common imports this module (SRC, REF)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark-run-id", default="benchmark_canonical")
    ap.add_argument("--ctrl-run-ids", default=cm.CTRL_RUNS, help="control runs (comma-separated; the models are spread over several runs)")
    ap.add_argument("--models", default="auto", help="comma-separated list of models, or auto (all models finished in the controls)")
    ap.add_argument("--only", default=None, help="first part of the challenge-domain names (comma-separated), for smoke runs")
    ap.add_argument("--ctrl-only", default=None, help="control names (comma-separated), for smoke runs")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--base-disjoint", action="store_true", help="unify the evaluation population to the lot-disjoint rows (see the description above)")
    a = ap.parse_args()
    out = os.path.join(config.RESULTS_DIR, "analysis", a.run_id)
    os.makedirs(out, exist_ok=True)
    df, y = load_data()
    lot = df[config.LOT_COL]
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", a.benchmark_run_id, "report", "per_fit.csv"))
    pf = pf[pf.regime.isin([SRC, REF])]
    run_ids = a.ctrl_run_ids.split(",")
    ctrl = pd.concat([pd.read_csv(os.path.join(config.RESULTS_DIR, "benchmark", "runs", r, "rows.csv")).assign(src_run=r) for r in run_ids])
    ctrl = ctrl[(ctrl.status == "ok") & ctrl.regime.isin([SRC, REF])]
    if a.ctrl_only:
        ctrl = ctrl[ctrl.candidate.isin(a.ctrl_only.split(","))]
    n_ctrl = ctrl.candidate.nunique()
    if n_ctrl == 0:
        raise SystemExit(f"no control domain: check --ctrl-only({a.ctrl_only}) or --ctrl-run-ids({a.ctrl_run_ids})")
    # whether the model is complete: for every control domain (number of repetitions = N_REPEATS_BENCHMARK) x two regimes
    cnt = ctrl.drop_duplicates(["candidate", "model", "repeat", "regime"]).groupby("model").size()
    complete = sorted(m for m, c in cnt.items() if c == n_ctrl * config.N_REPEATS_BENCHMARK * 2)
    partial = sorted(set(cnt.index) - set(complete))
    models = complete if a.models == "auto" else a.models.split(",")
    if a.models == "auto":
        log(f"[robust] {len(models)} models complete in the controls: {models} | models left out because they are unfinished: {partial}")
    if not models:
        raise SystemExit("no usable model (no model is finished in the controls)")
    miss = [m for m in models if m not in complete]
    if miss:
        raise SystemExit(f"models that are unfinished or missing in the controls: {miss}")
    run_of = ctrl.groupby("model").src_run.agg(lambda s: sorted(set(s)))
    bad = {m: v for m, v in run_of.items() if len(v) != 1}
    if any(m in bad for m in models):
        raise SystemExit(f"a model occurs in several control runs: { {m: bad[m] for m in models if m in bad} }")
    ctrl_dirs = {m: os.path.join(config.CACHE_DIR, "benchmark", "runs", run_of[m][0], "models") for m in models}
    pf = pf[pf.model.isin(models)]
    ctrl = ctrl[ctrl.model.isin(models)]
    if a.only:
        pf = pf[pf.candidate.str.startswith(tuple(a.only.split(",")))]
    if pf.empty:
        raise SystemExit(f"no challenge domain: check --only({a.only}) or the model list")
    for nm, g in list(pf.groupby("candidate")) + list(ctrl.groupby("candidate")):
        n = g.groupby("model").repeat.nunique()
        if set(n.index) != set(models):
            raise SystemExit(f"{nm}: the model set differs from the request {sorted(set(n.index) ^ set(models))}")
        if not (n == config.N_REPEATS_BENCHMARK).all():
            raise SystemExit(f"{nm}: the number of repetitions differs from config.N_REPEATS_BENCHMARK({config.N_REPEATS_BENCHMARK}) {n.to_dict()}")
    rows, cache = [], {}
    jobs = [("challenge", nm, g) for nm, g in pf.groupby("candidate", sort=False)] + \
           [("control", nm, g) for nm, g in ctrl.groupby("candidate", sort=False)]
    for group, name, g in jobs:
        yv, lots, preds, masks = load_domain(group, name, g, df, y, lot, models, ctrl_dirs, a.base_disjoint)
        if group == "challenge":
            if a.base_disjoint:   # lots are resampled among the lot-disjoint rows
                mk = masks["not_shared"]
                cache[name] = (yv[mk], lots[mk], {m: [(ps[mk], pr[mk]) for ps, pr in pairs] for m, pairs in preds.items()})
            else:
                cache[name] = (yv, lots, preds)
        for var in VARIANTS:
            mk = masks[var]
            for m in models:
                r, e = cell(preds[m], yv, mk)
                rows.append(dict(group=group, candidate=name, variant=var, model=m, n_rows=int(mk.sum()),
                                 n_lots=int(len(np.unique(lots[mk]))), rmse=r, mae=e))
        log(f"[robust] {group} {name[:14]:14s} rows {len(yv)} | not-shared rows {masks['not_shared'].mean():.1%} | halves {masks['half1'].sum()}/{masks['half2'].sum()}")
    c = pd.DataFrame(rows)
    json.dump(dict(models=models, ctrl_run_ids=run_ids, n_ctrl=int(n_ctrl), base_disjoint=bool(a.base_disjoint)), open(os.path.join(out, "robust_meta.json"), "w"), ensure_ascii=False)
    c.to_csv(os.path.join(out, "robust_cells.csv"), index=False)
    dom = c.groupby(["group", "candidate", "variant"])[["rmse", "mae", "n_rows", "n_lots"]].median().reset_index()
    ctrl_max = dom[dom.group == "control"].groupby("variant")[["rmse", "mae"]].max()
    tab = dom[dom.group == "challenge"].merge(ctrl_max, left_on="variant", right_index=True, suffixes=("", "_ctrl_max"))
    tab["exceed_rmse"] = tab.rmse > tab.rmse_ctrl_max
    tab["exceed_mae"] = tab.mae > tab.mae_ctrl_max
    tab["exceed_both"] = tab.exceed_rmse & tab.exceed_mae
    tab.to_csv(os.path.join(out, "robust_table.csv"), index=False)
    rng = np.random.default_rng(BOOT_SEED)
    thr = ctrl_max.loc["not_shared" if a.base_disjoint else "all"].to_dict()
    b = []
    for name, (yv, lots, preds) in cache.items():
        p = boot_exceed(yv, lots, preds, thr, a.n_boot, rng)
        b.append(dict(candidate=name, n_boot=a.n_boot, p_exceed_rmse=p["rmse"], p_exceed_mae=p["mae"]))
    pd.DataFrame(b).to_csv(os.path.join(out, "robust_boot.csv"), index=False)
    pv = tab.pivot(index="candidate", columns="variant", values="exceed_both")
    print("maximum over the controls (per variant):\n", ctrl_max.round(2).to_string())
    print("\nexceeding on both metrics (per variant):\n", pv[list(VARIANTS)].to_string())
    print("\ndomain values:\n", tab.pivot(index="candidate", columns="variant", values="rmse")[list(VARIANTS)].round(2).to_string())
    print("\nshare of resamplings exceeding:\n", pd.DataFrame(b).round(3).to_string())
    log("[robust] done")


if __name__ == "__main__":
    main()
