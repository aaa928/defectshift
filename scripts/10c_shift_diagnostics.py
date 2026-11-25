"""Three-fold shift diagnosis: feature-space OT distance (official OTDD with the label term switched off), DISDE (the core
computation of the official disde package) and the label shift (reused from 10a_shift_statistics).

Rows: A_dev (source) against B_tr + B_screen (target), the same rows as the shift statistics (10a_shift_statistics); A_heldout and
B_te are not used.
Features: the same set as the benchmark (candidates.feature_sets: the production-lot column and the categorical split-definition
columns are excluded).
Candidates: the 78 official candidates plus the 10 randomized lot-split controls (default; --no-placebo removes them). CPU only
(the script stops if a GPU is visible).

1) Feature-space OT distance (covariate): DatasetDistance of microsoft/otdd with the same dummy label for all rows and lambda_y =
   0, which switches the label term off (the official "ignore labels" mode builds pseudo labels with DBSCAN and is not used);
   lambda_x = 1, p = 2, entreg = 0.1, debiased_loss = True. Input: numerical columns = quantile transform fitted on A_dev only
   (QUANTILE_KW, uniform on [0, 1]; values outside the fitted range go to the boundary), missing values filled with the median of
   the transformed A_dev values; categorical columns = one-hot with the dataset-fixed vocabulary; the same transformation for both
   sides. In every repetition r the same n_used = min(1000, n_source, n_target) rows are drawn without replacement from both sides
   (seed OT_SEED + r) and the distance is computed OT_REPEATS times; the values per repetition, their mean and standard deviation
   are recorded (n_used < 1000 is flagged sample_limited; a side used completely stays fixed across repetitions). The number of
   rows after the draw is below the official maxsamples (10,000), so the package performs no subsampling of its own (checked). OT
   on the numerical columns only and on the categorical columns only is recorded from the same draw. The official code renumbers
   the target labels from 1, so the label column enters the regularization schedule of geomloss (the data diameter); a geomloss
   Sinkhorn divergence computed directly without a label column is therefore recorded in every repetition as a check. Own
   baseline: source versus source OT after A_dev is halved by production lot (the same number of rows; baseline_size_limited if a
   half has too few rows) and the difference (excess). The chance level is described, without a verdict, by the position among the
   10 controls and by the difference to the own baseline.
2) DISDE (concept): fit_overlap_decomp of the official disde of Cai, Namkoong and Yadlowsky (2023) as it is (cross-fitted domain
   classifier, shared distribution p(x)q(x)/(p(x)+q(x)) weights, weighted mean of per-sample performance values; the per-sample
   value is the squared error, the general loss of Alg. 1). Structure of App. D.1 and E.1.2 of the original paper: the source is
   split 80/20 and the evaluation model is trained on the 80%; the classifier and the decomposition use the source 20% plus all
   target rows with 3-fold cross-fitting. Our changes (recorded as condition columns): the 80/20 split and the cross-fitting folds
   are formed by production lot. Domain classifier = XGBClassifier(random_state=0) (the same default as the WhyShift package).
   Evaluation model: default = XGBRegressor library defaults (no tuning, no early stopping, integer category codes, missing values
   as they are), all 88 candidates; tuned = in every repetition a lot-wise search split inside the source 80% with 30 trials
   (hpo.tune, the same integer category representation as the screening), then trained again with an early-stopping split inside
   the 80% (the source 20% is used neither for training, search nor early stopping), for the final 7 domains.
   Diagnostics (App. D.2): domain-classifier AUC, mean probability versus target share, share of extreme probabilities,
calibration error (10 bins), effective sample size of the weights per side, and the standardized mean difference of the numerical
features before and after weighting. Nothing is dropped by a threshold; the decomposition with the shared distribution min(p, q)
(Eq. 2.3a) is recorded as well.
   Main reported quantity = the Y|X term on the absolute-error basis (abs_yx_shift, sign kept, and abs_yx_shift_pct, its
percentage of the source loss, the mean absolute error on the 20% source evaluation rows): the decomposition of the official
get_decomp with only the per-sample value changed to the absolute error, with the same predictions, cross-fitted probabilities and
weights. The decomposition on the squared-error basis (yx_shift, yx_shift_pct, the official fit_overlap_decomp) and the ratio to
the total difference are supplementary. Row-alignment check: in every fold the same classifier is trained and applied directly and
compared with the probabilities returned by the official cross-fitting.
   A failed (candidate, repetition) is kept as a row with status=failed and the run continues; the exit code is 1 if an entry
lacks a successful row (a rerun repeats only that entry).
3) Label shift: dmean_y of the shift statistics (the same rows, the same definition); only the mean is recorded here for comparison.

Output: results/shift_diag/<run_id>/{manifest.json, otdd.csv (candidate summary), otdd_reps.csv (per repetition), disde.csv}. Running again with the
same run-id skips the finished (candidate, repetition) entries.
Usage: CUDA_VISIBLE_DEVICES="" python scripts/10c_shift_diagnostics.py --run-id <run> [--no-placebo] [--only a,b] [--repeats 10]
       [--eval-model default|tuned] [--skip-otdd] [--skip-disde] [--threads 2]
Requires geomloss and the official otdd and disde packages (docs/reproducing.md, paragraph "Shift diagnostics").
"""
import argparse
import os
import sys
import time

if "--threads" in sys.argv:   # fix the thread count before sklearn, numpy and torch are called (when the CPU is shared with a training job)
    _n = sys.argv[sys.argv.index("--threads") + 1]
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_k] = _n

import geomloss  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from disde.dataset import Dataset  # noqa: E402
from disde.fit_predict import CrossFitIdx  # noqa: E402
from disde.loss_decomp import fit_overlap_decomp, get_decomp, get_min_weights  # noqa: E402
from disde.model import SKLearnPropModel  # noqa: E402
from otdd.pytorch.distance import DatasetDistance  # noqa: E402

from defectshift import config, hpo, manifest, regimes, registry  # noqa: E402
from defectshift.candidates import build_one, feature_sets, load_candidates, load_data  # noqa: E402
from defectshift.runio import append_row  # noqa: E402
from defectshift.splits import B_view, _strat_split, es_split, get_A  # noqa: E402
from defectshift.utils import log  # noqa: E402

SEED = 2027
SOURCE_EVAL_FRAC = 0.20          # source 80/20 of App. E.1.2 of the original DISDE paper
N_CROSSFIT = 3                   # 3-fold cross-fitting, same place
OTDD_KW = dict(method="precomputed_labeldist", λ_x=1.0, λ_y=0.0, p=2, entreg=0.1, loss="sinkhorn", debiased_loss=True,
               device="cpu", precision="single", verbose=0)
OTDD_MAXSAMPLES = 10000          # default of the official distance()
OT_N_MAX = 1000                  # upper limit of the number of rows drawn from both sides in every repetition
OT_REPEATS = 10
OT_SEED = 3000
QUANTILE_KW = dict(n_quantiles=1000, output_distribution="uniform", subsample=None, random_state=0)
EXTREME = 0.01                   # boundary for counting extreme probabilities (recorded only, not used for a verdict)
N_CAL_BINS = 10
CODE_COMMITS = dict(otdd="72f1b22", disde="eda6394")   # commits of the official repositories (microsoft/otdd, namkoong-lab/disde)


def candidate_rows(df, y, lot, name):
    """Source (A_dev) and target (B_tr + B_screen) rows and the features of one candidate."""
    slot, mA, mB = build_one(df, name)
    a = get_A(slot, df.index[mA], y, lot)
    bv = B_view(slot, df.index[mB], y, lot, "shift_statistics")
    assert set(bv) == {"B_tr", "B_screen"}
    cat_cols, num_cols = feature_sets(df, name)
    return slot, a["A_dev"], bv["B_tr"].append(bv["B_screen"]), cat_cols, num_cols


def otdd_features(df, src, tgt, cat_cols, num_cols):
    """Numerical = quantile transform fitted on A_dev (src) only + missing values filled with the median of the transformed src (a column that
    is entirely missing in src gets the constant 0.5); categorical = one-hot with the dataset-fixed vocabulary. The same transformation
    for source and target."""
    qt = QuantileTransformer(**QUANTILE_KW).fit(df.loc[src, num_cols].to_numpy(float))
    S, T = qt.transform(df.loc[src, num_cols].to_numpy(float)), qt.transform(df.loc[tgt, num_cols].to_numpy(float))
    med = np.full(S.shape[1], 0.5)
    observed = ~np.isnan(S).all(0)
    med[observed] = np.nanmedian(S[:, observed], axis=0)
    S, T = np.where(np.isnan(S), med, S), np.where(np.isnan(T), med, T)
    parts_s, parts_t = [S.astype(np.float32)], [T.astype(np.float32)]
    n_onehot = 0
    for c in cat_cols:
        vocab = np.sort(df[c].unique())
        parts_s.append((df.loc[src, c].to_numpy()[:, None] == vocab[None, :]).astype(np.float32))
        parts_t.append((df.loc[tgt, c].to_numpy()[:, None] == vocab[None, :]).astype(np.float32))
        n_onehot += len(vocab)
    return np.hstack(parts_s), np.hstack(parts_t), n_onehot


def feature_ot(Xs, Xt, seed=SEED, maxsamples=OTDD_MAXSAMPLES):
    """Official DatasetDistance (one dummy label, lambda_y = 0). Returns (distance, whether source was subsampled, whether target was subsampled).
    For the debiased computation the official code renumbers the target labels from 1 -> the label column (source 0, target 1) does not enter the
    cost (C2 = 0) but it enters the regularization schedule of geomloss (the data diameter). A value computed directly without a label column
    (direct_sinkhorn) is therefore recorded in every repetition."""
    zs, zt = torch.zeros(len(Xs), dtype=torch.long), torch.zeros(len(Xt), dtype=torch.long)
    D1 = torch.utils.data.TensorDataset(torch.from_numpy(Xs), zs)
    D2 = torch.utils.data.TensorDataset(torch.from_numpy(Xt), zt)
    np.random.seed(seed)   # for the maxsamples draw of the official code (np.random.choice)
    dist = DatasetDistance(D1, D2, **OTDD_KW).distance(maxsamples=maxsamples)
    return float(dist), len(Xs) > maxsamples, len(Xt) > maxsamples


def direct_sinkhorn(Xs, Xt):
    """The geomloss Sinkhorn divergence with the same settings (p, entreg, debias) computed directly from the features alone (no label column) - a check of the official value."""
    loss = geomloss.SamplesLoss(loss="sinkhorn", p=OTDD_KW["p"], debias=OTDD_KW["debiased_loss"],
                                blur=OTDD_KW["entreg"] ** (1 / OTDD_KW["p"]), backend="tensorized")
    return float(loss(torch.from_numpy(Xs), torch.from_numpy(Xt)))


def official_ot(Xs, Xt, seed):
    d, samp_s, samp_t = feature_ot(Xs, Xt, seed=seed)
    assert not samp_s and not samp_t, "the package subsampled by maxsamples"
    return d


def ot_repeats(Xs, Xt, n_num, src_lot, src_strat, n_max=OT_N_MAX, repeats=OT_REPEATS, seed=OT_SEED):
    """In every repetition r the same number n_used = min(n_max, n_source, n_target) of rows is drawn without replacement from both sides and
    the feature OT is computed (all columns, numerical only, categorical only; the full one with a direct-computation check). Own baseline: the
    source (A_dev) is halved by production lot (stratified by y > 0), n_base = min(n_used, rows of the half) rows are drawn from each half, and the
    source-versus-source OT is computed (all, numerical only, categorical only). baseline_size_limited if n_base < n_used. Returns a list of dicts per repetition."""
    n_used = min(n_max, len(Xs), len(Xt))
    cols = dict(all=slice(None), num=slice(0, n_num), cat=slice(n_num, None))
    out = []
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        i_s, i_t = rng.choice(len(Xs), n_used, replace=False), rng.choice(len(Xt), n_used, replace=False)
        assert len(i_s) == len(i_t) == n_used < OTDD_MAXSAMPLES
        row = dict(repeat=r, seed=seed + r, n_used=n_used)
        for k, c in cols.items():
            row[f"ot_{k}"] = official_ot(Xs[i_s][:, c], Xt[i_t][:, c], seed + r)
        row["ot_all_direct"] = direct_sinkhorn(Xs[i_s], Xt[i_t])
        h1, h2 = next(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed + r).split(np.zeros(len(Xs)), src_strat, src_lot))
        assert not set(src_lot[h1]) & set(src_lot[h2]), "the two baseline halves share a lot"
        n_base = min(n_used, len(h1), len(h2))
        j1, j2 = rng.choice(h1, n_base, replace=False), rng.choice(h2, n_base, replace=False)
        row.update(n_base=n_base, baseline_size_limited=n_base < n_used)
        for k, c in cols.items():
            row[f"base_{k}"] = official_ot(Xs[j1][:, c], Xs[j2][:, c], seed + r)
            row[f"excess_{k}"] = row[f"ot_{k}"] - row[f"base_{k}"]
        out.append(row)
    return out


def lot_folds(t, groups, k, seed):
    """Fold number of k folds stratified by t (0 = source evaluation, 1 = target) and grouped by production lot."""
    fold = np.full(len(t), -1)
    for f, (_, te) in enumerate(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(t)), t, groups)):
        fold[te] = f
    assert (fold >= 0).all()
    return fold


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else np.nan


def calibration_error(p, t, bins=N_CAL_BINS):
    """Binned calibration error of the probability p against the actual t (equal-width bins, mean of |mean p - mean t| weighted by the number of rows)."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((b == i).sum() * abs(p[b == i].mean() - t[b == i].mean()) for i in range(bins) if (b == i).any()) / len(p))


def weighted_smd(Z, t, w):
    """Per numerical feature |weighted mean difference| / pooled standard deviation before weighting. Returns (mean, maximum)."""
    s, q = t == 0, t == 1
    sd = np.sqrt((Z[s].var(0) + Z[q].var(0)) / 2)
    keep = sd > 0
    ms = (Z[s] * w[s, None]).sum(0) / w[s].sum()
    mq = (Z[q] * w[q, None]).sum(0) / w[q].sum()
    d = np.abs(ms - mq)[keep] / sd[keep]
    return float(d.mean()), float(d.max())


def eval_errors(df, y, lot, cat_cols, num_cols, src_tr, idx, eval_model, rep, threads):
    """Trains the evaluation model on src_tr and returns the errors (truth - prediction) of the rows idx. tuned searches and stops early inside src_tr only (CPU)."""
    feats = cat_cols + num_cols
    if eval_model == "default":
        m = XGBRegressor(random_state=0, n_jobs=threads, tree_method="hist", device="cpu")
        m.fit(df.loc[src_tr, feats], y.loc[src_tr])
        pred = m.predict(df.loc[idx, feats])
        info = dict(eval_params=str({k: v for k, v in m.get_params().items() if v is not None}))
    else:
        spec = registry.get("XGBoost")
        ctx = regimes.Ctx(df, y, cat_cols, num_cols, cat_mode="integer")
        tr, va = _strat_split(src_tr, y, lot, config.HPO_VAL_FRAC, SEED + 100 + rep, f"repetition {rep} evaluation-model search split")
        rec = hpo.tune(spec, ctx, tr, va, n_trials=config.HPO_N_TRIALS, seed=config.HPO_SEED)
        fit_, es_ = es_split(src_tr, y, lot, rep)
        f = regimes.fit(spec, ctx, fit_, es_, SEED + rep, params=rec["params"])
        pred = regimes.predict(f, ctx, idx)
        used = set(tr) | set(va) | set(fit_) | set(es_)
        assert used <= set(src_tr), "the tuned evaluation model used rows outside the source 80%"
        info = dict(eval_params=str(rec["params"]), eval_best_score=rec.get("best_score"), eval_rows_within_src_train=True)
        # The training device is not recorded in the row (regimes switches the device to cpu before prediction, so it cannot show the training
        # device). main() stops before the start if a GPU is visible, so the training is on the CPU - device="cpu" of the run manifest is the
        # record of the run unit.
    return y.loc[idx].to_numpy(float) - np.asarray(pred, float), info


def check_crossfit_alignment(X, t, fold, pi, threads):
    """Whether the probabilities pi returned by the official cross_fit_predict match our row order: in every fold the same classifier is trained and applied directly and compared (maximum absolute difference)."""
    assert len(fold) == len(t) and set(np.unique(fold)) == set(range(N_CROSSFIT))
    direct = np.full(len(t), np.nan)
    for k in range(N_CROSSFIT):
        te = fold == k
        m = XGBClassifier(random_state=0, n_jobs=threads, tree_method="hist", device="cpu").fit(X[~te], t[~te])
        direct[te] = m.predict_proba(X[te])[:, 1]
    assert not np.isnan(direct).any(), "a row never entered an evaluation fold"
    return float(np.max(np.abs(direct - pi)))


def disde_one(df, y, lot, name, src, tgt, cat_cols, num_cols, rep, eval_model, threads):
    seed = SEED + rep
    src_tr, src_ev = _strat_split(src, y, lot, SOURCE_EVAL_FRAC, seed, f"repetition {rep} DISDE source 80/20")
    assert not set(lot.loc[src_tr]) & set(lot.loc[src_ev]), "the source 80/20 share a lot"
    idx = src_ev.append(tgt)
    err, info = eval_errors(df, y, lot, cat_cols, num_cols, src_tr, idx, eval_model, rep, threads)
    loss, loss_abs = err ** 2, np.abs(err)
    t = np.r_[np.zeros(len(src_ev)), np.ones(len(tgt))].astype(int)
    feats = cat_cols + num_cols
    X = df.loc[idx, feats].to_numpy(np.float32)
    grp = lot.loc[idx].to_numpy()
    fold = lot_folds(t, grp, N_CROSSFIT, seed)
    for f in range(N_CROSSFIT):
        assert not set(grp[fold == f]) & set(grp[fold != f]), "a lot is shared between cross-fitting folds"
    data = Dataset(x=X, y=loss, t=t)
    cf = CrossFitIdx(len(data), N_CROSSFIT, remap=False)
    cf.map = fold.copy()                     # fold numbers of the official CrossFitIdx (map % num_splits) -> folds by lot
    prop = SKLearnPropModel(XGBClassifier(random_state=0, n_jobs=threads, tree_method="hist", device="cpu"))
    res, ex = fit_overlap_decomp(data, prop, cf, return_extras=True)
    pi, w = np.asarray(ex["props"], float), np.asarray(ex["w"], float)
    mn = get_decomp(data, get_min_weights(data, pi))
    ab = get_decomp(Dataset(x=X, y=loss_abs, t=t), w)   # absolute-error decomposition: the same predictions, cross-fitted probabilities and weights, only the per-sample value is |error|
    align = check_crossfit_alignment(X, t, fold, pi, threads)
    gap = res.eqrq - res.eprp
    s, q = t == 0, t == 1
    Z = df.loc[idx, num_cols]
    Z = Z.fillna(Z.loc[src_ev].median()).fillna(0).to_numpy(float)
    smd0 = weighted_smd(Z, t, np.ones(len(t)))
    smd1 = weighted_smd(Z, t, w)
    return dict(
        candidate=name, eval_model=eval_model, repeat=rep, split_unit="lot", crossfit_unit="lot", n_crossfit=N_CROSSFIT,
        n_src_train=len(src_tr), n_src_eval=len(src_ev), n_tgt=len(tgt), n_lots_eval=len(np.unique(grp)),
        e_p_rp=res.eprp, e_s_rp=res.esrp, e_s_rq=res.esrq, e_q_rq=res.eqrq,
        x_shift_p_to_s=res.esrp - res.eprp, yx_shift=res.esrq - res.esrp, x_shift_s_to_q=res.eqrq - res.esrq, total_gap=gap,
        yx_shift_pct=100 * (res.esrq - res.esrp) / res.eprp, x_shift_p_to_s_pct=100 * (res.esrp - res.eprp) / res.eprp,
        x_shift_s_to_q_pct=100 * (res.eqrq - res.esrq) / res.eprp, total_gap_pct=100 * gap / res.eprp,
        src_train_sha=hpo._idx_sha(src_tr), src_eval_sha=hpo._idx_sha(src_ev), tgt_sha=hpo._idx_sha(tgt),
        fold_sha=hpo._idx_sha(pd.Index(fold)),
        yx_ratio=(res.esrq - res.esrp) / gap if gap != 0 and np.isfinite(gap) else np.nan,
        yx_shift_min=mn.esrq - mn.esrp, yx_ratio_min=(mn.esrq - mn.esrp) / gap if gap != 0 and np.isfinite(gap) else np.nan,
        abs_e_p_rp=ab.eprp, abs_yx_shift=ab.esrq - ab.esrp, abs_total_gap=ab.eqrq - ab.eprp,
        abs_yx_shift_pct=100 * (ab.esrq - ab.esrp) / ab.eprp, abs_total_gap_pct=100 * (ab.eqrq - ab.eprp) / ab.eprp,
        abs_x_shift_p_to_s_pct=100 * (ab.esrp - ab.eprp) / ab.eprp, abs_x_shift_s_to_q_pct=100 * (ab.eqrq - ab.esrq) / ab.eprp,
        loss_check=float(abs(res.eprp - loss[s].mean()) + abs(res.eqrq - loss[q].mean())),
        abs_loss_check=float(abs(ab.eprp - loss_abs[s].mean()) + abs(ab.eqrq - loss_abs[q].mean())),
        crossfit_align_maxdiff=align,
        prop_auc=float(roc_auc_score(t, pi)), prop_mean=float(pi.mean()), t_mean=float(t.mean()),
        prop_frac_lo=float((pi < EXTREME).mean()), prop_frac_hi=float((pi > 1 - EXTREME).mean()),
        prop_q05_src=float(np.quantile(pi[s], .05)), prop_q95_src=float(np.quantile(pi[s], .95)),
        prop_q05_tgt=float(np.quantile(pi[q], .05)), prop_q95_tgt=float(np.quantile(pi[q], .95)),
        calib_err=calibration_error(pi, t), ess_src=ess(w[s]), ess_tgt=ess(w[q]),
        ess_src_frac=ess(w[s]) / s.sum(), ess_tgt_frac=ess(w[q]) / q.sum(),
        w_max_over_mean_src=float(w[s].max() / w[s].mean()), w_max_over_mean_tgt=float(w[q].max() / w[q].mean()),
        smd_mean_before=smd0[0], smd_max_before=smd0[1], smd_mean_after=smd1[0], smd_max_after=smd1[1],
        **info)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--no-placebo", action="store_true", help="leave out the 10 randomized lot-split controls (included by default as the chance-level reference)")
    ap.add_argument("--only", default=None, help="comma-separated candidates (for smoke runs and the tuned comparison)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--eval-model", choices=["default", "tuned"], default="default")
    ap.add_argument("--skip-otdd", action="store_true")
    ap.add_argument("--skip-disde", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    names = load_candidates(include_placebo=not args.no_placebo)
    if args.only:
        want = args.only.split(",")
        bad = [n for n in want if n not in load_candidates(include_placebo=True)]
        if bad:
            raise SystemExit(f"names that are not candidates: {bad}")
        names = [n for n in load_candidates(include_placebo=True) if n in want]
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():   # the default evaluation model and the classifier are fixed to the CPU, while the retuning path (hpo, regimes) uses a visible GPU, so run with the GPU hidden
        raise SystemExit('run with CUDA_VISIBLE_DEVICES="" (the diagnosis is CPU only: the device is the same for all evaluation models and does not compete with GPU jobs)')
    run_dir = os.path.join(config.RESULTS_DIR, "shift_diag", args.run_id)
    import disde
    import importlib.metadata as md
    import otdd
    versions = {p: md.version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost", "optuna", "torch", "geomloss", "POT")}
    status = manifest.write_or_check(run_dir, manifest.build(dict(
        stage="shift_diag", candidates=names, seed=SEED, repeats=args.repeats, eval_model=args.eval_model,
        source_eval_frac=SOURCE_EVAL_FRAC, n_crossfit=N_CROSSFIT, split_unit="lot", crossfit_unit="lot",
        otdd=dict(OTDD_KW, maxsamples=OTDD_MAXSAMPLES, dummy_label=True, n_max=OT_N_MAX, repeats=OT_REPEATS, seed=OT_SEED,
                  quantile=QUANTILE_KW, quantile_fit="A_dev", missing_after_quantile="A_dev median of transformed values"),
        code_commits=CODE_COMMITS,
        disde_file=disde.__file__, otdd_file=otdd.__file__, package_versions=versions, python=sys.executable, device="cpu",
        include_placebo=not args.no_placebo,
        rows="source = A_dev, target = B_tr + B_screen", features="candidates.feature_sets",
        skip_otdd=args.skip_otdd, skip_disde=args.skip_disde)))
    out_o, out_or, out_d = (os.path.join(run_dir, f) for f in ("otdd.csv", "otdd_reps.csv", "disde.csv"))

    def ok_rows(path):
        if not os.path.exists(path):
            return pd.DataFrame()
        t_ = pd.read_csv(path)
        return t_[t_.status == "ok"] if "status" in t_ else t_
    done_o = set(ok_rows(out_o).get("candidate", []))
    if os.path.exists(out_or):   # the per-repetition table is written before the summary -> repetition rows of a candidate without a summary (traces of an interruption or failure) are removed and recomputed
        f = pd.read_csv(out_or)
        f[f.candidate.isin(done_o)].to_csv(out_or, index=False)
    d_ok = ok_rows(out_d)
    done_d = set(zip(d_ok.candidate, d_ok.repeat)) if len(d_ok) else set()
    df, y = load_data()
    lot = df[config.LOT_COL]
    log(f"[shift_diag] run {args.run_id} ({status}) {len(names)} candidates, {args.repeats} repetitions, evaluation model {args.eval_model}")
    n_fail = 0
    for name in names:
        slot, src, tgt, cat_cols, num_cols = candidate_rows(df, y, lot, name)
        if not args.skip_otdd and name not in done_o:
            t0 = time.time()
            try:
                Xs, Xt, n_oh = otdd_features(df, src, tgt, cat_cols, num_cols)
                reps = ot_repeats(Xs, Xt, len(num_cols), lot.loc[src].to_numpy(), (y.loc[src] > 0).astype(int).to_numpy())
            except Exception as e:   # a failure is kept as a row and the run goes on to the next candidate (exit code 1 at the end)
                n_fail += 1
                append_row(out_o, dict(candidate=name, slot=slot, status="failed", error=f"{type(e).__name__}: {e}"[:500]))
                log(f"[shift_diag] {name:30s} feature OT failed: {type(e).__name__}: {e}")
            else:
                for r in reps:
                    append_row(out_or, dict(candidate=name, **r))
                R = pd.DataFrame(reps)
                n_used = int(R.n_used.iloc[0])
                ya, yb = y.loc[src].to_numpy(float), y.loc[tgt].to_numpy(float)
                summ = dict(candidate=name, slot=slot, status="ok", n_src=len(src), n_tgt=len(tgt), n_used=n_used,
                            sample_limited=n_used < OT_N_MAX, src_fixed=n_used == len(src), tgt_fixed=n_used == len(tgt),
                            n_base=int(R.n_base.min()), baseline_size_limited=bool(R.baseline_size_limited.any()),
                            n_num=len(num_cols), n_cat=len(cat_cols), n_onehot=n_oh, dim=Xs.shape[1], repeats=len(R),
                            direct_maxabsdiff=float((R.ot_all - R.ot_all_direct).abs().max()),
                            mean_y_src=float(ya.mean()), mean_y_tgt=float(yb.mean()), sec=time.time() - t0)
                for k in ("ot_all", "ot_num", "ot_cat", "base_all", "base_num", "base_cat", "excess_all", "excess_num", "excess_cat"):
                    summ[f"{k}_mean"], summ[f"{k}_sd"] = float(R[k].mean()), float(R[k].std(ddof=1))
                summ.update(ot_all_min=float(R.ot_all.min()), ot_all_max=float(R.ot_all.max()))
                append_row(out_o, summ)
                log(f"[shift_diag] {name:30s} feature OT {summ['ot_all_mean']:.3f} baseline {summ['base_all_mean']:.3f} "
                    f"(rows {len(src)}/{len(tgt)} -> {n_used}, dimension {Xs.shape[1]}, {time.time() - t0:.0f}s)")
        if args.skip_disde:
            continue
        for rep in range(args.repeats):
            if (name, rep) in done_d:
                continue
            t0 = time.time()
            try:
                r = disde_one(df, y, lot, name, src, tgt, cat_cols, num_cols, rep, args.eval_model, args.threads)
            except Exception as e:
                n_fail += 1
                append_row(out_d, dict(candidate=name, eval_model=args.eval_model, repeat=rep, status="failed",
                                       error=f"{type(e).__name__}: {e}"[:500], sec=time.time() - t0))
                log(f"[shift_diag] {name:30s} r{rep} DISDE failed: {type(e).__name__}: {e}")
                continue
            r.update(status="ok", sec=time.time() - t0)
            append_row(out_d, r)
            log(f"[shift_diag] {name:30s} r{rep} Y|X (absolute error) {r['abs_yx_shift_pct']:+.1f}% (squared error {r['yx_shift_pct']:+.1f}%) "
                f"AUC {r['prop_auc']:.3f} ESS {r['ess_src_frac']:.3f}/{r['ess_tgt_frac']:.3f} ({r['sec']:.0f}s)")
    # completeness: every candidate x (1 OT row, the DISDE repetitions) has a successful row
    lack_o = [] if args.skip_otdd else sorted(set(names) - set(ok_rows(out_o).get("candidate", [])))
    d_ok = ok_rows(out_d)
    have = set(zip(d_ok.candidate, d_ok.repeat)) if len(d_ok) else set()
    lack_d = [] if args.skip_disde else [(n, r_) for n in names for r_ in range(args.repeats) if (n, r_) not in have]
    if n_fail or lack_o or lack_d:
        log(f"[shift_diag] incomplete: {n_fail} failures in this run, {len(lack_o)} OT candidates without a successful row, {len(lack_d)} DISDE (candidate, repetition) entries "
            f"(e.g. {lack_o[:3]} {lack_d[:3]}). Running again with the same run-id repeats only those entries")
        sys.exit(1)
    log(f"[shift_diag] done: {len(names)} candidates, {args.repeats} repetitions")


if __name__ == "__main__":
    main()
