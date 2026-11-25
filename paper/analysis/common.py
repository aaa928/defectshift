"""Common loader of the supplementary paper analyses. It reads only saved predictions (no retraining, CPU only).
It reads the B_te predictions of one domain by the same rule as load_domain of verdict_robustness.py and also returns the row index and the split information.
Cell value (model x domain) = repetition mean of (source error / reference error - 1) x 100 per repetition, domain value = median over models (the same as in verdict_robustness.py).
"""
import os

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from defectshift import config  # noqa: E402
from defectshift.utils import read_per_fit
from defectshift.candidates import build_one, slot_dir  # noqa: E402
from defectshift.splits import B_view, get_A  # noqa: E402
import verdict_robustness as rb  # noqa: E402

SRC, REF = rb.SRC, rb.REF
CTRL_RUNS = "controls_batch_01,controls_batch_02,controls_tabnet,controls_batch_03,controls_node"


def short(name):
    """First part of a candidate name (x_time<=... -> x_time)."""
    return name.split("<")[0]


def load_challenge(benchmark_run_id, models=None):
    """per_fit of the challenge domains (source and reference regimes only). All 22 models if models is None."""
    pf = read_per_fit(os.path.join(config.RESULTS_DIR, "benchmark", "runs", benchmark_run_id, "report", "per_fit.csv"))
    pf = pf[pf.regime.isin([SRC, REF])]
    if models is not None:
        pf = pf[pf.model.isin(models)]
    return pf


def load_controls(run_ids, models):
    """Rows of the controls and the prediction folder per model. For every model, all 10 controls x repetitions x two regimes must exist."""
    run_ids = run_ids.split(",")
    ctrl = pd.concat([pd.read_csv(os.path.join(config.RESULTS_DIR, "benchmark", "runs", r, "rows.csv")).assign(src_run=r) for r in run_ids])
    ctrl = ctrl[(ctrl.status == "ok") & ctrl.regime.isin([SRC, REF]) & ctrl.model.isin(models)]
    n_ctrl = ctrl.candidate.nunique()
    cnt = ctrl.drop_duplicates(["candidate", "model", "repeat", "regime"]).groupby("model").size()
    miss = [m for m in models if cnt.get(m, 0) != n_ctrl * config.N_REPEATS_BENCHMARK * 2]
    if miss:
        raise SystemExit(f"models not finished in the controls: {miss}")
    run_of = ctrl.groupby("model").src_run.agg(lambda s: sorted(set(s)))
    if any(len(v) != 1 for v in run_of):
        raise SystemExit(f"a model occurs in several control runs: {run_of[run_of.map(len) != 1].to_dict()}")
    dirs = {m: os.path.join(config.CACHE_DIR, "benchmark", "runs", run_of[m][0], "models") for m in models}
    return ctrl, dirs


def load_domain(group, name, g, df, y, lot, models, ctrl_dirs=None):
    """One domain. Returns idx (B_te row index), yv, lots, preds{model: [(source prediction, reference prediction), ...repetitions]}, bv (B split), a (A split)."""
    slot, mA, mB = build_one(df, name)
    bv = B_view(slot, df.index[mB], y, lot, "benchmark")
    a = get_A(slot, df.index[mA], y, lot)
    preds, idx0 = {}, None
    for m in models:
        reps = sorted(g[g.model == m].repeat.unique())
        if len(reps) != config.N_REPEATS_BENCHMARK:
            raise SystemExit(f"{name} {m}: number of repetitions {len(reps)} != {config.N_REPEATS_BENCHMARK}")
        pairs = []
        for i in reps:
            z = {}
            for reg in (SRC, REF):
                if group == "challenge":
                    path = g[(g.model == m) & (g.repeat == i) & (g.regime == reg)].pred_path.iloc[0]
                else:
                    path = os.path.join(ctrl_dirs[m], slot_dir(slot), f"{m}__{reg}__r{i}.pred.npz")
                z[reg] = np.load(path)
            assert np.array_equal(z[SRC]["ood_idx"], z[REF]["ood_idx"])
            idx0 = z[SRC]["ood_idx"] if idx0 is None else idx0
            assert np.array_equal(idx0, z[SRC]["ood_idx"]), "the B_te row order differs between models and repetitions"
            pairs.append((z[SRC]["ood_pred"], z[REF]["ood_pred"]))
        preds[m] = pairs
    return dict(idx=idx0, yv=y.loc[idx0].to_numpy(), lots=lot.loc[idx0].to_numpy(), preds=preds, bv=bv, a=a)


def domain_value(preds, yv, mask):
    """Domain value (RMSE, MAE) on the mask rows: the median of the model cells (rb.cell)."""
    cells = np.array([rb.cell(p, yv, mask) for p in preds.values()])
    return float(np.median(cells[:, 0])), float(np.median(cells[:, 1]))
