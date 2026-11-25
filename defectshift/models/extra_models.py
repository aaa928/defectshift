"""The two additional deep-learning models (NODE, TabNet): their list and the tuning and training entry points. Kept separate
from the 20-model roster (registry.py) and the tuning list (hpo.TUNED).

Why separate: hpo.py and regimes.py are compatibility files (runio.COMPAT_FILES); changing them would block the hash checks of
the runs that reuse the main tuning run (sensitivity analysis, merging of runs). The two models are therefore attached through
this file and a dedicated runner (scripts/06a_benchmark_node_tabnet.py), and the merge, aggregation and sensitivity
scripts see both lists through spec_of(), tuned_all(), tune() and fit_predict().

- route "node": models/node.py (official Qwicen/node code, PyTorch)
- route "tabnet": models/tabnet.py -> models/tabnet_tf1_worker.py (official google-research/tabnet, TensorFlow 1, separate
  environment)
- Both are tuned models (OOD_match_tuned included) and early-stopping models (uses_es = True; the internal 80/20 lot-level split
  follows the same rule as the main benchmark).
"""
from defectshift import hpo, registry
from defectshift.registry import Spec

EXTRA_SPECS = {
    "NODE": Spec("NODE", "deep", "node", "NODE", True),
    "TabNet": Spec("TabNet", "deep", "tabnet", "TabNet", True),
}
EXTRA_TUNED = ("NODE", "TabNet")


def spec_of(name):
    """Spec of a model of the 20-model roster or of an additional model."""
    if name in registry.SPECS:
        return registry.SPECS[name]
    return EXTRA_SPECS[name]


def tuned_all():
    """All tuned models (hpo.TUNED of the 20 models + the additional models)."""
    return tuple(hpo.TUNED) + EXTRA_TUNED


def n_trials_final(name):
    """Search budget of the main run: NODE = every grid configuration (25), TabNet = config.HPO_N_TRIALS (30)."""
    from defectshift import config
    from defectshift.models import node
    assert name in EXTRA_TUNED, name
    return len(node.CONFIGS) if name == "NODE" else config.HPO_N_TRIALS


def tune(name, df, y, cat_cols, num_cols, train_idx, val_idx, seed, n_trials=None, max_epochs=None, device=None, progress_path=None):
    """Search with training on train_idx and validation on val_idx. n_trials: for NODE the first n grid configurations
    (None = all), for TabNet the number of TPE trials (None = the final budget). progress_path: per-configuration progress record
    of the NODE grid (resuming, node.tune_grid); not used by TabNet.
    Returns (chosen params, record dict: best_score, trials, sampler, ...)."""
    from defectshift.models import node, tabnet
    if name == "NODE":
        cfgs = None if n_trials is None else node.CONFIGS[:n_trials]
        return node.tune_grid(df, y, cat_cols, num_cols, train_idx, val_idx, seed, max_epochs=max_epochs, configs=cfgs,
                                       progress_path=progress_path)
    if name == "TabNet":
        return tabnet.tune(df, y, cat_cols, num_cols, train_idx, val_idx, n_trials or n_trials_final(name), seed=seed,
                                  max_epochs=max_epochs, device=device)
    raise ValueError(name)


def fit_predict(name, df, y, cat_cols, num_cols, fit_idx, es_idx, pred_sets, params, seed, max_epochs=None, save_prefix=None,
                device=None):
    """Training on fit_idx with early stopping on es_idx; predictions for pred_sets ({name: rows}) on the original scale.
    params=None means the official default configuration. With save_prefix the weights are saved (NODE <prefix>.pt, TabNet
    <prefix>.ckpt.*). Returns (prediction dict, record dict: train_sec, best_epoch, n_epochs, ...)."""
    from defectshift.models import node, tabnet
    if name == "NODE":
        return node.fit_predict(df, y, cat_cols, num_cols, fit_idx, es_idx, pred_sets, params or node.DEFAULT, seed,
                                         max_epochs=max_epochs, save_path=None if save_prefix is None else save_prefix + ".pt")
    if name == "TabNet":
        return tabnet.fit_predict(df, y, cat_cols, num_cols, fit_idx, es_idx, pred_sets, params, seed, max_epochs=max_epochs,
                                         ckpt_prefix=None if save_prefix is None else save_prefix + ".ckpt", device=device)
    raise ValueError(name)


def predict_saved(name, df, y, cat_cols, num_cols, fit_idx, params, seed, save_prefix, pred_sets, device=None):
    """Prediction from saved weights without training (save-and-reload check)."""
    from defectshift.models import node, tabnet
    if name == "NODE":
        return node.predict_saved(df, y, cat_cols, num_cols, fit_idx, seed, save_prefix + ".pt", pred_sets)
    return tabnet.predict_saved(df, y, cat_cols, num_cols, fit_idx, params, seed, save_prefix + ".ckpt", pred_sets,
                                       device=device)
