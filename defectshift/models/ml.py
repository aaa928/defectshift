"""
Builders of the classical and tree ML models and their Optuna hyperparameter search (regression only).

  LinearRegression (OLS), LinearModel (Ridge), kNN, SVM_rbf (SVR), DecisionTree, RandomForest, GBR (scikit-learn),
  XGBoost, LightGBM, CatBoost

Protocol:
  - The boosted trees (LightGBM, XGBoost, CatBoost) do not search the number of trees; it is set by early stopping on the
    validation set (50 rounds, at most 2000, the official configuration of Gorishniy et al. 2021). The search spaces are in
    _suggest_params (one source per model; Supplementary Material S7 of the paper lists them). The other models have no early stopping,
    and every trial is compared by its validation R^2.
  - Search: the same budget for every tuned model (OPTUNA_N_TRIALS = 30, TPE sampler seed 42).
  - LinearRegression has no hyperparameters and is trained once without search (_NO_SEARCH).

Devices: XGBoost (device="cuda", prediction on cpu), CatBoost (task_type="GPU"), LightGBM (LIGHTGBM_DEVICE = "cpu"),
scikit-learn models (linear, kNN, SVR, DT, RF, GBR) on the CPU.
"""
import os

import joblib
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb

from defectshift.utils import log, log_exception, free_memory

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
BOOSTING_MAX_ITER = 2000
OPTUNA_N_TRIALS = 30  # search budget applied equally to every tuned model
# The LightGBM device is fixed to the CPU rather than detected (scripts/checks/check_lgbm_device.py). The GPU build of
# the official distribution (OpenCL) trains no faster than the CPU in the actual search settings, and at its default
# single precision a retraining with the same seed changes the early-stopping round and the predictions.
LIGHTGBM_DEVICE = "cpu"

MODEL_NAMES = ["LinearRegression", "LinearModel", "kNN", "SVM_rbf", "DecisionTree", "RandomForest", "GBR",
               "XGBoost", "LightGBM", "CatBoost"]

_NO_SEARCH = {"LinearRegression"}  # models without hyperparameters: trained once without search


def _optional(trial, name, low, high, log=True):
    """The '? distribution' notation of the official search of Gorishniy et al. 2021 (bin/tune.py): choose
    optional_<name> from [False, True]; True draws from the distribution, False gives 0."""
    return trial.suggest_float(name, low, high, log=log) if trial.suggest_categorical(f"optional_{name}", [False, True]) else 0.0


def _suggest_params(task: str, name: str, trial: optuna.Trial) -> dict:
    """Optuna search space per model (sources in Supplementary Material S7 of the paper). Integer ranges whose source gives no
    distribution use a log scale if the upper bound is at least ten times the lower bound, uniform otherwise (Optuna
    default). Returned keys = estimator argument names."""
    if name == "LinearModel":    # Ridge: widened after a preliminary search reached both boundaries
        return dict(alpha=trial.suggest_float("alpha", 1e-4, 1e5, log=True))
    if name == "kNN":            # algorithm and leaf_size only affect search speed and keep their defaults
        return dict(n_neighbors=trial.suggest_int("n_neighbors", 1, 50, log=True),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                    p=trial.suggest_categorical("p", [1, 2, 3, 4, 5]))
    if name == "SVM_rbf":        # on the standardized target scale
        return dict(C=trial.suggest_float("C", 0.1, 1000.0, log=True),
                    gamma=trial.suggest_float("gamma", 1e-4, 10.0, log=True),
                    epsilon=trial.suggest_float("epsilon", 1e-4, 0.1, log=True))
    if name == "DecisionTree":
        # source range with two bounds widened (depth lower bound 3 -> 2, minimum leaf size upper bound 30 -> 100). The
        # criterion choice {squared_error, friedman_mse} of the source is not searched: friedman_mse is deprecated in
        # scikit-learn 1.9 and handled as squared_error with identical predictions, so squared_error (the default) is fixed.
        return dict(splitter=trial.suggest_categorical("splitter", ["best", "random"]),
                    max_depth=trial.suggest_int("max_depth", 2, 50, log=True),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 30, log=True),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100, log=True))
    if name == "RandomForest":
        no_depth = trial.suggest_categorical("max_depth_is_none", [False, True])
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 1000, log=True),
                    max_depth=None if no_depth else trial.suggest_int("max_depth", 3, 50, log=True),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20, log=True),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20, log=True),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", None]))
    if name == "GBR":
        # parameters and ranges of Grinsztajn et al. 2022, Table 7 (the weighted choice probabilities and LogNormal priors
        # of their random search are not reproduced). The loss is fixed to squared error like the other models. The
        # learning-rate prior LogNormal[log(0.01), log(10)] becomes log-uniform over its mean +- 2 standard deviations
        # (log scale), [1e-4, 1]. LogUniformInt[a.5, b.5] becomes the integer range after NumPy rounding.
        return dict(learning_rate=trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    n_estimators=trial.suggest_int("n_estimators", 10, 1000, log=True),
                    # the criterion choice [friedman_mse, squared_error] of Table 7 is not searched: both give identical
                    # predictions in scikit-learn 1.9, so the default (friedman_mse) is kept
                    max_depth=trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5]),
                    min_samples_split=trial.suggest_categorical("min_samples_split", [2, 3]),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 50, log=True),
                    min_impurity_decrease=trial.suggest_categorical("min_impurity_decrease", [0.0, 0.01, 0.02, 0.05]),
                    max_leaf_nodes=trial.suggest_categorical("max_leaf_nodes", [None, 5, 10, 15]))
    if name == "XGBoost":        # Gorishniy et al. 2021, Table 16 = official xgboost/tuning/0.toml
        return dict(max_depth=trial.suggest_int("max_depth", 3, 10),
                    min_child_weight=trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    learning_rate=trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
                    colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    gamma=_optional(trial, "gamma", 1e-8, 1e2),
                    reg_lambda=_optional(trial, "lambda", 1e-8, 1e2),
                    reg_alpha=_optional(trial, "alpha", 1e-8, 1e2))
    if name == "CatBoost":       # Gorishniy et al. 2021, Table 17 = official catboost/tuning/0.toml
        return dict(depth=trial.suggest_int("depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                    leaf_estimation_iterations=trial.suggest_int("leaf_estimation_iterations", 1, 10))
    if name == "LightGBM":
        # TabArena 2025, Table C.2 (trial labels follow the table, returned keys are LGBMRegressor arguments:
        # feature_fraction = colsample_bytree, bagging_fraction = subsample, min_data_in_leaf = min_child_samples,
        # lambda_l1, lambda_l2 = reg_alpha, reg_lambda). The fixed bagging_freq = 1 of the table is set in build_one_model.
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                    colsample_bytree=trial.suggest_float("feature_fraction", 0.4, 1.0),
                    subsample=trial.suggest_float("bagging_fraction", 0.7, 1.0),
                    num_leaves=trial.suggest_int("num_leaves", 2, 200, log=True),
                    min_child_samples=trial.suggest_int("min_data_in_leaf", 1, 64, log=True),
                    extra_trees=trial.suggest_categorical("extra_trees", [False, True]),
                    min_data_per_group=trial.suggest_int("min_data_per_group", 2, 100, log=True),
                    cat_l2=trial.suggest_float("cat_l2", 0.005, 2.0, log=True),
                    cat_smooth=trial.suggest_float("cat_smooth", 0.001, 100.0, log=True),
                    max_cat_to_onehot=trial.suggest_int("max_cat_to_onehot", 8, 100, log=True),
                    reg_alpha=trial.suggest_float("lambda_l1", 1e-4, 1.0),
                    reg_lambda=trial.suggest_float("lambda_l2", 1e-4, 2.0))
    raise ValueError(name)


def device_of(name: str) -> str:
    """Device on which the model actually trains (for the record)."""
    if name in ("XGBoost", "CatBoost"):
        return "cuda"
    if name == "LightGBM":
        return LIGHTGBM_DEVICE
    return "cpu"


def build_one_model(task: str, name: str, params: dict, seed: int = RANDOM_STATE):
    """One untrained regression model with the hyperparameters params. seed is the model's internal randomness (weight
    initialization, bagging and so on)."""
    assert task == "reg", "regression only"
    if name == "RandomForest":
        return RandomForestRegressor(n_jobs=-1, random_state=seed, **params)

    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=BOOSTING_MAX_ITER, random_state=seed,
                                 verbosity=-1, device=LIGHTGBM_DEVICE, subsample_freq=1, **params)   # TabArena bagging_freq = 1

    if name == "XGBoost":
        return xgb.XGBRegressor(n_estimators=BOOSTING_MAX_ITER, random_state=seed, n_jobs=-1,
                                tree_method="hist", device="cuda",
                                early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)

    if name == "CatBoost":
        # gpu_ram_part is lowered from its default 0.95: CatBoost holds that share of GPU memory for the whole process, which
        # can make other GPU models in the same or another process run out of memory.
        # As in the official configuration of Gorishniy et al. (catboost/tuning/0.toml), early_stopping_rounds and od_pval
        # are both passed to the constructor, which CatBoost applies as od_type=IncToDec, od_pval=0.001, od_wait=50; passing
        # early_stopping_rounds to fit would switch to od_type=Iter and override od_pval with 0. metric_period is not used
        # for early stopping and is not set. allow_writing_files=False: no training log files (no effect on predictions).
        return CatBoostRegressor(iterations=BOOSTING_MAX_ITER, random_state=seed, verbose=False,
                                 early_stopping_rounds=EARLY_STOPPING_ROUNDS, od_pval=0.001, allow_writing_files=False,
                                 task_type="GPU", devices="0", gpu_ram_part=0.3, **params)

    if name == "SVM_rbf":
        return SVR(kernel="rbf", **params)

    if name == "LinearModel":
        return Ridge(random_state=seed, **params)

    # Missing values: DecisionTree accepts them natively (scikit-learn 1.3+), kNN gets imputation and standardization,
    # GBR imputation only (protocol.ml_needs_scale).
    if name == "kNN":
        return KNeighborsRegressor(n_jobs=-1, **params)
    if name == "DecisionTree":
        return DecisionTreeRegressor(random_state=seed, **params)
    if name == "GBR":
        return GradientBoostingRegressor(random_state=seed, **params)

    if name == "LinearRegression":
        return LinearRegression(**params)  # OLS has no randomness and ignores the seed

    raise ValueError(name)


def fit_with_val(task: str, name: str, model, Xtr, ytr, Xval=None, yval=None):
    """Boosted trees stop early on the validation set; the other models are fitted directly."""
    if name == "XGBoost":
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        # Training uses device="cuda" (GPU histograms), but prediction inputs are pandas frames on the CPU; switching the
        # device to cpu for prediction avoids a device-transfer warning and overhead on every call. The trained ensemble is
        # valid regardless of the device.
        model.set_params(device="cpu")
    elif name == "LightGBM":
        # LightGBM 4.7.0 deprecates eval_set; eval_X and eval_y have the same meaning
        model.fit(Xtr, ytr, eval_X=Xval, eval_y=yval,
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    elif name == "CatBoost":
        model.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True, verbose=False)   # early-stopping settings are in the constructor
    else:
        model.fit(Xtr, ytr)
    return model


def _fit_and_score(task: str, name: str, params: dict, Xtr, ytr, Xval, yval):
    """Train one Optuna trial and return its validation R^2. On failure, log the exception and return None (only that
    trial is skipped), so that a rare failure of one configuration (convergence error, driver issue) does not end the run."""
    try:
        model = build_one_model(task, name, params)
        fit_with_val(task, name, model, Xtr, ytr, Xval, yval)
        return dict(model=model, params=params, score=r2_score(yval, model.predict(Xval)))
    except Exception:
        log_exception(f"[{task}][{name}] params={params} training failed, skipping this combination")
        return None
    finally:
        free_memory()


def tune_and_fit(task: str, name: str, Xtr, ytr, Xval, yval, n_trials: int = OPTUNA_N_TRIALS, extra_params=None,
                 trial_log=None):
    """Search n_trials times with Optuna (TPE) and return the (model, hyperparameters, score) with the best validation
    R^2. Models without hyperparameters (_NO_SEARCH, such as LinearRegression) are trained once without search. Only the
    best trial is kept in memory. extra_params = model arguments that are not searched (native categorical handling); they
    are added for training only and not to the returned params. With trial_log (a list), every trial's number, state,
    validation score, time and hyperparameters are appended."""
    extra = dict(extra_params or {})

    if name in _NO_SEARCH:
        r = _fit_and_score(task, name, dict(extra), Xtr, ytr, Xval, yval)
        if r is None:
            raise RuntimeError(f"[{task}][{name}] training failed (see the ERROR lines in the log)")
        return r["model"], {}, r["score"]

    best_holder = {}

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(task, name, trial)
        r = _fit_and_score(task, name, {**params, **extra}, Xtr, ytr, Xval, yval)
        if r is None:
            raise optuna.TrialPruned()
        r["params"] = params   # only the searched hyperparameters are returned
        if name in ("XGBoost", "LightGBM", "CatBoost"):   # to check whether a trial reaches the maximum round (2000)
            trial.set_user_attr("best_round", _round_of(r["model"]))
        score = r["score"]
        if not best_holder or score > best_holder["score"]:
            best_holder.clear()
            best_holder.update(r)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    if trial_log is not None:
        trial_log.extend(_trial_records(study))

    if not best_holder:
        raise RuntimeError(f"[{task}][{name}] all trials failed (see the ERROR lines in the log)")

    return best_holder["model"], best_holder["params"], best_holder["score"]


def _trial_records(study):
    """Optuna trial records (number, state, validation score, seconds, hyperparameters), with the search range of each
    numerical parameter (dist = [low, high, log scale], to see whether a chosen value lies at a range end) and per-trial
    training records (user_attrs: best boosting round, MLP/FT-T epochs)."""
    out = []
    for t in study.trials:
        sec = (t.datetime_complete - t.datetime_start).total_seconds() if t.datetime_complete and t.datetime_start else None
        dist = {k: [d.low, d.high, bool(d.log)] for k, d in t.distributions.items() if hasattr(d, "low")}
        out.append(dict(number=t.number, state=t.state.name, value=t.value, seconds=sec, params=dict(t.params),
                        dist=dist, user_attrs=dict(t.user_attrs)))
    return out


def _round_of(model):
    """Round chosen by early stopping of a boosted model (None if unavailable)."""
    for a in ("best_iteration", "best_iteration_"):
        try:
            v = getattr(model, a)
            if v is not None:
                return int(v)
        except Exception:
            pass
    try:
        return int(model.get_best_iteration())
    except Exception:
        return None


def fit_one(task: str, name: str, params: dict, Xtr, ytr, Xval=None, yval=None, seed=RANDOM_STATE):
    """Build one model with the final params and train it on (Xtr, ytr). seed is the model's internal randomness;
    Xval/yval are used by early-stopping models only."""
    model = build_one_model(task, name, params, seed=seed)
    fit_with_val(task, name, model, Xtr, ytr, Xval, yval)
    return model


def save_model(model, path: str):
    """Save a scikit-learn, XGBoost, CatBoost or LightGBM model with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log(f"[Model saved] -> {path}")


def load_model(path: str):
    return joblib.load(path)
