"""Single entry point of the training protocol for the ml, transformer and tabular_dl routes (preprocessing, training and
tuning calls). The training protocol constants and the way each library is called live only here; regimes and hpo call
only these functions. The foundation route is in models/foundation.py and is called by regimes directly.
"""
import pandas as pd

from defectshift.utils import cat_cardinalities, prep_features_cat_cont, set_seed

# Training protocol constants of the MLP and FT-Transformer. pytabkit (RealMLP, TabM) uses its own epochs and stopping.
TRANSFORMER_MAX_EPOCHS = 150
TRANSFORMER_PATIENCE = 16   # Gorishniy et al. (2021): stop after patience + 1 consecutive epochs without improvement
TRANSFORMER_BATCH_SIZE = 256

# Only distance- and gradient-based models need imputation and scaling; tree models take raw NaN.
ML_NEEDS_SCALE = {"LinearModel", "LinearRegression", "kNN", "SVM_rbf"}
ML_IMPUTE_ONLY = {"GBR"}  # scikit-learn's GBR rejects missing values but needs no scaling


# Categorical representation modes. "integer": every ML model takes the cat_* integer codes as numbers (domain
# screening, selection audit and the integer-coding sensitivity analysis). "benchmark": main benchmark — linear
# models, kNN and SVR use a one-hot encoding fitted on the training rows (dummies are not standardized); XGBoost,
# LightGBM and CatBoost use native categorical handling (vocabulary = fixed dataset schema); DT, RF and GBR keep the
# integer codes.
CAT_MODES = ("integer", "benchmark")
ML_ONEHOT = {"LinearRegression", "LinearModel", "kNN", "SVM_rbf"}
ML_NATIVE_CAT = {"XGBoost", "LightGBM", "CatBoost"}


def ml_cat_representation(model_name: str, cat_mode: str = "integer") -> str:
    """Categorical representation of one ML model: "integer" / "onehot" / "native"."""
    assert cat_mode in CAT_MODES, cat_mode
    if cat_mode == "benchmark" and model_name in ML_ONEHOT:
        return "onehot"
    if cat_mode == "benchmark" and model_name in ML_NATIVE_CAT:
        return "native"
    return "integer"


def ml_transform(df, feature_cols, model_name, train_idx, cat_mode="integer"):
    """Transform (DataFrame -> model input DataFrame) fitted on the training rows (train_idx). integer: the existing
    fit_feature_transform. onehot: numerical columns get training-row median imputation and standardization (with the
    floor), categorical columns a OneHotEncoder fitted on the training rows (unseen levels become all zeros), dummies stay
    0/1. native: numerical columns keep their missing values, categorical columns get a category dtype with the fixed
    vocabulary of the dataset (CatBoost keeps the integers and receives them as cat_features)."""
    from sklearn.preprocessing import OneHotEncoder
    from defectshift.utils import fit_feature_transform
    rep = ml_cat_representation(model_name, cat_mode)
    if rep == "integer":
        return fit_feature_transform(df, feature_cols, ml_needs_scale(model_name), train_idx)
    cat = [c for c in feature_cols if c.startswith("cat_")]
    num = [c for c in feature_cols if c not in cat]
    if rep == "native":
        dtypes = {c: pd.CategoricalDtype(sorted(df[c].dropna().unique())) for c in cat}

        def _native(X):
            X = X[feature_cols].copy()
            if model_name != "CatBoost":
                for c in cat:
                    X[c] = X[c].astype(dtypes[c])
            return X
        return _native
    num_tf = fit_feature_transform(df, num, True, train_idx)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float).fit(df.loc[train_idx, cat])
    names = list(enc.get_feature_names_out(cat))

    def _onehot(X):
        return pd.concat([num_tf(X[num]), pd.DataFrame(enc.transform(X[cat]), index=X.index, columns=names)], axis=1)
    return _onehot


def ml_model_extra(model_name, feature_cols, cat_mode="integer") -> dict:
    """Model arguments needed for native categorical handling (not hyperparameters; not stored in tuning results)."""
    if ml_cat_representation(model_name, cat_mode) != "native":
        return {}
    if model_name == "XGBoost":
        return {"enable_categorical": True}
    if model_name == "CatBoost":
        return {"cat_features": [c for c in feature_cols if c.startswith("cat_")]}
    return {}   # LightGBM treats category-dtype columns as categorical automatically


def ml_needs_scale(model_name: str):
    """Preprocessing mode. True = training-row median imputation and standardization, "impute" = imputation only,
    False = unchanged."""
    if model_name in ML_NEEDS_SCALE:
        return True
    if model_name in ML_IMPUTE_ONLY:
        return "impute"
    return False


def fit_ml(task, model_name, df, feature_cols, target, train_idx, val_idx, best_params, seed, cat_mode="integer"):
    """Train an ML (classical or tree) model. The preprocessing (imputation, standardization, categorical
    representation) is chosen by model name and cat_mode and fitted on the training rows.

    Returns (model, X_train_df_raw, y_train_ndarray, transform). transform applies the same medians, standardization and
    categorical transform to other rows (A_heldout, B_te); regimes.predict rebuilds the same function from the training rows.
    """
    from defectshift.models.ml import fit_one
    from defectshift.utils import fit_feature_transform
    set_seed(seed)
    transform = ml_transform(df, feature_cols, model_name, train_idx, cat_mode)
    Xtr = transform(df.loc[train_idx, feature_cols])
    ytr = target.loc[train_idx]
    # models without early stopping train without a validation set (val_idx=None)
    Xval = transform(df.loc[val_idx, feature_cols]) if val_idx is not None else None
    yval = target.loc[val_idx] if val_idx is not None else None
    model = fit_one(task, model_name, {**best_params, **ml_model_extra(model_name, feature_cols, cat_mode)},
                    Xtr, ytr, Xval, yval, seed=seed)
    return model, df.loc[train_idx, feature_cols], ytr.to_numpy(), transform


def tune_ml(task, model_name, df, feature_cols, target, hpo_train_idx, hpo_val_idx, n_trials, seed=999, cat_mode="integer",
            details=None):
    """Hyperparameter search of an ML model (once per domain, reused across the repetitions). Returns best_params.

    The global random state (Python, NumPy, PyTorch) is fixed to seed before the search (the Optuna sampler seed is fixed
    separately in ml.tune_and_fit), so that the search does not depend on the order of models in the process. The
    default 999 = config.HPO_SEED."""
    from defectshift.models.ml import tune_and_fit
    set_seed(seed)
    tf = ml_transform(df, feature_cols, model_name, hpo_train_idx, cat_mode)
    Xtr, Xval = tf(df.loc[hpo_train_idx, feature_cols]), tf(df.loc[hpo_val_idx, feature_cols])
    trials = []
    _, best_params, best_score = tune_and_fit(task, model_name, Xtr, target.loc[hpo_train_idx],
                                              Xval, target.loc[hpo_val_idx], n_trials=n_trials,
                                              extra_params=ml_model_extra(model_name, feature_cols, cat_mode),
                                              trial_log=trials)
    if details is not None:   # best validation score and trial log
        details.update(best_score=best_score, trials=trials, sampler="optuna.TPESampler(seed=42)")
    return best_params


def tune_transformer(task, model_name, df, cat_cols, num_cols, target, hpo_train_idx, hpo_val_idx, n_trials, seed=999,
                     max_epochs=TRANSFORMER_MAX_EPOCHS, patience=TRANSFORMER_PATIENCE, batch_size=TRANSFORMER_BATCH_SIZE,
                     details=None):
    """Search of the MLP and FT-Transformer (transformer.tune_and_fit). The global seed is fixed at the start, the
    preprocessing is fitted on the tuning training rows (prep_features_cat_cont), validation = hpo_val_idx (early stopping
    and selection). Returns the chosen hyperparameters."""
    from defectshift.models.transformer import tune_and_fit
    set_seed(seed)
    Xtr_tup, Xval_tup = prep_features_cat_cont(df, cat_cols, num_cols, hpo_train_idx, hpo_val_idx)
    trials = []
    _, best_params, best_score = tune_and_fit(task, model_name, Xtr_tup, target.loc[hpo_train_idx].to_numpy(),
                                              Xval_tup, target.loc[hpo_val_idx].to_numpy(),
                                              cat_cardinalities(df, cat_cols), len(num_cols),
                                              max_epochs=max_epochs, patience=patience, batch_size=batch_size,
                                              n_trials=n_trials, trial_log=trials)
    if details is not None:
        details.update(best_score=best_score, trials=trials, sampler="optuna.TPESampler(seed=42)")
    return best_params


def tune_dl2(task, model_name, df, feature_cols, cat_cols, target, hpo_train_idx, hpo_val_idx, n_trials, seed=999, details=None,
             n_epochs=None):
    """Official pytabkit HPO of RealMLP and TabM (n_trials random configurations). Returns the params for regimes.fit
    = {"hpo_fit_params": the full chosen configuration, "hpo_n_steps": n_trials}."""
    from defectshift.models.tabular_dl import hpo_pytabkit
    set_seed(seed)
    r = hpo_pytabkit(model_name, df.loc[hpo_train_idx, feature_cols], target.loc[hpo_train_idx],
                     df.loc[hpo_val_idx, feature_cols], target.loc[hpo_val_idx], cat_cols, n_trials, seed, n_epochs=n_epochs)
    if details is not None:
        details.update(best_score=r["val_r2"], best_alg_idx=r["best_alg_idx"], trials=None,
                       sampler=f"pytabkit {model_name}_HPO_Regressor random search (random_state={seed})")
    return {"hpo_fit_params": r["fit_params"], "hpo_n_steps": n_trials}


def fit_transformer(task, model_name, df, cat_cols, num_cols, target, train_idx, val_idx,
                    best_params, seed,
                    max_epochs=TRANSFORMER_MAX_EPOCHS, patience=TRANSFORMER_PATIENCE,
                    batch_size=TRANSFORMER_BATCH_SIZE):
    """Train an MLP or FT-Transformer (val_idx = early-stopping rows). best_params None means DEFAULT_PARAMS.

    Returns (model, params, Xtr_tup, y_train_ndarray)
    """
    from defectshift.models.transformer import fit_default
    set_seed(seed)
    Xtr_tup, Xval_tup = prep_features_cat_cont(df, cat_cols, num_cols, train_idx, val_idx)
    cardinalities = cat_cardinalities(df, cat_cols)
    model, params = fit_default(
        task, model_name, Xtr_tup, target.loc[train_idx].to_numpy(),
        Xval_tup, target.loc[val_idx].to_numpy(), cardinalities, len(num_cols),
        params=best_params, max_epochs=max_epochs, patience=patience, batch_size=batch_size,
    )
    return model, params, Xtr_tup, target.loc[train_idx].to_numpy()


def fit_dl2(task, model_name, df, feature_cols, cat_cols, target, train_idx, val_idx, seed, verbose=False, params=None,
            n_epochs=None):
    """Train RealMLP or TabM (pytabkit; val_idx = early-stopping rows). Without params the package's tuned defaults are
    used, otherwise the configuration chosen by the HPO. n_epochs: epoch cap for smoke tests only (None = package value).

    Returns (model, params, val_score, elapsed, X_train_df, y_train_ndarray)
    """
    from defectshift.models.tabular_dl import fit_model
    set_seed(seed)
    Xtr = df.loc[train_idx, feature_cols]
    Xval = df.loc[val_idx, feature_cols]
    ytr = target.loc[train_idx]
    model, params, val_score, elapsed = fit_model(
        task, model_name, Xtr, ytr, Xval, target.loc[val_idx], cat_cols, verbose=verbose,
        random_state=seed,  # the model seed changes with repetition and regime
        params=params,      # configuration chosen by the HPO (package tuned defaults otherwise)
        n_epochs=n_epochs,
    )
    return model, params, val_score, elapsed, Xtr, ytr.to_numpy()
