"""
Builders, training loop and Optuna hyperparameter search of the MLP and FT-Transformer (regression only).

  - Early stopping on the validation MSE: stop after patience + 1 consecutive epochs without improvement and restore the
    best state_dict (rule of Gorishniy et al. 2021). The restored best epoch (from 1) is kept in model._best_epoch.
  - Search: Optuna TPE sampler (seed 42), training and validation R^2 per trial (tune_and_fit).
  - Training without search uses DEFAULT_PARAMS (representative settings near the middle of the search space).

Libraries:
  - FT-Transformer: rtdl.FTTransformer (pip package `rtdl`)
  - MLP: rtdl.MLP (baseline of Gorishniy et al. 2021). It takes a single flat vector rather than (x_cat, x_cont), so
    _FlatBackboneWithEmbeddings embeds the categorical features and concatenates them with the numerical features.

GPU: used when torch.cuda.is_available(). free_memory() (gc + torch.cuda.empty_cache()) runs after every model.
"""
import os

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

import rtdl

from defectshift.utils import log, log_exception, free_memory

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPTUNA_N_TRIALS = 30

MODEL_NAMES = ["FT_Transformer", "MLP"]

# Representative settings for training without search (such as the library-default sensitivity analysis). Values of the
# search space that are missing here come from _LEGACY at build and training time.
DEFAULT_PARAMS = {
    "FT_Transformer": dict(d_token=64, n_blocks=2, lr=1e-3),
    "MLP": dict(d_layers=[256, 256], dropout=0.1, cat_emb_dim=8, lr=1e-3),
}
_LEGACY = dict(attention_dropout=0.2, ffn_dropout=0.1, residual_dropout=0.0, d_ffn_factor=2.0, weight_decay=1e-5)


def _optional(trial, name, low, high, log=False):
    """The '? distribution' notation of the official search of Gorishniy et al. 2021 (bin/tune.py): optional_<name>
    True draws from the distribution, False gives 0."""
    return trial.suggest_float(name, low, high, log=log) if trial.suggest_categorical(f"optional_{name}", [False, True]) else 0.0


def _suggest_params(name: str, trial: optuna.Trial) -> dict:
    """Optuna search space per model = Gorishniy et al. 2021, Tables 13 and 15, and their official configurations
    (output/california_housing/*/tuning/0.toml). The MLP categorical embedding size {8, 16, 32} is our setting; the
    original paper searches it only for the dataset with categorical features."""
    if name == "FT_Transformer":
        return dict(n_blocks=trial.suggest_int("n_layers", 1, 4),
                    d_token=trial.suggest_int("d_token", 64, 512, step=8),          # multiple of n_heads = 8 ('$d_token')
                    attention_dropout=trial.suggest_float("attention_dropout", 0.0, 0.5),
                    ffn_dropout=trial.suggest_float("ffn_dropout", 0.0, 0.5),
                    residual_dropout=_optional(trial, "residual_dropout", 0.0, 0.2),
                    d_ffn_factor=trial.suggest_float("d_ffn_factor", 2 / 3, 8 / 3),  # ReGLU: [1, 4] x 2/3
                    lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True))
    if name == "MLP":
        n_layers = trial.suggest_int("n_layers", 1, 8)                             # '$mlp_d_layers' 1, 8, 1, 512
        d_first = [trial.suggest_int("d_first", 1, 512)]
        d_middle = [trial.suggest_int("d_middle", 1, 512)] * (n_layers - 2) if n_layers > 2 else []
        d_last = [trial.suggest_int("d_last", 1, 512)] if n_layers > 1 else []
        return dict(d_layers=d_first + d_middle + d_last,
                    dropout=_optional(trial, "dropout", 0.0, 0.5),
                    cat_emb_dim=trial.suggest_categorical("cat_emb_dim", [8, 16, 32]),
                    lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                    weight_decay=_optional(trial, "weight_decay", 1e-6, 1e-3, log=True))
    raise ValueError(name)


class _FlatBackboneWithEmbeddings(nn.Module):
    """Thin wrapper for rtdl.MLP, which takes one flat vector instead of (x_cat, x_cont): one small nn.Embedding per
    categorical column, concatenated with the numerical features and passed to the backbone (rtdl.MLP). Unlike the
    FT-Transformer, which turns every feature into a token for attention, this model only embeds, concatenates and applies
    an MLP, the plain baseline against which attention is compared."""

    def __init__(self, cardinalities, cat_emb_dim: int, backbone: nn.Module):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, cat_emb_dim) for card in cardinalities])
        self.backbone = backbone

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_cont], dim=1)
        return self.backbone(x)


def build_one_model(name: str, params: dict, cardinalities, num_continuous: int):
    """One untrained PyTorch model with the hyperparameters params (d_out=1, raw regression output)."""
    if name == "FT_Transformer":
        p = {**_LEGACY, **params}
        return rtdl.FTTransformer.make_baseline(
            n_num_features=num_continuous, cat_cardinalities=list(cardinalities),
            d_token=p["d_token"], n_blocks=p["n_blocks"],
            attention_dropout=p["attention_dropout"], ffn_d_hidden=int(p["d_token"] * p["d_ffn_factor"]),
            ffn_dropout=p["ffn_dropout"], residual_dropout=p["residual_dropout"], d_out=1,
        )
    if name == "MLP":
        d_in = len(cardinalities) * params["cat_emb_dim"] + num_continuous
        backbone = rtdl.MLP.make_baseline(d_in=d_in, d_layers=params["d_layers"], dropout=params["dropout"], d_out=1)
        return _FlatBackboneWithEmbeddings(cardinalities=list(cardinalities), cat_emb_dim=params["cat_emb_dim"],
                                           backbone=backbone)
    raise ValueError(name)


def _forward_ft(model, x_cat, x_cont):
    return model(x_cont, x_cat).squeeze(-1)


def _forward_flat_backbone(model, x_cat, x_cont):
    return model(x_cont, x_cat).squeeze(-1)


FORWARD_FNS = {"FT_Transformer": _forward_ft, "MLP": _forward_flat_backbone}


def _to_tensor(x, dtype, device):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _forward_in_chunks(forward_fn, model, x_cat, x_cont, chunk_size):
    """Forward the validation or evaluation rows in chunks of the training batch size and concatenate; a single forward
    over all rows makes attention grow with batch x tokens^2 and easily runs out of memory."""
    outs = [forward_fn(model, x_cat[i:i + chunk_size], x_cont[i:i + chunk_size])
            for i in range(0, x_cat.shape[0], chunk_size)]
    return torch.cat(outs, dim=0)


def fit_with_val(task: str, name: str, model, Xtr, ytr, Xval, yval,
                  lr: float, weight_decay: float = _LEGACY["weight_decay"], max_epochs: int = 150, patience: int = 16,
                  batch_size: int = 256):
    """Xtr/Xval are (X_cat, X_cont) tuples. AdamW, MSE, no learning-rate schedule, restore of the best state_dict
    (Gorishniy et al. 2021). Weight decay: the FT-Transformer uses rtdl's optimization_param_groups (no decay on the
    tokenizer, normalization and biases, the rule of the official bin/ft_transformer.py); the MLP decays all parameters
    (official bin/mlp.py). Early stopping: any decrease of the validation MSE is an improvement; training stops after
    patience + 1 consecutive epochs without improvement (official zero.ProgressTracker(patience=16)). max_epochs is a
    computational cap that the original paper does not have. The restored best epoch (from 1) is kept in model._best_epoch
    and the number of trained epochs in model._n_epochs."""
    assert task == "reg", "regression only"
    model.to(DEVICE)
    forward_fn = FORWARD_FNS[name]
    groups = model.optimization_param_groups() if name == "FT_Transformer" else model.parameters()
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xtr_cat_t = _to_tensor(Xtr[0], torch.long, DEVICE)
    Xtr_cont_t = _to_tensor(Xtr[1], torch.float32, DEVICE)
    ytr_t = _to_tensor(np.asarray(ytr, dtype="float32"), torch.float32, DEVICE)
    Xval_cat_t = _to_tensor(Xval[0], torch.long, DEVICE)
    Xval_cont_t = _to_tensor(Xval[1], torch.float32, DEVICE)
    yval_t = _to_tensor(np.asarray(yval, dtype="float32"), torch.float32, DEVICE)

    n = len(ytr_t)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    n_bad = 0
    n_epochs = 0
    val_hist = []
    for epoch in range(max_epochs):
        n_epochs = epoch + 1
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(forward_fn(model, Xtr_cat_t[idx], Xtr_cont_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_out = _forward_in_chunks(forward_fn, model, Xval_cat_t, Xval_cont_t, batch_size)
            val_loss = loss_fn(val_out, yval_t).item()
        val_hist.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            n_bad = 0
        else:
            n_bad += 1
            if n_bad > patience:
                log(f"  [{name}] epoch {epoch+1}: early stop (best val_loss={best_val_loss:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model._best_epoch = best_epoch
    model._n_epochs = n_epochs
    model._val_hist = val_hist
    return model


def tail_gain(val_hist, k=30):
    """Decrease of the lowest validation MSE (standardized y, >= 0) during the last k epochs; None if at most k epochs
    were trained. Shows whether a trial that reached the epoch cap was still improving."""
    if len(val_hist) <= k:
        return None
    return float(min(val_hist[:-k]) - min(val_hist))


def fit_default(task: str, name: str, Xtr, ytr, Xval, yval, cardinalities, num_continuous,
                 params: dict = None, max_epochs: int = 150, patience: int = 16, batch_size: int = 256):
    """Train once with fixed hyperparameters, without search (the validation set serves early stopping only). Without
    params, DEFAULT_PARAMS[name]. Exceptions propagate."""
    if params is None:
        params = DEFAULT_PARAMS[name]
    try:
        model = build_one_model(name, params, cardinalities, num_continuous)
        fit_with_val(task, name, model, Xtr, ytr, Xval, yval, lr=params["lr"],
                     weight_decay=params.get("weight_decay", _LEGACY["weight_decay"]),
                     max_epochs=max_epochs, patience=patience, batch_size=batch_size)
        return model, params
    finally:
        free_memory()


def predict_scores(name: str, model, X, chunk_size: int = 256):
    """X=(X_cat, X_cont). Raw regression output."""
    model.eval()
    with torch.no_grad():
        x_cat = _to_tensor(X[0], torch.long, DEVICE)
        x_cont = _to_tensor(X[1], torch.float32, DEVICE)
        out = _forward_in_chunks(FORWARD_FNS[name], model, x_cat, x_cont, chunk_size)
    return out.cpu().numpy()


def predict_final(task: str, name: str, model, X):
    """(pred, pred), the same return shape as predict_final of the other model families."""
    assert task == "reg", "regression only"
    raw = predict_scores(name, model, X)
    return raw, raw


def _fit_and_score(task: str, name: str, params: dict, Xtr, ytr, Xval, yval, cardinalities, num_continuous,
                    max_epochs: int, patience: int, batch_size: int):
    """Train one Optuna trial and return its validation R^2. On failure, log the exception and return None (only that
    trial is skipped)."""
    try:
        model = build_one_model(name, params, cardinalities, num_continuous)
        fit_with_val(task, name, model, Xtr, ytr, Xval, yval, lr=params["lr"],
                     weight_decay=params.get("weight_decay", _LEGACY["weight_decay"]),
                     max_epochs=max_epochs, patience=patience, batch_size=batch_size)
        return dict(model=model, params=params, score=r2_score(yval, predict_scores(name, model, Xval)))
    except Exception:
        log_exception(f"[{task}][{name}] params={params} training failed, skipping this trial")
        return None
    finally:
        free_memory()


def tune_and_fit(task: str, name: str, Xtr, ytr, Xval, yval, cardinalities, num_continuous,
                  max_epochs: int = 150, patience: int = 16, batch_size: int = 256,
                  n_trials: int = OPTUNA_N_TRIALS, trial_log=None):
    """Search n_trials times with Optuna (TPE, seed 42) and return the (model, hyperparameters, score) with the best
    validation R^2. Trials share one GPU in sequence and are not parallelized; a failed trial does not stop the others.
    Only the best trial is kept in memory. With trial_log (a list), every trial's number, state, validation score,
    seconds and hyperparameters are appended."""
    best_holder = {}

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(name, trial)
        r = _fit_and_score(task, name, params, Xtr, ytr, Xval, yval, cardinalities, num_continuous,
                            max_epochs, patience, batch_size)
        if r is None:
            raise optuna.TrialPruned()
        # to check whether a trial reached the epoch cap
        trial.set_user_attr("n_epochs", r["model"]._n_epochs)
        trial.set_user_attr("best_epoch", r["model"]._best_epoch)
        trial.set_user_attr("tail_gain_30", tail_gain(r["model"]._val_hist))
        score = r["score"]
        if not best_holder or score > best_holder["score"]:
            best_holder.clear()
            best_holder.update(r)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    if trial_log is not None:
        from defectshift.models.ml import _trial_records
        trial_log.extend(_trial_records(study))

    if not best_holder:
        raise RuntimeError(f"[{task}][{name}] all trials failed (see the ERROR lines in the log)")

    return best_holder["model"], best_holder["params"], best_holder["score"]


def save_model(name: str, model, params: dict, cardinalities, num_continuous: int, path: str):
    """Save the state_dict with the metadata needed to rebuild the model (model name, hyperparameters, cardinalities,
    number of numerical features); the object itself is not pickled, so that checkpoints survive library updates."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(
        name=name, params=params, cardinalities=list(cardinalities),
        num_continuous=num_continuous, state_dict=model.state_dict(),
    ), path)
    log(f"[Model saved] -> {path}")


def load_model(path: str):
    """Rebuild a model from a checkpoint written by save_model. Returns (model, params)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model = build_one_model(ckpt["name"], ckpt["params"], ckpt["cardinalities"], ckpt["num_continuous"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    return model, ckpt["params"]
