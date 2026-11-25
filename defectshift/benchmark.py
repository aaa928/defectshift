"""Shared rules of the final benchmark (scripts/06_benchmark.py), also used by the NODE/TabNet runner, the
sensitivity runner and the aggregation: regimes and record keys, artifact paths, loading of the tuning
and foundation-qualification runs, training rows per regime and their checks, early-stopping rounds, prediction-file checks,
and memory measurements.
"""
import json
import os

import numpy as np
import psutil
import torch

from defectshift import config, hpo, manifest
from defectshift.candidates import slot_dir
from defectshift.models import foundation as fm
from defectshift.runio import FMQ_COMPAT_FILES, FMQ_COMPAT_PKGS, check_compat, check_same_inputs
from defectshift.splits import es_split

REGIMES = ("ID_full", "ID_match", "OOD_match", "OOD_match_tuned")
KEY = ("slot", "repeat", "model", "regime")


def artifact_prefix(model_dir, slot, model, regime, i):
    return os.path.join(model_dir, slot_dir(slot), f"{model}__{regime}__r{i}")


def load_hpo(hpo_run_id, final_sha, need_compat=True):
    """Manifest (args) and hash of a tuning run. It must come from the same domain selection (hash of final_domains.json)
    and from the same data and candidate list as now."""
    path = os.path.join(config.RESULTS_DIR, "benchmark", "hpo", hpo_run_id, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist (run the tuning first)")
    man = json.load(open(path))
    args = man["args"]
    if args.get("screening_final_domains_sha256") != final_sha:
        raise SystemExit("the tuning run comes from a different domain selection (final_domains.json hash differs)")
    check_same_inputs(man, f"tuning run {hpo_run_id}")
    if need_compat:   # search spaces, preprocessing, y handling, training functions and packages must match to reuse the chosen values (skipped for smoke runs)
        check_compat(man, f"tuning run {hpo_run_id}")
    return args, manifest.file_sha256(path)


# Check of a search record (model, side, domain, final budget, no epoch cap, hashes of the search rows). need_trials = None for smoke runs (--allow-nonfinal-hpo)
check_hpo_record = hpo.check_record


def check_fmq_weights(man, what):
    """The qualification run (its args, man) must have used the weight files that environment/weights.json pins now (repository, commit and
    sha256 of every file, per model of the run); a qualification made with other weights is refused. Raises SystemExit."""
    rec = man.get("foundation_weights")
    if rec is None:
        raise SystemExit(f"{what}: the manifest has no foundation_weights (run of an earlier protocol version)")
    models = man.get("models") or list(rec)
    bad = [n for n in models if n not in rec or rec[n] != fm.weights_record(n)]
    if bad:
        raise SystemExit(f"{what}: the foundation-model weights of {bad} differ from the pinned weights (environment/weights.json); rerun the qualification")


def load_fmq(fmq_run_id, final_sha):
    """Qualification verdicts (model -> passed) and the summary hash. The run must come from the same domain selection, and the
    three judged models (fm.QUALIFY_JUDGED) must all be in the summary (a missing one stops the run instead of counting as failed)."""
    run_dir = os.path.join(config.RESULTS_DIR, "benchmark", "fm_qualify", fmq_run_id)
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist (run the foundation-model qualification first)")
    full_man = json.load(open(os.path.join(run_dir, "manifest.json")))
    man = full_man["args"]
    if man.get("screening_final_domains_sha256") != final_sha:
        raise SystemExit("the qualification run comes from a different domain selection (final_domains.json hash differs)")
    # The memory, time and reproducibility verdicts depend on the device and the foundation packages: the qualification must be from the same GPU and packages
    cur_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if full_man.get("gpu") != cur_gpu:       # a qualification without a GPU has no memory measurement and already fails in 05_qualify_foundation
        raise SystemExit(f"qualification GPU ({full_man.get('gpu')}) differs from the current GPU ({cur_gpu})")
    # Input configuration, foundation wrappers, verdict code (per-file hashes), foundation packages and protocol constants must match to use the verdicts
    check_compat(full_man, f"qualification {fmq_run_id}", files=FMQ_COMPAT_FILES, pkgs=FMQ_COMPAT_PKGS)
    check_fmq_weights(man, f"qualification {fmq_run_id}")
    res = json.load(open(path))["results"]
    missing = [m for m in fm.QUALIFY_JUDGED if m not in res]
    if missing:
        raise SystemExit(f"judged models {missing} are missing from the qualification summary (rerun with all six)")
    return {k: bool(v["passed"]) for k, v in res.items()}, manifest.file_sha256(path)


def in_roster(spec, fm_pass):
    """Non-foundation models are included. Of the foundation models, the three judged ones by their verdict, the others (TabPFN,
    TabICL, TabFM) always."""
    return spec.route != "foundation" or spec.name not in fm.QUALIFY_JUDGED or fm_pass.get(spec.name, False)


def regime_rows(regime, spec, a_train, a_val, a_match, b_match, bv, y, lot, i):
    """(training rows, early-stopping rows) of a regime."""
    if regime == "ID_full":
        return (a_train, a_val) if spec.uses_es else (a_train, None)
    pool = a_match if regime == "ID_match" else b_match      # OOD_match_tuned uses the same rows as OOD_match
    return es_split(pool, y, lot, i) if spec.uses_es else (pool, None)


def check_rows(regime, fit_idx, es_idx, a, bv):
    """The evaluation rows (A_heldout, B_te) are not used for training, and the rows are within the range allowed for the regime.
    Explicit raises, not assert statements, so that the leakage guard also runs under python -O."""
    used = set(fit_idx) | (set() if es_idx is None else set(es_idx))
    if used & (set(a["A_heldout"]) | set(bv["B_te"])):
        raise AssertionError(f"{regime}: training rows overlap the evaluation rows")
    if regime in ("ID_full", "ID_match"):
        if not used <= set(a["A_dev"]):
            raise AssertionError(f"{regime}: rows outside A_dev")
    else:   # OOD_match, OOD_match_tuned: B_match within B_tr only (B_val serves the target-side search only)
        if not used <= set(bv["B_tr"]):
            raise AssertionError(f"{regime}: rows outside B_tr")


def rounds_used(cfg):
    """Number of iterations actually used after early stopping, as a count (libraries differ in counting from 0 or 1). (count,
    unit) or (None, None). XGBoost best_iteration counts from 0 (count = +1), LightGBM best_iteration_ from 1 (0 = early stopping
    not triggered -> not recorded), CatBoost best_iteration_ from 0 (+1). MLP and FT-Transformer: the restored best epoch
    (from 1, recorded by regimes)."""
    def num(k):
        v = cfg.get(k)
        return int(v) if isinstance(v, (int, float)) and v == v else None
    impl = cfg.get("impl")
    if impl == "XGBoost" and num("best_iteration") is not None:
        return num("best_iteration") + 1, "boosting rounds"
    if impl == "LightGBM" and num("best_iteration_"):
        return num("best_iteration_"), "boosting rounds"
    if impl == "CatBoost" and num("best_iteration_") is not None:
        return num("best_iteration_") + 1, "boosting rounds"
    if cfg.get("route") == "transformer" and num("best_epoch") is not None:
        return num("best_epoch"), "epochs"
    return None, None


PRED_KEYS = ("id_idx", "id_pred", "ood_idx", "ood_pred")


def pred_ok(prefix):
    """Whether the prediction file opens, has every key, has index and prediction arrays of equal length, and only finite
    predictions."""
    try:
        z = np.load(prefix + ".pred.npz")
        if not set(PRED_KEYS) <= set(z.files):
            return False
        return (len(z["id_idx"]) == len(z["id_pred"]) and len(z["ood_idx"]) == len(z["ood_pred"])
                and bool(np.isfinite(z["id_pred"]).all()) and bool(np.isfinite(z["ood_pred"]).all()))
    except Exception:
        return False


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mb():
    """Peak torch allocation (MB). Without a GPU it cannot be measured, so the value is empty (not 0)."""
    return round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None


def _mem():
    gpu_used = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu_used = round((total - free) / 2**20)
    return dict(gpu_used_mb=gpu_used, cpu_rss_mb=round(psutil.Process().memory_info().rss / 2**20))
