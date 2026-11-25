"""Shared rules of the sensitivity runner (scripts/07a_sensitivity.py), also used by the sensitivity report
(scripts/07c_sensitivity_report.py): the names of the NODE preprocessing arms and the paths of predictions and search records.
"""
import os

from defectshift import config
from defectshift.candidates import slot_dir

NODE_PREP_ARM = "node_prep_a"
NODE_C_CHECK_ARM = "node_c_check"


def pred_prefix(run_id, slot, model, regime, arm, i):
    return os.path.join(config.CACHE_DIR, "sensitivity", "runs", run_id, "preds", slot_dir(slot), f"{model}__{regime}__{arm}__r{i}")


def hpo_prefix(run_id, slot, model, split_seed):
    return os.path.join(config.CACHE_DIR, "sensitivity", "runs", run_id, "hpo", slot_dir(slot), f"{model}__source__split{split_seed}")
