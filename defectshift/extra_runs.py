"""Shared rules of the NODE/TabNet runner (scripts/06a_benchmark_node_tabnet.py), also used by the sensitivity runner:
the lineage record of the two models (their code, the official code, the additional packages, the TensorFlow 1 environment, the
CUDA allocator setting) and loading a NODE/TabNet tuning run.
"""
import importlib.metadata as md
import json
import os
import subprocess

from defectshift import config, manifest
from defectshift.models import node, tabnet
from defectshift.runio import check_compat, check_same_inputs

EXTRA_FILES = ("defectshift/models/extra_models.py", "defectshift/models/node.py",
               "defectshift/models/tabnet.py", "defectshift/models/tabnet_tf1_worker.py")
EXTRA_PKGS = ("qhoptim", "category-encoders", "tensorboardX")


def _pkg(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _sha_files(root, rels):
    return {r: manifest.file_sha256(os.path.join(root, r)) for r in rels}


def lineage():
    """Record of the two models' code, the official code, the additional packages and the TF environment (tuning and benchmark must match)."""
    node_lib = sorted(f"lib/{f}" for f in os.listdir(os.path.join(node.NODE_DIR, "lib")) if f.endswith(".py"))
    tf = subprocess.run([tabnet.TF_PYTHON, "-c", "import sys, numpy, tensorflow as tf; "
                         "print(sys.version.split()[0], numpy.__version__, tf.__version__)"],
                        capture_output=True, text=True, env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3"))
    return dict(extra_code_sha256=_sha_files(os.path.realpath(config.REPO_ROOT), EXTRA_FILES),
                vendor_sha256=dict(node=_sha_files(node.NODE_DIR, node_lib),
                                   tabnet=_sha_files(tabnet.TABNET_DIR, ["tabnet_model.py"])),
                vendor_commit=dict(node=node.NODE_COMMIT, tabnet=tabnet.TABNET_COMMIT),
                extra_packages={p: _pkg(p) for p in EXTRA_PKGS},
                tabnet_env=dict(python_path=tabnet.TF_PYTHON, versions=tf.stdout.strip() or None,
                                error=(tf.stderr[-300:] if tf.returncode else None)),
                cuda_env=dict(PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                              CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG")))


def load_extra_hpo(hpo_run_id, final_sha, strict):
    path = os.path.join(config.RESULTS_DIR, "benchmark", "hpo", hpo_run_id, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist (run 06a_benchmark_node_tabnet.py hpo first)")
    man = json.load(open(path))
    a = man["args"]
    if a.get("stage") != "benchmark_tuning_extra":
        raise SystemExit(f"{hpo_run_id} is not a NODE/TabNet tuning run (stage {a.get('stage')})")
    if a.get("screening_final_domains_sha256") != final_sha:
        raise SystemExit("the tuning run comes from a different domain selection (final_domains.json hash differs)")
    check_same_inputs(man, f"tuning run {hpo_run_id}")
    if strict:   # shared training code, packages and protocol constants (runio) + the two models' code, official code, additional packages and TF environment (lineage)
        check_compat(man, f"tuning run {hpo_run_id}")
        cur = lineage()
        bad = [k for k in ("extra_code_sha256", "vendor_sha256", "vendor_commit", "extra_packages", "tabnet_env", "cuda_env")
               if a.get(k) != cur[k]]   # cuda_env: tuning and training (benchmark, sensitivity) use the same allocator setting
        if bad:
            raise SystemExit(f"tuning run {hpo_run_id}: differs from the current code and environment of the two models {bad}")
    return a, manifest.file_sha256(path)
