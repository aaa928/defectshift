"""Paths and protocol constants. Every script reads its settings from here (protocol: docs/usage.md).

Paths can be overridden with environment variables (data and caches are not part of the repository):
  DEFECTSHIFT_DATA_PATH    path to data.parquet           (default: <repository>/data/data.parquet)
  DEFECTSHIFT_CACHE_DIR    split and model cache folder   (default: <repository>/cache)
  DEFECTSHIFT_RESULTS_DIR  results folder                 (default: <repository>/results)
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get("DEFECTSHIFT_DATA_PATH", os.path.join(REPO_ROOT, "data", "data.parquet"))
CACHE_DIR = os.environ.get("DEFECTSHIFT_CACHE_DIR", os.path.join(REPO_ROOT, "cache"))
RESULTS_DIR = os.environ.get("DEFECTSHIFT_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
CANDIDATES_CSV = os.path.join(REPO_ROOT, "splits", "candidates.csv")

# Protocol version, recorded in the manifest of every run. Merging results and checking the manuscript tables and
# figures accept only runs of the current version.
PROTOCOL_VERSION = "1.0"

TASK = "reg"
TARGET_COL = "y"
MIN_DOMAIN = 300

# Production lot: the grouping unit of every split (all records of a lot fall on one side). Not a model input: a new
# lot is unseen at deployment, and the lot identifier would let a model memorize sibling records.
LOT_COL = "cat_049"

# --- Source side A: A_heldout is fixed per candidate; tuning and repetition splits stay inside A_dev -------------
A_HELDOUT_FRAC = 0.20      # of A
A_HELDOUT_SEED = 2027
HPO_SEED = 999
HPO_VAL_FRAC = 0.20        # validation share of the tuning split of A_dev
A_VAL_FRAC_OF_DEV = 0.25   # A_val of repetition i = 25% of A_dev = 20% of A, so A_train = 60% of A
RANDOM_STATE = 42          # seed of the A_train/A_val split of repetition i = RANDOM_STATE + i; also the n* sampling seed

# --- Target side B: four subsets, fixed once per candidate ------------------------------------------------------
B_SPLIT_SEED = 7777
B_FRACS = dict(B_tr=0.48, B_val=0.12, B_screen=0.15, B_te=0.25)
B_TE_MIN = 500             # screening eligibility: floor(0.25 |B|) >= 500, computed from |B| without reading B_te

# --- Internal early-stopping split (early-stopping models only; same rule for A_match and B_match) -------------
ES_VAL_FRAC = 0.20
ES_SEED_BASE = 5000        # seed of repetition i = ES_SEED_BASE + i

# --- Repetitions and seeds --------------------------------------------------------------------------------------
N_REPEATS_SCREENING = 3
N_REPEATS_BENCHMARK = 5
# Model seed = RANDOM_STATE + offset + i. OOD_match_tuned shares the seed of OOD_match (same B_match, same internal
# 80/20 split, only the target-side hyperparameters differ), so their difference reflects the hyperparameters alone,
# within numerical nondeterminism.
SEED_OFFSET = dict(ID_full=0, ID_match=100, OOD_match=200, OOD_match_tuned=200)

# --- Random-split controls (negative controls): production lots assigned at random to A or B (stratified by y > 0)
# Target |B| = the 5th, 15th, ..., 95th percentiles of q_B = |B| / N over the 46 eligible process-condition
# candidates, times N (N = 14,276, rounded). Seed -> target |B|; the tests recompute these from the data. The actual
# |B| is approximate because lots are assigned whole (recorded in the result rows).
PLACEBO_B_TARGET = {9001: 2676, 9002: 3685, 9003: 4726, 9004: 5896, 9005: 6361,
                    9006: 6547, 9007: 7209, 9008: 7604, 9009: 8381, 9010: 9054}
PLACEBO_N_FOLDS = 100      # round(100 |B| / N) of 100 lot-level folds go to B, so q_B is approximated in steps of 0.01

# --- Models -----------------------------------------------------------------------------------------------------
SCREENING_MODELS = ("XGBoost", "TabPFN", "MLP", "TabM")
HPO_N_TRIALS = 30          # tuning budget of the benchmark, common to all tuned models (30 Optuna trials or 30 steps)
