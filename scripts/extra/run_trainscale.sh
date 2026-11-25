#!/bin/bash
# Training-row min-max scale sensitivity analysis (arm trainscale of scripts/07a_sensitivity.py).
# Usage: bash scripts/extra/run_trainscale.sh <screening-run-id> <hpo-run-id> <extra-hpo-run-id> [log-folder]
#   <screening-run-id>  run of scripts/02_select_domains.py (the selection of the challenge domains)
#   <hpo-run-id>        run of scripts/04_tune_hyperparameters.py (source-side search values of the benchmark)
#   <extra-hpo-run-id>  run of scripts/06a_benchmark_node_tabnet.py hpo (search values of TabNet)
# Stage 1 trains RealMLP, TabM and the six foundation models; stage 2 trains TabNet. Scope: the seven selected domains x repetitions
# 0-2 x ID_match and OOD_match. If a stage fails, the script stops there; running it again skips the finished trainings.
# Run it in the background (setsid nohup bash scripts/extra/run_trainscale.sh ... > trainscale.log 2>&1 &).
set -u
if [ $# -lt 3 ]; then
  echo "usage: bash scripts/extra/run_trainscale.sh <screening-run-id> <hpo-run-id> <extra-hpo-run-id> [log-folder]"
  exit 64
fi
SCREEN=$1; HPO=$2; EXTRA=$3; LOGS=${4:-logs}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PYTHON:-python}
ALLOC_ON=expandable_segments:True
mkdir -p "$LOGS"
stop() { echo "[trainscale] stopped: $1 exit code $2 $(date '+%F %T')"; exit 1; }

# Contention guard: look only at training processes whose argv[0] is python (editors and shells that merely contain the names are ignored)
if ps -eo args | grep -E "^[^ ]*python[^ ]* +(scripts/(extra/)?)?(06_benchmark|06a_benchmark_node_tabnet|07a_sensitivity|run_extra_controls|run_redrawn_target_splits|run_time_ordered)\.py" | grep -v grep > /dev/null; then
  echo "[trainscale] another training job is running; not started (avoids GPU contention)"
  exit 2
fi

cd "$ROOT"
COMMON=(--screening-run-id "$SCREEN" --hpo-run-id "$HPO" --arms trainscale)
echo "[trainscale] start $(date '+%F %T')"
echo "===== [trainscale] 1 trainscale_main start $(date '+%F %T') =====" >> "$LOGS/trainscale_main.log"
env -u PYTORCH_CUDA_ALLOC_CONF $PY scripts/07a_sensitivity.py "${COMMON[@]}" --run-id trainscale_main >> "$LOGS/trainscale_main.log" 2>&1
rc=$?; echo "[trainscale] 1 trainscale_main exit code $rc $(date '+%F %T')"; [ $rc -eq 0 ] || stop trainscale_main $rc
echo "===== [trainscale] 2 trainscale_tabnet start $(date '+%F %T') =====" >> "$LOGS/trainscale_tabnet.log"
PYTORCH_CUDA_ALLOC_CONF=$ALLOC_ON $PY scripts/07a_sensitivity.py "${COMMON[@]}" --extra-hpo-run-id "$EXTRA" --models NODE,TabNet --run-id trainscale_tabnet >> "$LOGS/trainscale_tabnet.log" 2>&1
rc=$?; echo "[trainscale] 2 trainscale_tabnet exit code $rc $(date '+%F %T')"; [ $rc -eq 0 ] || stop trainscale_tabnet $rc
echo "[trainscale] end $(date '+%F %T')"
