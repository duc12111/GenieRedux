#!/bin/bash
# Submit all 6 dual-codebook sweep training jobs, then all 6 eval jobs.
#
# Usage:
#   bash scripts/launch_dual_cb_sweep.sh train       # submit training only
#   bash scripts/launch_dual_cb_sweep.sh eval        # submit eval only
#   bash scripts/launch_dual_cb_sweep.sh all         # submit both (eval depends on train)

set -eo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-all}"
CB_SIZES=(8 16 32 64 128 256)

submit_train() {
  echo "Submitting training jobs..."
  local TRAIN_JIDS=()
  for CB_SIZE in "${CB_SIZES[@]}"; do
    JID=$(CB_SIZE=${CB_SIZE} sbatch --parsable \
      --job-name="dual_cb_train_cb${CB_SIZE}" \
      --comment="dual_cb_sweep_cb${CB_SIZE}" \
      scripts/train_dual_cb_sweep.sbatch)
    TRAIN_JIDS+=("${JID}")
    echo "  cb${CB_SIZE} -> job ${JID}"
  done
  echo "${TRAIN_JIDS[@]}"
}

submit_eval() {
  local DEPENDENCY="$1"
  echo "Submitting evaluation jobs..."
  for CB_SIZE in "${CB_SIZES[@]}"; do
    EXTRA_ARGS=()
    if [ -n "${DEPENDENCY}" ]; then
      EXTRA_ARGS+=(--dependency="afterok:${DEPENDENCY}")
    fi
    JID=$(CB_SIZE=${CB_SIZE} sbatch --parsable \
      --job-name="dual_cb_eval_cb${CB_SIZE}" \
      --comment="dual_cb_sweep_eval_cb${CB_SIZE}" \
      "${EXTRA_ARGS[@]}" \
      scripts/eval_dual_cb_sweep.sbatch)
    echo "  cb${CB_SIZE} -> job ${JID}"
  done
}

case "${MODE}" in
  train)
    submit_train
    ;;
  eval)
    submit_eval ""
    ;;
  all)
    JIDS_STR=$(submit_train | tail -1)
    DEP=$(echo "${JIDS_STR}" | tr ' ' ':')
    echo ""
    submit_eval "${DEP}"
    ;;
  *)
    echo "Usage: $0 {train|eval|all}"
    exit 1
    ;;
esac

echo ""
echo "Monitor with: squeue -u \$USER"
