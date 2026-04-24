#!/bin/bash
#SBATCH --job-name="test_c_probe"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=test_c_action_probe

# Test C: action probe on 50-game RetroAct Control subset.
# Trains a linear probe from 5 feature sources (small_vq, big_vq, encoder,
# pixel_delta, random) to ground-truth action labels.

set -eo pipefail

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux
cd /home/stud/nguyenti/storage/user/GenieRedux

CKPT="${CKPT:-checkpoints/tokenizer_dual_cb/dual_cb_train_cb64_allgames_inv0_flow0_deltanone/model-5000.pt}"
OUT_DIR="${OUT_DIR:-outputs/validation/R4_cb64}"
NUM_SAMPLES="${NUM_SAMPLES:-400}"
CB_SIZE="${CB_SIZE:-64}"
DELTA_REF="${DELTA_REF:-none}"

mkdir -p "$OUT_DIR"

echo "=== Test C: action probe on labelled subset ==="
echo "  checkpoint       : $CKPT"
echo "  cb size          : $CB_SIZE"
echo "  delta_ref        : $DELTA_REF"
echo "  num samples      : $NUM_SAMPLES"
echo "  output           : $OUT_DIR/test_c_action_probe.txt"

python scripts/test_c_action_probe.py \
  --checkpoint "$CKPT" \
  --small-codebook-size "$CB_SIZE" \
  --delta-ref "$DELTA_REF" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size 4 \
  --out "$OUT_DIR/test_c_action_probe.txt"

echo ""
echo "=== DONE ==="
date
