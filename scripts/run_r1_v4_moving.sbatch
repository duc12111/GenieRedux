#!/bin/bash
#SBATCH --job-name="r1_v4_moving"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=r1_v4_moving

# V4 with moving-only filter on R1's checkpoint.
# Previous V4 run on R1 reported R² ≈ 0.003 for all features including
# pixel_delta, which was static-patch swamping (85.6% of patches have
# ~zero flow, linear regressor predicts (0,0) constant). This run restricts
# to patches with |flow| > 0.2 to measure the real direction-prediction signal.

set -eo pipefail

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux
cd /home/stud/nguyenti/storage/user/GenieRedux

CKPT="${CKPT:-checkpoints/tokenizer_dual_cb/dual_cb_train_cb16_allgames_inv0_flow0/model-5000.pt}"
OUT_DIR="${OUT_DIR:-outputs/validation/R1_inv0}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
SUBSAMPLE="${SUBSAMPLE:-80000}"
THRESHOLD="${THRESHOLD:-0.2}"

mkdir -p "$OUT_DIR"

echo "=== V4 (moving-only, threshold=$THRESHOLD): flow R² ==="
echo "  checkpoint: $CKPT"
echo "  out:        $OUT_DIR/v4_flow_moving.txt"
python scripts/validate_v4_flow_r2.py \
  --checkpoint "$CKPT" \
  --small-codebook-size 16 \
  --num-samples "$NUM_SAMPLES" \
  --batch-size 4 \
  --subsample "$SUBSAMPLE" \
  --moving-only-threshold "$THRESHOLD" \
  --out "$OUT_DIR/v4_flow_moving.txt"

echo ""
echo "=== DONE ==="
date
