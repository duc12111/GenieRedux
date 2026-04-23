#!/bin/bash
#SBATCH --job-name="r1_probes"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=r1_probes

# Wave 2b: V3 (motion probe vs baselines) + V4 (flow R²) on R1's checkpoint.
# R1 = cb16 with Fix A (ema_update + kmeans_init + dead-code), no aux loss.
#
# Single job, sequential V3 then V4 (both need the same model loaded).

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

mkdir -p "$OUT_DIR"

echo "=== V3: motion probe vs baselines ==="
echo "  checkpoint: $CKPT"
echo "  out:        $OUT_DIR/v3_probe.txt"
python scripts/validate_v3_action_probe.py \
  --checkpoint "$CKPT" \
  --small-codebook-size 16 \
  --num-samples "$NUM_SAMPLES" \
  --batch-size 4 \
  --subsample "$SUBSAMPLE" \
  --out "$OUT_DIR/v3_probe.txt"

echo ""
echo "=== V4: Farnebäck flow R² ==="
echo "  checkpoint: $CKPT"
echo "  out:        $OUT_DIR/v4_flow.txt"
python scripts/validate_v4_flow_r2.py \
  --checkpoint "$CKPT" \
  --small-codebook-size 16 \
  --num-samples "$NUM_SAMPLES" \
  --batch-size 4 \
  --subsample "$SUBSAMPLE" \
  --out "$OUT_DIR/v4_flow.txt"

echo ""
echo "=== DONE ==="
date
