#!/bin/bash
#SBATCH --job-name="train_dual_cb"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=train_dual_cb

# End-to-end dual-codebook training.
# Big VQ (1024, from pretrained checkpoint) is frozen on frame 0.
# Small VQ (CB_SIZE, randomly initialised) is trained on rest-of-frames.
# Decoder + to_pixels are also trained so they adapt to the mixed quantisation.
# Encoder + patch embeddings + big VQ stay frozen.
#
# Usage:
#   sbatch scripts/train_dual_cb.sbatch                  # default cb16, 5 games
#   CB_SIZE=64 N_ENVS=10 sbatch scripts/train_dual_cb.sbatch

set -eo pipefail

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux
cd /home/stud/nguyenti/storage/user/GenieRedux

CB_SIZE="${CB_SIZE:-16}"
N_ENVS="${N_ENVS:-0}"            # 0 = all games (120 in retro_act_v0.0.0_g200_256)
STEPS="${STEPS:-5000}"
LAMBDA_INV="${LAMBDA_INV:-0}"    # appearance-invariance aux weight (B1a)
LAMBDA_FLOW="${LAMBDA_FLOW:-0}"  # optical-flow aux weight (B1b, deferred)

# Log-readable game-count tag: "all" when N_ENVS=0, else the number.
if [ "${N_ENVS}" = "0" ]; then
  GAMES_TAG="allgames"
else
  GAMES_TAG="${N_ENVS}games"
fi

echo "=== Dual-codebook training: small_codebook_size=${CB_SIZE}, n_envs=${N_ENVS} (${GAMES_TAG}), steps=${STEPS} ==="

python train_genie_redux.py \
  config=tokenizer_256_dual_cb \
  tokenizer.small_codebook_size="${CB_SIZE}" \
  train.n_envs="${N_ENVS}" \
  train.num_train_steps="${STEPS}" \
  train.lambda_inv="${LAMBDA_INV}" \
  train.lambda_flow="${LAMBDA_FLOW}" \
  train.sample_num_frames=7 \
  train.validate_every=500 \
  train.save_results_every=500 \
  train.wandb_name="dual_cb_train_cb${CB_SIZE}_${GAMES_TAG}_inv${LAMBDA_INV}_flow${LAMBDA_FLOW}"

# Logging cadence:
#   - train scalars (loss, vq loss, recon loss, lr): every 50 steps (default)
#   - validation scalars (loss, FID, PSNR): every 500 steps
#   - visual recon samples (wandb images + on-disk gifs): every 500 steps
#   - model checkpoints: every 1000 steps (default; keeps disk use reasonable)

echo "=== DONE ==="
date
