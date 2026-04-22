#!/bin/bash
#SBATCH --job-name="dual_cb_sweep_train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=dual_cb_sweep_train

# Dual-codebook controlled sweep: train a small VQ codebook with frozen encoder/decoder.
#
# Parameterised via env vars so the same script works for all codebook sizes:
#   CB_SIZE=64 sbatch scripts/train_dual_cb_sweep.sbatch
#
# Or submit all sizes at once:
#   for s in 8 16 32 64 128 256; do CB_SIZE=$s sbatch scripts/train_dual_cb_sweep.sbatch; done
#
# All runs use identical settings except codebook size:
#   - Pretrained encoder/decoder from the 1024-code big tokenizer (frozen)
#   - train_codebook_only=true  (only VQ project_in, project_out, codebook train)
#   - 5 games, 5000 steps, batch 1 x grad_accum 16

set -eo pipefail

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux
cd /home/stud/nguyenti/storage/user/GenieRedux

CB_SIZE="${CB_SIZE:-64}"

echo "=== Training dual-codebook sweep: codebook_size=${CB_SIZE} ==="

python train_genie_redux.py \
  config=tokenizer_256_small \
  tokenizer_fpath=checkpoints/genie_redux_guilded/model-100000.pt \
  tokenizer.codebook_size="${CB_SIZE}" \
  tokenizer.train_codebook_only=true \
  train.n_envs=5 \
  train.n_total_samples=50 \
  train.num_train_steps=5000 \
  train.batch_size=1 \
  train.grad_accum=16 \
  train.sample_num_frames=7 \
  train.wandb_name="dual_cb_sweep_cb${CB_SIZE}_5games"

echo "=== Training DONE: cb${CB_SIZE} ==="
date
