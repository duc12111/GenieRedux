#!/bin/bash
#SBATCH --job-name="tokenizer_cb16_scratch"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=tokenizer_cb16_scratch

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux

cd /home/stud/nguyenti/storage/user/GenieRedux && \
python train_genie_redux.py \
  config=tokenizer_256_small \
  tokenizer_fpath=null \
  tokenizer.codebook_size=16 \
  tokenizer.train_codebook_only=false \
  train.n_envs=1 \
  train.n_total_samples=50 \
  train.num_train_steps=5000 \
  train.batch_size=1 \
  train.grad_accum=16 \
  train.sample_num_frames=7 \
  train.wandb_name=tokenizer_cb16_scratch_1game
