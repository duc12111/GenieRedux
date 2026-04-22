#!/bin/bash
#SBATCH --job-name="eval_tok_ood"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=eval_tok_ood

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux
cd /home/stud/nguyenti/storage/user/GenieRedux

# ── Realistic (ood_realistic_256) ────────────────────────────────────────────

echo "=== 1/6 tokenizer_ft_cb16_3games REALISTIC ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb16_3games/model-5000.pt \
  tokenizer.codebook_size=16 \
  eval.dataset_root_dpath=data/ood_test \
  eval.dataset_name=ood_realistic_256 \
  eval.n_envs=0 \
  eval.model_name=tokenizer_ft_cb16_3games_realistic \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== 2/6 tokenizer_ft_cb64_5games REALISTIC ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb64_5games/model-5000.pt \
  tokenizer.codebook_size=64 \
  eval.dataset_root_dpath=data/ood_test \
  eval.dataset_name=ood_realistic_256 \
  eval.n_envs=0 \
  eval.model_name=tokenizer_ft_cb64_5games_realistic \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== 3/6 tokenizer_cb16_scratch_1game REALISTIC ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_cb16_scratch_1game/model-5000.pt \
  tokenizer.codebook_size=16 \
  eval.dataset_root_dpath=data/ood_test \
  eval.dataset_name=ood_realistic_256 \
  eval.n_envs=0 \
  eval.model_name=tokenizer_cb16_scratch_1game_realistic \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

# ── Pixel Art (ood_pixelart) ─────────────────────────────────────────────────

echo "=== 4/6 tokenizer_ft_cb16_3games PIXELART ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb16_3games/model-5000.pt \
  tokenizer.codebook_size=16 \
  eval.dataset_root_dpath=data/ood_test \
  eval.dataset_name=ood_pixelart \
  eval.n_envs=0 \
  eval.model_name=tokenizer_ft_cb16_3games_pixelart \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== 5/6 tokenizer_ft_cb64_5games PIXELART ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb64_5games/model-5000.pt \
  tokenizer.codebook_size=64 \
  eval.dataset_root_dpath=data/ood_test \
  eval.dataset_name=ood_pixelart \
  eval.n_envs=0 \
  eval.model_name=tokenizer_ft_cb64_5games_pixelart \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== 6/6 tokenizer_cb16_scratch_1game PIXELART ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_cb16_scratch_1game/model-5000.pt \
  tokenizer.codebook_size=16 \
  eval.dataset_root_dpath=data/ood_test \
  eval.dataset_name=ood_pixelart \
  eval.n_envs=0 \
  eval.model_name=tokenizer_cb16_scratch_1game_pixelart \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== ALL OOD EVALS DONE ==="
date
