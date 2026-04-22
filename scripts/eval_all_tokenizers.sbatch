#!/bin/bash
#SBATCH --job-name="eval_all_tokenizers"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=eval_all_tokenizers

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux
cd /home/stud/nguyenti/storage/user/GenieRedux

echo "=== 1/6 tokenizer_ft_cb16_3games IN-DISTRIBUTION ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb16_3games/model-5000.pt \
  tokenizer.codebook_size=16 \
  eval.dataset_root_dpath=data_generation/datasets \
  eval.dataset_name=retro_act_v0.0.0_g200_256_single \
  eval.model_name=tokenizer_ft_cb16_3games \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== 2/6 tokenizer_ft_cb16_3games OOD ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb16_3games/model-5000.pt \
  tokenizer.codebook_size=16 \
  eval.dataset_root_dpath=data_generation/datasets \
  eval.dataset_name=retro_act_v0.0.0_g200_256 \
  eval.n_envs=5 \
  eval.model_name=tokenizer_ft_cb16_3games_ood \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== 3/6 tokenizer_ft_cb64_5games IN-DISTRIBUTION ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb64_5games/model-5000.pt \
  tokenizer.codebook_size=64 \
  eval.dataset_root_dpath=data_generation/datasets \
  eval.dataset_name=retro_act_v0.0.0_g200_256_single \
  eval.model_name=tokenizer_ft_cb64_5games \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== 4/6 tokenizer_ft_cb64_5games OOD ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb64_5games/model-5000.pt \
  tokenizer.codebook_size=64 \
  eval.dataset_root_dpath=data_generation/datasets \
  eval.dataset_name=retro_act_v0.0.0_g200_256 \
  eval.n_envs=5 \
  eval.model_name=tokenizer_ft_cb64_5games_ood \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== 5/6 tokenizer_cb16_scratch_1game IN-DISTRIBUTION ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_cb16_scratch_1game/model-5000.pt \
  tokenizer.codebook_size=16 \
  eval.dataset_root_dpath=data_generation/datasets \
  eval.dataset_name=retro_act_v0.0.0_g200_256_single \
  eval.model_name=tokenizer_cb16_scratch_1game \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== 6/6 tokenizer_cb16_scratch_1game OOD ==="
python eval_genie_redux.py \
  config=tokenizer_256_small \
  model=tokenizer \
  tokenizer_fpath=checkpoints/tokenizer/tokenizer_cb16_scratch_1game/model-5000.pt \
  tokenizer.codebook_size=16 \
  eval.dataset_root_dpath=data_generation/datasets \
  eval.dataset_name=retro_act_v0.0.0_g200_256 \
  eval.n_envs=5 \
  eval.model_name=tokenizer_cb16_scratch_1game_ood \
  eval.batch_size=4 \
  eval.sample_num_frames=7 \
  eval.visualize_token_ids=true

echo "=== ALL EVALS DONE ==="
date
