#!/bin/bash
#SBATCH --job-name="eval_dual_cb_ft_3games"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=eval_dual_cb_ft_3games

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux
cd /home/stud/nguyenti/storage/user/GenieRedux

python eval_genie_redux.py \
  config=eval_dual_codebook \
  tokenizer_fpath=checkpoints/genie_redux_guilded/model-100000.pt \
  eval.small_tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb16_3games/model-5000.pt \
  eval.dataset_root_dpath=data_generation/datasets \
  eval.dataset_name=retro_act_v0.0.0_g200_256 \
  eval.n_envs=20 \
  eval.batch_size=16 \
  eval.sample_num_frames=7 \
  eval.model_name=dual_cb_ft_3games \
  eval.visualize_token_ids=true

echo "=== DONE ==="
date

