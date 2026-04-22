#!/bin/bash
#SBATCH --job-name="dual_cb_analysis_scratch1g"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=dual_cb_analysis_scratch1game

# Dual-codebook analysis aligned with tokenizer_cb16_scratch_1game:
# - Dataset retro_act_v0.0.0_g200_256_single has exactly one env on disk:
#     retro_coolspot-snes_v0.0.0_g200_256_single  ->  whitelist key: coolspot-snes
# - Invariance + same_game use that single-game dataset (matches eval under
#   outputs/evaluation/retro_act_v0.0.0_g200_256_single/tokenizer_cb16_scratch_1game/...).
# - Cross-game uses the multi-game dataset so a second env exists for B.

set -eo pipefail

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux
cd /home/stud/nguyenti/storage/user/GenieRedux

set -u

BIG_TOKENIZER_FPATH="${BIG_TOKENIZER_FPATH:-checkpoints/genie_redux_guilded/model-100000.pt}"
SMALL_TOKENIZER_FPATH="${SMALL_TOKENIZER_FPATH:-checkpoints/tokenizer/tokenizer_cb16_scratch_1game/model-5000.pt}"
DATASET_ROOT_DPATH="${DATASET_ROOT_DPATH:-data_generation/datasets}"
DATASET_SINGLE="${DATASET_SINGLE:-retro_act_v0.0.0_g200_256_single}"
DATASET_MULTI="${DATASET_MULTI:-retro_act_v0.0.0_g200_256}"
SEED="${SEED:-0}"
SMALL_CODEBOOK_SIZE="${SMALL_CODEBOOK_SIZE:-16}"

WHITELIST_PRETRAIN="${WHITELIST_PRETRAIN:-coolspot-snes}"
WHITELIST_OTHER="${WHITELIST_OTHER:-addamsfamily-nes}"

OUT_BASE="${OUT_BASE:-outputs/analysis/dual_cb_scratch1game_run${SLURM_JOB_ID:-}}"
INV_DIR="${OUT_BASE}/invariance_colorjitter"
SWAP_SAME_DIR="${OUT_BASE}/swap_same_game"
SWAP_CROSS_DIR="${OUT_BASE}/swap_cross_game"

echo "=== Experiment #2: color-jitter invariance (ID: ${WHITELIST_PRETRAIN} on ${DATASET_SINGLE}) ==="
python scripts/analyze_dual_codebook_invariance_colorjitter.py \
  --big-tokenizer-fpath "${BIG_TOKENIZER_FPATH}" \
  --small-tokenizer-fpath "${SMALL_TOKENIZER_FPATH}" \
  --small-codebook-size "${SMALL_CODEBOOK_SIZE}" \
  --dataset-root-dpath "${DATASET_ROOT_DPATH}" \
  --dataset-name "${DATASET_SINGLE}" \
  --num-samples 50 \
  --n-envs 1 \
  --motion-frames 7 \
  --jitter medium \
  --seed "${SEED}" \
  --transition-entropy \
  --out-dir "${INV_DIR}"

echo "=== Experiment #3a: code swap same_game (${WHITELIST_PRETRAIN}) ==="
python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \
  --big-tokenizer-fpath "${BIG_TOKENIZER_FPATH}" \
  --small-tokenizer-fpath "${SMALL_TOKENIZER_FPATH}" \
  --small-codebook-size "${SMALL_CODEBOOK_SIZE}" \
  --dataset-root-dpath "${DATASET_ROOT_DPATH}" \
  --dataset-name "${DATASET_SINGLE}" \
  --mode same_game \
  --whitelist-env "${WHITELIST_PRETRAIN}" \
  --num-pairs 20 \
  --seed "${SEED}" \
  --out-dir "${SWAP_SAME_DIR}"

echo "=== Experiment #3b: code swap cross_game (A=pretrain game, B=other) ==="
python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \
  --big-tokenizer-fpath "${BIG_TOKENIZER_FPATH}" \
  --small-tokenizer-fpath "${SMALL_TOKENIZER_FPATH}" \
  --small-codebook-size "${SMALL_CODEBOOK_SIZE}" \
  --dataset-root-dpath "${DATASET_ROOT_DPATH}" \
  --dataset-name "${DATASET_MULTI}" \
  --mode cross_game \
  --whitelist-env-a "${WHITELIST_PRETRAIN}" \
  --whitelist-env-b "${WHITELIST_OTHER}" \
  --num-pairs 20 \
  --seed "${SEED}" \
  --out-dir "${SWAP_CROSS_DIR}"

echo "=== DONE ==="
echo "Outputs under: ${OUT_BASE}"
date
