#!/bin/bash
#SBATCH --job-name="dual_cb_analysis"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=dual_cb_analysis

# Dual-codebook analysis: (1) color-jitter small-token invariance, (2) code-swap motion transfer.
# Edit *_FPATH / WHITELIST_* below if your checkpoints or game IDs differ.
# Big tokenizer checkpoint must contain tokenizer weights (vq.* / to_patch* / encoder.*), not dynamics-only.
#
# For tokenizer_cb16_scratch_1game (trained on retro_act_*_g200_256_single = Cool Spot only), use instead:
#   sbatch scripts/analyze_dual_codebook_scratch1game.sbatch
# Whitelist key for that game is: coolspot-snes (see dataset subfolder retro_coolspot-snes_*).

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
SMALL_TOKENIZER_FPATH="${SMALL_TOKENIZER_FPATH:-checkpoints/tokenizer/tokenizer_ft_cb16_3games/model-5000.pt}"
DATASET_ROOT_DPATH="${DATASET_ROOT_DPATH:-data_generation/datasets}"
DATASET_NAME="${DATASET_NAME:-retro_act_v0.0.0_g200_256}"
SEED="${SEED:-0}"
SMALL_CODEBOOK_SIZE="${SMALL_CODEBOOK_SIZE:-16}"

WHITELIST_ENV="${WHITELIST_ENV:-addamsfamily-nes}"
WHITELIST_ENV_A="${WHITELIST_ENV_A:-addamsfamily-nes}"
WHITELIST_ENV_B="${WHITELIST_ENV_B:-addamsfamily-nes}"

OUT_BASE="${OUT_BASE:-outputs/analysis/dual_cb_analysis_run${SLURM_JOB_ID:-}}"
INV_DIR="${OUT_BASE}/invariance_colorjitter"
SWAP_SAME_DIR="${OUT_BASE}/swap_same_game"
SWAP_CROSS_DIR="${OUT_BASE}/swap_cross_game"

echo "=== Experiment #2: color-jitter invariance ==="
python scripts/analyze_dual_codebook_invariance_colorjitter.py \
  --big-tokenizer-fpath "${BIG_TOKENIZER_FPATH}" \
  --small-tokenizer-fpath "${SMALL_TOKENIZER_FPATH}" \
  --small-codebook-size "${SMALL_CODEBOOK_SIZE}" \
  --dataset-root-dpath "${DATASET_ROOT_DPATH}" \
  --dataset-name "${DATASET_NAME}" \
  --num-samples 50 \
  --n-envs 20 \
  --motion-frames 7 \
  --jitter medium \
  --seed "${SEED}" \
  --transition-entropy \
  --out-dir "${INV_DIR}"

echo "=== Experiment #3a: code swap (same_game) ==="
python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \
  --big-tokenizer-fpath "${BIG_TOKENIZER_FPATH}" \
  --small-tokenizer-fpath "${SMALL_TOKENIZER_FPATH}" \
  --small-codebook-size "${SMALL_CODEBOOK_SIZE}" \
  --dataset-root-dpath "${DATASET_ROOT_DPATH}" \
  --dataset-name "${DATASET_NAME}" \
  --mode same_game \
  --whitelist-env "${WHITELIST_ENV}" \
  --num-pairs 20 \
  --seed "${SEED}" \
  --out-dir "${SWAP_SAME_DIR}"

echo "=== Experiment #3b: code swap (cross_game) ==="
python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \
  --big-tokenizer-fpath "${BIG_TOKENIZER_FPATH}" \
  --small-tokenizer-fpath "${SMALL_TOKENIZER_FPATH}" \
  --small-codebook-size "${SMALL_CODEBOOK_SIZE}" \
  --dataset-root-dpath "${DATASET_ROOT_DPATH}" \
  --dataset-name "${DATASET_NAME}" \
  --mode cross_game \
  --whitelist-env-a "${WHITELIST_ENV_A}" \
  --whitelist-env-b "${WHITELIST_ENV_B}" \
  --num-pairs 20 \
  --seed "${SEED}" \
  --out-dir "${SWAP_CROSS_DIR}"

echo "=== DONE ==="
echo "Outputs under: ${OUT_BASE}"
date
