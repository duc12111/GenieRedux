#!/bin/bash
#SBATCH --job-name="dual_cb_sweep_eval"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --exclude=node1,node7
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/nguyenti/slurm/logs/slurm-%j.out
#SBATCH --comment=dual_cb_sweep_eval

# Full evaluation suite for one dual-codebook sweep checkpoint.
# Runs: (1) reconstruction PSNR/SSIM/FID + token viz,
#        (2) color-jitter invariance,
#        (3) same-game code swap,
#        (4) cross-game code swap.
#
# Parameterised via CB_SIZE:
#   CB_SIZE=64 sbatch scripts/eval_dual_cb_sweep.sbatch
#
# Or evaluate all sizes:
#   for s in 8 16 32 64 128 256; do CB_SIZE=$s sbatch scripts/eval_dual_cb_sweep.sbatch; done

set -eo pipefail

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pwd; hostname; date
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate genie_redux
cd /home/stud/nguyenti/storage/user/GenieRedux

set -u

CB_SIZE="${CB_SIZE:-64}"

BIG_TOKENIZER_FPATH="${BIG_TOKENIZER_FPATH:-checkpoints/genie_redux_guilded/model-100000.pt}"
SMALL_TOKENIZER_FPATH="${SMALL_TOKENIZER_FPATH:-checkpoints/tokenizer/dual_cb_sweep_cb${CB_SIZE}_5games/model-5000.pt}"
DATASET_ROOT_DPATH="${DATASET_ROOT_DPATH:-data_generation/datasets}"
DATASET_NAME="${DATASET_NAME:-retro_act_v0.0.0_g200_256}"
SEED="${SEED:-0}"

WHITELIST_ENV_A="${WHITELIST_ENV_A:-coolspot-snes}"
WHITELIST_ENV_B="${WHITELIST_ENV_B:-addamsfamily-nes}"

OUT_BASE="${OUT_BASE:-outputs/analysis/dual_cb_sweep/cb${CB_SIZE}}"

echo "========================================================"
echo "  Dual-codebook sweep evaluation: CB_SIZE=${CB_SIZE}"
echo "  Small checkpoint: ${SMALL_TOKENIZER_FPATH}"
echo "========================================================"

# ── 1. Reconstruction eval (PSNR / SSIM / FID + token visualisation) ────────
echo ""
echo "=== 1/4: Reconstruction eval (dual-codebook mode) ==="
python eval_genie_redux.py \
  config=eval_dual_codebook \
  tokenizer_fpath="${BIG_TOKENIZER_FPATH}" \
  eval.small_tokenizer_fpath="${SMALL_TOKENIZER_FPATH}" \
  eval.small_codebook_size="${CB_SIZE}" \
  eval.dataset_root_dpath="${DATASET_ROOT_DPATH}" \
  eval.dataset_name="${DATASET_NAME}" \
  eval.n_envs=20 \
  eval.batch_size=16 \
  eval.sample_num_frames=7 \
  eval.model_name="dual_cb_sweep_cb${CB_SIZE}" \
  eval.visualize_token_ids=true

# ── 2. Color-jitter invariance (H1) ─────────────────────────────────────────
echo ""
echo "=== 2/4: Color-jitter invariance ==="
python scripts/analyze_dual_codebook_invariance_colorjitter.py \
  --big-tokenizer-fpath "${BIG_TOKENIZER_FPATH}" \
  --small-tokenizer-fpath "${SMALL_TOKENIZER_FPATH}" \
  --small-codebook-size "${CB_SIZE}" \
  --dataset-root-dpath "${DATASET_ROOT_DPATH}" \
  --dataset-name "${DATASET_NAME}" \
  --num-samples 50 \
  --n-envs 20 \
  --motion-frames 7 \
  --jitter medium \
  --seed "${SEED}" \
  --transition-entropy \
  --out-dir "${OUT_BASE}/invariance_colorjitter"

# ── 3. Code swap: same game ─────────────────────────────────────────────────
echo ""
echo "=== 3/4: Code swap (same_game: ${WHITELIST_ENV_A}) ==="
python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \
  --big-tokenizer-fpath "${BIG_TOKENIZER_FPATH}" \
  --small-tokenizer-fpath "${SMALL_TOKENIZER_FPATH}" \
  --small-codebook-size "${CB_SIZE}" \
  --dataset-root-dpath "${DATASET_ROOT_DPATH}" \
  --dataset-name "${DATASET_NAME}" \
  --mode same_game \
  --whitelist-env "${WHITELIST_ENV_A}" \
  --num-pairs 20 \
  --seed "${SEED}" \
  --out-dir "${OUT_BASE}/swap_same_game"

# ── 4. Code swap: cross game ────────────────────────────────────────────────
echo ""
echo "=== 4/4: Code swap (cross_game: A=${WHITELIST_ENV_A}, B=${WHITELIST_ENV_B}) ==="
python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \
  --big-tokenizer-fpath "${BIG_TOKENIZER_FPATH}" \
  --small-tokenizer-fpath "${SMALL_TOKENIZER_FPATH}" \
  --small-codebook-size "${CB_SIZE}" \
  --dataset-root-dpath "${DATASET_ROOT_DPATH}" \
  --dataset-name "${DATASET_NAME}" \
  --mode cross_game \
  --whitelist-env-a "${WHITELIST_ENV_A}" \
  --whitelist-env-b "${WHITELIST_ENV_B}" \
  --num-pairs 20 \
  --seed "${SEED}" \
  --out-dir "${OUT_BASE}/swap_cross_game"

echo ""
echo "=== ALL EVAL DONE: cb${CB_SIZE} ==="
echo "Outputs under: ${OUT_BASE}"
date
