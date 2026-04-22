#!/usr/bin/env python3
"""
Experiment #2: Appearance-perturbation invariance for the small codebook.

What it tests
1) Encode a short tokenization window using the dual-codebook tokenizer:
   - first frame quantized with the big (1024) VQ
   - remaining frames quantized with the small (e.g. 16) VQ
2) Apply appearance-only perturbation (color jitter) to the *same* frames
   (same jitter parameters for all frames in the window)
3) Re-encode and measure how much small-code token IDs flip.

If the small codebook represents motion (not appearance), then:
  - token flip rate for small IDs should be lower than what we'd see
    if the small IDs encoded appearance directly
  - flip rates should concentrate less in static/background patches

``--big-tokenizer-fpath`` must be a tokenizer training checkpoint (model-*.pt), not a
Genie dynamics-only save; Genie checkpoints typically omit tokenizer weights.

Example (matches dual-codebook analysis plan):

python scripts/analyze_dual_codebook_invariance_colorjitter.py \\
  --big-tokenizer-fpath checkpoints/genie_redux_guilded/tokenizer.pt \\
  --small-tokenizer-fpath checkpoints/tokenizer/tokenizer_ft_cb16_3games/model-5000.pt \\
  --small-codebook-size 16 \\
  --dataset-root-dpath data_generation/datasets \\
  --dataset-name retro_act_v0.0.0_g200_256 \\
  --num-samples 50 \\
  --jitter medium \\
  --out-dir outputs/analysis/dual_cb_invariance_ft_cb16_3games
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import os
import random
from dataclasses import dataclass
import cv2
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from PIL import Image

from data.data import DatasetOutputFormat, MultiEnvironmentDataset, TransformsGenerator
from models import Tokenizer
from models.components.vector_quantize import VectorQuantize
from scripts.dual_codebook_checkpoint import load_tokenizer_weights, set_seed


@dataclass
class JitterParams:
    brightness_factor: float
    contrast_factor: float
    saturation_factor: float
    hue_shift_degrees: float


def golden_ratio_lut(codebook_size: int) -> np.ndarray:
    """Return (codebook_size, 3) uint8 RGB LUT."""
    phi = 0.618033988749895
    hues = (np.arange(codebook_size) * phi) % 1.0
    hsv = np.ones((1, codebook_size, 3), dtype=np.float32)
    hsv[0, :, 0] = hues
    hsv[0, :, 1] = 0.9
    hsv[0, :, 2] = 0.95
    colors = (hsv_to_rgb(hsv).squeeze(0) * 255).astype(np.uint8)
    return colors


def apply_color_jitter_rgb_uint8(frames_uint8: np.ndarray, params: JitterParams) -> np.ndarray:
    """
    frames_uint8: (T, H, W, 3) uint8 RGB
    returns:      (T, H, W, 3) uint8 RGB
    """
    assert frames_uint8.ndim == 4 and frames_uint8.shape[-1] == 3

    # Convert to float in [0, 1]
    x = frames_uint8.astype(np.float32) / 255.0

    # Brightness
    x = np.clip(x * params.brightness_factor, 0.0, 1.0)

    # Contrast (per-frame, per-channel mean)
    mean = x.mean(axis=(1, 2), keepdims=True)
    x = np.clip((x - mean) * params.contrast_factor + mean, 0.0, 1.0)

    # Saturation
    # NTSC luma weights
    gray = np.dot(x[..., :3], np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
    gray = gray[..., None]
    x = np.clip(x * params.saturation_factor + gray * (1.0 - params.saturation_factor), 0.0, 1.0)

    # Hue via HSV (cv2 uses H in [0..179] for 0..360 degrees)
    hue_shift_units = params.hue_shift_degrees / 2.0
    out = []
    for i in range(x.shape[0]):
        rgb_uint8 = (x[i] * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv[..., 0]
        h = (h + hue_shift_units) % 180.0
        hsv[..., 0] = h
        rgb2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        out.append(rgb2)
    return np.stack(out, axis=0)


def sample_jitter_params(rng: random.Random, jitter: str) -> JitterParams:
    # Parameters roughly aligned with a "torchvision-like" interpretation:
    # hue in [-hue_strength, +hue_strength] * 360 degrees
    if jitter == "mild":
        brightness = 0.10
        contrast = 0.20
        saturation = 0.20
        hue = 0.05
    elif jitter == "medium":
        brightness = 0.30
        contrast = 0.40
        saturation = 0.50
        hue = 0.15
    elif jitter == "strong":
        brightness = 0.50
        contrast = 0.60
        saturation = 0.80
        hue = 0.25
    else:
        raise ValueError(f"Unknown jitter strength: {jitter}")

    def uni_factor(strength: float) -> float:
        lo = max(0.0, 1.0 - strength)
        hi = 1.0 + strength
        return rng.uniform(lo, hi)

    brightness_factor = uni_factor(brightness)
    contrast_factor = uni_factor(contrast)
    saturation_factor = uni_factor(saturation)

    # hue shift in turns (fraction of full rotation)
    hue_shift_degrees = rng.uniform(-hue, hue) * 360.0
    return JitterParams(
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
        saturation_factor=saturation_factor,
        hue_shift_degrees=hue_shift_degrees,
    )


def mean_transition_entropy(ids: torch.Tensor) -> float:
    """Mean over time boundaries of empirical entropy of (id_{t-1}, id_t) pair distribution (spatial)."""
    ids_cpu = ids.detach().cpu()
    t_small, _, _ = ids_cpu.shape
    if t_small < 2:
        return float("nan")
    max_id = int(ids_cpu.max().item()) + 1
    ents = []
    for t in range(1, t_small):
        a = ids_cpu[t - 1].reshape(-1).numpy().astype(np.int64)
        b = ids_cpu[t].reshape(-1).numpy().astype(np.int64)
        packed = a * max_id + b
        _, counts = np.unique(packed, return_counts=True)
        p = counts.astype(np.float64) / counts.sum()
        ents.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(ents))


def save_token_gif(
    out_dir: Path,
    filename: str,
    video_frames: np.ndarray,
    token_ids: np.ndarray,
    big_codebook_size: int,
    small_codebook_size: int,
    upscale: int = 8,
):
    """Save animated GIF: original frame on top, token-map on bottom.

    Frame 0 indices come from the big codebook; frames 1+ from the small codebook.
    Each gets its own color LUT so indices map correctly.

    Args:
        video_frames:       (T, C, H, W) float32 in [0,1] or uint8 in [0,255].
        token_ids:          (T_tokens, ph, pw) int indices (dual-codebook concatenated).
        big_codebook_size:  size of big VQ codebook (for frame 0).
        small_codebook_size: size of small VQ codebook (for frames 1+).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lut_big = golden_ratio_lut(big_codebook_size)
    lut_small = golden_ratio_lut(small_codebook_size)

    ids_np = token_ids if isinstance(token_ids, np.ndarray) else token_ids.detach().cpu().numpy()
    t_tokens = ids_np.shape[0]

    if video_frames.dtype != np.uint8:
        frames_uint8 = (np.clip(video_frames, 0, 1) * 255).astype(np.uint8)
    else:
        frames_uint8 = video_frames
    if frames_uint8.shape[1] == 3:
        frames_uint8 = frames_uint8.transpose(0, 2, 3, 1)

    h, w = frames_uint8.shape[1], frames_uint8.shape[2]
    n_frames = min(len(frames_uint8), t_tokens)

    gif_frames = []
    for t in range(n_frames):
        lut = lut_big if t == 0 else lut_small
        rgb_map = lut[ids_np[t]]
        tm_pil = Image.fromarray(rgb_map).resize((w, h), Image.NEAREST)
        tm_arr = np.array(tm_pil)
        combined = np.concatenate([frames_uint8[t], tm_arr], axis=0)
        gif_frames.append(Image.fromarray(combined))

    if gif_frames:
        gif_frames[0].save(
            out_dir / filename,
            save_all=True, append_images=gif_frames[1:], loop=0, duration=80,
        )


def save_recon_gif(
    out_dir: Path,
    filename: str,
    video_frames: np.ndarray,
    recon_video: torch.Tensor,
):
    """Save animated GIF: original frame on top, reconstruction on bottom.

    Args:
        video_frames: (T, C, H, W) float32 in [0,1] or uint8 in [0,255].
        recon_video:  (1, C, T, H, W) reconstruction tensor.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if video_frames.dtype != np.uint8:
        frames_uint8 = (np.clip(video_frames, 0, 1) * 255).astype(np.uint8)
    else:
        frames_uint8 = video_frames
    if frames_uint8.shape[1] == 3:
        frames_uint8 = frames_uint8.transpose(0, 2, 3, 1)

    recon = torch.clamp(recon_video[0], 0, 1).cpu()  # (C, T, H, W)
    n_frames = min(len(frames_uint8), recon.shape[1])

    gif_frames = []
    for t in range(n_frames):
        orig = frames_uint8[t]  # (H, W, 3)
        rec = (recon[:, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        combined = np.concatenate([orig, rec], axis=0)
        gif_frames.append(Image.fromarray(combined))

    if gif_frames:
        gif_frames[0].save(
            out_dir / filename,
            save_all=True, append_images=gif_frames[1:], loop=0, duration=80,
        )


def save_small_token_montage(
    out_dir: Path,
    sample_id: int,
    ids_small_orig: torch.Tensor,
    ids_small_jit: torch.Tensor,
    codebook_size: int,
    upscale: int = 8,
):
    """
    ids_small_*: (T_small, ph, pw) on CPU or GPU
    saves: orig_vs_jit_small_tokens_sample_{sample_id}.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lut = golden_ratio_lut(codebook_size)  # (K,3) uint8

    ids_small_orig_np = ids_small_orig.detach().cpu().numpy()
    ids_small_jit_np = ids_small_jit.detach().cpu().numpy()
    t_small, ph, pw = ids_small_orig_np.shape

    h_px = ph * upscale
    w_px = pw * upscale

    # 2 rows (orig, jit) x T_small columns
    montage = Image.new("RGB", (w_px * t_small, h_px * 2))

    for t in range(t_small):
        rgb_orig = lut[ids_small_orig_np[t]]  # (ph,pw,3)
        rgb_jit = lut[ids_small_jit_np[t]]

        # Upscale by repeating pixels (nearest-neighbor)
        rgb_orig_up = np.repeat(np.repeat(rgb_orig, upscale, axis=0), upscale, axis=1)
        rgb_jit_up = np.repeat(np.repeat(rgb_jit, upscale, axis=0), upscale, axis=1)

        montage.paste(Image.fromarray(rgb_orig_up), (w_px * t, 0))
        montage.paste(Image.fromarray(rgb_jit_up), (w_px * t, h_px))

    montage.save(out_dir / f"orig_vs_jit_small_tokens_sample_{sample_id}.png")


def main() -> None:
    epilog = r"""
Example:
  python scripts/analyze_dual_codebook_invariance_colorjitter.py \
    --big-tokenizer-fpath checkpoints/genie_redux_guilded/tokenizer.pt \
    --small-tokenizer-fpath checkpoints/tokenizer/tokenizer_ft_cb16_3games/model-5000.pt \
    --small-codebook-size 16 \
    --dataset-root-dpath data_generation/datasets \
    --dataset-name retro_act_v0.0.0_g200_256 \
    --num-samples 50 \
    --jitter medium \
    --out-dir outputs/analysis/dual_cb_invariance_ft_cb16_3games
"""
    parser = argparse.ArgumentParser(
        description="Dual-codebook small-token invariance under color jitter (analysis only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("--big-tokenizer-fpath", type=str, required=True)
    parser.add_argument("--small-tokenizer-fpath", type=str, required=True)
    parser.add_argument("--small-codebook-size", type=int, default=16)
    parser.add_argument("--dataset-root-dpath", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=20)
    parser.add_argument("--seq-length-input", type=int, default=15)
    parser.add_argument("--seq-step", type=int, default=16)
    parser.add_argument("--motion-frames", type=int, default=7, help="How many frames after the anchor (t in description).")
    parser.add_argument("--jitter", type=str, default="medium", choices=["mild", "medium", "strong"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-k", type=int, default=5, help="How many samples to save token-map montages for.")
    parser.add_argument("--upscale", type=int, default=8)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--transition-entropy",
        action="store_true",
        help="Add per-sample transition-entropy columns (pair entropy across space, mean over time).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset_folder = os.path.join(args.dataset_root_dpath, args.dataset_name)

    # Tokenization window: 1 anchor frame + motion frames
    num_first_frames = 1
    total_frames = num_first_frames + args.motion_frames

    # --- load big tokenizer ---
    big_tokenizer = Tokenizer(
        dim=512,
        codebook_size=1024,
        image_size=256,
        patch_size=8,
        temporal_patch_size=1,
        num_blocks=8,
        dim_head=64,
        heads=8,
        ff_mult=4.0,
        vq_loss_w=1.0,
        recon_loss_w=1.0,
    ).to(device)

    load_tokenizer_weights(big_tokenizer, args.big_tokenizer_fpath)
    big_tokenizer.eval()

    # --- load small VQ only ---
    small_vq = VectorQuantize(
        dim=512,
        codebook_size=args.small_codebook_size,
        learnable_codebook=True,
        ema_update=False,
        use_cosine_sim=True,
        commitment_weight=0.25,
        codebook_dim=32,
    ).to(device)
    small_tok_state = torch.load(args.small_tokenizer_fpath, map_location="cpu")
    vq_state = {k.replace("vq.", "", 1): v for k, v in small_tok_state["model"].items() if k.startswith("vq.")}
    small_vq.load_state_dict(vq_state, strict=True)
    small_vq.eval()

    # --- dataset ---
    transforms = TransformsGenerator.get_final_transforms(big_tokenizer.image_size, None)
    dataset = MultiEnvironmentDataset(
        dataset_folder,
        seq_length_input=args.seq_length_input,
        seq_step=args.seq_step,
        split_type="instance",
        split="all",
        transform=transforms["test"],
        format=DatasetOutputFormat.IVG,
        enable_cache=False,
        n_workers=1,
        n_envs=args.n_envs,
    )

    # --- sample indices deterministically ---
    total_len = len(dataset)
    if args.num_samples > total_len:
        sample_indices = list(range(total_len))
    else:
        # torch.randperm is deterministic under fixed seeds
        perm = torch.randperm(total_len, generator=torch.Generator().manual_seed(args.seed))
        sample_indices = perm[: args.num_samples].tolist()

    # --- metrics storage ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt, ph, pw = big_tokenizer.get_video_patch_shape(total_frames, num_first_frames=num_first_frames)
    assert pt == total_frames, "Expected temporal_patch_size=1 so pt should equal total_frames."
    n_first = ph * pw

    # small part has pt-1 time steps
    t_small = pt - 1

    frame_flip_cols = [f"flip_rate_small_frame_{i}" for i in range(t_small)]

    metrics_path = out_dir / "metrics.csv"
    extra_entropy_cols = (
        ["trans_entropy_mean_orig", "trans_entropy_mean_jit"] if args.transition_entropy else []
    )
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "token_flip_rate_small_mean", *frame_flip_cols, *extra_entropy_cols])

    print(f"Running invariance test on {len(sample_indices)} samples. total_frames={total_frames}, ph={ph}, pw={pw}")

    flip_rates = []
    trans_ent_orig_list = []
    trans_ent_jit_list = []
    per_frame_accum = [[] for _ in range(t_small)]

    for local_i, global_idx in enumerate(sample_indices):
        item = dataset[global_idx]
        input_frames = item["input_frames"]  # (F, C, H, W) float32 in [0,1]
        input_frames = input_frames[:total_frames]

        # Prepare anchor+motion video: (b,c,f,h,w)
        # input_frames is (T,C,H,W)
        videos = torch.from_numpy(input_frames).unsqueeze(0).to(device)  # (1,T,C,H,W)
        videos = videos.permute(0, 2, 1, 3, 4).contiguous()  # (1,C,T,H,W)

        with torch.no_grad():
            indices_orig = big_tokenizer.forward_dual_codebook(
                videos, small_vq, return_only_codebook_ids=True
            )  # (1, pt, ph, pw)

        ids_small_orig = indices_orig[:, 1:, :, :]  # (1, t_small, ph, pw)

        # --- apply appearance-only perturbation ---
        # Sample jitter params once per trajectory
        rng = random.Random(args.seed + global_idx)
        params = sample_jitter_params(rng, args.jitter)

        frames_uint8 = (input_frames * 255.0).clip(0, 255).astype(np.uint8)  # (T,C,H,W)
        frames_uint8 = frames_uint8.transpose(0, 2, 3, 1)  # (T,H,W,C)
        frames_jit_uint8 = apply_color_jitter_rgb_uint8(frames_uint8, params)  # (T,H,W,C)
        frames_jit_uint8 = frames_jit_uint8.transpose(0, 3, 1, 2)  # (T,C,H,W)
        frames_jit = torch.from_numpy(frames_jit_uint8).to(device).float() / 255.0  # (T,C,H,W)
        videos_jit = frames_jit.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # (1,C,T,H,W)

        with torch.no_grad():
            indices_jit = big_tokenizer.forward_dual_codebook(
                videos_jit, small_vq, return_only_codebook_ids=True
            )

        ids_small_jit = indices_jit[:, 1:, :, :]

        # --- metrics ---
        token_flip = (ids_small_orig != ids_small_jit).float()
        token_flip_rate_small_mean = token_flip.mean().item()
        flip_rates.append(token_flip_rate_small_mean)

        # Per small-frame (time step) flip rates
        per_frame = []
        for t in range(t_small):
            v = token_flip[:, t].mean().item()
            per_frame.append(v)
            per_frame_accum[t].append(v)

        te_o = mean_transition_entropy(ids_small_orig[0]) if args.transition_entropy else None
        te_j = mean_transition_entropy(ids_small_jit[0]) if args.transition_entropy else None
        if args.transition_entropy:
            trans_ent_orig_list.append(te_o)
            trans_ent_jit_list.append(te_j)

        row_tail = []
        if args.transition_entropy:
            row_tail = [te_o, te_j]

        # Save CSV
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([global_idx, token_flip_rate_small_mean, *per_frame, *row_tail])

        # Save montage, token-index GIFs, and reconstruction GIFs for a few samples
        if local_i < args.save_k:
            save_small_token_montage(
                out_dir=out_dir,
                sample_id=global_idx,
                ids_small_orig=ids_small_orig[0],
                ids_small_jit=ids_small_jit[0],
                codebook_size=args.small_codebook_size,
                upscale=args.upscale,
            )
            big_cb_size = big_tokenizer.vq.codebook_size
            all_ids_orig = indices_orig[0].detach().cpu().numpy()  # (pt, ph, pw)
            save_token_gif(
                out_dir, f"{global_idx}_orig_token_indices.gif",
                input_frames, all_ids_orig,
                big_codebook_size=big_cb_size,
                small_codebook_size=args.small_codebook_size,
                upscale=args.upscale,
            )
            all_ids_jit = indices_jit[0].detach().cpu().numpy()
            frames_jit_np = frames_jit.cpu().numpy()  # (T,C,H,W)
            save_token_gif(
                out_dir, f"{global_idx}_jit_token_indices.gif",
                frames_jit_np, all_ids_jit,
                big_codebook_size=big_cb_size,
                small_codebook_size=args.small_codebook_size,
                upscale=args.upscale,
            )
            with torch.no_grad():
                recon_orig = big_tokenizer.forward_dual_codebook(
                    videos, small_vq, return_recons_only=True
                )
                recon_jit = big_tokenizer.forward_dual_codebook(
                    videos_jit, small_vq, return_recons_only=True
                )
            save_recon_gif(out_dir, f"{global_idx}_orig.gif", input_frames, recon_orig)
            save_recon_gif(out_dir, f"{global_idx}_jit.gif", frames_jit_np, recon_jit)

        if (local_i + 1) % 10 == 0:
            mean_so_far = float(np.mean(flip_rates))
            print(f"[{local_i+1}/{len(sample_indices)}] mean small flip rate so far: {mean_so_far:.6f}")

    mean_flip = float(np.mean(flip_rates)) if flip_rates else 0.0
    std_flip = float(np.std(flip_rates)) if flip_rates else 0.0

    agg_frames = [float(np.mean(per_frame_accum[t])) if per_frame_accum[t] else 0.0 for t in range(t_small)]
    agg_row = ["__aggregate__", mean_flip, *agg_frames]
    if args.transition_entropy and trans_ent_orig_list:
        agg_row.extend(
            [
                float(np.nanmean(trans_ent_orig_list)),
                float(np.nanmean(trans_ent_jit_list)),
            ]
        )
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(agg_row)

    summary_path = out_dir / "summary.txt"
    summary_lines = [
        f"num_samples={len(flip_rates)}",
        f"small_codebook_size={args.small_codebook_size}",
        f"total_frames={total_frames} (anchor=1, motion={args.motion_frames})",
        f"jitter={args.jitter}",
        f"mean_small_token_flip_rate={mean_flip}",
        f"std_small_token_flip_rate={std_flip}",
    ]
    if args.transition_entropy and trans_ent_orig_list:
        summary_lines.append(
            f"mean_transition_entropy_orig={float(np.nanmean(trans_ent_orig_list))}"
        )
        summary_lines.append(
            f"mean_transition_entropy_jit={float(np.nanmean(trans_ent_jit_list))}"
        )
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Done. mean small flip rate={mean_flip:.6f} (+/- {std_flip:.6f}). Metrics saved to {out_dir}")


if __name__ == "__main__":
    main()

