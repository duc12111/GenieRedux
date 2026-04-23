#!/usr/bin/env python3
"""V2: per-code patch visualisation.

For each of K small-VQ codes, find N patches from the validation set that
were assigned that code, and render them as a temporal strip (prev / curr
/ next frame of that patch). The output is a grid image: one row per code,
N columns per code, each cell a 3-frame strip of the patch with a border.

Signal to read:
- Row per code shows VISUALLY SIMILAR MOTION across N examples → codebook
  is a motion vocabulary. Good.
- Row per code shows similar APPEARANCE (not motion) → codes encode
  appearance, not motion.
- Row per code looks random → code has no clear semantic content.

Usage:
    python scripts/validate_v2_per_code_patches.py \
      --checkpoint checkpoints/tokenizer_dual_cb/<run>/model-5000.pt \
      --small-codebook-size 16 \
      --num-per-code 6 \
      --out outputs/validation/<run>/v2_patches.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Subset

from data.data import DatasetOutputFormat, MultiEnvironmentDataset, TransformsGenerator
from models import DualCodebookTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--small-codebook-size", type=int, default=16)
    p.add_argument("--big-codebook-size", type=int, default=1024)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--dataset-root-dpath", default="data_generation/datasets")
    p.add_argument("--dataset-name", default="retro_act_v0.0.0_g200_256")
    p.add_argument("--num-samples", type=int, default=40,
                   help="validation windows to scan for patches")
    p.add_argument("--num-per-code", type=int, default=6,
                   help="examples per code in the output grid")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--upscale", type=int, default=4,
                   help="visual upscale factor for each patch (nearest-neighbour)")
    p.add_argument("--out", default="outputs/validation/v2_patches.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(0)

    model = DualCodebookTokenizer(
        dim=512,
        codebook_size=args.big_codebook_size,
        small_codebook_size=args.small_codebook_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_blocks=8, dim_head=64, heads=8, ff_mult=4,
    )
    sd = torch.load(args.checkpoint, map_location="cpu")["model"]
    model.load_state_dict(sd, strict=False)
    model.eval().to(args.device)

    K = args.small_codebook_size
    N = args.num_per_code
    ph = model.patch_size[0]
    pw = model.patch_size[1]

    transforms = TransformsGenerator.get_final_transforms(model.image_size, None)
    ds = MultiEnvironmentDataset(
        f"{args.dataset_root_dpath}/{args.dataset_name}",
        seq_length_input=args.num_frames - 1,
        seq_step=20,
        split_type="session", split="validation",
        transform=transforms["train"],
        format=DatasetOutputFormat.IVG,
        enable_cache=False, n_workers=2, n_envs=0,
        n_samples=max(1, args.num_samples // 100 + 1),
    )
    ds = Subset(ds, range(min(args.num_samples, len(ds))))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Per-code list of (frames_chw, patch_y, patch_x, frame_t)
    # where frames_chw is (C, 3, ph, pw) spanning [t-1, t, t+1] if available,
    # padded with copies when at window boundaries.
    per_code_examples = {k: [] for k in range(K)}
    max_scan = N * 3  # try to collect 3× target so we can subsample diverse examples

    with torch.no_grad():
        for batch in dl:
            videos = rearrange(batch["input_frames"], "b f c h w -> b c f h w").to(args.device)
            indices = model.forward_dual_codebook(videos, model.small_vq, return_only_codebook_ids=True)
            # indices: (B, T, h, w); we only care about rest frames (T>=1)
            B, T, h, w = indices.shape

            videos_cpu = videos.detach().cpu()  # (B, C, T_total, H, W)
            idx_cpu = indices.detach().cpu()

            for b in range(B):
                for t_rest in range(1, T):
                    # t_rest is the index in the code tensor (0 = frame 0)
                    frame_t = t_rest  # position in videos tensor (0-indexed)
                    for y in range(h):
                        for x in range(w):
                            code = int(idx_cpu[b, t_rest, y, x].item())
                            if len(per_code_examples[code]) >= max_scan:
                                continue
                            # Extract patch from 3 frames: [t-1, t, t+1] (clip at boundaries)
                            frames_idx = [max(0, frame_t - 1), frame_t,
                                          min(videos_cpu.shape[2] - 1, frame_t + 1)]
                            strip = []
                            for fi in frames_idx:
                                patch = videos_cpu[b, :, fi,
                                                   y * ph:(y + 1) * ph,
                                                   x * pw:(x + 1) * pw]
                                strip.append(patch)
                            strip = torch.stack(strip, dim=0)  # (3, C, ph, pw)
                            per_code_examples[code].append(strip)
            # stop scanning early if every code already has enough
            if all(len(per_code_examples[k]) >= max_scan for k in range(K)):
                break

    # Subsample N examples per code, evenly spaced
    for k in range(K):
        ex = per_code_examples[k]
        if len(ex) == 0:
            per_code_examples[k] = []
        elif len(ex) <= N:
            per_code_examples[k] = ex
        else:
            idx = np.linspace(0, len(ex) - 1, N).round().astype(int).tolist()
            per_code_examples[k] = [ex[i] for i in idx]

    # Build the grid image.
    # Each cell: 3 frames × (upscale * ph) pixels wide × (upscale * pw) pixels tall
    up = args.upscale
    cell_w = 3 * ph * up  # 3 frames horizontally within a cell
    cell_h = pw * up
    gap = 4  # px between cells
    grid_w = N * (cell_w + gap) - gap
    grid_h = K * (cell_h + gap) - gap

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.float32)  # white background

    def patch_to_np(p):
        """(C, ph, pw) float → (ph, pw, 3) in [0, 1]."""
        arr = p.numpy()
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = np.clip(arr, 0.0, 1.0)
        return np.transpose(arr, (1, 2, 0))  # (ph, pw, 3)

    def upscale_patch(arr2d, factor):
        return np.kron(arr2d, np.ones((factor, factor, 1)))

    for k in range(K):
        ex = per_code_examples[k]
        for i in range(N):
            cell_x0 = i * (cell_w + gap)
            cell_y0 = k * (cell_h + gap)
            if i >= len(ex):
                # mark empty cells with grey
                canvas[cell_y0:cell_y0 + cell_h, cell_x0:cell_x0 + cell_w] = 0.85
                continue
            strip = ex[i]  # (3, C, ph, pw)
            for fi in range(3):
                patch = patch_to_np(strip[fi])
                patch_up = upscale_patch(patch, up)
                x0 = cell_x0 + fi * ph * up
                canvas[cell_y0:cell_y0 + cell_h, x0:x0 + ph * up] = patch_up

    # Render with labels
    fig, ax = plt.subplots(1, 1, figsize=(max(6, N * 1.0), max(6, K * 0.8)))
    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([i * (cell_h + gap) + cell_h / 2 for i in range(K)])
    ax.set_yticklabels([f"code {k}  (n={len(per_code_examples[k])})" for k in range(K)])
    ax.set_title(
        f"Per-code patch examples (3-frame strip each)\n"
        f"K={K}, patch_size={ph}×{pw}, {N} examples per code"
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved per-code patches to {out_path}")

    # Also print usage summary
    used = sum(1 for k in range(K) if len(per_code_examples[k]) > 0)
    empty = [k for k in range(K) if len(per_code_examples[k]) == 0]
    print(f"codes with at least 1 example: {used}/{K}")
    if empty:
        print(f"codes with zero usage in scanned windows: {empty}")


if __name__ == "__main__":
    main()
