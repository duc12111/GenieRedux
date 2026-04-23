#!/usr/bin/env python3
"""Motion-conditional utilisation: distinguish static codes from motion codes.

Extends the basic utilisation check. Computes Farnebäck optical flow per
patch, splits patches into static (|flow| < threshold) and moving, then
reports the code distribution separately for each bucket.

Reads the codebook as: one or more "static" codes dominating stationary
patches + (hopefully) a diverse set of motion codes distributed across
moving patches.

Key outputs:
- overall entropy (all patches)
- motion-conditional entropy (moving patches only) — should be near
  log2(K - num_static_codes) if motion codes are well-spread
- per-code: count, mean |flow|, variance of (dx, dy)
- classification: which codes are "static" (mean |flow| < threshold)
  vs "motion" codes

Usage:
    python scripts/validate_motion_conditional.py \
      --checkpoint checkpoints/tokenizer_dual_cb/<run>/model-5000.pt \
      --small-codebook-size 16 \
      --num-samples 30 \
      --motion-threshold 0.2 \
      --out outputs/validation/<run>/motion_conditional.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Subset

from data.data import DatasetOutputFormat, MultiEnvironmentDataset, TransformsGenerator
from models import DualCodebookTokenizer


def compute_farneback_flow_patch(videos_chw: np.ndarray, patch_size: int) -> np.ndarray:
    """videos_chw: (B, C, T, H, W) float [0,1]. Returns (B, T-1, h, w, 2) patch-avg flow in patch-units."""
    B, C, T, H, W = videos_chw.shape
    h = H // patch_size
    w = W // patch_size
    out = np.zeros((B, T - 1, h, w, 2), dtype=np.float32)
    for b in range(B):
        lum = videos_chw[b].mean(axis=0)                         # (T, H, W)
        lum_u8 = np.clip(lum * 255.0, 0, 255).astype(np.uint8)
        for t in range(T - 1):
            flow = cv2.calcOpticalFlowFarneback(
                lum_u8[t], lum_u8[t + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )                                                    # (H, W, 2)
            flow_patch = flow.reshape(h, patch_size, w, patch_size, 2).mean(axis=(1, 3))
            out[b, t] = flow_patch / float(patch_size)
    return out


def entropy_bits(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


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
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--motion-threshold", type=float, default=0.2,
                   help="|flow| threshold (in patch units) to classify a patch as moving")
    p.add_argument("--out", default="outputs/validation/motion_conditional.txt")
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

    # Accumulators
    all_indices = []   # (N_total,) patches × all frames flattened
    all_flow_mag = []  # scalar |flow|
    all_flow = []      # (dx, dy)

    with torch.no_grad():
        for batch in dl:
            videos_t = rearrange(batch["input_frames"], "b f c h w -> b c f h w")
            videos_np = videos_t.numpy()
            videos_gpu = videos_t.to(args.device)

            indices = model.forward_dual_codebook(
                videos_gpu, model.small_vq, return_only_codebook_ids=True
            )                                                   # (B, T, h, w)
            # motion frames = indices[:, 1:, :, :] — corresponds to transitions 0→1, 1→2, ...
            # Farnebäck flow[t] corresponds to frame t → t+1, so flow[0] aligns with indices[:, 1]
            rest_idx = indices[:, 1:, :, :].cpu().numpy()       # (B, T-1, h, w)
            flow = compute_farneback_flow_patch(videos_np, args.patch_size)  # (B, T-1, h, w, 2)
            flow_mag = np.linalg.norm(flow, axis=-1)             # (B, T-1, h, w)

            all_indices.append(rest_idx.reshape(-1))
            all_flow_mag.append(flow_mag.reshape(-1))
            all_flow.append(flow.reshape(-1, 2))

    idx_all = np.concatenate(all_indices)
    fm_all = np.concatenate(all_flow_mag)
    f_all = np.concatenate(all_flow)
    total = idx_all.shape[0]

    moving_mask = fm_all > args.motion_threshold
    static_mask = ~moving_mask
    n_moving = int(moving_mask.sum())
    n_static = int(static_mask.sum())

    # Per-code stats
    per_code = []
    for k in range(K):
        code_mask = idx_all == k
        n_code = int(code_mask.sum())
        if n_code == 0:
            per_code.append({
                "k": k, "count": 0, "pct": 0.0,
                "mean_flow_mag": 0.0,
                "flow_var_x": 0.0, "flow_var_y": 0.0,
                "pct_of_code_moving": 0.0,
                "pct_of_moving_patches": 0.0,
                "pct_of_static_patches": 0.0,
            })
            continue
        code_flow = f_all[code_mask]
        code_flow_mag = fm_all[code_mask]
        # fraction of patches *assigned to this code* that are moving
        pct_of_code_moving = 100.0 * float((code_flow_mag > args.motion_threshold).sum()) / n_code
        # fraction of all moving patches that got this code
        pct_of_moving = 100.0 * float(((idx_all == k) & moving_mask).sum()) / max(1, n_moving)
        # fraction of all static patches that got this code
        pct_of_static = 100.0 * float(((idx_all == k) & static_mask).sum()) / max(1, n_static)
        per_code.append({
            "k": k,
            "count": n_code,
            "pct": 100.0 * n_code / total,
            "mean_flow_mag": float(code_flow_mag.mean()),
            "flow_var_x": float(code_flow[:, 0].var()),
            "flow_var_y": float(code_flow[:, 1].var()),
            "pct_of_code_moving": pct_of_code_moving,
            "pct_of_moving_patches": pct_of_moving,
            "pct_of_static_patches": pct_of_static,
        })

    # Classify codes as static vs motion based on mean |flow|
    mean_flow_mags = np.array([c["mean_flow_mag"] for c in per_code])
    static_codes = np.where(mean_flow_mags < args.motion_threshold)[0].tolist()
    motion_codes = [k for k in range(K) if k not in static_codes]

    # Overall and conditional entropies
    counts_all = np.bincount(idx_all, minlength=K)
    counts_moving = np.bincount(idx_all[moving_mask], minlength=K)
    counts_static = np.bincount(idx_all[static_mask], minlength=K)
    ent_all = entropy_bits(counts_all)
    ent_moving = entropy_bits(counts_moving)
    ent_static = entropy_bits(counts_static)
    # Motion-conditional entropy over the motion codes alone (drop static-code mass)
    counts_moving_motion_only = counts_moving.copy()
    for k in static_codes:
        counts_moving_motion_only[k] = 0
    ent_moving_motion_only = entropy_bits(counts_moving_motion_only)

    # Output
    lines = []
    lines.append(f"=== Motion-conditional small-VQ utilisation (K={K}) ===")
    lines.append(f"checkpoint               : {args.checkpoint}")
    lines.append(f"num samples              : {args.num_samples}")
    lines.append(f"motion threshold         : |flow| > {args.motion_threshold} patch-units")
    lines.append(f"total patches            : {total}")
    lines.append(f"  static (below thresh)  : {n_static} ({100.0*n_static/total:.1f}%)")
    lines.append(f"  moving (above thresh)  : {n_moving} ({100.0*n_moving/total:.1f}%)")
    lines.append("")
    lines.append(f"Code classification:")
    lines.append(f"  static codes    : {static_codes}  ({len(static_codes)})")
    lines.append(f"  motion codes    : {motion_codes}  ({len(motion_codes)})")
    lines.append("")
    lines.append(f"Entropy / {math.log2(K):.2f} bits (uniform across all K={K}):")
    lines.append(f"  overall           : {ent_all:.3f}  (effective codes: {2**ent_all:.2f})")
    lines.append(f"  on static patches : {ent_static:.3f}  (effective codes: {2**ent_static:.2f})")
    lines.append(f"  on moving patches : {ent_moving:.3f}  (effective codes: {2**ent_moving:.2f})")
    if motion_codes:
        lines.append(f"  on moving patches, motion codes only (drops static code mass):")
        lines.append(f"                      {ent_moving_motion_only:.3f}  / {math.log2(len(motion_codes)):.2f} bits (uniform over {len(motion_codes)} motion codes)")
        lines.append(f"                      effective motion codes: {2**ent_moving_motion_only:.2f}")
    lines.append("")
    lines.append(f"Per-code breakdown (sorted by count descending):")
    lines.append(f"  {'k':>3} {'count':>8} {'pct':>6} {'mean|flow|':>10} {'var(dx)':>8} {'var(dy)':>8} {'%moving':>8} {'%mov→k':>8} {'%sta→k':>8}  role")
    sorted_codes = sorted(per_code, key=lambda c: -c["count"])
    for c in sorted_codes:
        role = "STATIC" if c["k"] in static_codes else "motion"
        lines.append(
            f"  {c['k']:>3} {c['count']:>8} {c['pct']:>5.1f}% {c['mean_flow_mag']:>10.3f} "
            f"{c['flow_var_x']:>8.3f} {c['flow_var_y']:>8.3f} "
            f"{c['pct_of_code_moving']:>7.1f}% {c['pct_of_moving_patches']:>7.1f}% {c['pct_of_static_patches']:>7.1f}%  {role}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "total_patches": total,
        "n_static": n_static,
        "n_moving": n_moving,
        "static_codes": static_codes,
        "motion_codes": motion_codes,
        "entropy_overall_bits": ent_all,
        "entropy_moving_patches_bits": ent_moving,
        "entropy_moving_motion_only_bits": ent_moving_motion_only,
        "max_entropy_K_bits": math.log2(K),
        "max_entropy_motion_codes_bits": math.log2(max(1, len(motion_codes))),
        "per_code": per_code,
    }, indent=2))

    for line in lines:
        print(line)
    print(f"\nsaved to {out_path} and {json_path}")


if __name__ == "__main__":
    main()
