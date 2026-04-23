#!/usr/bin/env python3
"""Diagnostic: measure how many of the small VQ codes are actually used.

Loads a joint-trained DualCodebookTokenizer checkpoint, runs forward on N random
validation windows, tallies the small-VQ index distribution over frames 1+.
Reports unique codes used, top-k mass, and entropy.

Symptom of codebook collapse: few codes (<= K/4) carry most of the mass and
entropy is far below log2(K).

Example:
    python scripts/check_small_vq_utilization.py \
      --checkpoint checkpoints/tokenizer_dual_cb/dual_cb_train_cb16_allgames/model-5000.pt \
      --small-codebook-size 16 --num-samples 100
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from einops import rearrange
from torch.utils.data import DataLoader

from data.data import DatasetOutputFormat, MultiEnvironmentDataset, TransformsGenerator
from models import DualCodebookTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--small-codebook-size", type=int, default=16)
    p.add_argument("--big-codebook-size", type=int, default=1024)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--dataset-root-dpath", default="data_generation/datasets")
    p.add_argument("--dataset-name", default="retro_act_v0.0.0_g200_256")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--also-big", action="store_true",
                   help="additionally report big-VQ (frozen 1024-code) utilisation on frame 0")
    args = p.parse_args()

    torch.manual_seed(0)

    model = DualCodebookTokenizer(
        dim=512,
        codebook_size=args.big_codebook_size,
        small_codebook_size=args.small_codebook_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_blocks=8,
        dim_head=64,
        heads=8,
        ff_mult=4,
    )
    sd = torch.load(args.checkpoint, map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"warn: unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if missing:
        print(f"warn: missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    model.eval().to(args.device)

    transforms = TransformsGenerator.get_final_transforms(model.image_size, None)
    ds = MultiEnvironmentDataset(
        f"{args.dataset_root_dpath}/{args.dataset_name}",
        seq_length_input=args.num_frames - 1,
        seq_step=20,
        split_type="session",
        split="validation",
        transform=transforms["train"],
        format=DatasetOutputFormat.IVG,
        enable_cache=False,
        n_workers=args.n_workers,
        n_envs=0,
        n_samples=max(1, args.num_samples // 100 + 1),  # per-env sample
    )
    # Subset to args.num_samples total
    from torch.utils.data import Subset
    ds = Subset(ds, range(min(args.num_samples, len(ds))))
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_workers, pin_memory=True,
    )

    K = args.small_codebook_size
    K_big = args.big_codebook_size
    counts = torch.zeros(K, dtype=torch.long)
    big_counts = torch.zeros(K_big, dtype=torch.long) if args.also_big else None
    total_rest_positions = 0
    total_first_positions = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            # dataset yields (B, F, C, H, W); model expects (B, C, F, H, W)
            videos = rearrange(batch["input_frames"], "b f c h w -> b c f h w").to(args.device)
            # forward_dual_codebook returns (B, pt, ph, pw) indices (frame 0 + frames 1+ concatenated)
            indices = model.forward_dual_codebook(
                videos, model.small_vq, return_only_codebook_ids=True
            )
            # strip frame 0, keep frames 1-7
            rest = indices[:, 1:, :, :].cpu().reshape(-1)
            counts += torch.bincount(rest, minlength=K)
            total_rest_positions += rest.numel()
            if big_counts is not None:
                first = indices[:, :1, :, :].cpu().reshape(-1)
                big_counts += torch.bincount(first, minlength=K_big)
                total_first_positions += first.numel()

    print()
    print(f"=== small-VQ utilisation (K={K}) ===")
    print(f"total rest-frame positions observed: {total_rest_positions}")

    probs = counts.float() / max(1, total_rest_positions)
    used = int((counts > 0).sum().item())
    active_strict = int((probs >= 0.01).sum().item())  # at least 1% mass
    entropy = -(probs[probs > 0] * probs[probs > 0].log2()).sum().item()
    max_entropy = math.log2(K)
    perplexity = 2 ** entropy

    print(f"unique codes with any usage       : {used} / {K}")
    print(f"codes carrying >=1% of mass       : {active_strict} / {K}")
    print(f"entropy                           : {entropy:.3f} / {max_entropy:.3f} bits  (uniform={max_entropy:.3f})")
    print(f"perplexity (effective # codes)    : {perplexity:.2f} / {K}")
    print()
    print("distribution (code_id: count, %mass):")
    order = torch.argsort(counts, descending=True)
    for i, code in enumerate(order.tolist()):
        c = counts[code].item()
        pct = 100.0 * c / max(1, total_rest_positions)
        bar = "#" * int(pct / 2)
        print(f"  code {code:3d}: {c:8d}  {pct:5.1f}%  {bar}")

    if big_counts is not None:
        print()
        print(f"=== big-VQ utilisation (K={K_big}, FROZEN) ===")
        print(f"total frame-0 positions observed  : {total_first_positions}")
        big_probs = big_counts.float() / max(1, total_first_positions)
        big_used = int((big_counts > 0).sum().item())
        big_active_01 = int((big_probs >= 0.001).sum().item())  # ≥0.1% mass
        big_entropy = -(big_probs[big_probs > 0] * big_probs[big_probs > 0].log2()).sum().item()
        big_max_entropy = math.log2(K_big)
        big_perplexity = 2 ** big_entropy
        print(f"unique codes with any usage       : {big_used} / {K_big}")
        print(f"codes carrying >=0.1% of mass     : {big_active_01} / {K_big}")
        print(f"entropy                           : {big_entropy:.3f} / {big_max_entropy:.3f} bits")
        print(f"perplexity (effective # codes)    : {big_perplexity:.2f} / {K_big}")
        top20 = torch.argsort(big_counts, descending=True)[:20].tolist()
        print("top-20 big-VQ codes (code_id: %mass):")
        for code in top20:
            c = big_counts[code].item()
            pct = 100.0 * c / max(1, total_first_positions)
            print(f"  code {code:4d}: {pct:5.2f}%")


if __name__ == "__main__":
    main()
