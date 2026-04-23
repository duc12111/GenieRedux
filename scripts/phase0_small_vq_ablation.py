#!/usr/bin/env python3
"""Phase 0 D1: measure how much the small VQ contributes to reconstruction.

Runs forward_dual_codebook twice on the same batch:
  (a) normal: frame 0 → big VQ, frames 1-7 → small VQ, decode all
  (b) ablated: replace quantised rest features with zeros before decode

Compares PSNR/SSIM/MSE of the two reconstructions against ground truth.

Interpretation:
  * Large gap (>2 dB PSNR) → small VQ IS contributing → collapse means we
    lost something the decoder was using.
  * Small gap (<0.5 dB) → small VQ is architecturally redundant under the
    current shared encoder → Fix A alone won't help; need supervision or
    architectural change.

Example:
    python scripts/phase0_small_vq_ablation.py \
      --checkpoint checkpoints/tokenizer_dual_cb/dual_cb_train_cb16_allgames/model-5000.pt \
      --num-samples 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch.utils.data import DataLoader, Subset

from data.data import DatasetOutputFormat, MultiEnvironmentDataset, TransformsGenerator
from models import DualCodebookTokenizer


@torch.no_grad()
def run_ablated(model: DualCodebookTokenizer, videos: torch.Tensor) -> torch.Tensor:
    """Same as forward_dual_codebook's reconstruction path but with rest_q zeroed."""
    b, c, f = videos.shape[:3]
    h, w = model.patch_height_width

    first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]
    first_frame_tokens = model.to_patch_emb_first_frame(first_frame)
    rest_frames_tokens = model.to_patch_emb(rest_frames)
    tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)
    tokens = model.encode(tokens)

    first_tokens = tokens[:, :1, :, :, :]
    first_flat, ps_first = pack([first_tokens], "b * d")
    first_q, _, _ = model.vq(first_flat)

    (first_q,) = unpack(first_q, ps_first, "b * d")

    t_rest = f - 1
    rest_q_zeros = torch.zeros(b, t_rest, h, w, first_q.shape[-1], device=videos.device, dtype=first_q.dtype)
    tokens_q = torch.cat([first_q, rest_q_zeros], dim=1)
    return model.decode(tokens_q)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()


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
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
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
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        non_ema_missing = [k for k in missing if "cluster_size" not in k and "embed_avg" not in k]
        if non_ema_missing:
            print(f"warn: missing keys (non-EMA): {non_ema_missing[:5]}{'...' if len(non_ema_missing) > 5 else ''}")
    model.eval().to(args.device)

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

    psnr_full, psnr_ablated, mse_delta = [], [], []
    n_batches = 0
    with torch.no_grad():
        for batch in dl:
            videos = rearrange(batch["input_frames"], "b f c h w -> b c f h w").to(args.device)
            recon_full = model.forward_dual_codebook(videos, model.small_vq, return_recons_only=True)
            recon_ablated = run_ablated(model, videos)
            psnr_full.append(psnr(recon_full, videos))
            psnr_ablated.append(psnr(recon_ablated, videos))
            mse_full = F.mse_loss(recon_full, videos).item()
            mse_abl = F.mse_loss(recon_ablated, videos).item()
            mse_delta.append(mse_abl - mse_full)
            n_batches += 1

    def mean(xs):
        return sum(xs) / max(1, len(xs))

    print()
    print("=== Phase 0 D1: small-VQ ablation ===")
    print(f"batches processed : {n_batches}")
    print(f"PSNR (full)       : {mean(psnr_full):6.3f} dB")
    print(f"PSNR (ablated)    : {mean(psnr_ablated):6.3f} dB")
    print(f"PSNR delta        : {mean(psnr_full) - mean(psnr_ablated):+6.3f} dB  (full - ablated)")
    print(f"MSE delta         : {mean(mse_delta):+.6f}           (ablated - full)")
    print()
    delta = mean(psnr_full) - mean(psnr_ablated)
    if delta < 0.5:
        print(f"INTERPRETATION: delta < 0.5 dB → small VQ is ~redundant. Fix A alone")
        print(f"  likely won't help. Phase 2 supervision or Phase 3 architectural needed.")
    elif delta > 2.0:
        print(f"INTERPRETATION: delta > 2 dB → small VQ IS contributing to reconstruction.")
        print(f"  Fix A alone may suffice to restore a healthy codebook.")
    else:
        print(f"INTERPRETATION: mid-range delta → small VQ contributes somewhat.")
        print(f"  Fix A may help; supervision is a reasonable belt-and-suspenders.")


if __name__ == "__main__":
    main()
