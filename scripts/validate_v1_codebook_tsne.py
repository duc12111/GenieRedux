#!/usr/bin/env python3
"""V1: t-SNE visualisation of the small-VQ codebook.

Loads a DualCodebookTokenizer checkpoint, extracts the small-VQ codebook
embeddings, projects them to 2D with t-SNE, and saves a scatter plot
coloured by usage frequency.

Signal to read:
- Scattered clusters in 2D → codebook has structure (good)
- Uniform cloud → codes are indistinguishable, no learned organisation
- A few points far from the rest → potential "motion primitive" clusters

Usage:
    python scripts/validate_v1_codebook_tsne.py \
      --checkpoint checkpoints/tokenizer_dual_cb/<run>/model-5000.pt \
      --small-codebook-size 16 \
      --out outputs/validation/<run>/v1_tsne.png
"""
from __future__ import annotations

import argparse
import math
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
from sklearn.manifold import TSNE
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
                   help="number of validation windows to tally usage from")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--out", default="outputs/validation/v1_tsne.png")
    p.add_argument("--perplexity", type=float, default=5.0,
                   help="t-SNE perplexity; for K=16 codes keep this small")
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

    # Extract codebook — shape (1, K, d_cb) or (K, d_cb)
    codebook = model.small_vq._codebook.embed.detach().cpu()
    if codebook.ndim == 3:
        codebook = codebook.squeeze(0)
    K, d = codebook.shape
    assert K == args.small_codebook_size, f"codebook shape {codebook.shape} vs expected K={args.small_codebook_size}"

    # Tally usage on validation batch
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

    counts = torch.zeros(K, dtype=torch.long)
    total = 0
    with torch.no_grad():
        for batch in dl:
            videos = rearrange(batch["input_frames"], "b f c h w -> b c f h w").to(args.device)
            indices = model.forward_dual_codebook(videos, model.small_vq, return_only_codebook_ids=True)
            rest = indices[:, 1:, :, :].cpu().reshape(-1)
            counts += torch.bincount(rest, minlength=K)
            total += rest.numel()

    probs = counts.float() / max(1, total)

    # t-SNE to 2D. For small K (≤50), perplexity must be < K.
    perplexity = min(args.perplexity, max(2.0, (K - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca", learning_rate="auto")
    codebook_2d = tsne.fit_transform(codebook.numpy())  # (K, 2)

    # Compute code stats
    active = int((counts > 0).sum().item())
    entropy_nats = -(probs[probs > 0] * probs[probs > 0].log()).sum().item() if (probs > 0).any() else 0.0
    entropy_bits = entropy_nats / math.log(2)
    perplexity_eff = math.exp(entropy_nats)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Size scaled by usage frequency (log + floor so unused codes are still visible)
    sizes = 40 + 1200 * probs.numpy()
    colours = probs.numpy()
    sc = ax.scatter(codebook_2d[:, 0], codebook_2d[:, 1],
                    s=sizes, c=colours, cmap="viridis",
                    edgecolors="black", linewidths=0.5, alpha=0.85)
    for k in range(K):
        ax.annotate(str(k), codebook_2d[k], ha="center", va="center", fontsize=9)
    plt.colorbar(sc, ax=ax, label="usage frequency (fraction of patches)")
    ax.set_title(
        f"small-VQ codebook t-SNE (K={K}, d_cb={d})\n"
        f"active codes: {active}/{K}  |  "
        f"entropy: {entropy_bits:.2f}/{math.log2(K):.2f} bits  |  "
        f"effective codes: {perplexity_eff:.2f}"
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved t-SNE plot to {out_path}")
    print(f"codebook: K={K}, d_cb={d}, active={active}, entropy={entropy_bits:.2f} bits")


if __name__ == "__main__":
    main()
