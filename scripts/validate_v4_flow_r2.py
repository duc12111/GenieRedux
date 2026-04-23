#!/usr/bin/env python3
"""V4: Farnebäck optical-flow R² probe.

Tests whether small-VQ code embeddings linearly predict ground-truth optical flow.
Farnebäck flow is computed on each frame pair, patch-averaged to the small-VQ
grid resolution, then a Ridge regressor is fit: code_embedding → (dx, dy).

Reported: R² on held-out split, per dimension and overall. Higher R² → codes
linearly encode flow → direct evidence the codes are a motion representation.

Baselines computed alongside small_vq for context:
- big_vq:      embedding of the frame-0 anchor (should be FAR worse; it's appearance-only)
- pixel_delta: raw |Δpixel| per patch (should be strong; it's derived from the same pixels as flow)
- random:      noise, chance-level

Signal to read:
- small_vq R² ≥ 0.3 → motion representation quantitatively confirmed
- small_vq R² >> big_vq R² → codes encode motion BEYOND appearance (direct D1 claim)
- small_vq R² << pixel_delta R² → the bottleneck throws away a lot of motion info
  (might mean more codes needed, or supervision like this loss as an aux could help)

Usage:
    python scripts/validate_v4_flow_r2.py \
      --checkpoint checkpoints/tokenizer_dual_cb/<run>/model-5000.pt \
      --small-codebook-size 16 \
      --num-samples 60 \
      --out outputs/validation/<run>/v4_flow.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
from einops import pack, rearrange
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data.data import DatasetOutputFormat, MultiEnvironmentDataset, TransformsGenerator
from models import DualCodebookTokenizer


def compute_farneback_flow_patch(videos_chw: np.ndarray, patch_size: int) -> np.ndarray:
    """
    videos_chw: (B, C, T, H, W) float in [0, 1]  (numpy)
    Returns patch-averaged flow (B, T-1, h, w, 2) where h = H // patch_size.

    Uses cv2.calcOpticalFlowFarneback on the luminance channel of each consecutive
    pair. Flow is then mean-pooled over patch-size blocks. Output is normalised to
    patch-units (flow / patch_size) so 1.0 ≈ 1 patch of displacement, keeping
    things scale-independent.
    """
    B, C, T, H, W = videos_chw.shape
    h = H // patch_size
    w = W // patch_size
    out = np.zeros((B, T - 1, h, w, 2), dtype=np.float32)

    for b in range(B):
        # Luminance: simple mean across channels for retro games; fine for flow.
        lum = videos_chw[b].mean(axis=0)  # (T, H, W)
        lum_u8 = np.clip(lum * 255.0, 0, 255).astype(np.uint8)

        for t in range(T - 1):
            # cv2 Farnebäck: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
            flow = cv2.calcOpticalFlowFarneback(
                lum_u8[t], lum_u8[t + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )  # (H, W, 2) — [dx, dy]
            # Average-pool to patch grid
            flow_reshaped = flow.reshape(h, patch_size, w, patch_size, 2)
            flow_patch = flow_reshaped.mean(axis=(1, 3))  # (h, w, 2)
            # Normalise to patch units (keeps scale independent of patch_size)
            out[b, t] = flow_patch / float(patch_size)

    return out


@torch.no_grad()
def extract_code_embeddings(model, videos):
    """Run the full dual-codebook path and return:
    - small code embeddings: (B, T-1, h, w, d_cb)
    - big code embeddings broadcast to motion frames: (B, T-1, h, w, d_cb)
    - pixel deltas per patch per channel: (B, T-1, h, w, C)
    """
    B, C, T, H, W = videos.shape
    h, w = model.patch_height_width
    ph, pw = model.patch_size

    first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]
    first_tokens = model.to_patch_emb_first_frame(first_frame)
    rest_tokens = model.to_patch_emb(rest_frames)
    tokens = torch.cat((first_tokens, rest_tokens), dim=1)
    encoded = model.encode(tokens)

    first_enc = encoded[:, :1]
    first_flat, _ = pack([first_enc], "b * d")
    _, first_idx, _ = model.vq(first_flat)
    big_cb = model.vq._codebook.embed
    if big_cb.ndim == 3:
        big_cb = big_cb.squeeze(0)
    big_emb = big_cb[first_idx].reshape(B, h, w, -1)
    big_emb_expanded = big_emb.unsqueeze(1).expand(B, T - 1, h, w, big_emb.shape[-1])

    rest_enc = encoded[:, 1:]
    rest_flat, _ = pack([rest_enc], "b * d")
    _, rest_idx, _ = model.small_vq(rest_flat)
    small_cb = model.small_vq._codebook.embed
    if small_cb.ndim == 3:
        small_cb = small_cb.squeeze(0)
    small_emb = small_cb[rest_idx].reshape(B, T - 1, h, w, -1)

    # pixel deltas per channel, per patch (for baseline)
    vid_patch = videos.reshape(B, C, T, h, ph, w, pw)
    diffs = (vid_patch[:, :, 1:] - vid_patch[:, :, :-1]).abs()           # (B, C, T-1, h, ph, w, pw)
    pixel_delta_per_channel = diffs.mean(dim=(4, 6)).permute(0, 2, 3, 4, 1)  # (B, T-1, h, w, C)

    return small_emb.cpu().numpy(), big_emb_expanded.cpu().numpy(), pixel_delta_per_channel.cpu().numpy()


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
    p.add_argument("--num-samples", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--subsample", type=int, default=40000)
    p.add_argument("--moving-only-threshold", type=float, default=0.0,
                   help="if > 0, restrict the regression to patches where |flow| > threshold. "
                        "Use 0.2 (patch units) to filter out static patches.")
    p.add_argument("--out", default="outputs/validation/v4_flow.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    rng = np.random.default_rng(0)
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

    small_all, big_all, pd_all, flow_all = [], [], [], []

    print(f"computing flow + embeddings on {len(ds)} windows...")
    for batch in dl:
        videos_t = rearrange(batch["input_frames"], "b f c h w -> b c f h w")   # (B, C, T, H, W)
        videos_np = videos_t.numpy()
        videos_gpu = videos_t.to(args.device)

        flow_patch = compute_farneback_flow_patch(videos_np, args.patch_size)   # (B, T-1, h, w, 2)
        small, big, pd = extract_code_embeddings(model, videos_gpu)

        small_all.append(small.reshape(-1, small.shape[-1]))
        big_all.append(big.reshape(-1, big.shape[-1]))
        pd_all.append(pd.reshape(-1, pd.shape[-1]))
        flow_all.append(flow_patch.reshape(-1, 2))

    small_feat = np.concatenate(small_all, axis=0).astype(np.float32)
    big_feat = np.concatenate(big_all, axis=0).astype(np.float32)
    pd_feat = np.concatenate(pd_all, axis=0).astype(np.float32)
    flow = np.concatenate(flow_all, axis=0).astype(np.float32)
    random_feat = rng.standard_normal(small_feat.shape).astype(np.float32)

    # Optional: filter to moving patches only. Ridge R² on a (dx, dy) target
    # dominated by static (0, 0) patches will report artificially low numbers
    # across all features. Restricting to moving patches measures the real
    # direction-prediction signal.
    n_before = small_feat.shape[0]
    if args.moving_only_threshold > 0:
        flow_mag = np.linalg.norm(flow, axis=1)
        keep_mask = flow_mag > args.moving_only_threshold
        small_feat = small_feat[keep_mask]
        big_feat = big_feat[keep_mask]
        pd_feat = pd_feat[keep_mask]
        random_feat = random_feat[keep_mask]
        flow = flow[keep_mask]
        n_after = small_feat.shape[0]
        print(f"moving-only filter (|flow| > {args.moving_only_threshold}): "
              f"{n_after}/{n_before} patches retained ({100.0*n_after/n_before:.1f}%)")

    N = small_feat.shape[0]
    if N > args.subsample:
        keep = rng.choice(N, size=args.subsample, replace=False)
        small_feat = small_feat[keep]
        big_feat = big_feat[keep]
        pd_feat = pd_feat[keep]
        random_feat = random_feat[keep]
        flow = flow[keep]

    def probe(X):
        X_tr, X_te, y_tr, y_te = train_test_split(X, flow, test_size=0.2, random_state=0)
        model_r = Ridge(alpha=1.0)
        model_r.fit(X_tr, y_tr)
        y_pred = model_r.predict(X_te)
        r2_overall = r2_score(y_te, y_pred)
        r2_dx = r2_score(y_te[:, 0], y_pred[:, 0])
        r2_dy = r2_score(y_te[:, 1], y_pred[:, 1])
        return r2_overall, r2_dx, r2_dy

    result_lines = []
    header = f"{'feature':<14}{'R² (overall)':>16}{'R² (dx)':>12}{'R² (dy)':>12}"
    print(header); result_lines.append(header)
    result_lines.append("-" * len(header))

    summary = {}
    for name, X in [("small_vq", small_feat), ("big_vq", big_feat),
                    ("pixel_delta", pd_feat), ("random", random_feat)]:
        r2, r2x, r2y = probe(X)
        line = f"{name:<14}{r2:>16.4f}{r2x:>12.4f}{r2y:>12.4f}"
        print(line); result_lines.append(line)
        summary[name] = {"r2_overall": float(r2), "r2_dx": float(r2x), "r2_dy": float(r2y)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(result_lines) + "\n")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved flow-R² results to {out_path} and {json_path}")

    s = summary["small_vq"]["r2_overall"]
    b = summary["big_vq"]["r2_overall"]
    print()
    print("quick reads:")
    print(f"  small_vq R² overall      : {s:+.3f}  (want ≥ 0.3)")
    print(f"  small_vq vs big_vq       : {s:+.3f} vs {b:+.3f}  (want small_vq > big_vq → motion beyond appearance)")
    print(f"  small_vq vs pixel_delta  : {s:+.3f} vs {summary['pixel_delta']['r2_overall']:+.3f}  (pixel_delta is near-upper-bound)")


if __name__ == "__main__":
    main()
