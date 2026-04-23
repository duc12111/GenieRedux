#!/usr/bin/env python3
"""V3: linear probe — do small-VQ codes carry motion info beyond baselines?

Compares five feature sources as predictors of motion-related targets on held-out
validation data. Reports a table: each feature source × each target.

Feature sources (per patch, per frame-transition):
  small_vq    — small-VQ code EMBEDDING (codebook[index], shape (codebook_dim,))
  big_vq      — big-VQ code embedding (from frame 0, broadcast to motion frames)
  pixel_delta — per-patch mean |frame_t - frame_{t-1}| (appearance-agnostic motion proxy)
  encoder     — post-encoder pre-VQ feature (shape (d,)); continuous baseline with no bottleneck
  random      — Gaussian noise of similar dim (chance-level baseline)

Targets (same patch, same frame):
  pixel_motion  — continuous Ridge regression target: mean |Δpixel| over the patch
  token_change  — binary logistic: did the small-VQ code change vs. prev frame?

Signal to read — small-VQ should beat big-VQ + pixel-delta + random on pixel_motion
R², OR at least be comparable, if the codes carry structured motion info.

Usage:
    python scripts/validate_v3_action_probe.py \
      --checkpoint checkpoints/tokenizer_dual_cb/<run>/model-5000.pt \
      --small-codebook-size 16 \
      --num-samples 60 \
      --out outputs/validation/<run>/v3_probe.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from einops import pack, rearrange, unpack
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data.data import DatasetOutputFormat, MultiEnvironmentDataset, TransformsGenerator
from models import DualCodebookTokenizer


@torch.no_grad()
def extract_all_features(model, videos):
    """Run encoder + VQs, collect every feature source we want to compare.

    Returns a dict of per-patch feature arrays, each shaped (B*(T-1)*h*w, feat_dim),
    plus the targets for the same positions. All arrays are aligned position-by-position.
    """
    B, C, T, H, W = videos.shape
    h, w = model.patch_height_width
    ph, pw = model.patch_size

    # Patch-embed and encode (same path as forward_dual_codebook)
    first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]
    first_tokens = model.to_patch_emb_first_frame(first_frame)           # (B,1,h,w,d)
    rest_tokens = model.to_patch_emb(rest_frames)                        # (B,T-1,h,w,d)
    tokens = torch.cat((first_tokens, rest_tokens), dim=1)               # (B,T,h,w,d)
    encoded = model.encode(tokens)                                       # (B,T,h,w,d)

    # Big VQ on frame 0 → (B, h*w) indices, look up embeddings
    first_enc = encoded[:, :1]
    first_flat, ps_first = pack([first_enc], "b * d")
    _, first_idx, _ = model.vq(first_flat)                               # (B, h*w)
    big_cb = model.vq._codebook.embed
    if big_cb.ndim == 3:
        big_cb = big_cb.squeeze(0)
    big_emb = big_cb[first_idx]                                          # (B, h*w, d_cb)
    big_emb = big_emb.reshape(B, h, w, -1)                               # (B, h, w, d_cb)

    # Small VQ on frames 1+ → (B, (T-1)*h*w) indices, look up embeddings
    rest_enc = encoded[:, 1:]
    rest_flat, ps_rest = pack([rest_enc], "b * d")
    _, rest_idx, _ = model.small_vq(rest_flat)
    small_cb = model.small_vq._codebook.embed
    if small_cb.ndim == 3:
        small_cb = small_cb.squeeze(0)
    small_emb = small_cb[rest_idx]                                       # (B, (T-1)*h*w, d_cb)
    small_emb = small_emb.reshape(B, T - 1, h, w, -1)

    # Small-VQ indices reshaped for token_change target
    small_idx = rest_idx.reshape(B, T - 1, h, w)

    # Pixel deltas: per-patch |frame_t - frame_{t-1}|, mean over channel and pixels
    # Shape into (B, T, h, ph, w, pw) then diff.
    vid = videos
    vid_patch = vid.reshape(B, C, T, h, ph, w, pw)
    diffs = (vid_patch[:, :, 1:] - vid_patch[:, :, :-1]).abs()           # (B, C, T-1, h, ph, w, pw)
    pixel_delta_scalar = diffs.mean(dim=(1, 4, 6))                       # (B, T-1, h, w)
    # pixel_delta feature vector = per-channel means (richer than scalar)
    pixel_delta_per_channel = diffs.mean(dim=(4, 6))                     # (B, C, T-1, h, w)
    pixel_delta_feat = pixel_delta_per_channel.permute(0, 2, 3, 4, 1)    # (B, T-1, h, w, C)

    # Encoder features (post-encoder pre-VQ) for frames 1+
    encoder_feat = encoded[:, 1:]                                        # (B, T-1, h, w, d)

    # Big-VQ embedding broadcast to motion frames (static appearance reference)
    big_emb_expanded = big_emb.unsqueeze(1).expand(B, T - 1, h, w, big_emb.shape[-1])  # (B,T-1,h,w,d_cb)

    # Targets
    # pixel_motion = per-patch scalar |Δpixel| (continuous target)
    pixel_motion_target = pixel_delta_scalar                              # (B, T-1, h, w)
    # token_change between consecutive motion frames (binary)
    # frame 1 vs frame 0-in-rest: small_idx[:, 0] is frame 1's small code; we compare
    # consecutive small frames starting from frame 2
    # → binary target has shape (B, T-2, h, w); align the feature arrays by dropping last time step.
    token_change_target = (small_idx[:, 1:] != small_idx[:, :-1]).to(torch.long)  # (B, T-2, h, w)

    def flatten_all_time(x):
        # (B, T-1, h, w, feat) → (B*(T-1)*h*w, feat)
        return x.reshape(-1, x.shape[-1])

    def flatten_scalar(x):
        # (B, T-1, h, w) → (B*(T-1)*h*w,)
        return x.reshape(-1)

    features = {
        "small_vq":    flatten_all_time(small_emb),
        "big_vq":      flatten_all_time(big_emb_expanded),
        "pixel_delta": flatten_all_time(pixel_delta_feat),
        "encoder":     flatten_all_time(encoder_feat),
    }
    targets_full = {
        "pixel_motion": flatten_scalar(pixel_motion_target),              # aligned with all of T-1
        "small_idx":    flatten_scalar(small_idx),                        # used to build token_change
    }

    # token_change aligns to a DIFFERENT slice (needs consecutive motion frames)
    # drop last time step of each feature to line up with token_change_target which is (B, T-2, h, w)
    def drop_last_time(x):
        # (B, T-1, h, w, feat) → (B, T-2, h, w, feat) → flatten
        return x[:, :-1].reshape(-1, x.shape[-1])
    features_tc = {k: drop_last_time(v_5d) for k, v_5d in [
        ("small_vq", small_emb),
        ("big_vq", big_emb_expanded),
        ("pixel_delta", pixel_delta_feat),
        ("encoder", encoder_feat),
    ]}
    tc_target = token_change_target.reshape(-1).cpu().numpy()

    out = {
        "features": {k: v.cpu().numpy() for k, v in features.items()},
        "features_tc": {k: v.cpu().numpy() for k, v in features_tc.items()},
        "targets": {
            "pixel_motion": targets_full["pixel_motion"].cpu().numpy(),
            "token_change": tc_target,
        },
    }
    return out


def run_probe(X_train, X_test, y_train, y_test, task):
    """Fit a linear model appropriate for the task, return the held-out metric."""
    if task == "regression":
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return r2
    elif task == "classification":
        model = LogisticRegression(max_iter=500, n_jobs=-1)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))
    else:
        raise ValueError(task)


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
    p.add_argument("--num-samples", type=int, default=60,
                   help="validation windows to extract features from")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--subsample", type=int, default=40000,
                   help="max positions to keep for fitting (full set can be huge)")
    p.add_argument("--out", default="outputs/validation/v3_probe.txt")
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

    # Accumulate features across batches
    features_acc = {k: [] for k in ["small_vq", "big_vq", "pixel_delta", "encoder"]}
    features_tc_acc = {k: [] for k in ["small_vq", "big_vq", "pixel_delta", "encoder"]}
    pm_target, tc_target = [], []

    print(f"extracting features on {len(ds)} windows...")
    for batch_idx, batch in enumerate(dl):
        videos = rearrange(batch["input_frames"], "b f c h w -> b c f h w").to(args.device)
        out = extract_all_features(model, videos)
        for k, v in out["features"].items():
            features_acc[k].append(v)
        for k, v in out["features_tc"].items():
            features_tc_acc[k].append(v)
        pm_target.append(out["targets"]["pixel_motion"])
        tc_target.append(out["targets"]["token_change"])

    features = {k: np.concatenate(v, axis=0) for k, v in features_acc.items()}
    features_tc = {k: np.concatenate(v, axis=0) for k, v in features_tc_acc.items()}
    pm_target = np.concatenate(pm_target)
    tc_target = np.concatenate(tc_target)

    # Add random baselines (same dim as small_vq)
    d_small = features["small_vq"].shape[1]
    features["random"] = rng.standard_normal((features["small_vq"].shape[0], d_small)).astype(np.float32)
    features_tc["random"] = rng.standard_normal((features_tc["small_vq"].shape[0], d_small)).astype(np.float32)

    # Subsample if too big
    N_full = features["small_vq"].shape[0]
    if N_full > args.subsample:
        keep = rng.choice(N_full, size=args.subsample, replace=False)
        features = {k: v[keep] for k, v in features.items()}
        pm_target = pm_target[keep]
    N_tc_full = features_tc["small_vq"].shape[0]
    if N_tc_full > args.subsample:
        keep_tc = rng.choice(N_tc_full, size=args.subsample, replace=False)
        features_tc = {k: v[keep_tc] for k, v in features_tc.items()}
        tc_target = tc_target[keep_tc]

    # Run probes
    result_lines = []
    header = f"{'feature':<14}{'pixel_motion R²':>20}{'token_change acc':>22}"
    print(header); result_lines.append(header)
    result_lines.append("-" * len(header))

    summary = {}
    for source in ["small_vq", "big_vq", "pixel_delta", "encoder", "random"]:
        # Regression on pixel_motion (continuous)
        X = features[source]
        y = pm_target
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        r2 = run_probe(X_tr, X_te, y_tr, y_te, "regression")

        # Classification on token_change (binary)
        X_tc = features_tc[source]
        y_tc = tc_target
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_tc, y_tc, test_size=0.2, random_state=0)
        acc = run_probe(X_tr2, X_te2, y_tr2, y_te2, "classification")

        line = f"{source:<14}{r2:>20.4f}{acc:>22.4f}"
        print(line); result_lines.append(line)
        summary[source] = {"pixel_motion_r2": float(r2), "token_change_acc": float(acc)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(result_lines) + "\n")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved probe results to {out_path} and {json_path}")

    # Interpretation hint
    small_r2 = summary["small_vq"]["pixel_motion_r2"]
    big_r2 = summary["big_vq"]["pixel_motion_r2"]
    pd_r2 = summary["pixel_delta"]["pixel_motion_r2"]
    rand_r2 = summary["random"]["pixel_motion_r2"]
    print()
    print("quick reads:")
    print(f"  small_vq vs random       : {small_r2:+.3f} vs {rand_r2:+.3f}  (want small_vq > random)")
    print(f"  small_vq vs big_vq       : {small_r2:+.3f} vs {big_r2:+.3f}  (want small_vq > big_vq → codes encode MOTION beyond appearance)")
    print(f"  small_vq vs pixel_delta  : {small_r2:+.3f} vs {pd_r2:+.3f}  (want small_vq ≳ pixel_delta → codes encode structured motion)")


if __name__ == "__main__":
    main()
