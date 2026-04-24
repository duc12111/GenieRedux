#!/usr/bin/env python3
"""Test C: do small-VQ codes linearly predict game actions?

Restricts to the 50-game RetroAct Control subset (the only games with
labelled actions). For each window, extracts five feature sources and
trains a linear probe targeting the window-aggregated action label.

Feature sources:
  small_vq    — histogram over small-VQ indices for frames 1-7
  big_vq      — histogram over big-VQ indices for frame 0
  encoder     — mean-pooled post-encoder pre-VQ features over frames 1-7
  pixel_delta — per-channel mean |Δpixel| summed over patch grid
  random      — Gaussian noise of similar dim

Target: majority-vote action class across the window.

Pass criterion:
  small_vq accuracy ≥ 2× chance (e.g. > 2/N_classes)
  AND
  small_vq accuracy ≥ 1.3× big_vq accuracy

Example:
    python scripts/test_c_action_probe.py \
      --checkpoint checkpoints/tokenizer_dual_cb/dual_cb_train_cb64_allgames_inv0_flow0_deltanone/model-5000.pt \
      --small-codebook-size 64 \
      --num-samples 400 \
      --out outputs/validation/R4_cb64/test_c_action_probe.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from einops import pack, rearrange, unpack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data.data import DatasetOutputFormat, MultiEnvironmentDataset, TransformsGenerator
from models import DualCodebookTokenizer


def load_control_whitelist(csv_path: str) -> list[str]:
    names = []
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            name = row.get("game", "").strip().lower()
            if name:
                names.append(name)
    return names


@torch.no_grad()
def extract_feature_histograms(model, videos, actions):
    """Per-window features + action label.

    Returns dict of feature arrays (batch_dim, feat_dim) and labels (batch_dim,).
    """
    B, C, T, H, W = videos.shape
    h, w = model.patch_height_width
    ph, pw = model.patch_size
    K_small = model.small_vq.codebook_size
    K_big = model.vq.codebook_size

    # Forward through the dual-codebook to get indices + pre-VQ encoder features
    first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]
    first_tokens = model.to_patch_emb_first_frame(first_frame)
    rest_tokens = model.to_patch_emb(rest_frames)
    tokens = torch.cat((first_tokens, rest_tokens), dim=1)
    encoded = model.encode(tokens)

    first_enc = encoded[:, :1]
    first_flat, ps_first = pack([first_enc], "b * d")
    _, first_idx, _ = model.vq(first_flat)
    # Apply delta_ref for small VQ if configured (matches training-time quantisation)
    rest_enc = encoded[:, 1:]
    delta_ref = getattr(model, "delta_ref", "none")
    if delta_ref == "anchor":
        rest_enc = rest_enc - first_enc.expand_as(rest_enc)
    elif delta_ref == "rolling":
        prev = torch.cat((first_enc, rest_enc[:, :-1]), dim=1)
        rest_enc = rest_enc - prev
    rest_flat, ps_rest = pack([rest_enc], "b * d")
    _, rest_idx, _ = model.small_vq(rest_flat)

    small_idx = rest_idx.reshape(B, T - 1, h, w).cpu().numpy()
    big_idx = first_idx.reshape(B, 1, h, w).cpu().numpy()

    # small_vq histogram per window
    small_hist = np.zeros((B, K_small), dtype=np.float32)
    for b in range(B):
        counts = np.bincount(small_idx[b].reshape(-1), minlength=K_small)
        small_hist[b] = counts / counts.sum()

    # big_vq histogram per window
    big_hist = np.zeros((B, K_big), dtype=np.float32)
    for b in range(B):
        counts = np.bincount(big_idx[b].reshape(-1), minlength=K_big)
        big_hist[b] = counts / counts.sum()

    # Encoder features: mean-pool over (T-1, h, w) → (B, d). Use pre-delta features.
    encoder_feat = encoded[:, 1:].mean(dim=(1, 2, 3)).cpu().numpy()  # (B, d)

    # Pixel-delta: per-channel mean |Δpixel| over (T-1, patch H, patch W) → (B, C)
    vid_patch = videos.reshape(B, C, T, h, ph, w, pw)
    diffs = (vid_patch[:, :, 1:] - vid_patch[:, :, :-1]).abs()  # (B, C, T-1, h, ph, w, pw)
    pixel_delta = diffs.mean(dim=(2, 3, 4, 5, 6)).cpu().numpy()  # (B, C)

    # Action target: actions shape (B, T, num_classes) one-hot
    # Majority vote across motion frames (excluding frame 0)
    # actions is torch.Tensor coming from batch
    if actions is not None:
        a = actions.cpu().numpy()  # could be (B, T, C) or (B, T)
        if a.ndim == 3:
            # one-hot or multi-hot: take argmax per frame, majority across motion frames
            per_frame = a[:, 1:].argmax(axis=-1)  # (B, T-1)
        else:
            per_frame = a[:, 1:]
        # Majority
        labels = np.array([np.bincount(per_frame[b]).argmax() for b in range(B)])
    else:
        labels = np.zeros(B, dtype=np.int64)

    return {
        "small_vq": small_hist,
        "big_vq": big_hist,
        "encoder": encoder_feat,
        "pixel_delta": pixel_delta,
        "labels": labels,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--small-codebook-size", type=int, default=64)
    p.add_argument("--big-codebook-size", type=int, default=1024)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--delta-ref", default="none", choices=["none", "anchor", "rolling"])
    p.add_argument("--dataset-root-dpath", default="data_generation/datasets")
    p.add_argument("--dataset-name", default="retro_act_v0.0.0_g200_256")
    p.add_argument("--control-csv",
                   default="data_generation/annotations/RetroAct_v0.1_control_GenieRedux-G-50_sublist.csv")
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--out", default="outputs/validation/test_c_action_probe.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    whitelist = load_control_whitelist(args.control_csv)
    print(f"control whitelist: {len(whitelist)} games")

    model = DualCodebookTokenizer(
        dim=512,
        codebook_size=args.big_codebook_size,
        small_codebook_size=args.small_codebook_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_blocks=8, dim_head=64, heads=8, ff_mult=4,
        delta_ref=args.delta_ref,
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
        n_samples=max(1, args.num_samples // max(1, len(whitelist)) + 1),
        whitelist=whitelist,
    )
    if len(ds) == 0:
        raise RuntimeError("whitelist produced empty dataset; check game-name normalisation")
    ds = Subset(ds, range(min(args.num_samples, len(ds))))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    features = {k: [] for k in ["small_vq", "big_vq", "encoder", "pixel_delta"]}
    labels_all = []
    n_batches = 0
    with torch.no_grad():
        for batch in dl:
            videos = rearrange(batch["input_frames"], "b f c h w -> b c f h w").to(args.device)
            actions = batch.get("actions", None)
            if actions is not None:
                actions = actions.to(args.device)
            out = extract_feature_histograms(model, videos, actions)
            for k in features:
                features[k].append(out[k])
            labels_all.append(out["labels"])
            n_batches += 1

    features = {k: np.concatenate(v, axis=0) for k, v in features.items()}
    labels = np.concatenate(labels_all)
    print(f"extracted {len(labels)} windows, {n_batches} batches")

    # Random feature baseline
    features["random"] = rng.standard_normal((len(labels), features["small_vq"].shape[1])).astype(np.float32)

    # Analyse label distribution
    n_classes = len(np.unique(labels))
    chance = np.bincount(labels).max() / len(labels)  # majority-class baseline
    uniform_chance = 1.0 / n_classes

    # Train linear probes
    results = {}
    for name, X in features.items():
        if len(np.unique(labels)) < 2:
            results[name] = {"acc": 0.0, "bal_acc": 0.0, "note": "only one class"}
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.25, random_state=0, stratify=labels if n_classes > 1 else None)
        if len(np.unique(y_tr)) < 2:
            results[name] = {"acc": 0.0, "bal_acc": 0.0, "note": "only one class in train"}
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        bal = balanced_accuracy_score(y_te, y_pred)
        results[name] = {"acc": float(acc), "bal_acc": float(bal)}

    # Pass criterion
    small_acc = results["small_vq"]["acc"]
    big_acc = results["big_vq"]["acc"]
    random_acc = results["random"]["acc"]
    pass_chance = small_acc >= 2 * uniform_chance
    pass_vs_big = big_acc > 0 and small_acc >= 1.3 * big_acc
    pass_gate = pass_chance and pass_vs_big

    # Write results
    lines = []
    lines.append("=== Test C: action probe on labelled subset ===")
    lines.append(f"checkpoint                : {args.checkpoint}")
    lines.append(f"control whitelist games   : {len(whitelist)}")
    lines.append(f"extracted windows         : {len(labels)}")
    lines.append(f"action classes observed   : {n_classes}")
    lines.append(f"uniform chance            : {uniform_chance:.3f}")
    lines.append(f"majority-class baseline   : {chance:.3f}")
    lines.append(f"label distribution (top 10): {dict(zip(*np.unique(labels, return_counts=True)))}")
    lines.append("")
    lines.append(f"{'feature':<14}{'accuracy':>12}{'balanced acc':>16}{'ratio vs big_vq':>18}")
    lines.append("-" * 60)
    for name in ["small_vq", "big_vq", "encoder", "pixel_delta", "random"]:
        r = results[name]
        ratio = (r["acc"] / big_acc) if big_acc > 0 else 0.0
        lines.append(f"{name:<14}{r['acc']:>12.4f}{r['bal_acc']:>16.4f}{ratio:>18.2f}×")
    lines.append("")
    lines.append("Pass criteria:")
    lines.append(f"  small_vq ≥ 2 × uniform chance ({2 * uniform_chance:.3f}) : {'YES' if pass_chance else 'NO'}  (got {small_acc:.3f})")
    lines.append(f"  small_vq ≥ 1.3 × big_vq accuracy                        : {'YES' if pass_vs_big else 'NO'}  (got {small_acc / max(big_acc, 1e-9):.2f}×)")
    lines.append("")
    lines.append(f"PASS GATE                                                : {'YES' if pass_gate else 'NO'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "n_windows": int(len(labels)),
        "n_classes": int(n_classes),
        "uniform_chance": float(uniform_chance),
        "majority_chance": float(chance),
        "results": results,
        "pass_gate": bool(pass_gate),
    }, indent=2))

    print("\n".join(lines))
    print(f"\nsaved to {out_path} and {json_path}")


if __name__ == "__main__":
    main()
