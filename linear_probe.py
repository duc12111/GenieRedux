"""Linear probing: measure how much motion information survives VQ."""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from einops import pack, rearrange, unpack
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import DataLoader

from data.data import (
    DatasetOutputFormat,
    MultiEnvironmentDataset,
    TransformsGenerator,
)
from models import construct_model

import logging
logging.basicConfig(level=logging.INFO)
from tools.logger import getLogger
log = getLogger(__name__)


@torch.no_grad()
def extract_embeddings(tokenizer, videos):
    """Run encoder + VQ, return (pre_vq, post_vq, indices) as numpy."""
    b, c, f, *_ = videos.shape
    first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]
    first_frame_tokens = tokenizer.to_patch_emb_first_frame(first_frame)
    rest_frames_tokens = tokenizer.to_patch_emb(rest_frames)
    tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)
    *_, h, w, _ = tokens.shape
    tokens_encoded = tokenizer.encode(tokens)
    pre_vq = tokens_encoded.cpu().numpy()
    tokens_flat, packed_fhw_shape = pack([tokens_encoded], "b * d")
    tokens_q, indices, _ = tokenizer.vq(tokens_flat)
    post_vq = rearrange(tokens_q, "b (t h w) d -> b t h w d", h=h, w=w).cpu().numpy()
    (indices,) = unpack(indices, packed_fhw_shape, "b *")
    indices = indices.cpu().numpy().reshape(b, -1, h, w)
    return pre_vq, post_vq, indices


def compute_probe_targets(videos, indices):
    """Vectorised probe labels for every (t, h, w) excluding last frame."""
    b, t, h, w = indices.shape
    _, c, T, H, W = videos.shape
    ph_scale = H // h
    pw_scale = W // w
    token_change = (indices[:, :-1] != indices[:, 1:]).astype(np.int32).reshape(-1)
    next_token = indices[:, 1:].reshape(-1).astype(np.int64)
    vid_np = videos.cpu().numpy()
    vid_patches = vid_np.reshape(b, c, T, h, ph_scale, w, pw_scale)
    n_t = min(t - 1, T - 1)
    patch_diffs = np.abs(vid_patches[:, :, :n_t] - vid_patches[:, :, 1:n_t + 1])
    pixel_motion = patch_diffs.mean(axis=(1, 4, 6)).reshape(-1).astype(np.float32)
    return {
        "token_change": token_change,
        "next_token": next_token,
        "pixel_motion": pixel_motion,
    }


def flatten_embeddings(emb):
    """(b, t, h, w, d) -> (b*(t-1)*h*w, d), dropping the last time step."""
    return emb[:, :-1].reshape(-1, emb.shape[-1])


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = construct_model(args)
    if getattr(args, "tokenizer_fpath", None):
        tok_state = torch.load(args.tokenizer_fpath, map_location="cpu")
        model.load_state_dict(tok_state["model"])
        del tok_state
    model = model.to(device).eval()
    tokenizer = model
    codebook_size = tokenizer.vq.codebook_size

    dataset_folder = f"{args.eval.dataset_root_dpath}/{args.eval.dataset_name}"
    transforms = TransformsGenerator.get_final_transforms(model.image_size, None)
    eval_whitelist = getattr(args.eval, "whitelist", None)
    if eval_whitelist is not None and len(eval_whitelist) == 0:
        eval_whitelist = None
    test_data = MultiEnvironmentDataset(
        dataset_folder,
        seq_length_input=args.eval.num_frames - 1,
        seq_step=args.eval.seq_step,
        split_type="instance",
        split="all",
        transform=transforms["test"],
        format=DatasetOutputFormat.IVG,
        enable_cache=False,
        n_workers=getattr(args.eval, "n_data_workers", 4),
        n_envs=getattr(args.eval, "n_envs", 0),
        whitelist=eval_whitelist,
    )
    loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    num_first_frames = args.eval.num_first_frames
    sample_num_frames = args.eval.sample_num_frames
    total_frames = num_first_frames + sample_num_frames
    num_samples = int(getattr(args.probe, "num_samples", 20))
    model_name = args.eval.model_name
    save_dir = (
        Path(args.eval.save_root_dpath)
        / args.eval.dataset_name / model_name / "linear_probe"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    log.i(f"Linear probing | codebook={codebook_size} | frames={total_frames} | samples={num_samples}")
    log.i(f"Output dir: {save_dir}")

    all_pre_vq, all_post_vq = [], []
    all_targets = {"token_change": [], "next_token": [], "pixel_motion": []}
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        videos = batch["input_frames"].to(device)[:, :total_frames]
        videos = rearrange(videos, "b f c h w -> b c f h w")
        pre_vq, post_vq, indices = extract_embeddings(tokenizer, videos)
        targets = compute_probe_targets(videos, indices)
        all_pre_vq.append(flatten_embeddings(pre_vq))
        all_post_vq.append(flatten_embeddings(post_vq))
        for k in all_targets:
            all_targets[k].append(targets[k])
        if (i + 1) % 5 == 0:
            log.i(f"  Collected {i + 1}/{num_samples} samples")

    X_pre = np.concatenate(all_pre_vq, axis=0)
    X_post = np.concatenate(all_post_vq, axis=0)
    targets = {k: np.concatenate(v, axis=0) for k, v in all_targets.items()}
    n, d = X_pre.shape
    log.i(f"Probing vectors: {n}, embedding dim: {d}")

    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr, te = perm[:split], perm[split:]
    results = {}

    y = targets["token_change"]
    for tag, X in [("pre_vq", X_pre), ("post_vq", X_post)]:
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        clf.fit(X[tr], y[tr])
        acc = accuracy_score(y[te], clf.predict(X[te]))
        results[f"token_change/{tag}"] = acc
        log.i(f"  token_change | {tag}: accuracy = {acc:.4f}")

    y = targets["next_token"]
    for tag, X in [("pre_vq", X_pre), ("post_vq", X_post)]:
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        clf.fit(X[tr], y[tr])
        acc = accuracy_score(y[te], clf.predict(X[te]))
        results[f"next_token/{tag}"] = acc
        log.i(f"  next_token   | {tag}: accuracy = {acc:.4f}")

    y = targets["pixel_motion"]
    for tag, X in [("pre_vq", X_pre), ("post_vq", X_post)]:
        reg = Ridge(alpha=1.0)
        reg.fit(X[tr], y[tr])
        r2 = r2_score(y[te], reg.predict(X[te]))
        results[f"pixel_motion/{tag}"] = r2
        log.i(f"  pixel_motion | {tag}: R2 = {r2:.4f}")

    col_hdr = f"{'Task':<28s} {'Pre-VQ':>10s} {'Post-VQ':>10s} {'Delta':>10s}"
    row_sep = "-" * 58
    lines = [
        "Linear Probe Results: Motion Information in Tokenizer Embeddings",
        "=" * 70, "",
        f"Model        : {model_name}",
        f"Codebook     : {codebook_size}",
        f"Samples      : {num_samples}",
        f"Vectors      : {n}  (dim={d})",
        f"Train / test : {split} / {n - split}",
        "", col_hdr, row_sep,
    ]
    for task in ["token_change", "next_token", "pixel_motion"]:
        metric = "R2" if task == "pixel_motion" else "acc"
        pre = results[f"{task}/pre_vq"]
        post = results[f"{task}/post_vq"]
        delta = post - pre
        label = f"{task} ({metric})"
        lines.append(f"{label:<28s} {pre:>10.4f} {post:>10.4f} {delta:>+10.4f}")
    lines += [
        "",
        "Interpretation:",
        "  Delta < 0 means VQ bottleneck discards information for that task.",
        "  Large negative delta = strong evidence motion info is lost at VQ.",
    ]
    summary = "\n".join(lines)
    (save_dir / "probe_results.txt").write_text(summary)
    log.i(f"\n{summary}")
    log.i(f"\nSaved to {save_dir / 'probe_results.txt'}")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
