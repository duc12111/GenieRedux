"""Token editing script for discovering what each discrete code means.

Six modes:
  replace_id         – Replace every occurrence of token X with the background
                       token, decode, compare visually.
  spatial_ablation   – Fill spatial quadrants with the background token to see
                       what each region encodes.
  codebook_sweep     – At one patch position try every codebook entry and
                       produce a grid of decoded frames.
  frame_inject       – Auto-detect non-background patches on target frame(s),
                       replace them with the background token, decode the full
                       sequence to observe temporal propagation.  Runs weak
                       (single-frame), medium (frame 4), and strong
                       (frame 2 onward) variants.
  transition_matrix  – Build a codebook_size x codebook_size transition
                       heatmap showing token-to-token changes across
                       consecutive frames.  Saves per-sample and aggregate.
  temporal_diff_maps – Highlight which patches change token ID between
                       consecutive frames, overlaid on the original video.

Usage (Hydra overrides – note the '+' for the new edit.* keys):
  python edit_tokens.py config=tokenizer_256_small model=tokenizer \
    tokenizer_fpath=checkpoints/tokenizer/tokenizer_ft_cb16_3games/model-5000.pt \
    tokenizer.codebook_size=16 \
    eval.dataset_root_dpath=data_generation/datasets \
    eval.dataset_name=retro_act_v0.0.0_g200_256_single \
    eval.model_name=tokenizer_ft_cb16_3games \
    +edit.mode=replace_id +edit.num_samples=5
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from einops import rearrange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image, ImageDraw
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_color_lut(codebook_size):
    """Golden-ratio hue spacing – same palette as eval_genie_redux.py."""
    phi = 0.618033988749895
    hues = (np.arange(codebook_size) * phi) % 1.0
    hsv = np.ones((1, codebook_size, 3))
    hsv[0, :, 0] = hues
    hsv[0, :, 1] = 0.9
    hsv[0, :, 2] = 0.95
    return (hsv_to_rgb(hsv).squeeze(0) * 255).astype(np.uint8)


def token_colormap(indices_2d, colors, upscale=8):
    """(ph, pw) int array  →  (ph*upscale, pw*upscale, 3) uint8 RGB."""
    rgb = colors[indices_2d]
    return np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)


def detect_bg_token(indices):
    """Most-frequent token id in a flat index array (assumed background)."""
    return int(np.bincount(indices.flatten()).argmax())


def video_to_numpy_frames(video_tensor):
    """(C, T, H, W) float tensor → list of (H, W, 3) uint8 arrays."""
    vid = torch.clamp(video_tensor, 0, 1)
    return [
        (vid[:, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        for t in range(vid.shape[1])
    ]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _add_border(pil_img, color="red", width=3):
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    draw.rectangle([0, 0, w - 1, h - 1], outline=color, width=width)
    return pil_img


def save_comparison_grid(
    orig_frames, edited_frames, idx_orig, idx_edit,
    colors, save_path, highlight_frames=None, upscale=8,
):
    """4-row PNG grid: input | edited recon | token map orig | token map edit.

    orig_frames / edited_frames : list of (H, W, 3) uint8
    idx_orig / idx_edit         : (pt, ph, pw)  numpy int arrays
    highlight_frames            : set of frame indices that get a red border
    """
    highlight_frames = highlight_frames or set()
    n = min(len(orig_frames), len(edited_frames), idx_orig.shape[0])
    h, w = orig_frames[0].shape[:2]
    grid = Image.new("RGB", (w * n, h * 4))

    for t in range(n):
        imgs = [
            Image.fromarray(orig_frames[t]),
            Image.fromarray(edited_frames[t]),
            Image.fromarray(token_colormap(idx_orig[t], colors, upscale)).resize(
                (w, h), Image.NEAREST
            ),
            Image.fromarray(token_colormap(idx_edit[t], colors, upscale)).resize(
                (w, h), Image.NEAREST
            ),
        ]
        if t in highlight_frames:
            imgs = [_add_border(im) for im in imgs]
        for row, im in enumerate(imgs):
            grid.paste(im, (w * t, h * row))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(save_path)


def save_comparison_gif(orig_frames, edited_frames, save_path, duration=120):
    """GIF with original stacked above edited recon, one composite per step."""
    frames = []
    for orig, edit in zip(orig_frames, edited_frames):
        combined = np.concatenate([orig, edit], axis=0)
        frames.append(Image.fromarray(combined))
    if not frames:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        save_path, save_all=True, append_images=frames[1:],
        loop=0, duration=duration,
    )


# ---------------------------------------------------------------------------
# Mode 1 – replace_id
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_replace_id(tokenizer, video, indices_4d, bg_token, colors,
                   save_dir, codebook_size, device):
    """Replace every occurrence of each token id with bg, decode, compare."""
    save_dir.mkdir(parents=True, exist_ok=True)
    input_frames = video_to_numpy_frames(video)

    orig_recon = tokenizer.decode_from_codebook_indices(
        torch.tensor(indices_4d.reshape(1, -1), device=device)
    )
    orig_recon_frames = video_to_numpy_frames(orig_recon[0])

    for tid in range(codebook_size):
        if tid == bg_token:
            continue
        count = int((indices_4d == tid).sum())
        if count == 0:
            continue

        edited = indices_4d.copy()
        edited[edited == tid] = bg_token

        edited_recon = tokenizer.decode_from_codebook_indices(
            torch.tensor(edited.reshape(1, -1), device=device)
        )
        edited_frames = video_to_numpy_frames(edited_recon[0])

        save_comparison_grid(
            input_frames, edited_frames,
            indices_4d[0], edited[0], colors,
            save_dir / f"replace_token{tid}_with_bg.png",
        )
        save_comparison_gif(
            input_frames, edited_frames,
            save_dir / f"replace_token{tid}_with_bg.gif",
        )
        log.i(f"  replace_id: token {tid} (count={count}) → bg {bg_token}")

    save_comparison_grid(
        input_frames, orig_recon_frames,
        indices_4d[0], indices_4d[0], colors,
        save_dir / "original_reconstruction.png",
    )
    save_comparison_gif(
        input_frames, orig_recon_frames,
        save_dir / "original_reconstruction.gif",
    )


# ---------------------------------------------------------------------------
# Mode 2 – spatial_ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_spatial_ablation(tokenizer, video, indices_4d, bg_token, colors,
                         save_dir, device):
    """Ablate spatial quadrants and center on every frame, decode, compare."""
    save_dir.mkdir(parents=True, exist_ok=True)
    input_frames = video_to_numpy_frames(video)
    _, pt, ph, pw = indices_4d.shape

    quadrants = {
        "top_left":     (0, ph // 2, 0, pw // 2),
        "top_right":    (0, ph // 2, pw // 2, pw),
        "bottom_left":  (ph // 2, ph, 0, pw // 2),
        "bottom_right": (ph // 2, ph, pw // 2, pw),
        "center":       (ph // 4, 3 * ph // 4, pw // 4, 3 * pw // 4),
    }

    for name, (h0, h1, w0, w1) in quadrants.items():
        edited = indices_4d.copy()
        edited[0, :, h0:h1, w0:w1] = bg_token

        edited_recon = tokenizer.decode_from_codebook_indices(
            torch.tensor(edited.reshape(1, -1), device=device)
        )
        edited_frames = video_to_numpy_frames(edited_recon[0])

        save_comparison_grid(
            input_frames, edited_frames,
            indices_4d[0], edited[0], colors,
            save_dir / f"ablate_{name}.png",
        )
        save_comparison_gif(
            input_frames, edited_frames,
            save_dir / f"ablate_{name}.gif",
        )
        log.i(f"  spatial_ablation: {name} done")


# ---------------------------------------------------------------------------
# Mode 3 – codebook_sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_codebook_sweep(tokenizer, video, indices_4d, colors, save_dir,
                       codebook_size, device, sweep_t=0,
                       sweep_h=None, sweep_w=None):
    """Try every codebook entry at one position; save a grid of results."""
    save_dir.mkdir(parents=True, exist_ok=True)
    _, pt, ph, pw = indices_4d.shape
    if sweep_h is None:
        sweep_h = ph // 2
    if sweep_w is None:
        sweep_w = pw // 2

    orig_token = int(indices_4d[0, sweep_t, sweep_h, sweep_w])

    ncols = int(np.ceil(np.sqrt(codebook_size)))
    nrows = int(np.ceil(codebook_size / ncols))

    h_px, w_px = video.shape[2], video.shape[3]
    grid = Image.new("RGB", (w_px * ncols, h_px * nrows), color=(40, 40, 40))

    for tid in range(codebook_size):
        edited = indices_4d.copy()
        edited[0, sweep_t, sweep_h, sweep_w] = tid

        edited_recon = tokenizer.decode_from_codebook_indices(
            torch.tensor(edited.reshape(1, -1), device=device)
        )
        frame = video_to_numpy_frames(edited_recon[0])[sweep_t]
        pil_frame = Image.fromarray(frame)

        if tid == orig_token:
            _add_border(pil_frame, color="green", width=4)

        row, col = divmod(tid, ncols)
        grid.paste(pil_frame, (col * w_px, row * h_px))

    grid.save(save_dir / f"sweep_t{sweep_t}_h{sweep_h}_w{sweep_w}.png")
    log.i(
        f"  codebook_sweep: pos=({sweep_t},{sweep_h},{sweep_w}), "
        f"orig_token={orig_token}, grid {nrows}x{ncols}"
    )


# ---------------------------------------------------------------------------
# Mode 4 – frame_inject
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_frame_inject(tokenizer, video, indices_4d, colors, save_dir,
                     inject_configs, device):
    """Auto-detect non-bg patches on target frames, replace with bg, decode.

    inject_configs: list of dicts with keys 'name' and 'target_frames'.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    input_frames = video_to_numpy_frames(video)
    _, pt, ph, pw = indices_4d.shape

    for cfg in inject_configs:
        name = cfg["name"]
        target_frames = cfg["target_frames"]

        edited = indices_4d.copy()
        highlight = set()

        for fk in target_frames:
            if fk >= pt:
                continue
            frame_tokens = edited[0, fk]
            bg_token = int(np.bincount(frame_tokens.flatten()).argmax())
            non_bg_mask = frame_tokens != bg_token
            n_replaced = int(non_bg_mask.sum())
            edited[0, fk][non_bg_mask] = bg_token
            highlight.add(fk)
            log.i(
                f"  frame_inject '{name}': frame {fk}, "
                f"replaced {n_replaced} non-bg patches → token {bg_token}"
            )

        edited_recon = tokenizer.decode_from_codebook_indices(
            torch.tensor(edited.reshape(1, -1), device=device)
        )
        edited_frames = video_to_numpy_frames(edited_recon[0])

        save_comparison_grid(
            input_frames, edited_frames,
            indices_4d[0], edited[0], colors,
            save_dir / f"inject_{name}.png",
            highlight_frames=highlight,
        )
        save_comparison_gif(
            input_frames, edited_frames,
            save_dir / f"inject_{name}.gif",
        )
        log.i(f"  frame_inject '{name}' → saved")


# ---------------------------------------------------------------------------
# Mode 5 – transition_matrix
# ---------------------------------------------------------------------------

def _save_transition_heatmaps(transition, codebook_size, save_path_prefix,
                               title_suffix=""):
    """Save full and motion-only transition heatmaps as PNGs."""
    save_dir = Path(save_path_prefix).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(transition, cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Token ID at t+1")
    ax.set_ylabel("Token ID at t")
    ax.set_title(f"Token Transition Matrix{title_suffix}")
    ax.set_xticks(range(codebook_size))
    ax.set_yticks(range(codebook_size))
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{save_path_prefix}_full.png", dpi=150)
    plt.close(fig)

    motion_only = transition.copy().astype(float)
    np.fill_diagonal(motion_only, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(motion_only, cmap="hot", interpolation="nearest")
    ax.set_xlabel("Token ID at t+1")
    ax.set_ylabel("Token ID at t")
    ax.set_title(f"Motion Transitions (no self){title_suffix}")
    ax.set_xticks(range(codebook_size))
    ax.set_yticks(range(codebook_size))
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{save_path_prefix}_motion.png", dpi=150)
    plt.close(fig)


def run_transition_matrix(indices_4d, codebook_size, save_dir):
    """Build codebook_size x codebook_size transition matrix. Returns it."""
    save_dir.mkdir(parents=True, exist_ok=True)
    _, pt, ph, pw = indices_4d.shape

    transition = np.zeros((codebook_size, codebook_size), dtype=np.int64)
    for t in range(pt - 1):
        curr = indices_4d[0, t].flatten()
        nxt = indices_4d[0, t + 1].flatten()
        np.add.at(transition, (curr, nxt), 1)

    _save_transition_heatmaps(
        transition, codebook_size,
        str(save_dir / "transition_matrix"),
        title_suffix=" (sample)",
    )

    total = transition.sum()
    self_trans = int(np.trace(transition))
    log.i(f"  transition_matrix: total={total}, self={self_trans} "
          f"({100 * self_trans / max(total, 1):.1f}%), "
          f"motion={total - self_trans}")

    return transition


# ---------------------------------------------------------------------------
# Mode 6 – temporal_diff_maps
# ---------------------------------------------------------------------------

def run_temporal_diff_maps(video, indices_4d, save_dir):
    """Highlight patches that change token ID between consecutive frames."""
    save_dir.mkdir(parents=True, exist_ok=True)
    input_frames = video_to_numpy_frames(video)
    _, pt, ph, pw = indices_4d.shape
    h_px, w_px = video.shape[2], video.shape[3]
    ph_scale = h_px // ph
    pw_scale = w_px // pw

    overlay_frames = []
    change_counts = []
    for t in range(pt):
        frame = input_frames[t].copy()
        if t > 0:
            changed = indices_4d[0, t] != indices_4d[0, t - 1]
            change_counts.append(int(changed.sum()))
            mask = np.repeat(
                np.repeat(changed, ph_scale, axis=0), pw_scale, axis=1
            )
            mask = mask[:h_px, :w_px]
            red = np.array([255, 50, 50], dtype=np.float32)
            frame[mask] = (
                frame[mask].astype(np.float32) * 0.5 + red * 0.5
            ).astype(np.uint8)
        overlay_frames.append(Image.fromarray(frame))

    if overlay_frames:
        overlay_frames[0].save(
            save_dir / "temporal_diff.gif",
            save_all=True, append_images=overlay_frames[1:],
            loop=0, duration=200,
        )

    combined = []
    for t in range(len(overlay_frames)):
        orig = Image.fromarray(input_frames[t])
        combo = Image.new("RGB", (w_px * 2, h_px))
        combo.paste(orig, (0, 0))
        combo.paste(overlay_frames[t], (w_px, 0))
        combined.append(combo)

    if combined:
        combined[0].save(
            save_dir / "temporal_diff_sidebyside.gif",
            save_all=True, append_images=combined[1:],
            loop=0, duration=200,
        )

    if change_counts:
        log.i(f"  temporal_diff: changed patches per frame: "
              f"{change_counts} (mean={np.mean(change_counts):.1f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model ----
    model = construct_model(args)
    if getattr(args, "tokenizer_fpath", None):
        tok_state = torch.load(args.tokenizer_fpath, map_location="cpu")
        model.load_state_dict(tok_state["model"])
        del tok_state
    model = model.to(device).eval()
    tokenizer = model

    codebook_size = tokenizer.vq.codebook_size
    colors = build_color_lut(codebook_size)

    # ---- dataset ----
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

    # ---- edit config ----
    mode = args.edit.mode
    num_samples = int(getattr(args.edit, "num_samples", 5))
    num_first_frames = args.eval.num_first_frames
    sample_num_frames = args.eval.sample_num_frames
    total_frames = num_first_frames + sample_num_frames

    model_name = args.eval.model_name
    run_id = getattr(args.eval, "samples_run_id", None)
    if run_id not in (None, ""):
        model_name = f"{model_name}_{run_id}"
    base_dir = (
        Path(args.eval.save_root_dpath)
        / args.eval.dataset_name
        / model_name
        / "token_edits"
    )

    agg_transition = (
        np.zeros((codebook_size, codebook_size), dtype=np.int64)
        if mode == "transition_matrix" else None
    )

    log.i(f"Mode: {mode} | codebook_size: {codebook_size} | "
          f"frames: {total_frames} | samples: {num_samples}")
    log.i(f"Output dir: {base_dir}")

    for i, batch in enumerate(loader):
        if i >= num_samples:
            break

        videos = batch["input_frames"].to(device)[:, :total_frames]
        videos = rearrange(videos, "b f c h w -> b c f h w")

        indices = tokenizer(videos, return_only_codebook_ids=True)
        pt, ph, pw = tokenizer.get_video_patch_shape(
            total_frames, num_first_frames
        )
        indices_4d = indices.cpu().numpy().reshape(1, pt, ph, pw)

        bg_token = detect_bg_token(indices_4d)
        sample_colors = colors.copy()
        sample_colors[bg_token] = [180, 180, 180]

        video_tensor = videos[0]  # (C, T, H, W)
        sample_dir = base_dir / mode / f"sample_{i}"

        log.i(f"--- Sample {i} | bg_token={bg_token} ---")

        if mode == "replace_id":
            run_replace_id(
                tokenizer, video_tensor, indices_4d, bg_token,
                sample_colors, sample_dir, codebook_size, device,
            )

        elif mode == "spatial_ablation":
            run_spatial_ablation(
                tokenizer, video_tensor, indices_4d, bg_token,
                sample_colors, sample_dir, device,
            )

        elif mode == "codebook_sweep":
            sweep_t = int(getattr(args.edit, "sweep_t", 0))
            sweep_h = getattr(args.edit, "sweep_h", None)
            sweep_w = getattr(args.edit, "sweep_w", None)
            if sweep_h is not None:
                sweep_h = int(sweep_h)
            if sweep_w is not None:
                sweep_w = int(sweep_w)
            run_codebook_sweep(
                tokenizer, video_tensor, indices_4d, sample_colors,
                sample_dir, codebook_size, device,
                sweep_t=sweep_t, sweep_h=sweep_h, sweep_w=sweep_w,
            )

        elif mode == "frame_inject":
            inject_configs = [
                {"name": "inject_frame2", "target_frames": [2]},
                {"name": "inject_frame4", "target_frames": [4]},
                {"name": "inject_frame2_onward",
                 "target_frames": list(range(2, pt))},
            ]
            run_frame_inject(
                tokenizer, video_tensor, indices_4d, sample_colors,
                sample_dir, inject_configs, device,
            )

        elif mode == "transition_matrix":
            sample_mat = run_transition_matrix(
                indices_4d, codebook_size, sample_dir,
            )
            agg_transition += sample_mat

        elif mode == "temporal_diff_maps":
            run_temporal_diff_maps(
                video_tensor, indices_4d, sample_dir,
            )

        else:
            log.i(f"Unknown mode: {mode}")
            return

    if agg_transition is not None and agg_transition.sum() > 0:
        agg_dir = base_dir / mode / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)
        _save_transition_heatmaps(
            agg_transition, codebook_size,
            str(agg_dir / "transition_matrix"),
            title_suffix=" (all samples)",
        )
        total = agg_transition.sum()
        self_trans = int(np.trace(agg_transition))
        log.i(f"Aggregate transition matrix: total={total}, "
              f"self={self_trans} ({100 * self_trans / max(total, 1):.1f}%)")

    log.i("All done!")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
