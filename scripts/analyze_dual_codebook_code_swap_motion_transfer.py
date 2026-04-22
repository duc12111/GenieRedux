#!/usr/bin/env python3
""" 
Experiment #3: Code swapping for "small codes = motion".

Pairing definition (self-supervised; no action labels)
  - Trajectory A provides small-code indices for all later frames
  - Trajectory B provides big-code indices for the anchor frame
  - We mix (big from B, small from A), decode, and compare the mixed endpoint
    reconstruction to both A and B endpoints.

This is an analysis-only script (no training).

Example usage (save outputs under outputs/analysis/)

``--big-tokenizer-fpath`` must be a tokenizer training checkpoint (model-*.pt), not a
Genie dynamics-only save.

Same-game pairing (restrict to one env, sample two windows within it):
python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \\
  --big-tokenizer-fpath checkpoints/genie_redux_guilded/tokenizer.pt \\
  --small-tokenizer-fpath checkpoints/tokenizer/tokenizer_cb16_scratch_1game/model-5000.pt \\
  --small-codebook-size 16 \\
  --dataset-root-dpath data_generation/datasets \\
  --dataset-name retro_act_v0.0.0_g200_256 \\
  --mode same_game \\
  --whitelist-env addamsfamily-nes \\
  --num-pairs 20 \\
  --out-dir outputs/analysis/dual_cb_swap_samegame_cb16_scratch

Cross-game pairing (sample A from env A, B from env B):
python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \\
  --big-tokenizer-fpath checkpoints/genie_redux_guilded/tokenizer.pt \\
  --small-tokenizer-fpath checkpoints/tokenizer/tokenizer_cb16_scratch_1game/model-5000.pt \\
  --small-codebook-size 16 \\
  --dataset-root-dpath data_generation/datasets \\
  --dataset-name retro_act_v0.0.0_g200_256 \\
  --mode cross_game \\
  --whitelist-env-a addamsfamily-nes \\
  --whitelist-env-b smb-nes \\
  --num-pairs 20 \\
  --out-dir outputs/analysis/dual_cb_swap_crossgame_cb16_scratch
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os
import random
import csv
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from matplotlib.colors import hsv_to_rgb

from data.data import DatasetOutputFormat, MultiEnvironmentDataset, TransformsGenerator
from models import Tokenizer
from models.components.vector_quantize import VectorQuantize
from scripts.dual_codebook_checkpoint import load_tokenizer_weights, set_seed
from training.evaluation import FidelityEvaluator


def golden_ratio_lut(codebook_size: int) -> np.ndarray:
    phi = 0.618033988749895
    hues = (np.arange(codebook_size) * phi) % 1.0
    hsv = np.ones((1, codebook_size, 3), dtype=np.float32)
    hsv[0, :, 0] = hues
    hsv[0, :, 1] = 0.9
    hsv[0, :, 2] = 0.95
    colors = (hsv_to_rgb(hsv).squeeze(0) * 255).astype(np.uint8)
    return colors


def indices_to_token_map(ids_small_2d: np.ndarray, codebook_size: int, upscale: int) -> Image.Image:
    lut = golden_ratio_lut(codebook_size)
    rgb = lut[ids_small_2d]  # (ph,pw,3)
    rgb_up = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    return Image.fromarray(rgb_up)


def save_token_gif(
    out_dir: Path,
    filename: str,
    video_tensor: torch.Tensor,
    token_ids: torch.Tensor,
    big_codebook_size: int,
    small_codebook_size: int,
    upscale: int = 8,
):
    """Save animated GIF: original frame on top, token-map on bottom.

    Frame 0 indices come from the big codebook; frames 1+ from the small codebook.

    Args:
        video_tensor:        (1, C, T, H, W) on any device.
        token_ids:           (1, pt, ph, pw) on any device (dual-codebook concatenated).
        big_codebook_size:   size of big VQ codebook (for frame 0).
        small_codebook_size: size of small VQ codebook (for frames 1+).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lut_big = golden_ratio_lut(big_codebook_size)
    lut_small = golden_ratio_lut(small_codebook_size)

    ids_np = token_ids[0].detach().cpu().numpy()  # (pt, ph, pw)
    vid = torch.clamp(video_tensor[0], 0, 1).cpu()  # (C, T, H, W)
    t_vid = vid.shape[1]
    t_tok = ids_np.shape[0]
    n_frames = min(t_vid, t_tok)

    h, w = vid.shape[2], vid.shape[3]
    gif_frames = []
    for t in range(n_frames):
        orig = (vid[:, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # (H,W,3)
        lut = lut_big if t == 0 else lut_small
        rgb_map = lut[ids_np[t]]
        tm_pil = Image.fromarray(rgb_map).resize((w, h), Image.NEAREST)
        tm_arr = np.array(tm_pil)
        combined = np.concatenate([orig, tm_arr], axis=0)
        gif_frames.append(Image.fromarray(combined))

    if gif_frames:
        gif_frames[0].save(
            out_dir / filename,
            save_all=True, append_images=gif_frames[1:], loop=0, duration=80,
        )


def save_recon_gif(
    out_dir: Path,
    filename: str,
    video_tensor: torch.Tensor,
    recon_video: torch.Tensor,
):
    """Save animated GIF: original frame on top, reconstruction on bottom.

    Args:
        video_tensor: (1, C, T, H, W) on any device.
        recon_video:  (1, C, T, H, W) reconstruction tensor.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = torch.clamp(video_tensor[0], 0, 1).cpu()  # (C, T, H, W)
    rec = torch.clamp(recon_video[0], 0, 1).cpu()
    n_frames = min(vid.shape[1], rec.shape[1])

    gif_frames = []
    for t in range(n_frames):
        orig = (vid[:, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        recon = (rec[:, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        combined = np.concatenate([orig, recon], axis=0)
        gif_frames.append(Image.fromarray(combined))

    if gif_frames:
        gif_frames[0].save(
            out_dir / filename,
            save_all=True, append_images=gif_frames[1:], loop=0, duration=80,
        )


def tensor_endpoint_to_pil(t: torch.Tensor) -> Image.Image:
    t = torch.clamp(t, 0.0, 1.0).cpu()
    return ToPILImage()(t)


def main() -> None:
    epilog = r"""
Examples:

  Same-game:
  python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \
    --big-tokenizer-fpath checkpoints/genie_redux_guilded/tokenizer.pt \
    --small-tokenizer-fpath checkpoints/tokenizer/tokenizer_cb16_scratch_1game/model-5000.pt \
    --small-codebook-size 16 \
    --dataset-root-dpath data_generation/datasets \
    --dataset-name retro_act_v0.0.0_g200_256 \
    --mode same_game \
    --whitelist-env addamsfamily-nes \
    --num-pairs 20 \
    --out-dir outputs/analysis/dual_cb_swap_samegame_cb16_scratch

  Cross-game:
  python scripts/analyze_dual_codebook_code_swap_motion_transfer.py \
    --big-tokenizer-fpath checkpoints/genie_redux_guilded/tokenizer.pt \
    --small-tokenizer-fpath checkpoints/tokenizer/tokenizer_cb16_scratch_1game/model-5000.pt \
    --small-codebook-size 16 \
    --dataset-root-dpath data_generation/datasets \
    --dataset-name retro_act_v0.0.0_g200_256 \
    --mode cross_game \
    --whitelist-env-a addamsfamily-nes \
    --whitelist-env-b smb-nes \
    --num-pairs 20 \
    --out-dir outputs/analysis/dual_cb_swap_crossgame_cb16_scratch
"""
    parser = argparse.ArgumentParser(
        description="Dual-codebook code swap: big anchor from B, small motion from A (analysis only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument('--big-tokenizer-fpath', type=str, required=True)
    parser.add_argument('--small-tokenizer-fpath', type=str, required=True)
    parser.add_argument('--small-codebook-size', type=int, default=16)
    parser.add_argument('--dataset-root-dpath', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--num-pairs', type=int, default=20)
    parser.add_argument('--n-envs', type=int, default=20)
    parser.add_argument('--seq-length-input', type=int, default=15)
    parser.add_argument('--seq-step', type=int, default=16)
    parser.add_argument('--motion-frames', type=int, default=7)
    parser.add_argument('--mode', type=str, default='same_game', choices=['same_game', 'cross_game'])
    parser.add_argument('--whitelist-env', type=str, default=None,
                        help='For same_game: restrict dataset to this single env (game folder key).')
    parser.add_argument('--whitelist-env-a', type=str, default=None,
                        help='For cross_game: env for trajectory A.')
    parser.add_argument('--whitelist-env-b', type=str, default=None,
                        help='For cross_game: env for trajectory B.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-k', type=int, default=5)
    parser.add_argument('--upscale', type=int, default=8)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()

    if args.mode == "same_game" and args.whitelist_env is None:
        parser.error("same_game requires --whitelist-env (single game key).")
    if args.mode == "cross_game":
        if args.whitelist_env_a is None or args.whitelist_env_b is None:
            parser.error("cross_game requires --whitelist-env-a and --whitelist-env-b.")

    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == 'cpu') else 'cpu')

    dataset_folder = os.path.join(args.dataset_root_dpath, args.dataset_name)

    # Tokenization window: anchor frame (big) + subsequent frames (small)
    num_first_frames = 1
    total_frames = num_first_frames + args.motion_frames

    # --- load big tokenizer ---
    big_tokenizer = Tokenizer(
        dim=512,
        codebook_size=1024,
        image_size=256,
        patch_size=8,
        temporal_patch_size=1,
        num_blocks=8,
        dim_head=64,
        heads=8,
        ff_mult=4.0,
        vq_loss_w=1.0,
        recon_loss_w=1.0,
    ).to(device)

    load_tokenizer_weights(big_tokenizer, args.big_tokenizer_fpath)
    big_tokenizer.eval()

    # --- load small VQ only ---
    small_vq = VectorQuantize(
        dim=512,
        codebook_size=args.small_codebook_size,
        learnable_codebook=True,
        ema_update=False,
        use_cosine_sim=True,
        commitment_weight=0.25,
        codebook_dim=32,
    ).to(device)

    small_ckpt = torch.load(args.small_tokenizer_fpath, map_location='cpu')
    vq_state = {k.replace('vq.', '', 1): v for k, v in small_ckpt['model'].items() if k.startswith('vq.')}
    small_vq.load_state_dict(vq_state, strict=True)
    small_vq.eval()

    metrics = FidelityEvaluator(device)

    # --- dataset(s) ---
    transforms = TransformsGenerator.get_final_transforms(big_tokenizer.image_size, None)

    def _make_dataset(whitelist=None):
        return MultiEnvironmentDataset(
            dataset_folder,
            seq_length_input=args.seq_length_input,
            seq_step=args.seq_step,
            split_type='instance',
            split='all',
            transform=transforms['test'],
            format=DatasetOutputFormat.IVG,
            enable_cache=False,
            n_workers=1,
            n_envs=args.n_envs,
            whitelist=whitelist,
        )

    use_two_datasets = (args.mode == 'cross_game'
                        and args.whitelist_env_a is not None
                        and args.whitelist_env_b is not None)

    if use_two_datasets:
        dataset_a = _make_dataset(whitelist=[args.whitelist_env_a])
        dataset_b = _make_dataset(whitelist=[args.whitelist_env_b])
    else:
        dataset_a = _make_dataset(whitelist=[args.whitelist_env])
        dataset_b = dataset_a

    pt, ph, pw = big_tokenizer.get_video_patch_shape(total_frames, num_first_frames=num_first_frames)
    assert pt == total_frames, 'Expected temporal_patch_size=1'

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_a = torch.Generator().manual_seed(args.seed)
    gen_b = torch.Generator().manual_seed(args.seed + 1)
    perm_a = torch.randperm(len(dataset_a), generator=gen_a)
    perm_b = torch.randperm(len(dataset_b), generator=gen_b)

    pool_a = min(len(dataset_a), args.num_pairs * 10)
    pool_b = min(len(dataset_b), args.num_pairs * 10)
    candidates_a = perm_a[:pool_a].tolist()
    candidates_b = perm_b[:pool_b].tolist()

    env_groups = {}
    valid_envs = []
    if args.mode == "same_game" and not use_two_datasets:
        csum = dataset_a.cummulative_size
        for idx in candidates_a:
            env_id = int(np.argmax(csum > idx))
            env_groups.setdefault(env_id, []).append(idx)
        valid_envs = [k for k, v in env_groups.items() if len(v) >= 2]
        if len(valid_envs) == 0:
            raise RuntimeError(
                "same_game: no environment has >= 2 windows in the candidate pool. "
                "Try a larger dataset slice, different --seq-step, or another --whitelist-env."
            )

    def dataset_env_id_in(ds, global_idx: int) -> int:
        csum = ds.cummulative_size
        return int(np.argmax(csum > global_idx))

    def load_window(ds, global_idx: int):
        item = ds[global_idx]
        input_frames = item['input_frames'][:total_frames]  # (T,C,H,W)
        videos = torch.from_numpy(input_frames).unsqueeze(0).to(device)  # (1,T,C,H,W)
        videos = videos.permute(0, 2, 1, 3, 4).contiguous()  # (1,C,T,H,W)
        return videos, item

    def decode_mixed(big_indices_flat: torch.Tensor, small_indices_flat: torch.Tensor) -> torch.Tensor:
        big_proj = big_tokenizer.vq.get_output_from_indices(big_indices_flat)  # (1,n_first,dim)
        small_proj = small_vq.get_output_from_indices(small_indices_flat)      # (1,n_rest,dim)
        tokens_flat = torch.cat([big_proj, small_proj], dim=1)                 # (1,pt*ph*pw,dim)
        d = tokens_flat.shape[-1]
        tokens = tokens_flat.view(1, pt, ph, pw, d)
        return big_tokenizer.decode(tokens)  # (1,c,t,H,W)

    metrics_path = out_dir / 'swap_pairs_metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pair_idx','idxA','idxB','envA','envB',
            'instance_id_A','instance_id_B','tgt_frame_id_A','tgt_frame_id_B',
            'psnr_mix_to_A','psnr_mix_to_B','ssim_mix_to_A','ssim_mix_to_B'
        ])

    saved = 0
    save_budget = min(args.save_k, args.num_pairs)
    psnr_a_list = []
    psnr_b_list = []
    ssim_a_list = []
    ssim_b_list = []

    def _int_field(x):
        if isinstance(x, torch.Tensor):
            return int(x.item())
        if isinstance(x, (list, np.ndarray)):
            return int(x[0])
        return int(x)

    for pair_idx in range(args.num_pairs):
        idxA = None
        idxB = None

        for _ in range(50):
            if args.mode == 'same_game' and not use_two_datasets:
                env_id = random.choice(valid_envs)
                idx_list = env_groups[env_id]
                idxA, idxB = random.sample(idx_list, 2)
            elif use_two_datasets:
                idxA = random.choice(candidates_a)
                idxB = random.choice(candidates_b)
            else:
                idxA = random.choice(candidates_a)
                idxB = random.choice(candidates_b if dataset_b is not dataset_a else candidates_a)
                if dataset_b is dataset_a and idxB == idxA:
                    continue

            videos_A, itemA = load_window(dataset_a, idxA)
            videos_B, itemB = load_window(dataset_b, idxB)

            tgtA = _int_field(itemA['tgt_frame_id'])
            tgtB = _int_field(itemB['tgt_frame_id'])
            if tgtA == tgtB:
                continue
            break

        if idxA is None or idxB is None:
            print(f'[pair {pair_idx}] skipping: no valid pair')
            continue

        with torch.no_grad():
            indices_A = big_tokenizer.forward_dual_codebook(videos_A, small_vq, return_only_codebook_ids=True)
            indices_B = big_tokenizer.forward_dual_codebook(videos_B, small_vq, return_only_codebook_ids=True)

        big_indices_mixed = indices_B[:, 0:1, :, :].reshape(1, -1)    # anchor from B: (1, ph*pw)
        small_indices_mixed = indices_A[:, 1:, :, :].reshape(1, -1)  # motion from A: (1, (pt-1)*ph*pw)

        with torch.no_grad():
            recon_mixed = decode_mixed(big_indices_mixed, small_indices_mixed)

        endpoint_recon = recon_mixed[:, :, -1, :, :]  # (1,c,H,W)
        endpoint_A = videos_A[:, :, -1, :, :]
        endpoint_B = videos_B[:, :, -1, :, :]

        psnr_A = metrics.psnr(endpoint_A.unsqueeze(2), endpoint_recon.unsqueeze(2))
        psnr_B = metrics.psnr(endpoint_B.unsqueeze(2), endpoint_recon.unsqueeze(2))
        ssim_A = metrics.ssim(endpoint_recon, endpoint_A)
        ssim_B = metrics.ssim(endpoint_recon, endpoint_B)

        envA = dataset_env_id_in(dataset_a, idxA)
        envB = dataset_env_id_in(dataset_b, idxB)
        instanceA = _int_field(itemA['instance_id'])
        instanceB = _int_field(itemB['instance_id'])
        tgtA = _int_field(itemA['tgt_frame_id'])
        tgtB = _int_field(itemB['tgt_frame_id'])

        psnr_a_list.append(psnr_A)
        psnr_b_list.append(psnr_B)
        ssim_a_list.append(ssim_A)
        ssim_b_list.append(ssim_B)

        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([pair_idx, idxA, idxB, envA, envB,
                             instanceA, instanceB, tgtA, tgtB,
                             psnr_A, psnr_B, ssim_A, ssim_B])

        if saved < save_budget:
            W = endpoint_recon.shape[-1]
            H = endpoint_recon.shape[-2]
            comp = Image.new('RGB', (W * 3, H))
            comp.paste(tensor_endpoint_to_pil(endpoint_A[0]), (0, 0))
            comp.paste(tensor_endpoint_to_pil(endpoint_recon[0]), (W, 0))
            comp.paste(tensor_endpoint_to_pil(endpoint_B[0]), (2 * W, 0))
            comp.save(out_dir / f'pair_{pair_idx}_endpoint_A_mixed_B.png')

            ids_small_A_last = indices_A[0, -1, :, :].detach().cpu().numpy()
            map_img = indices_to_token_map(ids_small_A_last, args.small_codebook_size, args.upscale)
            map_img.save(out_dir / f'pair_{pair_idx}_A_last_small_tokens.png')

            big_cb_size = big_tokenizer.vq.codebook_size
            save_token_gif(
                out_dir, f'pair_{pair_idx}_A_token_indices.gif',
                videos_A, indices_A,
                big_codebook_size=big_cb_size,
                small_codebook_size=args.small_codebook_size,
                upscale=args.upscale,
            )
            save_token_gif(
                out_dir, f'pair_{pair_idx}_B_token_indices.gif',
                videos_B, indices_B,
                big_codebook_size=big_cb_size,
                small_codebook_size=args.small_codebook_size,
                upscale=args.upscale,
            )
            save_token_gif(
                out_dir, f'pair_{pair_idx}_mixed_token_indices.gif',
                recon_mixed, indices_A,
                big_codebook_size=big_cb_size,
                small_codebook_size=args.small_codebook_size,
                upscale=args.upscale,
            )

            with torch.no_grad():
                recon_A = big_tokenizer.forward_dual_codebook(videos_A, small_vq, return_recons_only=True)
                recon_B = big_tokenizer.forward_dual_codebook(videos_B, small_vq, return_recons_only=True)
            save_recon_gif(out_dir, f'pair_{pair_idx}_A.gif', videos_A, recon_A)
            save_recon_gif(out_dir, f'pair_{pair_idx}_B.gif', videos_B, recon_B)
            save_recon_gif(out_dir, f'pair_{pair_idx}_mixed.gif', videos_A, recon_mixed)

            saved += 1

        print(f'[pair {pair_idx}] psnr_mix->A={psnr_A:.3f}, psnr_mix->B={psnr_B:.3f}, ssim->A={ssim_A:.3f}, ssim->B={ssim_B:.3f}')

    if psnr_a_list:
        agg = [
            '__aggregate__', '', '', '', '',
            '', '', '', '',
            float(np.mean(psnr_a_list)), float(np.mean(psnr_b_list)),
            float(np.mean(ssim_a_list)), float(np.mean(ssim_b_list)),
        ]
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(agg)
        summary_path = out_dir / 'summary.txt'
        summary_path.write_text(
            f"num_pairs_written={len(psnr_a_list)}\n"
            f"mean_psnr_mix_to_A={float(np.mean(psnr_a_list))}\n"
            f"mean_psnr_mix_to_B={float(np.mean(psnr_b_list))}\n"
            f"mean_ssim_mix_to_A={float(np.mean(ssim_a_list))}\n"
            f"mean_ssim_mix_to_B={float(np.mean(ssim_b_list))}\n"
        )

    print(f'Done. Metrics saved to: {metrics_path}')


if __name__ == '__main__':
    main()
