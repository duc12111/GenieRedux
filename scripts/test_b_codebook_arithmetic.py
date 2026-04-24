#!/usr/bin/env python3
"""Test B: is there linear compositional structure in the small-VQ codebook?

Uses motion-conditional stats (from scripts/validate_motion_conditional.py
output JSON) to categorise motion codes by (intensity, direction) buckets,
then tests whether embedding arithmetic generalises across direction:

    embed[fast_horiz] - embed[slow_horiz] ≈ embed[fast_vert] - embed[slow_vert]

If the codebook has linear structure, the intensity-shift direction vector
should be similar regardless of which direction-category we compute it in.

Pass criterion: median cosine similarity ≥ 0.5 across ≥ 10 valid 4-tuples.

Example:
    python scripts/test_b_codebook_arithmetic.py \
      --checkpoint checkpoints/tokenizer_dual_cb/dual_cb_train_cb64_allgames_inv0_flow0_deltanone/model-5000.pt \
      --motion-stats outputs/validation/R4_cb64/motion_conditional.json \
      --small-codebook-size 64 \
      --out outputs/validation/R4_cb64/test_b_arithmetic.txt
"""
from __future__ import annotations

import argparse
import itertools
import json
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
from sklearn.manifold import TSNE

from models import DualCodebookTokenizer


def categorise_codes(per_code_stats: list[dict], motion_codes: list[int]):
    """Assign each motion code to an (intensity_tier, direction_type) bucket.

    intensity_tier ∈ {low, mid, high}  based on mean|flow| tertiles within motion codes
    direction_type ∈ {horizontal, mixed, vertical}  based on var_dy / (var_dx + var_dy)
        (low y-fraction → horizontal; high → vertical)
    """
    stats_by_k = {s["k"]: s for s in per_code_stats}
    mags = np.array([stats_by_k[k]["mean_flow_mag"] for k in motion_codes])
    # Per-code y-fraction: var_dy / (var_dx + var_dy)
    y_fracs = []
    for k in motion_codes:
        s = stats_by_k[k]
        denom = s["flow_var_x"] + s["flow_var_y"] + 1e-9
        y_fracs.append(s["flow_var_y"] / denom)
    y_fracs = np.array(y_fracs)

    # Tertile boundaries
    mag_lo, mag_hi = np.percentile(mags, [33.33, 66.67])
    yf_lo, yf_hi = np.percentile(y_fracs, [33.33, 66.67])

    buckets = {}
    for k, mag, yf in zip(motion_codes, mags, y_fracs):
        if mag < mag_lo:
            it = "low"
        elif mag < mag_hi:
            it = "mid"
        else:
            it = "high"
        if yf < yf_lo:
            dt = "horizontal"
        elif yf < yf_hi:
            dt = "mixed"
        else:
            dt = "vertical"
        buckets.setdefault((it, dt), []).append(k)
    return buckets, mags, y_fracs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--motion-stats", required=True,
                   help="path to motion_conditional.json from validate_motion_conditional.py")
    p.add_argument("--small-codebook-size", type=int, default=64)
    p.add_argument("--big-codebook-size", type=int, default=1024)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--delta-ref", default="none")
    p.add_argument("--out", default="outputs/validation/test_b_arithmetic.txt")
    p.add_argument("--tsne-out", default=None,
                   help="optional t-SNE plot path; defaults to sibling of --out")
    args = p.parse_args()

    # Load model just to get the codebook embeddings
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

    codebook = model.small_vq._codebook.embed.detach().cpu()
    if codebook.ndim == 3:
        codebook = codebook.squeeze(0)
    codebook = codebook.numpy()  # (K, d_cb)
    K, d_cb = codebook.shape
    assert K == args.small_codebook_size

    with open(args.motion_stats) as f:
        stats = json.load(f)
    motion_codes = stats["motion_codes"]
    static_codes = stats["static_codes"]
    per_code_stats = stats["per_code"]

    buckets, mags, y_fracs = categorise_codes(per_code_stats, motion_codes)

    # Print bucket summary
    lines = []
    lines.append("=== Test B: codebook arithmetic ===")
    lines.append(f"checkpoint                : {args.checkpoint}")
    lines.append(f"codebook shape            : {codebook.shape}")
    lines.append(f"static codes              : {static_codes}")
    lines.append(f"motion codes              : {len(motion_codes)}")
    lines.append("")
    lines.append(f"Categorisation (intensity × direction):")
    lines.append(f"  {'bucket':<25} {'count':>6}  codes")
    for it in ["low", "mid", "high"]:
        for dt in ["horizontal", "mixed", "vertical"]:
            codes = buckets.get((it, dt), [])
            lines.append(f"  {(it + ',' + dt):<25} {len(codes):>6}  {codes}")
    lines.append("")

    # ---- Analogy test 1: intensity-shift across directions ----
    # (slow_H, fast_H, slow_V, fast_V) — does (fast_H - slow_H) ≈ (fast_V - slow_V)?
    analogies = []
    intensity_pairs = [("low", "high"), ("low", "mid"), ("mid", "high")]
    direction_pairs = [("horizontal", "vertical"),
                       ("horizontal", "mixed"),
                       ("mixed", "vertical")]

    for (lo_it, hi_it) in intensity_pairs:
        for (d1, d2) in direction_pairs:
            lo_d1 = buckets.get((lo_it, d1), [])
            hi_d1 = buckets.get((hi_it, d1), [])
            lo_d2 = buckets.get((lo_it, d2), [])
            hi_d2 = buckets.get((hi_it, d2), [])
            if not (lo_d1 and hi_d1 and lo_d2 and hi_d2):
                continue
            # Cartesian product of codes within each bucket, or just take bucket centroids
            cent_lo_d1 = codebook[lo_d1].mean(axis=0)
            cent_hi_d1 = codebook[hi_d1].mean(axis=0)
            cent_lo_d2 = codebook[lo_d2].mean(axis=0)
            cent_hi_d2 = codebook[hi_d2].mean(axis=0)
            # Intensity-shift vectors under each direction
            v_d1 = cent_hi_d1 - cent_lo_d1
            v_d2 = cent_hi_d2 - cent_lo_d2
            cs = cosine(v_d1, v_d2)
            analogies.append({
                "type": "intensity_shift_across_direction",
                "description": f"({lo_it}→{hi_it}) intensity in {d1} vs {d2}",
                "cosine": cs,
            })

    # ---- Analogy test 2: direction-shift across intensities ----
    # (H_slow, V_slow, H_fast, V_fast) — does (V_fast - H_fast) ≈ (V_slow - H_slow)?
    for (d1, d2) in direction_pairs:
        for (lo_it, hi_it) in intensity_pairs:
            lo_d1 = buckets.get((lo_it, d1), [])
            lo_d2 = buckets.get((lo_it, d2), [])
            hi_d1 = buckets.get((hi_it, d1), [])
            hi_d2 = buckets.get((hi_it, d2), [])
            if not (lo_d1 and lo_d2 and hi_d1 and hi_d2):
                continue
            cent_lo_d1 = codebook[lo_d1].mean(axis=0)
            cent_lo_d2 = codebook[lo_d2].mean(axis=0)
            cent_hi_d1 = codebook[hi_d1].mean(axis=0)
            cent_hi_d2 = codebook[hi_d2].mean(axis=0)
            v_lo = cent_lo_d2 - cent_lo_d1
            v_hi = cent_hi_d2 - cent_hi_d1
            cs = cosine(v_lo, v_hi)
            analogies.append({
                "type": "direction_shift_across_intensity",
                "description": f"({d1}→{d2}) direction in {lo_it} vs {hi_it} intensity",
                "cosine": cs,
            })

    # ---- Analogy test 3: motion-vs-static direction ----
    # Centroid of motion codes vs centroid of static codes — is there a consistent
    # "motion direction" vector across different motion buckets?
    if static_codes:
        static_centroid = codebook[static_codes].mean(axis=0)
        motion_centroids_per_bucket = {}
        for bucket_key, codes in buckets.items():
            if codes:
                motion_centroids_per_bucket[bucket_key] = codebook[codes].mean(axis=0)
        bucket_keys = list(motion_centroids_per_bucket.keys())
        # Pairwise cosines between "motion - static" vectors
        for i, bk_i in enumerate(bucket_keys):
            for bk_j in bucket_keys[i + 1:]:
                v_i = motion_centroids_per_bucket[bk_i] - static_centroid
                v_j = motion_centroids_per_bucket[bk_j] - static_centroid
                cs = cosine(v_i, v_j)
                analogies.append({
                    "type": "motion_minus_static_consistency",
                    "description": f"{bk_i} vs {bk_j}",
                    "cosine": cs,
                })

    # Aggregate
    lines.append(f"Analogies tested          : {len(analogies)}")
    if analogies:
        by_type = {}
        for a in analogies:
            by_type.setdefault(a["type"], []).append(a["cosine"])
        for t, vals in by_type.items():
            vals = np.array(vals)
            lines.append(f"  {t:<45}: n={len(vals):3d}  mean={vals.mean():+.3f}  median={np.median(vals):+.3f}  ±{vals.std():.3f}")
        all_cs = np.array([a["cosine"] for a in analogies])
        overall_median = float(np.median(all_cs))
        overall_mean = float(all_cs.mean())
        lines.append("")
        lines.append(f"Overall: n={len(all_cs)}  mean={overall_mean:+.3f}  MEDIAN={overall_median:+.3f}")
        lines.append(f"Random baseline expected: ~0.00 (for d_cb={d_cb}, any random pair)")
        lines.append("")
        pass_gate = overall_median >= 0.5 and len(all_cs) >= 10
        lines.append(f"PASS GATE (median ≥ 0.5 across ≥ 10 analogies): {'YES' if pass_gate else 'NO'}")
        if overall_median >= 0.3:
            lines.append(f"  (median {overall_median:+.3f} suggests SOME linear structure)")
        elif overall_median >= 0.1:
            lines.append(f"  (median {overall_median:+.3f} suggests WEAK linear structure)")
        else:
            lines.append(f"  (median {overall_median:+.3f} suggests NO linear structure)")
    else:
        lines.append("No analogies could be tested (insufficient bucket coverage)")
        pass_gate = False

    # Write top detailed analogies
    lines.append("")
    lines.append("Top analogies by cosine:")
    for a in sorted(analogies, key=lambda x: -x["cosine"])[:15]:
        lines.append(f"  {a['cosine']:+.3f}  [{a['type']:<42}]  {a['description']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "overall_median_cosine": float(overall_median) if analogies else None,
        "overall_mean_cosine": float(overall_mean) if analogies else None,
        "n_analogies": len(analogies),
        "pass_gate": pass_gate,
        "analogies": analogies,
        "buckets": {f"{k[0]}_{k[1]}": v for k, v in buckets.items()},
    }, indent=2))

    # t-SNE plot coloured by (intensity, direction) bucket
    tsne_path = args.tsne_out or str(out_path.with_name("test_b_tsne.png"))
    perplexity = min(5.0, max(2.0, (K - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca", learning_rate="auto")
    codebook_2d = tsne.fit_transform(codebook)
    fig, ax = plt.subplots(figsize=(10, 9))
    # Plot static codes in grey
    if static_codes:
        ax.scatter(codebook_2d[static_codes, 0], codebook_2d[static_codes, 1],
                   s=200, marker="s", c="grey", edgecolors="black", label="static", alpha=0.7)
    # Plot motion codes coloured by direction, sized by intensity
    direction_colours = {"horizontal": "tab:blue", "mixed": "tab:green", "vertical": "tab:red"}
    intensity_sizes = {"low": 80, "mid": 160, "high": 280}
    for (it, dt), codes in buckets.items():
        if codes:
            ax.scatter(codebook_2d[codes, 0], codebook_2d[codes, 1],
                       s=intensity_sizes[it], c=direction_colours[dt],
                       edgecolors="black", alpha=0.75,
                       label=f"{it},{dt}")
    for k in range(K):
        ax.annotate(str(k), codebook_2d[k], ha="center", va="center", fontsize=7)
    ax.set_title(f"cb{K} codebook t-SNE by (intensity, direction)\n"
                 f"overall median cosine arithmetic: {overall_median:+.3f}  "
                 f"[{'PASS' if pass_gate else 'FAIL'} gate]")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(tsne_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(lines))
    print(f"\nsaved to {out_path} and {json_path} and {tsne_path}")


if __name__ == "__main__":
    main()
