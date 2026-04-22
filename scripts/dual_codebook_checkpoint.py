"""Shared helpers for dual-codebook analysis scripts."""

from __future__ import annotations

import random

import numpy as np
import torch

from models import Tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_tokenizer_weights(tokenizer: Tokenizer, checkpoint_fpath: str) -> None:
    """Load tokenizer weights from a tokenizer-only checkpoint or a full module dict.

    GenieRedux / GenieReduxGuided checkpoints often omit tokenizer weights in ``model``;
    in that case a clear error is raised.
    """
    ckpt = torch.load(checkpoint_fpath, map_location="cpu")
    sd = ckpt["model"]

    tok_sd = {
        k.replace("tokenizer.", "", 1): v
        for k, v in sd.items()
        if isinstance(k, str) and k.startswith("tokenizer.")
    }
    if len(tok_sd) > 0:
        tokenizer.load_state_dict(tok_sd, strict=True)
        return

    looks_like_tokenizer = any(
        isinstance(k, str)
        and (
            k.startswith("vq.")
            or k.startswith("to_patch")
            or k.startswith("encode")
            or k.startswith("decoder")
        )
        for k in sd
    )
    if not looks_like_tokenizer:
        raise ValueError(
            "This checkpoint does not contain tokenizer weights (no tokenizer.* prefix and no "
            "vq./to_patch*/encoder/decoder keys). Use a tokenizer training checkpoint (model-*.pt), "
            "e.g. the file used as tokenizer_fpath for the Genie run."
        )

    tokenizer.load_state_dict(sd, strict=True)
