import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from torch import nn

from models.components import STViViT
from models.components.vector_quantize import VectorQuantize

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(numer, denom):
    return (numer % denom) == 0


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def cast_tuple(val, l=1):
    return val if isinstance(val, tuple) else (val,) * l


def log_wandb_all_losses(
    accelerator,
    vq_loss,
    recon_loss,
    step,
    is_training=True,
):
    """
    Logs various losses to Weights & Biases (wandb).

    Args:
        accelerator: The Hugging Face Accelerator object.
        vq_loss (torch.Tensor): The Vector Quantization loss.
        recon_loss (torch.Tensor): The reconstruction loss.
        step (int): The current training step.
        is_training (bool): Whether the model is in training mode. Default is True.

    Returns:
        None
    """
    if not exists(accelerator) or not accelerator.is_main_process:
        return
    mode = "Train" if is_training else "Validation"
    accelerator.log({f"{mode} VQ loss": vq_loss.item()}, step=step)
    accelerator.log({f"{mode} reconstruction loss": recon_loss.item()}, step=step)

    return


class Tokenizer(STViViT):
    """
    A neural network module for tokenizing video data.

    This class implements a tokenizer that can encode video frames into discrete tokens
    and decode them back to video frames. It uses a combination of spatial and temporal
    transformers along with vector quantization.

    Attributes:
        wandb_mode (str): The mode for Weights & Biases logging.
        force_cpu (bool): Whether to force CPU usage instead of GPU.
        vq_loss_w (float): The weight for the Vector Quantization loss.
        recon_loss_w (float): The weight for the reconstruction loss.
        gp_weight (float): The weight for gradient penalty.
        image_size (tuple): The size of the input images (height, width).
        patch_size (tuple): The size of the patches (height, width).
        temporal_patch_size (int): The size of temporal patches.
        spatial_rel_pos_bias (ContinuousPositionBias): The spatial relative position bias.
        to_patch_emb_first_frame (nn.Sequential): Embedding layer for the first frame.
        to_patch_emb (nn.Sequential): Embedding layer for the rest of the frames.
        encoder (STTransformer): The encoder transformer.
        vq (VectorQuantize): The Vector Quantization layer.
        decoder (STTransformer): The decoder transformer.
        to_pixels_first_frame (nn.Sequential): Layer to convert tokens to pixels for the first frame.
        to_pixels (nn.Sequential): Layer to convert tokens to pixels for the rest of the frames.
        config (dict): Configuration dictionary containing all initialization parameters.
    """

    def __init__(
        self,
        *,
        dim=512,
        codebook_size=1024,
        image_size=64,
        patch_size=4,
        temporal_patch_size=1,
        num_blocks=8,
        wandb_mode="disabled",
        codebook_dim=32,
        dim_head=64,
        heads=8,
        channels=3,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4.0,
        vq_loss_w=1.0,
        recon_loss_w=1.0,
        enable_decoder=True,
        train_decoder_only=False,
    ):
        """
        Initializes the Tokenizer.

        Args:
            dim (int): The dimension of the model. Default is 512.
            codebook_size (int): The size of the codebook for vector quantization. Default is 1024.
            image_size (int or tuple): The size of the input images. Default is 64.
            patch_size (int or tuple): The size of the patches. Default is 4.
            temporal_patch_size (int): The size of temporal patches. Default is 1.
            num_blocks (int): The number of transformer blocks. Default is 8.
            wandb_mode (str): The mode for Weights & Biases logging. Default is "disabled".
            codebook_dim (int): The dimension of the codebook. Default is 32.
            dim_head (int): The dimension of each attention head. Default is 64.
            heads (int): The number of attention heads. Default is 8.
            channels (int): The number of input channels. Default is 3.
            attn_dropout (float): The dropout rate for attention layers. Default is 0.0.
            ff_dropout (float): The dropout rate for feedforward layers. Default is 0.0.
            ff_mult (float): The multiplier for the feedforward dimension. Default is 4.0.
            vq_loss_w (float): The weight for the Vector Quantization loss. Default is 1.0.
            recon_loss_w (float): The weight for the reconstruction loss. Default is 1.0.
        """
        super().__init__(
            dim=dim,
            codebook_size=codebook_size,
            image_size=image_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            num_blocks=num_blocks,
            wandb_mode=wandb_mode,
            codebook_dim=codebook_dim,
            dim_head=dim_head,
            heads=heads,
            channels=channels,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult,
            vq_loss_w=vq_loss_w,
            recon_loss_w=recon_loss_w,
            enable_decoder=enable_decoder,
        )

        self.train_decoder_only = train_decoder_only

    def calculate_video_token_mask(self, videos, video_frame_mask):
        """
        Calculate a mask for video tokens based on the input video and frame mask.

        Args:
            videos (torch.Tensor): Input video tensor.
            video_frame_mask (torch.Tensor): Mask indicating valid frames in the video.

        Returns:
            torch.Tensor: Mask for video tokens.
        """
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        # Ensure the number of frames (minus the first frame) is divisible by temporal patch size
        assert torch.all(
            ((video_frame_mask.sum(dim=-1) - 1) % self.temporal_patch_size) == 0
        ), (
            "number of frames must be divisible by temporal patch size, subtracting off the first frame"
        )

        # Split mask into first frame and rest of the frames
        first_frame_mask, rest_frame_mask = (
            video_frame_mask[:, :1],
            video_frame_mask[:, 1:],
        )

        # Reshape rest frame mask to group by temporal patch size
        rest_vq_mask = rearrange(
            rest_frame_mask, "b (f p) -> b f p", p=self.temporal_patch_size
        )

        # Combine first frame mask with the rest, considering any valid frame in temporal patch
        video_mask = torch.cat((first_frame_mask, rest_vq_mask.any(dim=-1)), dim=-1)

        # Repeat mask for each spatial patch
        return repeat(video_mask, "b f -> b (f hw)", hw=(h // ph) * (w // pw))

    def get_video_patch_shape(self, num_frames, num_first_frames=1):
        """
        Calculate the shape of video patches.

        Args:
            num_frames (int): Total number of frames in the video.
            num_first_frames (int): Number of frames to be treated separately (default: 1).

        Returns:
            tuple: Shape of video patches (frames, height, width).
        """
        patch_frames = 0

        # Handle first frames separately
        if num_first_frames > 0:
            num_frames -= num_first_frames
            patch_frames += num_first_frames

        # Calculate remaining patch frames
        patch_frames += num_frames // self.temporal_patch_size

        return (patch_frames, *self.patch_height_width)

    @property
    def image_num_tokens(self):
        """
        Calculate the number of tokens in a single image.

        Returns:
            int: Number of tokens in an image.
        """
        return int(self.image_size[0] / self.patch_size[0]) * int(
            self.image_size[1] / self.patch_size[1]
        )

    def frames_per_num_tokens(self, num_tokens):
        """
        Calculate the number of frames represented by a given number of tokens.

        Args:
            num_tokens (int): Number of tokens.

        Returns:
            int: Number of frames represented by the tokens.
        """
        tokens_per_frame = self.image_num_tokens

        assert (num_tokens % tokens_per_frame) == 0, (
            f"number of tokens must be divisible by number of tokens per frame {tokens_per_frame}"
        )
        assert num_tokens > 0

        pseudo_frames = num_tokens // tokens_per_frame
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    def num_tokens_per_frames(self, num_frames, num_first_frames=1):
        """
        Calculate the number of tokens needed to represent a given number of frames.

        Args:
            num_frames (int): Number of frames.
            num_first_frames (int): Number of frames to be treated separately (default: 1).

        Returns:
            int: Number of tokens needed to represent the frames.
        """
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        # Handle first frames separately
        if num_first_frames > 0:
            num_frames -= num_first_frames
            total_tokens += num_first_frames * image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        # Calculate tokens for remaining frames
        return (
            total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens
        )

    def forward(
        self,
        videos=None,
        mask=None,
        return_recons=False,
        return_recons_only=False,
        return_only_codebook_ids=False,
        accelerator_tracker=None,
        step=0,
        log_every=50,
        **kwargs,
    ):
        """
        Forward pass of the Tokenizer model.

        Args:
            videos (torch.Tensor): Input video tensor.
            mask (torch.Tensor): Mask for variable-length videos.
            return_recons (bool): Whether to return reconstructions.
            return_recons_only (bool): Whether to return only reconstructions.
            return_only_codebook_ids (bool): Whether to return only codebook indices.
            accelerator_tracker: Accelerator for logging.
            step (int): Current step for logging.
            log_every (int): Frequency of logging.

        Returns:
            torch.Tensor or tuple: Loss or tuple of (loss, reconstructions).
        """
        assert videos is not None, "video must be provided"
        assert videos.ndim == 5

        b, c, f, *image_dims, device = *videos.shape, videos.device

        # Validate input dimensions
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f
        assert divisible_by(f - 1, self.temporal_patch_size), (
            f"number of frames ({f}) minus one ({f - 1}) must be divisible by temporal patch size ({self.temporal_patch_size})"
        )

        # Split video into first frame and rest frames
        first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]

        # Embed patches
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb(rest_frames)

        # Concatenate tokens
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)

        shape = tokens.shape
        *_, h, w, _ = shape

        # Encode tokens
        tokens = self.encode(tokens)

        # Quantize
        tokens, packed_fhw_shape = pack([tokens], "b * d")

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(videos, mask)
        tokens, indices, vq_loss = self.vq(tokens, mask=vq_mask)

        if self.train_decoder_only:
            tokens = tokens.detach()

        if return_only_codebook_ids:
            (indices,) = unpack(indices, packed_fhw_shape, "b *")
            return indices

        tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=h, w=w)

        # Decode tokens
        recon_video = self.decode(tokens)

        if return_recons_only:
            returned_recon = recon_video
            return returned_recon

        # Compute losses
        if exists(mask):
            # Variable-length video / images training
            recon_loss = F.mse_loss(videos, recon_video, reduction="none")
            recon_loss = recon_loss[repeat(mask, "b t -> b c t", c=c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(videos, recon_video)

        # Combine losses
        if self.train_decoder_only:
            loss = self.recon_loss_w * recon_loss
        else:
            loss = self.vq_loss_w * vq_loss + self.recon_loss_w * recon_loss

        # Log losses if needed
        if self.wandb_mode != "disabled" and step % log_every == 0:
            log_wandb_all_losses(
                accelerator_tracker,
                vq_loss,
                recon_loss,
                step,
                self.training,
            )

        if return_recons:
            returned_recon = recon_video
            return loss, returned_recon

        return loss

    def forward_dual_codebook(
        self,
        videos,
        small_vq,
        return_only_codebook_ids=False,
        return_recons_only=False,
    ):
        """Forward pass using dual codebooks: self.vq (big, frozen) for frame 0,
        small_vq for frames 1+.

        Used by analysis scripts (small_vq passed as argument) and internally by
        DualCodebookTokenizer.forward() (which passes self.small_vq).

        Args:
            videos: (B, C, T, H, W)
            small_vq: VectorQuantize instance for frames 1+
            return_only_codebook_ids: if True, return (B, pt, ph, pw) concatenated indices
            return_recons_only: if True, return reconstructed video tensor

        Returns:
            indices (B, pt, ph, pw), or recon video, or (small_vq_loss, recon_loss)
        """
        assert videos is not None and videos.ndim == 5

        b, c, f, *image_dims = videos.shape
        h, w = self.patch_height_width  # spatial patch grid dims

        # Embed patches
        first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)   # (B, 1, h, w, d)
        rest_frames_tokens = self.to_patch_emb(rest_frames)               # (B, t_rest, h, w, d)
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)  # (B, t, h, w, d)

        # Encode (output still (B, t, h, w, d))
        tokens = self.encode(tokens)

        # Split into frame-0 and rest
        first_tokens = tokens[:, :1, :, :, :]   # (B, 1, h, w, d)
        rest_tokens  = tokens[:, 1:, :, :, :]   # (B, t_rest, h, w, d)

        # Optional delta-reference: subtract a reference from rest features
        # before quantisation so the small VQ's input is motion-only by
        # construction. The reference is added back before decoding so the
        # decoder sees full features.
        delta_ref = getattr(self, "delta_ref", "none")
        if delta_ref == "anchor":
            # subtract frame-0 features from every motion frame
            rest_reference = first_tokens.expand_as(rest_tokens)
            rest_tokens = rest_tokens - rest_reference
        elif delta_ref == "rolling":
            # subtract previous frame's features: frame t's ref is frame t-1
            # (for t_rest frames indexed 0..t_rest-1, ref is the preceding
            #  frame; for the first rest frame, its ref is frame 0)
            prev_tokens = torch.cat(
                (first_tokens, rest_tokens[:, :-1, :, :, :]), dim=1
            )  # (B, t_rest, h, w, d) — ref_t = original rest_token at t-1 or first
            rest_reference = prev_tokens
            rest_tokens = rest_tokens - rest_reference
        else:
            rest_reference = None  # no-op; small_vq quantises raw features

        # Flatten spatial+temporal for VQ
        first_flat, ps_first = pack([first_tokens], "b * d")   # (B, h*w, d)
        rest_flat,  ps_rest  = pack([rest_tokens],  "b * d")   # (B, t_rest*h*w, d)

        # Quantize frame 0 with big VQ (self.vq)
        first_q, first_indices, first_vq_loss = self.vq(first_flat)

        # Quantize frames 1+ with small VQ
        rest_q, rest_indices, small_vq_loss = small_vq(rest_flat)

        if return_only_codebook_ids:
            t_rest = (f - 1) // self.temporal_patch_size
            (first_indices,) = unpack(first_indices, ps_first, "b *")
            (rest_indices,)  = unpack(rest_indices,  ps_rest,  "b *")
            first_indices = first_indices.reshape(b, 1,      h, w)
            rest_indices  = rest_indices.reshape( b, t_rest, h, w)
            return torch.cat([first_indices, rest_indices], dim=1)  # (B, pt, h, w)

        # Reconstruct — combine quantized tokens back to (B, t, h, w, d) for decode
        (first_q,) = unpack(first_q, ps_first, "b * d")
        (rest_q,)  = unpack(rest_q,  ps_rest,  "b * d")
        # If delta-mode, rest_q now holds quantised DELTAS. Add the reference
        # back so the decoder sees full features (delta + reference = estimated
        # original features, up to quantisation error on the delta).
        if rest_reference is not None:
            rest_q = rest_q + rest_reference
        tokens_q = torch.cat([first_q, rest_q], dim=1)   # (B, t, h, w, d)
        recon_video = self.decode(tokens_q)

        if return_recons_only:
            return recon_video

        return small_vq_loss, recon_video


class DualCodebookTokenizer(Tokenizer):
    """Tokenizer that uses two separate VQ codebooks:
    - self.vq  (big, 1024 codes): frozen after loading pretrained weights, handles frame 0
    - self.small_vq (small, configurable): trained from scratch, handles frames 1+

    The decoder and to_pixels layers are trained to adapt to the mixed quantisation.
    Encoder, patch embeddings, and big VQ stay frozen.

    Usage in training:
        model = DualCodebookTokenizer(small_codebook_size=16, ...)
        # load pretrained big tokenizer weights (strict=False, small_vq keys are ignored)
        model.load_state_dict(big_ckpt, strict=False)
        model.freeze_big_components()
        # optimise only model.trainable_parameters()
    """

    def __init__(self, *, small_codebook_size=16, codebook_dim=32,
                 delta_ref: str = "none", **kwargs):
        super().__init__(codebook_dim=codebook_dim, **kwargs)
        # Codebook-health settings (Fix A): EMA updates enable dead-code
        # resampling, and k-means init spreads the initial codes over the
        # actual feature distribution. Together these prevent the collapse
        # observed in run 1528367 where 95.6% of patches picked a single code.
        self.small_vq = VectorQuantize(
            dim=kwargs["dim"],
            codebook_size=small_codebook_size,
            codebook_dim=codebook_dim,
            use_cosine_sim=True,
            commitment_weight=0.25,
            ema_update=True,
            kmeans_init=True,
            threshold_ema_dead_code=2,
        )
        # Delta-reference mode: if "none" (default), small_vq quantises
        # post-encoder features directly. If "anchor", small_vq quantises
        # (feat_t - feat_0) so the input is "change from frame-0 anchor" by
        # construction. If "rolling", small_vq quantises (feat_t - feat_{t-1})
        # for instantaneous motion. The decoder still reconstructs the full
        # frame by combining first_q with (rest_q + reference features) via
        # its learned layers.
        if delta_ref not in ("none", "anchor", "rolling"):
            raise ValueError(
                f"delta_ref must be one of 'none', 'anchor', 'rolling'; got {delta_ref!r}"
            )
        self.delta_ref = delta_ref
        self.config["small_codebook_size"] = small_codebook_size
        self.config["delta_ref"] = delta_ref

    def encode_rest_features(self, videos):
        """Run the encoder path and return post-encoder pre-VQ features for
        frames 1+ only. Shape (B, T-1, h, w, d). Used by the invariance
        auxiliary loss, which needs to compare encoded features for the
        original and colour-jittered versions of the same window.
        """
        assert videos is not None and videos.ndim == 5
        first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb(rest_frames)
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)
        tokens = self.encode(tokens)
        return tokens[:, 1:, :, :, :]

    def codebook_logits(self, features):
        """Compute cosine-similarity logits against the small VQ codebook.

        features: (..., d)  →  logits: (..., K).
        Uses the same project_in + cosine-sim convention as the small_vq
        forward pass. Differentiable through the codebook, not through the
        encoder features (the encoder is frozen).
        """
        import torch.nn.functional as F
        x = self.small_vq.project_in(features)
        # Codebook stored in self.small_vq._codebook.embed, shape (1, K, d_cb)
        codebook = self.small_vq._codebook.embed.squeeze(0)
        x = F.normalize(x, dim=-1)
        cb = F.normalize(codebook, dim=-1)
        logits = x @ cb.T
        return logits

    def freeze_big_components(self):
        """Freeze encoder, patch embeddings, spatial pos bias, and big VQ.

        Call this after loading pretrained weights so only the decoder, to_pixels
        layers, and small_vq receive gradient updates.
        """
        for module in (
            self.to_patch_emb_first_frame,
            self.to_patch_emb,
            self.encoder,
            self.vq,
            self.spatial_rel_pos_bias,
        ):
            for p in module.parameters():
                p.requires_grad_(False)

    def trainable_parameters(self):
        """Return only the parameters that should be optimised."""
        return (
            list(self.decoder.parameters())
            + list(self.to_pixels_first_frame.parameters())
            + list(self.to_pixels.parameters())
            + list(self.small_vq.parameters())
        )

    def forward(
        self,
        videos=None,
        mask=None,
        return_recons=False,
        return_recons_only=False,
        return_only_codebook_ids=False,
        accelerator_tracker=None,
        step=0,
        log_every=50,
        **kwargs,
    ):
        assert videos is not None and videos.ndim == 5

        if return_only_codebook_ids or return_recons_only:
            return self.forward_dual_codebook(
                videos,
                self.small_vq,
                return_only_codebook_ids=return_only_codebook_ids,
                return_recons_only=return_recons_only,
            )

        small_vq_loss, recon_video = self.forward_dual_codebook(videos, self.small_vq)

        b, c = videos.shape[:2]
        if exists(mask):
            recon_loss = F.mse_loss(videos, recon_video, reduction="none")
            recon_loss = recon_loss[repeat(mask, "b t -> b c t", c=c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(videos, recon_video)

        loss = self.vq_loss_w * small_vq_loss + self.recon_loss_w * recon_loss

        if self.wandb_mode != "disabled" and step % log_every == 0:
            log_wandb_all_losses(
                accelerator_tracker,
                small_vq_loss,
                recon_loss,
                step,
                self.training,
            )

        if return_recons:
            return loss, recon_video

        return loss
