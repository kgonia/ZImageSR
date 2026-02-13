from __future__ import annotations

import torch
import torch.nn as nn

from zimagesr.training.transformer_utils import call_transformer, vae_decode_to_pixels


@torch.no_grad()
def one_step_sr(
    transformer: nn.Module,
    vae: nn.Module,
    lr_latent: torch.Tensor,
    tl: float,
    t_scale: float,
    vae_sf: float,
    cap_feats_2d: torch.Tensor,
    sr_scale: float = 1.0,
) -> "Image.Image":  # noqa: F821  — PIL lazy import
    """Run one-step super-resolution inference.

    Args:
        transformer: LoRA-wrapped transformer (in eval mode).
        vae: VAE decoder.
        lr_latent: ``(1, C, H, W)`` degraded latent.
        tl: Truncation level TL.
        t_scale: Transformer's ``t_scale`` config value (usually 1000.0).
        vae_sf: VAE ``scaling_factor``.
        cap_feats_2d: ``(seq_len, cap_dim)`` null caption features.
        sr_scale: Inference correction scale for ``v(TL)``.

    Returns:
        PIL Image of the super-resolved output.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    device = lr_latent.device
    dtype = lr_latent.dtype

    TL_t = torch.tensor([tl * t_scale], device=device, dtype=dtype)
    TL_bc = torch.tensor([tl], device=device, dtype=dtype).view(1, 1, 1, 1)

    v = call_transformer(
        transformer,
        latents=lr_latent,
        timestep=TL_t,
        cap_feats_2d=cap_feats_2d,
    )
    z0_hat = lr_latent - (sr_scale * v) * TL_bc

    autocast_dt = torch.bfloat16 if device.type == "cuda" else None
    pixels = vae_decode_to_pixels(vae, z0_hat, vae_sf, autocast_dtype=autocast_dt)

    img_tensor = pixels[0].clamp(0, 1).float().cpu()
    return TF.to_pil_image(img_tensor)
