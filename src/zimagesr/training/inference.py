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
    vae_sf: float,
    cap_feats_2d: torch.Tensor,
    sr_scale: float = 1.0,
    refine_steps: int = 1,
) -> "Image.Image":  # noqa: F821  — PIL lazy import
    """Run super-resolution inference with optional multi-step refinement.

    Args:
        transformer: LoRA-wrapped transformer (in eval mode).
        vae: VAE decoder.
        lr_latent: ``(1, C, H, W)`` degraded latent.
        tl: Truncation level TL.
        vae_sf: VAE ``scaling_factor``.
        cap_feats_2d: ``(seq_len, cap_dim)`` null caption features.
        sr_scale: Inference correction scale for ``v(TL)``.
        refine_steps: Number of Euler integration steps from ``t=TL`` to ``0``.
            ``1`` reproduces the original one-step update.

    Returns:
        PIL Image of the super-resolved output.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    device = lr_latent.device
    dtype = lr_latent.dtype

    if refine_steps < 1:
        raise ValueError(f"refine_steps must be >= 1, got {refine_steps}")

    bsz = int(lr_latent.shape[0])
    z = lr_latent
    t_edges = torch.linspace(tl, 0.0, steps=refine_steps + 1, device=device, dtype=dtype)
    for idx in range(refine_steps):
        t_now = t_edges[idx]
        dt = t_now - t_edges[idx + 1]
        t_batch = torch.full((bsz,), t_now, device=device, dtype=dtype)
        dt_bc = torch.full((1, 1, 1, 1), dt, device=device, dtype=dtype)
        v = call_transformer(
            transformer,
            latents=z,
            timestep=t_batch,
            cap_feats_2d=cap_feats_2d,
        )
        z = z - (sr_scale * v) * dt_bc
    z0_hat = z

    autocast_dt = torch.bfloat16 if device.type == "cuda" else None
    pixels = vae_decode_to_pixels(vae, z0_hat, vae_sf, autocast_dtype=autocast_dt)

    img_tensor = pixels[0].clamp(0, 1).float().cpu()
    return TF.to_pil_image(img_tensor)
